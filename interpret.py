import sys
import re
import ctypes
import inspect
import operator
from contextlib import contextmanager
from pycparser import c_generator, c_ast, c_lexer, c_parser, preprocess_file

from continuation import Continuation

from inspect import signature, getsourcelines, getmembers


import c_utils

# TODO: check results for overflow???
binops = {
    '+': lambda type_: (type_, operator.add),
    '-': lambda type_: (type_, operator.sub),
    '*': lambda type_: (type_, operator.mul),
    # TODO: division by zero runtime error
    '/': lambda type_: (type_, (operator.truediv if is_float_type(type_) else operator.floordiv)),
    # TODO: assert something about type_ being int,
    '%': lambda type_: (type_, operator.mod),

    '|': lambda type_: (type_, operator.or_),
    '&': lambda type_: (type_, operator.and_),
    '^': lambda type_: (type_, operator.xor),
    '<<': lambda type_: (type_, operator.lshift),
    # TODO: something with unsigned right shift??,
    '>>': lambda type_: (type_, operator.rshift),

    # TODO: any issues with falsiness?,
    # TODO: what type does a boolean comparison return?,
    # TODO: these are returning True instead of 1
    '==': lambda type_: (['int'], operator.eq),
    '!=': lambda type_: (['int'], operator.ne),
    '<': lambda type_: (['int'], operator.lt),
    '<=': lambda type_: (['int'], operator.le),
    '>': lambda type_: (['int'], operator.gt),
    '>=': lambda type_: (['int'], operator.ge),

    # && and || are special, because they can short circuit. By the time we call this function, short-circuiting
    # should already have been checked for
    '&&': lambda type_: (['int'], lambda lval, rval: 1 if rval else 0),
    '||': lambda type_: (['int'], lambda lval, rval: 1 if rval else 0)
}

def assert_false(str_ = ''):
    assert False, str_

unops = {
   '+': lambda: operator.pos,
   '-': lambda: operator.neg,
   '~': lambda: operator.inv,
   '!': lambda: operator.not_,
   'sizeof': lambda: assert_false("TODO")
   }

# shouldn't call these directly, since they require accessing memory/locals
for k in ['*', '&', 'p++', '++p', 'p--', '--p']:
    unops[k] = lambda _: assert_false("Shouldn't call these unops")

python_type_map = {
    '_Bool': ctypes.c_bool,
    'char': ctypes.c_char,
    'unsigned char': ctypes.c_bool,
    'short': ctypes.c_bool,
    'unsigned short': ctypes.c_bool,
    'int': ctypes.c_bool,
    'unsigned int': ctypes.c_bool,
    'long': ctypes.c_bool,
    'unsigned long': ctypes.c_bool,
    'long long': ctypes.c_bool,
    'unsigned long long': ctypes.c_bool,
    'float': ctypes.c_bool,
    'double': ctypes.c_bool,
    'long double': ctypes.c_bool,
}

int_types = set(['_Bool'])
for t in ['char', 'short', 'int', 'long', 'long long']:
    int_types.add(t)
    int_types.add('unsigned ' + t)
float_types = set(['float', 'double', 'long double'])

builtin_funcs = { name: func for name, func in inspect.getmembers(c_utils, inspect.isfunction) }

# TODO: this is wildly incomplete?
def cast_to_c_val(val):
    if is_pointer_type(val.type) or is_func_type(val.type):
        # TODO: use the other types from https://docs.python.org/3/library/ctypes.html?
        return val
    if val.type[0] in python_type_map:
        assert len(type_) == 1, "This should always be true, right??"
        return python_type_map[val.type[0]](val.value).value
    assert False, val.type

def cast_to_python_val(val):
    if is_int_type(val.expanded_type):
        return int(val.value)
    elif is_float_type(val.expanded_type):
        return float(val.value)
    elif is_string_type(val.expanded_type) or is_pointer_type(val.expanded_type):
        return val.value
    assert False, val.expanded_type

def expand_type(type_, typedefs):
    if type_ == []: return
    if is_func_type(type_):
        return (type_[0], expand_type(type_[1], typedefs), type_[2], expand_type(type_[3], typedefs))
    elif isinstance(type_, list):
        ret = [expand_type(t, typedefs) for t in type_]
        if isinstance(ret[0], list):
            ret = [item for sublist in ret[0] for item in sublist]
        return ret
    elif type_ in typedefs:
        return typedefs[type_]
    else:
        # TODO: how to handle this? Could still pass a continuation, but only to help with this error?
        #my_assert(type_ in python_type_map or type_.startswith('[') or type_ == '*' or type_ == '...', 'Invalid type')
        return type_


def is_func_type(type_):
    #return type_[0].startswith('(')
    return isinstance(type_, tuple)

def is_float_type(type_):
    return len(type_) == 1 and type_[0] in float_types

def is_int_type(type_):
    return len(type_) == 1 and type_[0] in int_types

def is_pointer_type(type_):
    if isinstance(type_, Val):
        type_ = type_.expanded_type
    return type_[0].startswith('[') or type_[0] == '*'

def is_string_type(type_):
    return len(type_) == 2 and is_pointer_type(type_[0]) and type_[1] == 'char'

def types_match(type1, type2):
    return type1 == type2

def is_valid_return_value(functype, val):
    return types_match(functype[0][1], val.type)

def is_void_function(type_):
    return type_[0][1] == 'void'

# TODO: cache results from particularly common subtrees?
# TODO: asserts shouldn't really be asserts, since then broken student code will crash this code
# TODO: handle sizeof
# can't inherit from genericvisitor, since we need to pass continuations everywhere

class Address(object):
    __slots__ = ['base', 'offset']
    def __init__(self, base, offset):
        self.base = base
        self.offset = offset

    def __str__(self):
        return 'Address(' + str(self.base) + ', ' + str(self.offset) + ')'
        return str(self)

    # TODO: is __str__ necessary at this point? will str() automatically call repr if __str__ doesn't exist??
    def __repr__(self):
        return str(self)

class Flow(object):
    __slots__ = ['type', 'value']
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value

    def __str__(self):
        return 'Flow(' + str(self.type) + ', ' + str(self.value) + ')'

    def __repr__(self):
        return str(self)

class Memory(object):
    __slots__ = ['type', 'expanded_type', 'name', 'len', 'array', 'segment']
    def __init__(self, type_, expanded_type, name, len_, array, segment):
        self.type = type_
        self.expanded_type = expanded_type
        self.name = name
        self.len = len_
        self.array = array
        self.segment = segment


class Val(object):
    __slots__ = ['type', 'expanded_type', 'value']
    def __init__(self, type_, expanded_type, value):
        self.type = type_
        self.value = value
        self.expanded_type = expanded_type

    def __str__(self):
        return 'Val(' + str(self.type) + ', ' + str(self.value) + ')'

    def __repr__(self):
        return str(self)

# make this a subclass, so the Continuation module doesn't need to know anything about
# Interpreters
class InterpContinuation(Continuation):

    __slots__ = ['interpreter']
    def __init__(self, interpreter):
        self.interpreter = interpreter
        super().__init__()

    # These are all just convenience functions for common interpreter ops

    def visit(self, node):
        assert isinstance(node, c_ast.Node)
        self.apply(lambda: self.interpreter.visit(node))
        return self

    def push_context(self, context):
        self.apply(lambda: self.interpreter.push_context(context))
        return self

    def pop_context(self):
        # the nice thing about wrapping things in a lambda instead of passing
        # self.interpreter.pop_context directly is that now we know where the
        # lambda was defined. Not particularly interesting in this case, though.

        # TODO: provide more information about the caller, so that we have better
        # line numbers for the lambda printing? Almost like inlining...
        self.apply(lambda: self.interpreter.pop_context())
        return self

    def push_scope(self):
        self.apply(lambda: self.interpreter.push_scope())
        return self

    def pop_scope(self):
        self.apply(lambda: self.interpreter.pop_scope())
        return self

    # TODO: extend passthrough so we add in where the value was generated to each Val that's
    # passed through?
    # okay I actually did this, but it's not helpful for things pulled from memory. Remember in
    # memory the origin of a value??

class Interpreter(object):
    def __init__(self, ast, require_decls=True):
        self.require_decls = require_decls

        self.ast = ast
        self.k = InterpContinuation(self)
        self.rk = InterpContinuation(self)

        self.k

        self.stdin = None
        self.stdout = ''
        self.stderr = ''
        self.filesystem = {}

        self.scope = [[{}]]
        self.context = ['global']

        # values are tuples of type, num_elements, and array of elements
        self.memory = {}
        self.string_constants = {}

        # TODO: remove this hard-coding
        self.typedefs = {'string': ['*', 'char'], 'size_t': ['int']}
        self.memory_init('NULL', 'void', 0, [], 'NULL')

        # handle reading in all the typedefs, funcdefs, and everything, before we try to load in builtins (which may use
        # typedefs)
        self.visit(ast)
        self.run()

        for name in builtin_funcs:
            type_, func = builtin_funcs[name](self)
            val = self.make_val(type_, name)
            self.update_scope(name, val)
            self.memory_init(name, type_, 1,
                    [(lambda func: lambda args: self.k.passthrough(lambda: func(args)))(func)], 'text')
        self.run()

    def step(self):
        return self.k.step()

    def run(self):
        while True:
            ret = self.step()
            if ret is not None:
                pass
                #if isinstance(ret, c_ast.Node):
                #    ret.show(showcoord=True)
                #else:
                #    print(ret)
            else:
                break

    def make_val(self, type_, val):
        # TODO: associate the expanded_type with types_ instead of with vals?
        assert isinstance(type_, list)
        expanded_type = expand_type(type_, self.typedefs)
        return Val(type_, expanded_type, val)

    # TODO: this isn't nearly completely enough. for example, upcast int to long long
    def implicit_cast(self, val1, val2):
        self.k.info('implicit_cast')
        if is_int_type(val1.type) and is_float_type(val2.type):
            val1.value=float(val1.value)
            val1.type=val2.type
        elif is_float_type(val1.type) and is_int_type(val2.type):
            val2.value=float(val2.value)
            val2.type=val1.type
        # can only compare against NULL pointer if it's an int
        elif is_int_type(val1.type) and is_pointer_type(val2.type):
            self.k.kassert((lambda val: lambda: val == 0)(val1.value), 'blah8 ' + str(val1.value))
            val1.value = Address('NULL', 0)
            val1.type = val2.type
        elif is_int_type(val2.type) and is_pointer_type(val1.type):
            self.k.kassert((lambda val: lambda: val == 0)(val2.value), 'blah9 ' + str(val2.value))
            val2.value = Address('NULL', 0)
            val2.type = val1.type
        self.k.passthrough(lambda: val1).passthrough(lambda: val2)

    def pop_context(self):
        self.context.pop()

    def push_context(self, context):
        self.context.append(context)

    def pop_func_scope(self):
        self.k.info('pop func scope').apply(lambda: self.scope.pop())

    def push_func_scope(self):
        # append the global map
        self.k.info('push func scope').apply(lambda: self.scope.append([shelf.scope[0][0]]))

    def pop_scope(self):
        self.k.info('pop scope').apply(lambda: self.scope[-1].pop())

    def push_scope(self):
        self.k.info('push scope').apply(lambda: self.scope[-1].append({}))

    def update_scope(self, id_, val):

        self.scope[-1][-1][id_] = val

        self.k.info(('update_scope', id_, val))


    def update_memory(self, base, offset, val):
        # TODO: check types
        self.memory[base].array[offset] = val.value
        self.k.info(('memory-update', base, offset, val.value))

    def memory_init(self, name, type_, len_, array, segment):
        expanded_type = expand_type(type_, self.typedefs)
        self.memory[name] = Memory(type_, expanded_type, name, len_, array, segment)
        (self.k
        .info(('memory-init', name, type_, len_, array, segment)))
        #.passthrough(self.memory[name]))

    #def update_memory(self, arr, val):
    #    (self.k
    #    .info((name, self.memory))
    #    .passthrough(lambda: self.memory[arr].seti))

    def handle_string_const(self, type_):
        # TODO: check how it's being declared, since we might be doing a mutable character array
        self.k.info('strconst')

        if n.value in self.string_constants:
            name = self.string_constants[n.value]
        else:
            const_num = len(self.string_constants)
            # should be unique in memory, since it has a space in the name
            name = 'strconst ' + str(const_num)
            self.string_constants[n.value] = name
            # append 0 for the \0 character
            array = bytes(n.value, 'latin-1') + bytes([0])
            self.memory_init(name, type_, len(array), array, 'rodata')
        self.k.passthrough(lambda: self.make_val(type_, Address(name, 0)))

    # executes a program from the main function
    # shouldn't call this multiple times, since memory might be screwed up (global values not reinitialized, etc.)
    # TODO: _start??
    def setup_main(self, argv, stdin):

        # TODO: need to reset global variables to initial values as well
        #for i in list(self.memory.keys()):
        #    if self.memory[i]['segment'] in ['heap', 'stack', 'argv']:
        #        del(self.memory[i])

        # TODO: validate argv

        # TODO: no matching pop_scope. Return could do that, but not all functions call
        # return. Make this an explicit call to a "FuncCall" node?
        self.push_scope()
        id_map = self.scope[-1][-1]

        id_map['argc'] = self.make_val(['int'], len(argv) + 1)

        # this is the name of the spot in self.memory
        # the second index is where in the array this id references
        id_map['argv'] = self.make_val(['*', '*', 'char'], Address('argv', 0))

        # TODO: environment variables as well?
        self.memory_init('argv', ['*', 'char'], len(argv) + 1,
            [Address('argv[' + str(i) + ']', 0) for i in range(len(argv))] + [('NULL', 0)], 'argv')

        for i in range(len(argv)):
            array = bytearray(argv[i], 'latin-1') + bytearray([0])
            self.memory_init('argv[' + str(i) + ']', ['char'], len(array), array, 'argv')
            #for j in range(len(argv[i])):
            #    self.memory['argv[' + str(i) + '][' + str(j) + ']'] = ('char', 1, [argv[i][j]])


        self.stdout = ''
        self.stderr = ''
        self.stdin = stdin

        self.k.visit(self.memory['main'].array[0])

    def visit(self, node):
        #node.show(showcoord=True)
        method = 'visit_' + node.__class__.__name__
        ret = getattr(self, method)(node)
        assert ret is None

    def visit_ArrayRef(self, n):
        # TODO: shouldn't be allowed to assign to symbols / consts? Actually, symbols would fall under visit_ID
        # TODO: nested array refs?? argv[1][2]. need to handle dims appropriately

        (self.k.info(n)
        # we want to get rid of any lvalue context if we are eventually assigning to the array
        .push_context('arrayref')
        .visit(n.name)
        # TODO: we can have .expect take the type, to assert then and there?
        .expect(lambda arr: self.k.visit(n.subscript).expect(lambda idx:
            (self.k.pop_context()
            .kassert(lambda: is_pointer_type(idx) != is_pointer_type(arr), "Only one can be an address"))
            # TODO: this is an example of why everything should be lambdas. we can't actually
            # print idx before this
            .if_(is_pointer_type(idx), lambda: self.k.passthrough(lambda: idx.value).passthrough(lambda: arr.value))
            .else_(lambda: self.k.passthrough(lambda: arr.value).passthrough(lambda: idx.value))))

        .expect(lambda arr: self.k.expect(lambda idx:
            self.k
            .kassert(lambda: idx + arr.offset < self.memory[arr.base].len,
                'Out of bounds array:{} idx:{} length:{}'.format(arr.base, idx, self.memory[arr.base].len))
            .if_(self.context[-1] == 'lvalue',
                lambda: self.k
                .kassert(lambda: self.memory[arr.base].segment not in ['rodata', 'text', 'NULL'],
                         self.memory[arr.base].segment)
                .passthrough(lambda: lambda val:
                    self.update_memory(arr.base, idx+arr.offset, val)))
            .else_(lambda: self.k
                .passthrough(lambda: self.make_val(self.memory[arr.base].type,
                             self.memory[arr.base].array[idx + arr.offset]))))))

    def visit_Assignment(self, n):
        # TODO: handle others
        assert n.op == '='

        (self.k.info(n)
        .push_context('lvalue')
        .visit(n.lvalue)
        .pop_context()
        .expect(lambda assignment_op: self.k
            .push_context('rvalue')
            .visit(n.rvalue)
            .pop_context()
            .expect(lambda val:
                # assignments return the value of the assignment
                self.k.apply(lambda: assignment_op(val)).passthrough(lambda: val))))

    def visit_Cast(self, n):
        # TODO: validate
        (self.k.info(n)
        .visit(n.to_type)
        .expect(lambda type_: self.k
            .visit(n.expr).expect(lambda val:
                self.k.passthrough(lambda: self.make_val(type_, val.value)))))


    def visit_BinaryOp(self, n):
        assert n.op in binops

        # TODO cast to c type nonsense
        (self.k.info(n)
        .visit(n.left)
        .expect(lambda lval:
            self.k
            .if_(n.op == '&&' and not lval.value, lambda: self.k
                .passthrough(lambda: self.make_val(['_Bool'], 0)))
            .elseif(n.op == '||' and lval.value, lambda: self.k
                .passthrough(lambda: self.make_val(['_Bool'], 1)))
            .else_(lambda: self.k
                .visit(n.right)
                .expect(lambda rval: self.k
                    .apply(lambda: self.implicit_cast(lval, rval))
                    .expect(lambda lval: self.k.expect(lambda rval: self.k
                        .kassert(lambda: n.op != '%' or is_int_type(lval.type), lval.type)
                        .kassert(lambda: n.op != '%' or rval != 0, "Can't mod by zero")
                        .kassert(lambda: n.op != '/' or rval != 0, "Can't divide by zero")
                        .passthrough(lambda: binops[n.op](lval.type))
                        .expect(lambda typeval: self.k.passthrough(
                            lambda: self.make_val(typeval[0], typeval[1](lval.value, rval.value))))))))))


        # TODO: only currently supported pointer math is between a pointer for addition and subtraction,
        # and two pointers for subtraction. Could use like, xor support or something?

        #add_back = None
        #if is_pointer_type(ltype) or is_pointer_type(rtype):
        #    if n.op == '==' or n.op == '!=':
        #        # specifically for NULL handling
        #        if not is_pointer_type(ltype) or not is_pointer_type(rtype):
        #            type_, lval, rval = implicit_cast(ltype, lval, rtype, rval)
        #            rtype = ltype = type_

        #    if is_pointer_type(ltype) and is_pointer_type(rtype):
        #        assert n.op in ['-', '!=', '<', '>', '<=', '>=', '==']
        #        if n.op == '-':
        #            (larr, lval), (rarr, rval), type_ = lval, rval, ltype
        #            assert larr == rarr
        #        else: assert ltype == rtype
        #    else:
        #        assert n.op in ['+', '-']
        #        if is_pointer_type(ltype):
        #            (add_back, lval), type_ = lval, ltype
        #        else:
        #            (add_back, rval), type_ = rval, rtype

        #else:


        #val = op(lval, rval)
        #if add_back is not None:
        #    val = (add_back, val)
        #return type_, val


    def visit_Break(self, n):
        #TODO assert continuation['break'] is not None, 'Break in of invalid context'
        self.k.info(n).passthrough(lambda: Flow('Break'))

    def visit_Compound(self, n):
        # TODO: do we actually ever even care to stop on this node?
        (self.k.info(n)
        .loop_var(n.block_items)
        # TODO: can't use loop, since we need to shortcut
        .loop((lambda stmt:
            self.k
            .visit(stmt)
            .expect(lambda flow:
                self.k
                .if_(isinstance(flow, Flow) and flow.type != 'Normal', lambda: self.k
                    # pass through to the looper to determine whether we should shortcircuit
                    .passthrough(lambda: True).passthrough(lambda: flow))
                .else_(lambda: self.k
                    .passthrough(lambda: False).passthrough(lambda: Flow('Normal'))))), shortcircuit=True)
        .expect(lambda flows:
            self.k.passthrough(lambda: Flow('Normal', None) if isinstance(flows[-1], Val) else flows[-1])))

    def visit_Constant(self, n):
        # TODO: necessary to expand the type??
        self.k.info(n)
        if not is_string_type(n.type):
            self.k.passthrough(lambda: self.make_val([n.type], cast_to_python_val(self.make_val([n.type], n.value))))
        else:
                self.handle_string_const(n.value)

    def visit_Continue(self, n):
        #assert continuations['continue'] is not None, 'Continue in invalid context'
        # TODO: put the loop in scope context
        (self.k.info(n).passthrough(lambda: Flow('Continue')))

    # name: the variable being declared
    # quals: list of qualifiers (const, volatile)
    # funcspec: list function specifiers (i.e. inline in C99)
    # storage: list of storage specifiers (extern, register, etc.)
    # type: declaration type (probably nested with all the modifiers)
    # init: initialization value, or None
    # bitsize: bit field size, or None
    def decl_helper(self, n, type_):
        # TODO: compare n.type against type_ for validity
        # TODO: funcdecls might be declared multiple times?
        # TODO: doesn't return the name
        (self.k.info(n)
        .if_(n.type, lambda: self.k.visit(n.type))
        # The function type should be the next passthrough
        .expect(lambda type_: self.k
            .if_(n.init, lambda: self.k
                .push_context('rvalue')
                .visit(n.init)
                .pop_context()
                # TODO: validate the type
                .expect(lambda val: self.k.apply(lambda:
                    self.update_scope(id_=n.name, val=self.make_val(type_, val.value))))))
        .passthrough(lambda: Flow('Normal')))

    def visit_Decl(self, n):
        # TODO: can n.type ever be None? only for funcdecl?
        if not n.type: return lambda type_: self.decl_helper(n, type_)
        else: self.decl_helper(n, None)


    def visit_DeclList(self, n):
        (self.k.info(n)
        .loop_var(n.decls)
        .loop(lambda decl:
            self.k.visit(decl))
        # do something?
        .expect(lambda flows: None)
        .passthrough(lambda: Flow('Normal')))

    def visit_ExprList(self, n):
        (self.k.info(n)
        .loop_var(n.exprs)
        .loop(lambda expr: self.k.visit(expr)))

    def visit_ID(self, n):
        def helper():
            for i in range(len(self.scope[-1])-1, -1, -1):
                if n.name in self.scope[-1][i]:
                    id_map = self.scope[-1][i]
                    self.k.passthrough(lambda: id_map)
                    return
            else:
                (self.k
                .kassert(lambda: not self.require_decls, 'Undeclared identifier: ' + n.name)
                .passthrough(lambda: self.scope[-1][-1]))

        (self.k.info(n)
        .apply(helper)
        .expect(lambda id_map:
            self.k
            .if_(self.context[-1] == 'lvalue', lambda: self.k
                .passthrough(lambda: lambda val: operator.setitem(id_map, n.name, val)))
            .else_(lambda: self.k
                .kassert(lambda: n.name in id_map, "Undeclared identifier")
                .kassert(lambda: id_map[n.name].value is not None, "Uninitialized variable")
                .passthrough(lambda: id_map[n.name]))))

    def visit_FileAST(self, n):
        # TODO: put global map placement in here??
        (self.k.info(n)
        .loop_var(n.ext)
        .loop(lambda ext: self.k.visit(ext))
        # don't really care about the return
        .expect(lambda exts: None))

    def visit_For(self, n):
        def for_inner():
            self.k.info(n)
            if n.cond: self.k.visit(n.cond)
            else: self.k.passthrough(lambda: self.make_val(['_Bool'], True))

            (self.k.expect(lambda cond: self.k
                # TODO: need to cast types
                .if_(cond.value, lambda: self.k
                    .visit(n.stmt)
                    .expect(lambda flow:
                        (self.k
                        .if_(flow.type == 'Normal' or flow.type == 'Continue', lambda: self.k
                            .visit(n.next)
                            .expect(lambda _: None)
                            .apply(for_inner))
                        .elseif(flow.type == 'Return', lambda: self.k.passthrough(lambda: flow))
                        .else_(lambda: self.k
                            .kassert(lambda: flow.type == 'Break', 'Invalid flow to for loop')
                            .passthrough(lambda: Flow('Normal'))))))
                .else_(lambda: self.k
                    .pop_scope()
                    .passthrough(lambda: Flow('Normal')))))


        # TODO: can we declare a variable in both for (int i) { int i; ??
        self.k.info(n)
        # TODO only do this if n.init exists??
        self.push_scope()
        if n.init:
            self.k.visit(n.init)
            self.k.expect(lambda flow: None)
        self.k.apply(for_inner)

    def visit_FuncCall(self, n):
        # TODO: something about unused return value?
        # TODO: parse args for params / check types
        # TODO: deal with scope!!
        #if self.memory[n.name]['type'][0][0] == '(builtin)':
        # TODO: this needs to be expanded, since we might have weird types in c_utils?

        (self.k.info(n).visit(n.name)
        .expect(lambda name: self.k
            .if_(n.args, lambda: self.k.visit(n.args))
            .else_(lambda: self.k.passthrough(lambda: []))
            .expect(lambda args: self.k
                .if_(self.memory[name.value].type[0][0] == '(builtin)', lambda: self.k
                    .apply(lambda: self.memory[name.value].array[0](args)))
                .else_(lambda: self.k
                    .kassert(lambda: self.memory[name.value].type[0][0] == '(user-defined)', 'blah3')
                    # TODO: make these push_func_scope
                    .apply(self.push_func_scope)
                    .visit(self.memory[name.value].array[0])
                    .apply(self.pop_func_scope)
                    # TODO: check return type
                    .expect(lambda flow:
                        self.k
                        .kassert(lambda: flow.type == 'Return', 'Didn\'t return from FuncCall')
                        .passthrough(lambda: flow.value))))))



    def visit_Typedef(self, n):
        # TODO: something with memory
        self.k.info(n).visit(n.type)#.expect(lambda type_: None)

    def visit_TypeDecl(self, n):
        self.k.info(n).visit(n.type)

    def visit_ArrayDecl(self, n):
        self.k.info(n).visit(n.type)
        if n.dim:
            (self.k
            .visit(n.dim)
            .expect(lambda dim: self.k.passthrough(lambda: str(dim.value))))
        else: self.k.passthrough(lambda: '')
        self.k.expect(lambda type_: self.k.expect(lambda dim:
            self.k.passthrough(lambda: ['[' + dim + ']'] + type_)))


    def visit_IdentifierType(self, n):
        self.k.info(n).passthrough(lambda: n.names)

    def visit_ParamList(self, n):
        # TODO: can param.name be empty?
        (self.k.info(n)
        .loop_var(n.params)
        .loop(lambda param:
            self.k
            .visit(param.type)
            .expect(lambda ptype: self.k.passthrough(lambda: (param.name, ptype)))))

    def visit_FuncDecl(self, n):
        # only the funcdef funcdecl will necessarily have parameter names
        # TODO: this can theoretically appear inside of funcdefs, but we would step over it
        (self.k.info(n).visit(n.type)
        .expect(lambda ret_type: self.k
            .if_(n.args, lambda: self.k.visit(n.args))
            .else_(lambda: self.k.expect(lambda ret_type: self.k.passthrough(lambda: ([], []))))
            .expect(lambda params:
                self.k.passthrough(lambda: [('(user-defined)', ret_type,
                    [param[0] for param in params],
                    [param[1] for param in params])]))))

        # TODO: typedefs should be scoped
        #continuation.passthrough(lambda k:
        #    lambda ret_type: lambda params:
        #    expand_type([('(user-defined)', ret_type, params[0], params[1])], self.typedefs, k))



    def visit_FuncDef(self, n):
        # TODO: check against prior funcdecls?
        (self.k.info(n)
        .visit(n.decl)
        .expect(lambda val:
            self.k.apply(lambda: self.memory_init(n.decl.name, val.type, 1, [n.body], 'text')))
        .passthrough(lambda: Flow('Normal')))


    def visit_If(self, n):
        self.k.info(n)
        if n.cond: self.k.visit(n.cond)
        # TODO: cprrect default?
        else: self.k.passthrough(lambda: self.make_val(['_Bool'], True))

        self.k.expect(lambda cond: self.k
            .if_(cond.value, lambda: self.k.visit(n.iftrue))
            .elseif(n.iffalse, lambda: self.k.visit(n.iffalse))
            # if we didn't execute any branches, we need to
            # generate Flow for this block
            .else_(lambda: self.k.passthrough(lambda: Flow('Normal'))))

    def visit_PtrDecl(self, n):
        (self.k.info(n)
        .visit(n.type)
        .expect(lambda type_: self.k.passthrough(lambda: ['*'] + type_)))

    def visit_Struct(self, n):
        # TODO: finish this
        (self.k.info(n)
        .loop_var(n.decls)
        .loop(lambda decl: self.k.visit(decl)))
        # TODO do something with memory

    def visit_Typename(self, n):
        # TODO: stuff with n.quals
        (self.k.info(n)
        # TODO: don't know when this is not None
        .kassert(lambda: not n.name, 'uh oh')
        .visit(n.type))

    # TODO: account for overflow in binop/unop?
    def visit_UnaryOp(self, n):
        assert n.op in unops

        # TODO: are the semantics of this correct? like, x = ++x
        def inc_dec(val, assignment):
            new_value = val.value + (1 if n.op in ['p++', '++p'] else -1)
            new_value = self.make_val(val.type, new_value)

            (self.k
            .apply(lambda: assignment(new_value))
            .passthrough(lambda: val if n.op in ['p++', 'p--'] else new_value))

        # TODO: use types
        def helper(val):
            if n.op in ['p++', 'p--', '++p', '--p']:
                (self.k
                .push_context('lvalue')
                .visit(n.expr)
                .pop_context()
                .expect(lambda assignment_op:
                    self.k.apply(lambda: inc_dec(val, assignment_op))))
            elif n.op == '*':
                assert False
                #assert is_pointer_type(type_)
                #arr, offset = val
                ## TODO: is this fine as long as we never actually dereference. like, maybe
                ## just for sizeof purposes?
                #assert offset < self.memory[arr].len
                ## TODO: dereference type_
                #return type_[1:], self.memory[arr].base[offset]
            elif n.op == '&':
                assert False
                #assert is_pointer_type(type_)
                #arr, offset = val
                ## TODO: add pointer type, but only if not func pointer/constant array
                #return ['*'] + type_, self.memory[arr].base
            else:
                # TODO: anything else with the type?
                value = unops[n.op](val.value)
                value = self.make_val(val.type, value)
                self.k.passthrough(lambda: value)

        self.k.info(n).visit(n.expr).expect(helper)


    # TODO: detect infinite loop??
    def visit_While(self, n):
        self.k.info(n)

        if n.cond: self.k.visit(n.cond)
        # TODO: is True the right default? can the condition even be empty?
        else: self.k.passthrough(lambda: self.make_val(['_Bool'], True))

        (self.k.expect(lambda cond:
            self.k
            .if_(cond.value, lambda: self.k
                .visit(n.stmt)
                .expect(lambda flow: self.k
                    .kassert(lambda: isinstance(flow, Flow), 'blah5')
                    .if_(flow.type == 'Continue' or flow.type == 'Normal', lambda: self.k.visit(n))
                    .elseif(flow.type == 'Break', lambda: self.k.passthrough(lambda: Flow('Normal')))
                    .else_(lambda: self.k
                        .kassert(lambda: flow.type == 'Return', 'blah6')
                        .passthrough(lambda: flow))))
            .else_(lambda: self.k.passthrough(lambda: Flow('Normal')))))




    def visit_Return(self, n):
        self.k.info(n)
        if n.expr: self.k.visit(n.expr)
        # TODO: we very explicitly want this to pass through to self.k.expect, not to the upper continuation
        else: self.k.passthrough(lambda: None)
        self.k.expect(lambda val: self.k.passthrough(lambda: Flow('Return', val)))

        # TODO: can we even write a return outside of a function?
        # TODO: we need to check this elsewhere
        #assert continuations['return'] is not None, 'Return in invalid context'

def main():
    parser = c_parser.CParser()
    try:
        # TODO: use a canonical set of header files, so we know what to expect?
        #cfile = preprocess_file(sys.argv[1], cpp_path='clang', cpp_args=['-E', '-nostdinc', r'-I../repair/fake_libc_include'])
        cfile = preprocess_file(sys.argv[1], cpp_path='clang', cpp_args=['-E', '-nostdinc', r'-Iclib/build/include'])
        #cpp_args = [r'-D__attribute__(x)=', r'-D__builtin_va_list=int',
                    #r'-D_Noreturn=', r'-Dinline=', r'-D__volatile__=', '-E']
        #cfile = preprocess_file(sys.argv[1], cpp_path='clang', cpp_args=['-E'])
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        sys.exit(1)

    interpret = Interpreter(ast)
    # some kind of JIT after the first execution?
    interpret.setup_main(['./vigenere', 'HELlO'], 'wOrld\n')
    interpret.run()
    assert len(interpret.k.passthroughs) == 1 == 1
    # TODO: why is this passthrough double wrapped in context?
    ret = interpret.k.get_passthrough(0)
    assert len(interpret.k.passthroughs) == 0
    print(ret)
    print(ret.type, ret.value)
    print(interpret.stdout)

if __name__=='__main__':
    main()
