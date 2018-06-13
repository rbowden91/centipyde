import sys
import re
import ctypes
import inspect
import operator
from contextlib import contextmanager
from pycparser import c_generator, c_ast, c_lexer, c_parser, preprocess_file

from inspect import signature


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

unops = {
   '+': lambda type_: (type_, operator.pos),
   '-': lambda type_: (type_, operator.neg),
   '~': lambda type_: (type_, operator.inv),
   '!': lambda type_: (type_, operator.not_),
   'sizeof': lambda type_: (type_, 'TODO'),

   # shouldn't call these directly, since they require accessing memory/locals
   '*': lambda type_: (None, None),
   '&': lambda type_: (None, None),
   'p++': lambda type_: (None, None),
   '++p': lambda type_: (None, None),
   'p--': lambda type_: (None, None),
   '--p': lambda type_: (None, None)
}

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

def cast_to_c_val(type_, val):
    if is_pointer_type(type_) or is_func_type(type_):
        # TODO: use the other types from https://docs.python.org/3/library/ctypes.html?
        return val
    if type_[0] in python_type_map:
        assert len(type_) == 1, "This should always be true, right??"
        return python_type_map[type_[0]](val).value
    assert False, type_

def cast_to_python_val(type_, val):
    if is_int_type(type_):
        return int(val)
    elif is_float_type(type_):
        return float(val)
    elif is_string_type(type_) or is_pointer_type(type_):
        return val
    assert False, type_

def expand_type(type_, typedefs):
    if is_func_type(type_):
        return (type_[0], expand_type(type_[1], typedefs), type_[2], expand_type(type_[3], typedefs))
    elif isinstance(type_, list):
        return [expand_type(t) for t in type_]
    elif type_ in typedefs:
        return typedefs[type_]
    else:
        # TODO: how to handle this? Could still pass a continuation, but only to help with this error?
        my_assert(type_ in python_type_map or type_.startswith('[') or type_ == '*' or type_ == '...', 'Invalid type')
        return type_


def is_func_type(type_):
    #return type_[0].startswith('(')
    return isinstance(type_, tuple)

def is_float_type(type_):
    return len(type_) == 1 and type_[0] in float_types

def is_int_type(type_):
    return len(type_) == 1 and type_[0] in int_types

def is_pointer_type(type_):
    return type_[0].startswith('[') or type_[0] == '*'

def is_string_type(type_):
    return len(type_) == 2 and is_pointer_type(type_[0]) and type_[1] == 'char'

def types_match(type1, type2):
    return type1 == type2

def is_valid_return_value(functype, val):
    return types_match(functype[0][1], val.type)

def is_void_function(type_):
    return type_[0][1] == 'void'


def make_cont(func):
    if isinstance(func, Continuation):
        return func
    else:
        return Continuation(func)

def my_assert(condition, string, continuation):
    assert condition, string
    continuation(None)


# TODO: cache results from particularly common subtrees?
# TODO: asserts shouldn't really be asserts, since then broken student code will crash this code
# TODO: handle sizeof
# can't inherit from genericvisitor, since we need to pass continuations everywhere

class Info(object):
    __slots__ = ['args', 'continuation']

    def __init__(self, args, continuation):
        self.args = args
        assert isinstance(continuation, Continuation)
        self.continuation = continuation

    def __str__(self):
        return 'Info(' + str(self.args) + ', ' + str(self.continuation) + ')'

class Continuation(object):
    __slots__ = ['func']
    def __init__(self):

        self.continuations = []
        self.passthroughs = []
        self.if_id = 1
        self.completed_if_id = set()

        # No nested loops, for now
        self.loop_passthroughs = None


    def cprint(self, val):
        old_func = self.func
        def inner(k):
            print(val)
            old_func(k)
        self.func = inner
        return self

    def apply(self, val):
        # does literally nothing right now
        #assert len(signature(func).parameters) == 0
        #self.continuations.append(('apply', func))
        return self

    def handle(self, arg):
        if arg[0] == 'apply':
            # TODO: can we just call this in apply()
            arg[1]()
        elif arg[0] == 'expect':
            arg[1](self.passthroughs.pop(0))
        elif arg[0] == 'info':
            return arg[1]
        elif arg[0] == 'kassert':
            assert arg[1](), arg[2]
        elif arg[0] == 'loop':
            # TODO: just move loop_var into the loop call?
            self.passthroughs = self.loop_passthroughs + [self.passthroughs]
        elif arg[0] == 'loop_var':
            assert(self.continuations[-1][0] == 'loop')
            loop = self.continuations.pop(-1)
            loop_var = arg[1]
            self.loop_passthroughs = self.passthroughs
            self.passthroughs = []
            if loop_var is None:
                self.continuations.pop(-1)
                self.passthroughs.append([])
            else:
                for i in range(len(arg[1])-1, -1, -1):
                    loop(loop_var[i])
                    if loop[2]:
                        loop(self.passthroughs)

        elif arg[0] == 'passthrough':
            ret = arg[1]
            assert ret is not None
            self.passthroughs.append(ret)

        # TODO: lots of copy-paste
        elif arg[0] == 'if_':
            cond = arg[1]
            if cond:
                self.completed_if_id.add(arg[3])
                arg[2]()

        elif arg[0] == 'elseif':
            if arg[3] not in self.completed_if_id and arg[1]():
                self.completed_if_id.add(arg[2])

        elif arg[0] == 'else_':
            if arg[2] not in self.completed_if_id:
                self.completed_if_id.add(arg[2])
                arg[1]()

        self.continue_()

    def expect(self, func):
        assert len(signature(func).parameters) == 1
        self.continuations.append(('expect', func))
        return self

    def info(self, info):
        assert len(signature(func).parameters) == 1
        self.continuations.append(('info', info))
        return self

    def kassert(self, cond, str_):
        self.continuations.append(('kassert', info))
        return self

    def loop(self, func, list_=False, shortcircuit=False):
        assert len(signature(func).parameters) == 1
        self.continuations.append(('loop', loop, list_, shortcircuit))
        return self

    def loop_var(self, var):
        assert var is None or isinstance(var, list)
        self.continuations.append(('loop_var', loop_var))
        return self

    def passthrough(self, val):
        self.continuations.append(('passthrough', val))
        return self

    def if_(self, cond, func):
        assert len(signature(func).parameters) == 0
        self.continuations.append(('if_', cond, func, self.if_id))
        self.if_id += 1
        return self

    def elseif(self, cond, func):
        assert len(signature(func).parameters) == 0
        self.continuations.append(('elseif', cond, func, self.if_id))
        return self

    def else_(self):
        self.continuations.append(('else_', func, self.if_id))
        return self

    # This is just a convenience function because visiting nodes is so common
    def visit(self, node):
        #self.apply(lambda: self.visit(node))
        self.apply(self.visit(node))
        return self



class Address(object):
    __slots__ = ['base', 'offset']
    def __init__(self, base, offset):
        self.base = base
        self.offset = offset

class Val(object):
    __slots__ = ['type', 'expanded_type', 'value']
    def __init__(self, type_, expanded_type, value):
        self.type = type_
        self.value = value
        self.expanded_type = expanded_type

class Flow(object):
    __slots__ = ['type', 'value']
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value

class Interpreter(object):
    def __init__(self, ast, require_decls=True):
        self.require_decls = require_decls

        self.ast = ast
        self.continuation = None
        self.reverse_k = None

        self.stdin = None
        self.stdout = ''
        self.stderr = ''
        self.filesystem = {}

        self.scope = []
        self.context = None

        self.global_map = {}

        # values are tuples of type, num_elements, and array of elements
        self.memory = {}
        self.string_constants = {}

        # TODO: remove this hard-coding
        self.typedefs = {'string': ['*', 'char'], 'size_t': ['int']}

        for name in builtin_funcs:
            type_, func = builtin_funcs[name](self)
            val = self.make_val(type_, name)
            self.global_map[name] = val
            self.memory_init(name, type_, val.expanded_type, 1, [(lambda func: lambda args, k: k(func(args)))(func)],
                             'text')

        self.memory_init('NULL', 'void', 'void', 0, [], 'NULL')

    def make_val(self, type_, val):
        # TODO: associate the expanded_type with types_ instead of with vals?
        expanded_type = expand_type(type_, self.typedefs)
        return Val(type_, expanded_type, val)

    # TODO: this isn't nearly completely enough. for example, upcast int to long long
    def implicit_cast(val1, val2):
        self.k.info('implicit_cast')
        if is_int_type(val1.type) and is_float_type(val2.type):
            val1.value=float(val1)
            val1.type=val2.type
        elif is_float_type(type1) and is_int_type(type2):
            val2.value=float(val2)
            val2.type=val1.type
        # can only compare against NULL pointer if it's an int
        elif is_int_type(type1) and is_pointer_type(type2):
            self.k.myassert(lambda: val1 == 0)
            val1.value = Address('NULL', 0)
            val1.type = val2.type
        elif is_int_type(type2) and is_pointer_type(type1):
            self.k.myassert(lambda: val2 == 0)
            val2.value = Address('NULL', 0)
            val2.type = val1.type
        self.k.passthrough(lambda: val1).passthrough(lambda: val2)

    def pop_context(self):
        self.context.pop()

    def push_context(self, context):
        self.context.push(context)

    def pop_scope(self):
        self.k.info('pop scope').apply(self.scope.pop())

    def push_scope(self):
        self.k.info('push scope').apply(self.scope.append({}))

    def update_scope(self, id_, val):

        scope[0][-1][id_] = val

        # TODO: use "self.wrap" or something instead, which automatically calls make_cont?
        self.k.info(('update_scope', id_, val))

    def memory_init(self, name, type_, expanded_type, len_, base, segment):
        self.memory[name] = {
            'type': type_,
            'expanded_type': expanded_type,
            'name': name,
            'len': len_,
            'base': base,
            'segment': segment
        }
        (self.k
        .info(('memory', name))
        .passthrough(lambda: self.memory[name]))

    #def update_memory(self, arr, val):
    #    (self.k
    #    .info((name, self.memory))
    #    .passthrough(lambda: self.memory[arr].seti))

    def handle_string_const(self, type_):
        # TODO: check how it's being declared, since we might be doing a mutable character array
        self.k.info('strconst')


        if n.value in self.string_constants:
            name = self.string_constants[n.value]
            self.k.wrap(lambda _: name)
        else:
            const_num = len(self.string_constants)
            # should be unique in memory, since it has a space in the name
            name = 'strconst ' + str(const_num)
            self.string_constants[n.value] = name
            # append 0 for the \0 character
            array = bytes(n.value, 'latin-1') + bytes([0])
            self.memory_init(name, type_, len(array), array, 'rodata')
        self.k.passthrough(Address(name, 0))

    # executes a program from the main function
    # shouldn't call this multiple times, since memory might be screwed up (global values not reinitialized, etc.)
    # TODO: _start??
    def setup_main(self, argv, stdin, continuation):
        continuation = make_cont(continuation)

        # TODO: need to reset global variables to initial values as well
        for i in list(self.memory.keys()):
            if self.memory[i]['segment'] in ['heap', 'stack', 'argv']:
                del(self.memory[i])

        # TODO: validate argv

        id_map = {}

        id_map['argc'] = ('int', len(argv) + 1)

        # this is the name of the spot in self.memory
        # the second index is where in the array this id references
        id_map['argv'] = (['*', '*', 'char'], ('argv', 0))

        # TODO: environment variables as well?
        self.memory_init('argv', ['*', 'char'], len(argv) + 1,
            [('argv[' + str(i) + ']', 0) for i in range(len(argv))] + [('NULL', 0)], 'argv')

        for i in range(len(argv)):
            array = bytearray(argv[i], 'latin-1') + bytearray([0])
            self.memory_init('argv[' + str(i) + ']', ['char'], len(array), array, 'argv')
            #for j in range(len(argv[i])):
            #    self.memory['argv[' + str(i) + '][' + str(j) + ']'] = ('char', 1, [argv[i][j]])


        self.stdout = ''
        self.stderr = ''
        self.stdin = stdin

        # call the body of the main function
        continuation.wrap(lambda flow: \
            'Exiting with default code 0' if flow.type == 'Normal' \
            else 'Exiting with code ' + str(flow.value))

        return self.k.visit(self.memory['main']['base'][0], ([self.global_map, id_map], 'funcbody'), continuation)

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        ret = func(node, scope, continuation)
        assert ret is None

    def visit_ArrayRef(self, n, scope, continuation):
        # TODO: shouldn't be allowed to assign to symbols / consts? Actually, symbols would fall under visit_ID
        # TODO: nested array refs?? argv[1][2]. need to handle dims appropriately

        (self.k.info(n)
        .visit(n.name)
        .visit(n.subscript)
        .expect(lambda arr: self.k.expect(lambda idx:
            (self.k
            .kassert(is_pointer_type(idx.type) != is_pointer_type(arr.type), "Only one can be an address"))
            .if_(is_pointer_type(idx), lambda: self.k.passthrough(idx).passthrough(arr))
            .else_(lambda: self.k.passthrough(arr).passthrough(idx))))

        .expect(lambda arr: self.k.expect(lambda idx:
            (self.k
            .kassert(idx + arr.offset < self.memory[arr]['len'],
                'Out of bounds array:{} idx:{} length:{}'.format(arr, idx, self.memory[arr]['len']))
            .if_(self.context[-1] == 'lvalue',
                lambda: self.k
                .kassert(self.memory[arr]['segment'] not in ['rodata', 'text', 'NULL'], self.memory[arr]['segment'])
                .passthrough(lambda val: operator.setitem(self.memory[arr]['base'], idx+arr.offset, val)))
            .else_(lambda: self.k
                .passthrough(self.make_val(self.memory[arr]['type'], self.memory[arr]['base'][idx + arr.offset])))))))

    def visit_Assignment(self, n):
        # TODO: handle others
        assert n.op == '='

        (self.k.info(n)
        .push_context('lvalue')
        .visit(lvalue)
        .pop_context()
        .push_context('rvalue')
        .visit(rvalue)
        .pop_context()
        .expect(lambda assignment_op: self.k.expect(lambda val:
            # assignments return the value of the assignment
            assignment_op(val) and self.k.passthrough(lambda: val))))

    def visit_Cast(self, n):
        # TODO: validate
        (self.k.info(n)
        .visit(n.to_type)
        .visit(n.expr)
        .expect(lambda type_: self.k.expect(lambda val: self.k.passthrough(self.make_val(type_, val.value)))))


    def visit_BinaryOp(self, n):
        assert n.op in binops

        # TODO cast to c type nonsense
        (self.k.info(n)
        .visit(n.left)
        .expect(lambda lval:
            self.k
            .if_(n.op == '&&' and not lval.value, lambda: self.k
                .passthrough(self.make_val(['_Bool'], 0)))
            .else_if(n.op == '||' and lval.value, lambda: self.k
                .passthrough(self.make_val(['_Bool'], 1)))
            .else_(lambda: self.k
                .visit(n.right)
                .expect(lambda rval: self.implicit_cast(lval, rval))))
        .expect(lambda lval: self.k.expect(lambda rval:
            self.k
            .kassert(n.op != '%' or is_int_type(lval.type), lval.type)
            .kassert(n.op != '%' or rval != 0, "Can't mod by zero")
            .kassert(n.op != '/' or rval != 0, "Can't divide by zero")
            .passthrough(lambda: self.make_val(lval.type, binops[n.op](lval.type)(lval.value, rval.value))))))


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
        self.k.info(n).passthrough(Flow('Break'))

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
                .if_(not isinstance(flow, Flow) or flow.type != 'Normal', lambda: self.k
                    # pass through to the looper to determine whether we should shortcircuit
                    .passthrough(flow).passthrough(True))
                .else_(lambda: self.k.passthrough(False)))), shortcircuit=True)
        .expect(lambda flows: self.k.passthrough(Flow('Normal', None) if isinstance(flows[-1], Val) else flows[-1])))

    def visit_Constant(self, n):
        # TODO: necessary to expand the type??
        (self.k.info(n)
        .passthrough(lambda:
                cast_to_python_val(self.make_val(n.type, n.value)) if not is_string_type(n.type) else
                self.handle_string_const(n.value)))

    def visit_Continue(self, n):
        #assert continuations['continue'] is not None, 'Continue in invalid context'
        # TODO: put the loop in scope context
        (self.k.info(n).passthrough(Flow('Continue')))

    # name: the variable being declared
    # quals: list of qualifiers (const, volatile)
    # funcspec: list function specifiers (i.e. inline in C99)
    # storage: list of storage specifiers (extern, register, etc.)
    # type: declaration type (probably nested with all the modifiers)
    # init: initialization value, or None
    # bitsize: bit field size, or None
    def visit_Decl(self, n, type_=None):
        # TODO: compare n.type against type_ for validity
        # TODO: funcdecls might be declared multiple times?
        # TODO: doesn't return the name
        (self.k.info(n)
        .if_(n.type, lambda: self.k.visit(n.type))
        .else_(lambda: self.k.kassert(type_ is not None))
        .if_(n.init, lambda: self.k
            .apply(self.push_context('rvalue'))
            .visit(n.init)
            .apply(self.pop_context())
            .expect(lambda val: self.update_scope(id_=n.name, val=val)))
        .else_(lambda: self.k
            # DeclList will apply the type if we need it to
            .expect(lambda type_: self.k.passthrough(lambda: self.make_val(type_, None)))))


    def visit_DeclList(self, n):
        (self.k.info(n)
        .loop_var(lambda: n.decls)
        .loop(lambda decl: lambda vals: self.k.visit(decl, vals[0].type if len(vals) else None), list_=True))

    def visit_ExprList(self, n):
        (self.k.info(n)
        .loop_var(n.exprs)
        .loop(lambda expr: self.k.visit(expr)))

    def visit_ID(self, n):
        def helper():
            for i in range(len(self.scope)-1, -1, -1):
                if n.name in self.scope[i]:
                    id_map = self.scope[i]
                    self.k.passthrough(id_map)
                    return
            else:
                self.k.kassert(not self.require_decls, n.name)

        (self.k.info(n)
        .apply(helper())
        .expect(lambda id_map:
            self.k
            .if_(self.context[-1] == 'lvalue', lambda k: self.k
                .expect(lambda val: operator.setitem(id_map, n.name, val)))
            .else_(lambda: self.k
                .kassert(n.name in id_map, "Undeclared identifier")
                .kassert(id_map[n.name].value is not None, "Uninitialized variable")
                .passthrough(id_map[n.name]))))

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
            else: self.k.passthrough(self.make_val('_Bool', True))

            (self.k.expect(lambda cond:
                self.k
                # TODO: need to cast types
                .if_(cond.value, lambda: self.k
                    .expect(lambda flow:
                        (self.k
                        .if_(flow.type == 'Normal' or flow.type == 'Continue', lambda: self.k
                            .visit(n.next)
                            .visit(for_inner))
                        .else_if(flow.type == 'Return', lambda: self.k.passthrough(flow))
                        .else_(lambda: self.k
                            .kassert(flow.type == 'Break', 'blah')
                            .passthrough(Flow('Normal'))))))
                .else_(lambda: self.k
                    .apply(self.pop_scope())
                    .passthrough(Flow('Normal')))))


        # TODO: can we declare a variable in both for (int i) { int i; ??
        self.k.info(n)
        # TODO only do this if n.init exists??
        self.push_scope()
        if n.init:
            self.k.visit(n.init)
        self.k.apply(for_inner())

    def visit_FuncCall(self, n):
        # TODO: something about unused return value?
        # TODO: parse args for params / check types
        # TODO: deal with scope!!
        #if self.memory[n.name]['type'][0][0] == '(builtin)':
        # TODO: this needs to be expanded, since we might have weird types in c_utils?

        self.k.info(n).visit(n.name)
        if n.args: self.k.visit(n.args)
        else: self.k.passthrough([])

        (self.k.expect(lambda name: self.k.expect(lambda args:
            (self.k
            .if_(self.memory[n.name]['type'][0][0] == '(builtin)', lambda: self.k
                .apply(self.memory[name.value]['base'][0]([self.make_val(['string'], "hello")], k)))
            .else_(lambda: self.k
                .kassert(self.memory[n.name]['type'][0][0] == '(user-defined)')
                .apply(self.push_func_scope())
                .visit(self.memory[name]['base'][0]))
                .apply(self.pop_func_scope())
                # TODO: check return type
                .expect(lambda flow:
                    (self.k
                    .kassert(flow.type == 'Return', 'Didn\'t return from FuncCall')
                    .passthrough(flow.value)))))))



    def visit_Typedef(self, n):
        self.k.info(n).visit(n.type)

    def visit_TypeDecl(self, n):
        self.k.info(n).visit(n.type)

    def visit_ArrayDecl(self, n):
        self.k.info(n).visit(n.type)
        if n.dim:
            (self.k
            .visit(n.dim)
            .expect(lambda dim: self.k.passthrough(str(dim.value))))
        else: self.k.passthrough('')
        self.k.expect(lambda type_: self.k.expect(lambda dim: ['[' + dim + ']'] + type_))


    def visit_IdentifierType(self, n):
        self.k.info(n).passthrough(n.names)

    def visit_ParamList(self, n):
        # TODO: can param.name be empty?
        (self.k.info(n)
        .loop_var(n.params)
        .loop(lambda param:
            self.k
            .visit(param.type)
            .expect(lambda ptype: (param.name, ptype))))

    def visit_FuncDecl(self, n):
        # only the funcdef funcdecl will necessarily have parameter names
        # TODO: this can theoretically appear inside of funcdefs, but we would step over it
        self.k.info(n).visit(n.type)

        if n.args: self.k.visit(n.args)
        else: self.k.expect(lambda ret_type: self.k.passthrough((ret_type, [], [])))

        # TODO: typedefs should be scoped
        #continuation.passthrough(lambda k:
        #    lambda ret_type: lambda params:
        #    expand_type([('(user-defined)', ret_type, params[0], params[1])], self.typedefs, k))
        (self.k
        .expect(lambda ret_type:
            self.k.expect(lambda params: [('(user-defined)', ret_type, params[0], params[1])])))



    def visit_FuncDef(self, n):
        # TODO: check against prior funcdecls?
        (self.k.info(n)
        .visit(n.decl)
        .expect(lambda type_: self.memory_init(n.decl.name, type_, 1, [n.body], 'text')))


    def visit_If(self, n):
        self.k.info(n)
        if n.cond: self.k.visit(n.cond)
        # TODO: cprrect default?
        else: self.k.passthrough(self.make_val('_Bool', True))

        self.k.expect(lambda cond:
            self.k.visit(n.iftrue if cond else n.iffalse))

    def visit_PtrDecl(self, n):
        (self.k.info(n)
        .visit(n.type)
        .expect(lambda type_: self.k.passthrough(['*'] + type_)))

    def visit_Struct(self, n):
        # TODO: finish this
        (self.k.info(n)
        .loop_var(n.decls)
        .loop(lambda decl: self.k.visit(decl)))

    def visit_Typename(self, n):
        # TODO: stuff with n.quals
        (self.k.info(n)
        # TODO: don't know when this is not None
        .kassert(not n.name, 'uh oh')
        .visit(n.type))

    # TODO: account for overflow in binop/unop?
    def visit_UnaryOp(self, n):
        assert n.op in unops
        assert False

        # TODO: use types
        type_, val = self.k.visit(n.expr)
        if n.op == 'p++' or n.op == 'p--' or n.op == '++p' or n.op == '--p':
            with self.context('lvalue'):
                assignment_op = self.k.visit(n.expr)
            # TODO: this doesn't handle post-increment correctly
            if n.op == 'p++' or n.op == '++p':
                assignment_op(val + 1)
            else:
                assignment_op(val - 1)
            return type_, val
        elif n.op == '*':
            assert is_pointer_type(type_)
            arr, offset = val
            # TODO: is this fine as long as we never actually dereference. like, maybe
            # just for sizeof purposes?
            assert offset < self.memory[arr]['len']
            # TODO: dereference type_
            return type_[1:], self.memory[arr]['base'][offset]
        elif n.op == '&':
            assert is_pointer_type(type_)
            arr, offset = val
            # TODO: add pointer type, but only if not func pointer/constant array
            return ['*'] + type_, self.memory[arr]['base']
        else:
            return type_, unops[n.op](type_)(val)

    # TODO: detect infinite loop??
    def visit_While(self, n):
        self.k.info(n)

        if n.cond: self.k.visit(n.cond)
        # TODO: is True the right default? can the condition even be empty?
        else: self.k.passthrough(self.make_val('_Bool', True))

        (self.k.expect(lambda cond:
            self.k
            .if_(cond.value, lambda: self.k
                .visit(n.stmt)
                .expect(lambda flow: self.k
                    .kassert(lambda: isinstance(flow, Flow))
                    .if_(flow.type == 'Continue' or flow.type == 'Normal', lambda: self.k.visit(n))
                    .else_if(flow.type == 'Break', lambda: self.k.passthrough(Flow('Normal')))
                    .else_(lambda: self.k
                        .kassert(flow.type == 'Return', 'blah')
                        .passthrough(flow))))
            .else_(lambda: self.k.passthrough(lambda: Flow('Normal')))))




    def visit_Return(self, n):
        self.k.info(n)
        if n.expr: self.k.visit(n.expr)
        # TODO: we very explicitly want this to pass through to self.k.expect, not to the upper continuation
        else: self.k.passthrough(None)
        self.k.expect(lambda val: self.k.passthrough(lambda: Flow('Return', val)))

        # TODO: can we even write a return outside of a function?
        # TODO: we need to check this elsewhere
        #assert continuations['return'] is not None, 'Return in invalid context'

def main():
    interpret = Interpreter()
    parser = c_parser.CParser()
    try:
        # TODO: use a canonical set of header files, so we know what to expect?
        cfile = preprocess_file(sys.argv[1], cpp_path='clang', cpp_args=['-E', '-nostdinc', r'-I../repair/fake_libc_include'])
        #cpp_args = [r'-D__attribute__(x)=', r'-D__builtin_va_list=int',
                    #r'-D_Noreturn=', r'-Dinline=', r'-D__volatile__=', '-E']
        #cfile = preprocess_file(sys.argv[1], cpp_path='clang', cpp_args=['-E'])
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        sys.exit(1)

    def done(out):
        print(out, interpret.stdout)
        sys.exit(1)

    # some kind of JIT after the first execution?
    k = lambda _: interpret.setup_main(['./vigenere', 'HELlO'], 'wOrld\n', Continuation(done))

    info = interpret.visit(ast, [[interpret.global_map], 'global'], k)
    while info is not None:
        if hasattr(info.args, 'show'):
            print(info.args.show())
        print(info)
        info = info.continuation.func(None)
        assert isinstance(info, Info), info

if __name__=='__main__':
    main()
