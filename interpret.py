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

def expand_type(type_, typedefs, continuation = None):
    is_continuation = continuation is not None
    if continuation is not None:
        continuation = make_cont(continuation)
        continuation.wrap(lambda ret: ret)
    else:
        continuation = make_cont(lambda ret: ret)

    if is_func_type(type_):
        continuation.wrap(lambda ret_type: lambda ptypes: (type_[0], ret_type, type_[2], ptypes))
        for j in range(len(type_[3])-1,-1,-1):
            ptype = type_[3][j]
            continuation.wrap(
                lambda ret_type: lambda ptypes: expand_type(ptype, typedefs,
                lambda ptype: ret_type, ptypes + [ptype]))
            continuation.wrap(lambda _: expand_type(type_[1], typedefs, lambda ret_type: (ret_type, [])))
    elif isinstance(type_, list):
        for i in range(len(type_)-1, -1, -1):
            continuation.wrap((
                lambda t:
                lambda ret: expand_type(t, typedefs,
                lambda t: ret + [t]))(type_[i]))
        continuation.wrap(lambda _: [])
    elif type_ in typedefs:
        continuation.wrap(lambda _: typedefs[type_])
    else:
        continuation.wrap(
            lambda _: my_assert(type_ in python_type_map or type_.startswith('[') or type_ == '*' or type_ == '...',
                                'Invalid type',
            lambda _: type_))
    if not is_continuation:
        return continuation.func(None)
    else:
        return Info(('expand_type'), continuation)

# TODO: this isn't nearly completely enough. for example, upcast int to long long
def implicit_cast(type1, val1, type2, val2):
    if is_int_type(type1) and is_float_type(type2):
        return type2, float(val1), val2
    elif is_float_type(type1) and is_int_type(type2):
        return type1, val1, float(val2)
    elif is_int_type(type1) and is_pointer_type(type2):
        assert val1 == 0
        return type2, ('NULL', 0), val2
    elif is_int_type(type2) and is_pointer_type(type1):
        assert val2 == 0
        return type1, val1, ('NULL', 0)
    return type1, val1, val2

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
    def __init__(self, func):
        assert callable(func)
        assert len(signature(func).parameters) == 1
        self.func = func

    def cprint(self, val):
        old_func = self.func
        def inner(k):
            print(val)
            old_func(k)
        self.func = inner
        return self

    # order is index, arg, k
    def loop(self, args, passthrough=False, index=False, k=None):
        assert k is not None
        for i in range(len(args)-1,-1,-1):
            func = k
            if index:
                func = func(i)
            func = func(args[i])
            if passthrough:
                self.func = func(self.func)
                if isinstance(self.func, Info):
                    self.func = (lambda k: lambda _: k)(self.func)
            else:
                if isinstance(func, Info):
                    func = (lambda k: lambda _: k)(func)
                self.wrap(func)
        assert callable(self.func)
        return self

    def passthrough(self, func):
        assert callable(func)
        assert len(signature(func).parameters) == 1
        self.func = func(self.func)
        if isinstance(self.func, Info):
            self.func = (lambda k: lambda _: k)(self.func)
        assert callable(self.func)
        return self

    def wrap(self, k):
        if isinstance(k, Info):
            k = (lambda k: lambda _: k)(k)
        assert callable(k)
        assert len(signature(k).parameters) == 1
        old_func = self.func
        def new_func(args):
            func = k
            if isinstance(args, tuple):
                for arg in args:
                    func = func(arg)
                return old_func(func)
            else:
                return old_func(func(args))
        self.func = new_func
        assert callable(self.func)
        return self

class Address(object):
    __slots__ = ['base', 'offset']
    def __init__(self, base, offset):
        self.base = base
        self.offset = offset

class Val(object):
    __slots__ = ['type', 'value']
    def __init__(self, type_, value):
        self.type = type_
        self.value = value

class Interpreter(object):
    def __init__(self, require_decls=True):
        self.require_decls = require_decls

        self.stdin = None
        self.stdout = ''
        self.stderr = ''
        self.filesystem = {}

        self.global_map = {}

        # values are tuples of type, num_elements, and array of elements
        self.memory = {}
        self.string_constants = {}

        # TODO: remove this hard-coding
        self.typedefs = {'string': ['*', 'char'], 'size_t': ['int']}

        for name in builtin_funcs:
            type_, func = builtin_funcs[name](self)
            type_ = expand_type(type_, self.typedefs)
            self.global_map[name] = Val(type_, name)
            self.memory_init(name, type_, 1, [(lambda func: lambda args, k: k(func(args)))(func)], 'text')

        self.memory_init('NULL', 'void', 0, [], 'NULL')

    def extend_scope(self, scope, continuation):
        continuation = make_cont(continuation)
        continuation.wrap(scope + [{}])
        return Info('extend_scope', continuation)

    def update_scope(self, scope, id_=None, val=None, context=None, continuation=None):
        assert continuation is not None
        assert context is not None or id_ is not None

        if context is not None:
            scope = scope.copy()
            scope[1] = context
        if id_ is not None:
            scope[0][-1][id_] = val

        # TODO: use "self.wrap" or something instead, which automatically calls make_cont?
        continuation = make_cont(continuation)
        continuation.wrap(lambda _: scope)
        return Info((id_, val, context), continuation)

    def memory_init(self, name, type_, len_, base, segment, continuation = None):
        self.memory[name] = {
            'type': type_,
            'name': name,
            'len': len_,
            'base': base,
            'segment': segment
        }
        if continuation:
            continuation = make_cont(continuation)
            continuation.wrap(lambda _: self.memory[name])
            return Info(('memory', name, self.memory[name]), continuation)

    def update_memory(self, arr, val, continuation):
        continuation = make_cont(continuation)
        continuation.func(sef.memory[arr])
        return (name, self.memory), None, lambda _: continuation

    def handle_string_const(self, type_, scope, continuation):
        # TODO: check how it's being declared, since we might be doing a mutable character array
        continuation.wrap(lambda name: Address(name, 0))

        if n.value in self.string_constants:
            name = self.string_constants[n.value]
            continuation.wrap(lambda _: name)
        else:
            const_num = len(self.string_constants)
            # should be unique in memory, since it has a space in the name
            name = 'strconst ' + str(const_num)
            self.string_constants[n.value] = name
            # append 0 for the \0 character
            array = bytes(n.value, 'latin-1') + bytes([0])
            continuation.wrap(self.memory_init(name, type_, len(array), array, 'rodata', lambda _: name))
        return Info(('strconst'), continuation)

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
            'Exiting with default code 0' if flow is None \
            else 'Exiting with code ' + str(flow.value))

        return self.visit(self.memory['main']['base'][0], ([self.global_map, id_map], 'funcbody'), continuation)

    def visit(self, node, scope, continuation):
        method = 'visit_' + node.__class__.__name__
        if not isinstance(continuation, Continuation):
            continuation = Continuation(continuation)
        func = getattr(self, method)
        assert func is not None, "Unimplemented Node Type!"
        ret = func(node, scope, continuation)
        assert isinstance(ret, Info)
        return ret

    def visit_ArrayRef(self, n, scope, continuation):
        # TODO: shouldn't be allowed to assign to symbols / consts? Actually, symbols would fall under visit_ID
        # TODO: nested array refs?? argv[1][2]. need to handle dims appropriately

        if scope[1] == 'lvalue':
            continuation.wrap(lambda arr: lambda idx: \
                my_assert(self.memory[arr]['segment'] not in ['rodata', 'text', 'NULL'], self.memory[arr]['segment'],
                lambda _: (lambda val: operator.setitem(self.memory[arr]['base'], idx+arr.offset, val))))
        else:
            continuation.wrap(
                lambda arr: lambda idx: Val(self.memory[arr]['type'], self.memory[arr]['base'][idx+arr.offset]))


        continuation.wrap(lambda arr: lambda idx: \
            my_assert(idx + arr.offset < self.memory[arr]['len'],
                'Out of bounds array:{} idx:{} length:{}'.format(arr, idx, self.memory[arr]['len']),
                lambda _: arr, idx))

        continuation.wrap(
            self.visit(n.name, scope,
            lambda arr: self.visit(n.subscript, scope,
            # swap the two, in case the student happened to do something like 2[argv], which is
            # technically valid
            lambda idx: my_assert(is_pointer_type(idx_type) != is_pointer_type(arr_type),
                        "Only one can be an address",
            lambda _: (idx, arr) if is_pointer_type(idx) else (arr, idx)))))

        return Info(n, continuation)


    def visit_Assignment(self, n, scope, continuation):
        # TODO: handle others
        assert n.op == '='

        continuation.passthrough(lambda k:
            self.update_scope(scope, context='lvalue', continuation=
            lambda scope: self.visit(n.lvalue, scope,
            lambda assignment_op: self.update_scope(scope, context='rvalue', continuation=
            lambda scope: self.visit(n.rvalue, scope,
            # TODO:validate types
            lambda val: assignment_op(val, k))))))

        return Info(n, continuation)

    def visit_Cast(self, n, scope, continuation):
        continuation.wrap(
            self.visit(n.expr, scope,
            lambda val: self.visit(n.to_type, scope,
            # TODO: validate
            lambda type_: Val(type_, val.value))))
        return Info(n, continuation)


    def visit_BinaryOp(self, n, scope, continuation):
        assert n.op in binops

        # TODO cast to c type nonsense
        continuation.wrap(
            lambda lval: lambda rval: my_assert(n.op != '%' or is_int_type(lval.type), lval.type,
            lambda _: my_assert(n.op != '%' or rval != 0, "Can't mod by zero",
            lambda _: my_assert(n.op != '/' or rval != 0, "Can't divide by zero",
            lambda _: Val(lval.type, binops[n.op](lval.type)(lval.value, rval.value))))))

        continuation.wrap(
            self.visit(n.left, scope,
            # && and || are special, because they can short circuit
            lambda lval: Val(['_Bool'], 0) if n.op == '&&' and not lval \
                         else Val(['_Bool'], 1) if n.op == '||' and lval \
                         else self.visit(n.right, scope,
                         lambda rval: implicit_cast(lval, rval))))

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
        return Info(n, continuation)


    def visit_Break(self, n, scope, continuation):
        #assert continuation['break'] is not None, 'Break in of invalid context'
        continuation.wrap(lambda _: Val('Break', None))
        return Info(n, continuation)

    def visit_Compound(self, n, scope, continuation):

        continuation.wrap(
            lambda flow: Val('Normal', None) if flow is None \
            else flow)

        if n.block_items:
            continuation.loop(n.block_items, passthrough=True, k=
                lambda stmt: lambda k:
                lambda flow: self.visit(stmt, scope, k) if flow is None \
                             else k(flow))
        continuation.wrap(lambda _: None)

        # TODO: make sure that 'Normal', etc. isn't defined as an actual type
        # TODO: do we actually ever even care to stop on this node?
        return Info(n, continuation)

    def visit_Constant(self, n, scope, continuation):
        # TODO: necessary to expand the type??
        continuation.wrap(
            expand_type(n.type, self.typedefs,
            lambda type_:
                cast_to_python_val(Val(type_, n.value)) if not is_string_type(type_) else
                self.handle_string_const(type_, scope)))

        return Info(n, continuation)

    def visit_Continue(self, n, scope, continuations):
        #assert continuations['continue'] is not None, 'Continue in invalid context'
        # TODO: put the loop in scope context
        continuation.wrap(lambda _: Val('Continue', None))
        return Info(n, continuation)

    # name: the variable being declared
    # quals: list of qualifiers (const, volatile)
    # funcspec: list function specifiers (i.e. inline in C99)
    # storage: list of storage specifiers (extern, register, etc.)
    # type: declaration type (probably nested with all the modifiers)
    # init: initialization value, or None
    # bitsize: bit field size, or None
    def visit_Decl(self, n, scope, continuation):
        # TODO: compare n.type against type_ for validity
        # TODO: funcdecls might be declared multiple times?
        # TODO: doesn't return the name
        continuation.passthrough(lambda k:
            lambda val: self.update_scope(scope, id_=n.name, val=val, continuation=k))

        if n.init:
            continuation.wrap(
                lambda type_: self.update_scope(scope, context='rvalue', continuation=self.visit(n.init, scope,
                lambda val: Val(type_, val))))
        else:
            continuation.wrap(lambda type_: Val(type_, None))

        if n.type:
            continuation.passthrough(lambda k:
                self.visit(n.type, scope,
                lambda type_: expand_type(type_, self.typedefs, k)))
        else: continuation.wrap(lambda type_: type_)

        if n.type:
            return Info(n, continuation)
        else:
            return lambda type_: Info(n, (lambda _: continuation.func(type_)))

    def visit_DeclList(self, n, scope, continuation):
        helper = lambda i: lambda vals: lambda decl: decl(vals[0].type) if i else decl
        # TODO: can maybe pull a bit more of this into loop
        continuation.loop(n.decls, index=True, k=
            lambda i: lambda decl:
            lambda vals: helper(i, vals,
                self.visit(decl, scope,
                lambda val: vals + [val])))
        continuation.wrap(lambda _: [])
        return Info(n, continuation)

    def visit_ExprList(self, n, scope, continuation):
        continuation.loop(n.exprs, k=
            lambda expr:
            lambda vals: self.visit(expr, scope,
            lambda val: vals + [val]))
        continuation.wrap(lambda _: [])
        return Info(n, continuation)

    def visit_ID(self, n, scope, continuation):
        for i in range(len(scope[0])-1, -1, -1):
            if n.name in scope[0][i]:
                id_map = scope[0][i]
                break
        else:
            # TODO: change to my_assert
            assert not self.require_decls, n.name

        continuation.passthrough(lambda k:
            # TODO: this is almost certainly wrong
            ((lambda val: operator.setitem(id_map, n.name, val) or k(None)) if scope[1] == 'lvalue'
                else lambda _: my_assert(n.name in id_map, "Undeclared identifier",
                 lambda _: my_assert(id_map[n.name] is not None, "Uninitialized variable",
                 lambda _: k(id_map[n.name])))))

        return Info(n, continuation)

    def visit_FileAST(self, n, scope, continuation):
        # TODO: put global map placement in here??
        # Necessary lambda _?
        continuation.loop(n.ext, passthrough=True, k=
            lambda ext: lambda k:
            lambda _: self.visit(ext, scope, k))
        return Info(n, continuation)

    def visit_For(self, n, scope, continuation):
        def for_inner(scope, continuation):
            # TODO: scope probably has to be updated for continuations
            continuation.passthrough(lambda k: \
                lambda cond: self.visit(n.stmt, scope,
                lambda flow:
                    self.visit(n.next, scope,
                               lambda _: for_inner(scope, k)) if flow is None or flow.type == 'Continue' \
                    else k(None) if flow.type == 'Break' or flow.type != 'Return' \
                    else k(flow)))
            # TODO: is True the right default? can the condition even be empty?
            continuation.passthrough(lambda k:
                lambda _: self.visit(n.cond, k) if n.cond else k(Val('_Bool', True)))

            return Info(n, continuation)

        # TODO: can we declare a variable in both for (int i) { int i; ??
        continuation.passthrough(lambda k: lambda scope: for_inner(n, scope, k))
        if n.init:
            continuation.wrap(
                extend_scope(scope,
                lambda scope: self.visit(n.init, scope,
                lambda _: scope)))
        else:
            continuation.wrap(lambda _: scope)
        return Info(n, continuation)


    def visit_FuncCall(self, n, scope, continuation):
        # TODO: something about unused return value?
        # TODO: parse args for params / check types
        # TODO: deal with scope!!
        #if self.memory[n.name]['type'][0][0] == '(builtin)':
        # TODO: this needs to be expanded, since we might have weird types in c_utils?
        continuation.wrap(
            lambda val: expand_type(val.type, self.typedefs,
            lambda type_: Val(type_, val.value)))
        continuation.passthrough(lambda k:
            lambda name: self.memory[name.value]['base'][0]([Val(['string'], "hello")], k))

        #elif self.memory[name]['type'][0][0] == '(user-defined)':
        #    # TODO: need to fix scope!!
        #    val = self.visit(self.memory[name]['base'][0])
        #    if flow is None:
        #        continuation = my_assert(is_void_function(type_), "Didn't provide return value for non-void function",
        #                continuation)
        #    else:
        #        continuation = my_assert(flow[0] == 'Return', "didn't return from FuncCall",
        #                lambda _: my_assert(is_valid_return_value(type_, flow[1]), "Invalid return value", lambda _:
        #                    continuation.func(flow[1])))
        #else:
        #    assert False

        #if n.args:
        #    continuation.wrap(
        #        lambda name: self.visit(n.args, scope,
        #        lambda args: (name, args)))
        #else:
        #    continuation.wrap(lambda name: (name, []))

        continuation.passthrough(lambda k: self.visit(n.name, scope, k))

        return Info(n, continuation)

    def visit_Typedef(self, n, scope, continuation):
        continuation.passthrough(lambda k: self.visit(n.type, scope, k))
        return Info(n, continuation)

    def visit_TypeDecl(self, n, scope, continuation):
        continuation.passthrough(lambda k: self.visit(n.type, scope, k))
        return Info(n, continuation)

    def visit_ArrayDecl(self, n, scope, continuation):
        continuation.wrap(lambda type_: lambda dim: ['[{}]'.format(dim)] + type_)
        if n.dim:
            continuation.wrap(lambda type_: self.visit(n.dim, scope, lambda dim: (type_, dim.value)))
        else:
            continuation.wrap(lambda type_: (type_, ''))
        continuation.passthrough(lambda k: self.visit(n.type, scope, k))
        return Info(n, continuation)

    def visit_IdentifierType(self, n, scope, continuation):
        continuation.wrap(lambda _: n.names)
        return Info(n, continuation)

    def visit_ParamList(self, n, scope, continuation):
        # TODO: can param.name be empty?
        if n.params:
            continuation.loop(n.params, k=
                lambda param:
                lambda params:
                self.visit(param.type, scope,
                lambda ptype: (params + [(param.name, ptype)])))
        continuation.wrap(lambda _: [])
        return Info(n, continuation)

    def visit_FuncDecl(self, n, scope, continuation):
        # only the funcdef funcdecl will necessarily have parameter names
        # TODO: this can theoretically appear inside of funcdefs, but we would step over it

        # TODO: typedefs should be scoped
        #continuation.passthrough(lambda k:
        #    lambda ret_type: lambda params:
        #    expand_type([('(user-defined)', ret_type, params[0], params[1])], self.typedefs, k))

        #if n.args:
        #    continuation.wrap(
        #        lambda ret_type: self.visit(n.args, scope,
        #        lambda param_names: lambda param_types: (ret_type, param_names, param_types)))
        #else:
        #    continuation.wrap(lambda ret_type: (ret_type, [], []))

        #continuation.passthrough(lambda k: self.visit(n.type, scope, k))

        continuation.wrap(lambda _: [('user-defined)', 'int', [], [])])
        return Info(n, continuation)

    def visit_FuncDef(self, n, scope, continuation):
        # TODO: check against prior funcdecls?
        continuation.passthrough(lambda k:
            self.visit(n.decl, scope,
            lambda type_: self.memory_init(n.decl.name, type_, 1, [n.body], 'text', k)))
        return Info(n, continuation)


    def visit_If(self, n, scope, continuation):
        continuation.passthrough(lambda k:
            lambda cond: self.visit(n.iftrue, scope, k) if cond \
            else self.visit(n.iffalse, scope, k))

        continuation.passthrough(lambda k:
            self.visit(n.cond, scope, k) if n.cond \
            else k(Val('_Bool', True)))

        return Info(n, continuation)

    def visit_PtrDecl(self, n, scope, continuation):
        continuation.wrap(
            lambda _: self.visit(n.type, scope,
            lambda type_: ['*'] + type_))
        return Info(n, continuation)

    def visit_Struct(self, n, scope, continuation):
        # TODO: finish this
        if n.decls:
            continuation.loop(n.decls, k=
                lambda decl:
                lambda vals: self.visit(decl, scope,
                lambda val: (vals + [val])))
        continuation.wrap(lambda _: [])

        return Info(n, continuation)


    def visit_Typename(self, n, scope, continuation):
        # TODO: don't know when this is not None
        # TODO: stuff with n.quals
        if n.name: assert False
        continuation.passthrough(lambda k: self.visit(n.type, scope, k))
        return Info(n, continuation)

    # TODO: account for overflow in binop/unop?
    def visit_UnaryOp(self, n):
        assert n.op in unops
        assert False

        # TODO: use types
        type_, val = self.visit(n.expr)
        if n.op == 'p++' or n.op == 'p--' or n.op == '++p' or n.op == '--p':
            with self.context('lvalue'):
                assignment_op = self.visit(n.expr)
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
    def visit_While(self, n, scope, continuation):
        continuation.passthrough(lambda k:
            lambda cond: self.visit(n.stmt, scope,
            lambda flow: self.visit(n, scope, k) if flow is None or flow.type == 'Continue' \
                         else k(None) if flow.type == 'Break' or flow.type != 'Return' \
                         else k(flow)))
        continuation.wrap(lambda _:
            self.visit(n.cond, scope, continuation) if n.cond \
            # TODO: is True the right default? can the condition even be empty?
            else Val('_Bool', True))

        return Info(n, continuation)\

    def visit_Return(self, n, scope, continuation):
        # TODO: can we even write a return outside of a function?
        # TODO: we need to check this elsewhere
        #assert continuations['return'] is not None, 'Return in invalid context'
        continuation.wrap(lambda val: Val('Return', val))
        continuation.passthrough(lambda k: (self.visit(n.expr, scope, k) if n.expr else k(None)))
        return Info(n, continuation)

def main():
    interpret = Interpreter()
    parser = c_parser.CParser()
    try:
        cfile = preprocess_file(sys.argv[1], cpp_path='clang', cpp_args=['-E', r'-nostdinc', r'-Ifake_libc_include'])
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
