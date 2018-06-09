import sys
import re
import ctypes
import inspect
import operator
from contextlib import contextmanager
from pycparser import c_generator, c_ast, c_lexer, c_parser, preprocess_file

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
    ret = []

    assert isinstance(type_, list)
    for t in type_:
        if isinstance(t, tuple):
            # this must be a function type
            ret.append((t[0], expand_type(t[1], typedefs), t[2], [expand_type(ptype, typedefs) for ptype in t[3]]))
        elif t in typedefs:
            ret.extend(typedefs[t])
        else:
            assert t in python_type_map or t.startswith('[') or t == '*' or t == '...'
            ret.append(t)
    return ret

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
    return type_[0] == '()'

def is_float_type(type_):
    return len(type_) == 1 and type_[0] in float_types

def is_int_type(type_):
    return len(type_) == 1 and type_[0] in int_types

def is_pointer_type(type_):
    return type_[0].startswith('[') or type_[0] == '*'

def is_string_type(type_):
    return len(type_) == 2 and is_pointer_type(type_[0]) and type_[1] == 'char'



# TODO: cache results from particularly common subtrees?
# TODO: asserts shouldn't really be asserts, since then broken student code will crash this code
# TODO: handle sizeof
class Interpreter(c_ast.NodeVisitor):
    def __init__(self, require_decls=True):
        self.require_decls = require_decls

        self.stdin = None
        self.stdout = ''
        self.stderr = ''
        self.filesystem = {}

        self.id_map = [{}]
        self.type_map = [{}]

        # values are tuples of type, num_elements, and array of elements
        self.memory = {}
        self.contexts = [None]
        self.string_constants = {}

        # TODO: remove this hard-coding
        self.typedefs = {'string': ['*', 'char'], 'size_t': ['int']}

        for name in builtin_funcs:
            type_, func = builtin_funcs[name](self)
            type_ = expand_type(type_, self.typedefs)
            self.id_map[0][name] = (type_, name)
            self.memory_init(name, type_, 1, [func], 'text')

        self.memory_init('NULL', 'void', 0, [], 'NULL')

    @contextmanager
    def scope(self):
        self.id_map.append({})
        self.type_map.append({})
        try:
            yield
        except:
            raise
        else:
            self.id_map.pop()
            self.type_map.pop()

    @contextmanager
    def context(self, context):
        self.contexts.append(context)
        try:
            yield
        except:
            raise
        else:
            self.contexts.pop()

    def memory_init(self, name, type_, len_, array, segment):
        self.memory[name] = {
            'type': type_,
            'name': name,
            'len': len_,
            'array': array,
            'segment': segment
        }

    # executes a program from the main function
    # shouldn't call this multiple times, since memory might be screwed up (global values not reinitialized, etc.)
    # TODO: _start??
    def execute(self, argv, stdin=''):
        # TODO: need to reset global variables to initial values as well
        for i in list(self.memory.keys()):
            if self.memory[i]['segment'] in ['heap', 'stack', 'argv']:
                del(self.memory[i])

        # TODO: validate argv

        self.id_map[-1]['argc'] = ('int', len(argv) + 1)

        # this is the name of the spot in self.memory
        # the second index is where in the array this id references
        self.id_map[-1]['argv'] = (['*', '*', 'char'], ('argv', 0))

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
        exitcode = self.visit(self.memory['main']['array'][0])
        return exitcode


    def visit_ArrayRef(self, n):
        # TODO: technically can flip order of these things?
        # TODO: nested array refs?? argv[1][2]
        with self.context('array_ref'):
            idx_type, idx = self.visit(n.subscript)
            arr_type, arr = self.visit(n.name)

        # exactly one of idx and arr must be a pointer type
        assert is_pointer_type(idx_type) != is_pointer_type(arr_type)

        # swap the two, in case the student happened to do something like 2[argv], which is technically valid
        if is_pointer_type(idx_type):
            idx_type, idx, arr_type, arr = arr_type, arr, idx_type, idx
        arr, offset = arr
        idx += offset

        assert idx < self.memory[arr]['len'], 'Out of bounds array:{} idx:{} length:{}'.format(arr, idx,
                self.memory[arr]['len'])

        # TODO: shouldn't be allowed to assign to symbols / consts? Actually, symbols would fall under visit_ID
        if self.contexts[-1] == 'assignment_lvalue':
            assert self.memory[arr]['segment'] not in ['rodata', 'text', 'NULL'], self.memory[arr]['segment']
            return lambda val: operator.setitem(self.memory[arr]['array'], idx, val)
        else:
            return self.memory[arr]['type'], self.memory[arr]['array'][idx]

    # TODO: pointer deref?
    def visit_Assignment(self, n):
        with self.context('assignment_lvalue'):
            assignment_op = self.visit(n.lvalue)

        if n.op != '=':
            op = node.op[-1]
            node = c_ast.BinaryOp(op, n.lvalue, n.rvalue)
        else:
            node = n.rvalue
        with self.context('assignment_rvalue'):
            # TODO: validate type!
            type_, val = self.visit(n.rvalue)
            assignment_op(val)
        return None, None

    def visit_Cast(self, n):
        # TODO: validate this?
        type_ = self.visit(n.to_type)
        old_type, val = self.visit(n.expr)
        return type_, val


    def visit_BinaryOp(self, n):
        assert n.op in binops

        ltype, lval = self.visit(n.left)

        # && and || are special, because they can short circuit
        if n.op == '&&' and not lval:
            return ['int'], 0
        elif n.op == '||' and lval:
            return ['int'], 1

        rtype, rval = self.visit(n.right)

        # TODO: only currently supported pointer math is between a pointer for addition and subtraction, and two pointers for subtraction. Could use like, xor support or something?

        add_back = None
        if is_pointer_type(ltype) or is_pointer_type(rtype):
            if n.op == '==' or n.op == '!=':
                # specifically for NULL handling
                if not is_pointer_type(ltype) or not is_pointer_type(rtype):
                    type_, lval, rval = implicit_cast(ltype, lval, rtype, rval)
                    rtype = ltype = type_

            if is_pointer_type(ltype) and is_pointer_type(rtype):
                assert n.op in ['-', '!=', '<', '>', '<=', '>=', '==']
                if n.op == '-':
                    (larr, lval), (rarr, rval), type_ = lval, rval, ltype
                    assert larr == rarr
                else: assert ltype == rtype
            else:
                assert n.op in ['+', '-']
                if is_pointer_type(ltype):
                    (add_back, lval), type_ = lval, ltype
                else:
                    (add_back, rval), type_ = rval, rtype

        else:
            type_, lval, rval = implicit_cast(ltype, lval, rtype, rval)

        if n.op == '%':
            assert is_int_type(type_), type_
            assert rval != 0
        if n.op == '/':
            assert rval != 0

        type_, op = binops[n.op](type_)
        # TODO cast to c type nonsense
        val = op(lval, rval)
        if add_back is not None:
            val = (add_back, val)
        return type_, val


    def visit_Break(self, n):
        return 'break', None

    def visit_Compound(self, n):
        with self.scope():
            if n.block_items:
                for stmt in n.block_items:
                    with self.context('stmt'):
                        ret, retval = self.visit(stmt)
                    if ret is not None:
                        return ret, retval
        return None, None

    def visit_Constant(self, n):
        # TODO: necessary to expand the type??
        type_ = expand_type([n.type], self.typedefs)
        if is_string_type(type_):
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
            return type_, (name, 0)
        else:
            return type_, cast_to_python_val(type_, n.value)

    def visit_Continue(self, n):
        return 'continue', None

    # name: the variable being declared
    # quals: list of qualifiers (const, volatile)
    # funcspec: list function specifiers (i.e. inline in C99)
    # storage: list of storage specifiers (extern, register, etc.)
    # type: declaration type (probably nested with all the modifiers)
    # init: initialization value, or None
    # bitsize: bit field size, or None
    def visit_Decl(self, n):
        with self.context('assignment_rvalue'):
            type_, val = self.visit(n.init) if n.init else (None, None)

        # TODO: compare n.type against type_ for validity
        # TODO: funcdecls might be declared multiple times?
        type_ = expand_type(self.visit(n.type), self.typedefs) if n.type is not None else self.type
        self.id_map[-1][n.name] = type_, val

        if self.contexts[-1] == 'stmt':
            return None, None
        else:
            # doesn't return val
            return type_, n.name

    def visit_DeclList(self, n):
        self.visit(n.decls[0])
        # set this so later declarations have access to the type, like "int x = 2, y = 3;"
        self.type = self.id_map[-1][n.decls[0].name][0]
        [self.visit(decl) for decl in n.decls[1:]]
        self.type = None
        return None, None

    def visit_ExprList(self, n):
        types = []
        vals = []
        for expr in n.exprs:
            type_, val = self.visit(expr)
            types.append(type_)
            vals.append(val)
        return types, vals

    def visit_ID(self, n):
        for i in range(len(self.id_map)-1, -1, -1):
            if n.name in self.id_map[i]:
                id_map = self.id_map[i]
                break
        else:
            assert not self.require_decls

        if self.contexts[-1] == 'assignment_lvalue':
            return lambda val: operator.setitem(id_map, n.name, val)
        else:
            # check use of uninitialized values
            assert n.name in id_map and id_map[n.name] is not None
            return id_map[n.name]

    def visit_FileAST(self, n):
        [self.visit(ext) for ext in n.ext]

    def visit_For(self, n):
        with self.scope():
            with self.context('expr'):
                if n.init: self.visit(n.init)

            while True:
                with self.context('expr'):
                    ctype, cond = not n.cond or self.visit(n.cond)
                if not cond:
                    break

                with self.context('stmt'):
                    ret, retval = self.visit(n.stmt)

                # TODO: kind of annoying that we have to propagate this all the way up the stack. Can we do better with
                # some kind of continuation passing? Does this really even have a huge cost?
                if ret == 'break' or ret == 'return':
                    return ret, retval
                with self.context('expr'):
                    self.visit(n.next)
        return None, None

    def visit_FuncCall(self, n):
        # TODO: check types?
        type_, name = self.visit(n.name)

        with self.context('expr'):
            arg_types, args = self.visit(n.args) if n.args else ([], [])

        # TODO: parse args for params / check types
        if self.memory[name]['type'][0][0] == '(builtin)':
            type_, val = self.memory[name]['array'][0](args)
            # TODO: this needs to be expanded, since we might have weird types in c_utils?
            type_ = expand_type(type_, self.typedefs)
        elif self.memory[name]['type'][0][0] == '(user-defined)':
            with self.context(None):
                # TODO: need to fix scope!!
                type_, val = self.visit(self.memory[name]['array'][0])
        else:
            assert False

        if self.contexts[-1] != 'stmt':
            return type_, val
        else:
            # we are a statement-level function call
            return None, None

    def visit_TypeDecl(self, n):
        return self.visit(n.type)

    # TODO: account for dims
    def visit_ArrayDecl(self, n):
        # self.visit(n.dim)
        return ['[]'] + self.visit(n.type)

    def visit_IdentifierType(self, n):
        # TODO: account for overflow in binop/unop?
        return n.names

    def visit_ParamList(self, n):
        # TODO: can param.name be empty?
        if n.params:
            return [param.name for param in n.params], [self.visit(param.type) for param in n.params]
        else:
            return []

    def visit_FuncDecl(self, n):
        # only the funcdef funcdecl will necessarily have parameter names
        param_names, param_types = self.visit(n.args)
        ret_type = self.visit(n.type)
        type_ = expand_type([('(user-defined)', ret_type, param_names, param_types)], self.typedefs)

        return type_

    def visit_FuncDef(self, n):
        # TODO: check against prior funcdecls?
        type_, name = self.visit(n.decl)

        self.memory_init(name, type_, 1, [n.body], 'text')
        return None, None

    def visit_If(self, n):
        # XXX can the condition actually be left off?
        if n.cond:
            with self.context('if_cond'):
                type_, cond = self.visit(n.cond)
        else:
            type_, cond = '_Bool', True

        if cond:
            return self.visit(n.iftrue)
        elif n.iffalse:
            return self.visit(n.iffalse)
        else:
            return None, None

    def visit_Typename(self, n):
        # TODO: don't know when this is not None
        # TODO: stuff with n.quals
        if n.name: assert False
        return self.visit(n.type)

    def visit_UnaryOp(self, n):
        assert n.op in unops

        # TODO: use types
        type_, val = self.visit(n.expr)
        if n.op == 'p++' or n.op == 'p--' or n.op == '++p' or n.op == '--p':
            with self.context('assignment_lvalue'):
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
            return type_[1:], self.memory[arr]['array'][offset]
        elif n.op == '&':
            assert is_pointer_type(type_)
            arr, offset = val
            # TODO: add pointer type, but only if not func pointer/constant array
            return ['*'] + type_, self.memory[arr]['array']
        else:
            return type_, unops[n.op](type_)(val)

    # TODO: detect infinite loop??
    def visit_While(self, n):
        while True:
            with self.context('expr'):
                ctype, cond = not n.cond or self.visit(n.cond)
            if not cond:
                break
            with self.context('stmt'):
                ret, retval = self.visit(n.stmt)
            if ret == 'break' or ret == 'return':
                return ret, retval
        return None, None

    def visit_Return(self, n):
        type_, ret = self.visit(n.expr) if n.expr else None
        return 'return',  ret

if __name__=='__main__':
    interpret = Interpreter()
    parser = c_parser.CParser()
    try:
        cfile = preprocess_file(sys.argv[1], cpp_path='clang', cpp_args=['-E', r'-I../fake_libc_include'])
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        sys.exit(1)

    # some kind of JIT after the first execution?
    interpret.visit(ast)
    interpret.execute(['./vigenere', 'HELlO'], 'wOrld\n')
    print(interpret.stdout)
    interpret.execute(['./vigenere', 'HELlO'], 'wOoorld\n')
    print(interpret.stdout)
