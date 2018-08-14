# Can't use generators, because then we can't readily reverse?
import sys
import os
import subprocess

# TODO: eventually handle this. pycparser doesn't play nicely with mypy yet
from pycparser import c_ast, c_parser # type: ignore

from .crt import CRuntime
from .continuation import Continuation



def preprocess_file(file_, is_code=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    include_path = os.path.join(dir_path, 'clib/build/include')
    cpp_args = [r'cpp', r'-E', r'-g3', r'-gdwarf-2', r'-nostdinc', r'-I' + include_path,
                r'-D__attribute__(x)=', r'-D__builtin_va_list=int', r'-D_Noreturn=', r'-Dinline=', r'-D__volatile__=']
    cpp_args.append(file_ if not is_code else '-')

    # reading from stdin
    proc = subprocess.Popen(cpp_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, encoding='latin-1')
    stdout, stderr = proc.communicate(bytes(file_, 'latin-1') if is_code else None)
    if len(stderr) != 0:
        print('Uh oh! Stderr messages', proc.stderr)
    elif proc.returncode != 0:
        print('Uh oh! Nonzero error code')
    else:
        return stdout

# TODO: use getcwd for filename?
def init_interpreter(file_, is_code=False):
    parser = c_parser.CParser()
    # TODO: need to check for errors
    try:
        cfile = preprocess_file(file_, is_code)
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        sys.exit(1)

    return Interpreter(ast)

# TODO: cache results from particularly common subtrees?

# Make this a subclass, so the Continuation module doesn't need to know anything about
# Interpreters. This is mostly just convenience methods.
class InterpContinuation(Continuation):

    __slots__ = ['interpreter']
    def __init__(self, interpreter):
        self.interpreter = interpreter
        super().__init__()

    def visit(self, node):
        assert isinstance(node, c_ast.Node)
        self.apply(lambda: self.interpreter.visit(node))
        return self

    def push_context(self, context):
        (self.apply(lambda: self.interpreter.context.append(context),
                    lambda: self.interpreter.context.pop()))
        return self

    def pop_context(self):
        # the nice thing about wrapping things in a lambda instead of passing
        # self.interpreter.pop_context directly is that now we know where the
        # lambda was defined. Not particularly interesting in this case, though.

        # TODO: provide more information about the caller, so that we have better
        # line numbers for the lambda printing? Almost like inlining... Actually, better yet,
        # keep around the full lambda stacktrace wooo
        (self.passthrough(lambda: self.interpreter.context[-1])
        .expect(lambda context:
            self.apply(lambda: self.interpreter.context.pop(),
                       lambda: self.interpreter.context.append(context))))
        return self

    # call out to the C runtime, which might touch some global state
    def side_effect(self, *args):
        (forward, reverse, error, result) = self.interpreter.crt.side_effect(args)
        self.apply(forward, reverse)
        return self


    #def push_scope(self):
    #    (self.info(('push_scope',{}), ('pop_scope',{}))
    #    .apply(lambda: self.cwrapper.scope[-1].append({}),
    #           lambda: self.cwrapper.scope[-1].pop()))
    #    return self

    #def pop_scope(self):
    #    (self.passthrough(lambda: self.cwrapper.scope[-1][-1])
    #    .expect(lambda scope:
    #        self.info(('pop_scope',scope),('push_scope',scope))
    #        .apply(lambda: self.cwrapper.scope[-1].pop(),
    #               lambda: self.cwrapper.scope[-1].append(scope))))
    #    return self


    #def declare_typedef(self, name, type_):
    #    (self.info(('declare_typedef', name, type_), ('remove_typedef', name))
    #    .apply(lambda: self.cwrapper.declare_typedef(name, type_),
    #           lambda: self.cwrapper.remove_typedef(name))
    #    .passthrough(lambda: Flow('Normal')))
    #    return self

    #def declare_union(self, name, decls):
    #    (self.apply(lambda: self.passthrough(self.cwrapper.declare_union(name, decls)),
    #                lambda: self.cwrapper.remove_union(name))
    #    .expect(lambda union:
    #        self.info(('declare_union', union), ('remove_union', union))
    #        .passthrough(lambda: union)))
    #    return self

    ## TODO: update struct, after a forward declaration??
    #def declare_struct(self, name, decls):
    #    (self.apply(lambda: self.passthrough(self.cwrapper.declare_struct(name, decls)),
    #                lambda: self.cwrapper.remove_struct(name))
    #    .expect(lambda struct:
    #        self.info(('declare_struct', struct), ('remove_struct', struct))
    #        .passthrough(lambda: struct)))
    #    return self

    #def pop_func_scope(self):
    #    (self.passthrough(lambda: self.cwrapper.scope[-1][-1])
    #    .expect(lambda scope:
    #        self.info(('pop_func_scope',scope),('push_func_scope',scope))
    #        .apply(lambda: self.cwrapper.scope.pop(),
    #               lambda: self.cwrapper.scope.append(scope))))
    #    return self

    #def push_func_scope(self, scope=None):
    #    # append the global map
    #    (self.passthrough(lambda: [self.cwrapper.scope[-1][0]])
    #    .expect(lambda scope:
    #        (self.info(('push_func_scope',scope), ('pop_func_scope',scope))
    #        .apply(lambda: self.cwrapper.scope.append([{}]),
    #            lambda: self.cwrapper.scope.pop()))))
    #    return self

    ## TODO: preemptively expand, so that expand_type doesn't have to expand until no changes
    ## are made?

    ## XXX XXX XXX: these assignment values should passthrough
    #def declare_id(self, id_, val):
    #    (self.k.info(('declare_id', {'name': id_, 'value': val}), ('remove_id', {'name': id_}))
    #    .apply(lambda: self.cwrapper.declare_id(id_, val),
    #           lambda: self.cwrapper.remove_id(id_)))

    #def update_id(self, id_, val, scope=-1):
    #    (self.k.passthrough(lambda: self.scope[-1]['ids'][id_])
    #    .expect(lambda old_val:
    #        self.k.info(('update_id', {'name': id_, 'value': val, 'scope': scope}),
    #                    ('update_id', {'name': id_, 'value': old_val, 'scope': scope}))
    #        .apply(lambda: self.k.passthrough(self.cwrapper.update_id(id_, val, scope)),
    #               lambda: self.cwrapper.update_id(id_, oldval, scope))))

    ## TODO: need to handle stack pointers eventually
    #def allocate_memory(self, name, type_, len_, array, segment):
    #    (self
    #    .apply(lambda: self.k.passthrough(lambda: self.cwrapper.allocate_memory(name, type_, len_, array, segment)),
    #           lambda: self.cwrapper.deallocate_memory(name))
    #    .expect(lambda memory:
    #            self.k.info(('allocate_memory', memory), ('deallocate_memory', name))
    #            .passthrough(lambda: memory)))

    #def deallocate_memory(self, name):
    #    # TODO
    #    pass


    #def update_memory(self, base, offset, val):
    #    # TODO: use a getter instead of directly accessing?
    #    # TODO: check types
    #    (self.k.passthrough(lambda: self.cwrapper.memory[base].array[offset])
    #    .expect(lambda old_val:
    #        self.k.info(('update_memory', {'name': id_, 'base': base, 'offset': offset, 'value': val}),
    #                    ('update_memory', {'name': id_, 'base': base, 'offset': offset, 'value': old_val}))
    #        .apply(lambda: self.k.passthrough(self.cwrapper.update_memory(base, offset, val)),
    #               lambda: self.cwrapper.update_memory(base, offset, oldval))))

    ## TODO: this should really be done at the beginning of the program, to pre-load everything into memory...
    ## TODO: also, globals. Oof. Those aren't on the stack...
    #def handle_string_const(self, type_):
    #    # TODO: check how it's being declared, since we might be doing a mutable character array
    #    # TODO: move this into cwrapper
    #    self.k.info('strconst')

    #    if n.value in self.string_constants:
    #        new = False
    #        name = self.cwrapper.string_constants[n.value]
    #    else:
    #        new = True
    #        const_num = len(self.string_constants)
    #        # should be unique in memory, since it has a space in the name
    #        name = 'strconst ' + str(const_num)
    #        self.cwrapper[n.value] = name
    #        # append 0 for the \0 character
    #        array = bytes(n.value, 'latin-1') + bytes([0])
    #        self.memory_init(name, type_, len(array), array, 'rodata')
    #    self.k.passthrough(lambda: self.make_val(type_, Address(name, 0)))

    # TODO: extend passthrough so we add in where the value was generated to each Val that's
    # passed through?
    # okay I actually did this, but it's not helpful for things pulled from memory. Remember in
    # memory the origin of a value??

# This only knows about "flows". No other values are really relevant
class Flow(object):
    __slots__ = ['type', 'value']
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value

    def __str__(self):
        return 'Flow(' + str(self.type) + ', ' + str(self.value) + ')'

    def __repr__(self):
        return str(self)

    def to_dict(self):
        return {
            'class': 'Flow',
            'type': self.type,
            'value': self.value,
        }

class Interpreter(object):
    # TODO: pass require_decls through to cwrapper
    def __init__(self, ast, require_decls=True):
        # If false, we don't require type declarations before assigning to a variable (unlike regular C)
        self.require_decls = require_decls

        self.ast = ast
        self.k = InterpContinuation(self)

        # context isn't that interesting. it's just what "context" we're in at a given node. In particular, are we being
        # actively parsed as an lvalue or an rvalue
        self.context = ['global']

        # read in all the typedefs, funcdefs, globals, etc.
        # this is effectively "compiling", but only by looking at top-level things
        # TODO: Probably need to make explicit the iteration over globals
        self.runtime = CRuntime(require_decls)
        self.visit(ast)
        self.run()

        # so that we don't have to directly access the continuation
        self.step = self.k.step
        self.revstep = self.k.revstep

    # run until we can't process things anymore
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
        # get the return value and store it in interpreter here?

    # We don't really need the caching anymore that pycparser uses, since we no longer visit nodes multiple times
    def visit(self, node):
        #node.show(showcoord=True)
        method = 'visit_' + node.__class__.__name__
        ret = getattr(self, method)(node)
        assert ret is None

    def visit_ArrayDecl(self, n):
        (self.k.info(n)
        .visit(n.type)
        .expect(lambda type_: self.k
            .if_(n.dim, lambda: self.k.visit(n.dim).expect(lambda dim: self.k.passthrough(lambda: str(dim.value))))
            .else_(lambda: self.k.passthrough(lambda: ''))
            .expect(lambda dim: self.k.passthrough(lambda: [type_ + '[' + dim + ']'] + type_))))

    def visit_ArrayRef(self, n):
        # TODO: shouldn't be allowed to assign to symbols / consts? Actually, symbols would fall under visit_ID
        # TODO: nested array refs?? argv[1][2]. need to handle dims appropriately

        (self.k.info(n)
        # we want to get rid of any lvalue context if we are eventually assigning to the array
        .push_context('arrayref')
        .visit(n.name)
        .expect(lambda arr: self.k.visit(n.subscript).expect(lambda idx:
            (self.k.pop_context()
            # if they did something like 2[arr], which is legal, crt will handle that
            # XXX XXX XXX XXX
            .side_effect('')))))


       #     .kassert(lambda: is_pointer_type(idx) != is_pointer_type(arr), "Exactly one must be an address"))
       #     # TODO: this is an example of why everything should be lambdas. we can't actually
       #     # print idx before this
       #     .if_(is_pointer_type(idx), lambda: self.k.passthrough(lambda: idx.value).passthrough(lambda: arr.value))
       #     .else_(lambda: self.k.passthrough(lambda: arr.value).passthrough(lambda: idx.value))))

       # .expect(lambda arr: self.k.expect(lambda idx:
       #     self.k
       #     .kassert(lambda: idx + arr.offset < self.memory[arr.base].len,
       #         'Out of bounds array:{} idx:{} length:{}'.format(arr.base, idx, self.memory[arr.base].len))
       #     .if_(self.context[-1] == 'lvalue',
       #         lambda: self.k
       #         .kassert(lambda: self.memory[arr.base].segment not in ['rodata', 'text', 'NULL'],
       #                  self.memory[arr.base].segment)
       #         .passthrough(lambda: lambda val:
       #             self.update_memory(arr.base, idx+arr.offset, val)))
       #     .else_(lambda: self.k
       #         .passthrough(lambda: self.make_val(self.memory[arr.base].type,
       #                      self.memory[arr.base].array[idx + arr.offset]))))))

    def visit_Assignment(self, n):
        # TODO: handle others
        def assign_helper(lval, rval, assignment_op):
            if n.op == '-=':
                value = lval.value - rval.value
            elif n.op == '+=':
                value = lval.value + rval.value
            else:
                assert False, 'Unsupported unop'
            value = self.make_val(lval.type, value)
            self.k.apply(lambda: assignment_op(value)).passthrough(lambda: value)

        (self.k.info(n)
        .push_context('lvalue')
        .visit(n.lvalue)
        .pop_context()
        .expect(lambda assignment_op: self.k
            .push_context('rvalue')
            .visit(n.rvalue)
            .expect(lambda val: self.k
                # XXX XXX XXX
                .side_effect('update', 'variable', n.op))
                # assignments return the value of the assignment
                # TODO: handle more than just these cases
                #.if_(n.op == '=', lambda: self.k.apply(lambda: assignment_op(val)).passthrough(lambda: val))
                ## not really an rvalue, but we need it
                #.else_(lambda: self.k.visit(n.lvalue).expect(lambda lval: self.k
                #    .apply(lambda: assign_helper(lval, val, assignment_op)))))
            .pop_context()))

    def visit_BinaryOp(self, n):
        assert n.op in binops

        # TODO cast to c type nonsense
        (self.k.info(n)
        .visit(n.left)
        .expect(lambda lval: self.k
            # XXX XXX XXX XXX XXX
        ))
            #.if_(n.op == '&&' and not lval.value, lambda: self.k
            #    .passthrough(lambda: self.make_val(['_Bool'], 0)))
            #.elseif(n.op == '||' and lval.value, lambda: self.k
            #    .passthrough(lambda: self.make_val(['_Bool'], 1)))
            #.else_(lambda: self.k
            #    .visit(n.right)
            #    .expect(lambda rval: self.k
            #        .apply(lambda: self.implicit_cast(lval, rval))
            #        .expect(lambda lval: self.k.expect(lambda rval: self.k
            #            .kassert(lambda: n.op != '%' or is_int_type(lval.type), lval.type)
            #            .kassert(lambda: n.op != '%' or rval != 0, "Can't mod by zero")
            #            .kassert(lambda: n.op != '/' or rval != 0, "Can't divide by zero")
            #            .passthrough(lambda: binops[n.op](lval.type))
            #            .expect(lambda typeval: self.k.passthrough(
            #                lambda: self.make_val(typeval[0], typeval[1](lval.value, rval.value))))))))))


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

    def visit_Cast(self, n):
        # TODO: validate
        (self.k.info(n)
        .visit(n.to_type)
        .expect(lambda type_: self.k
            .visit(n.expr).expect(lambda val:
                # either we can let the cast go through automatically, and only complain that it was invalid
                # when it gets used, or complain immediately upon casting. In the latter case, it's a side effect
                self.k.passthrough(lambda: self.k.side_effect('cast', type_, val)))))

    def visit_Compound(self, n):
        # TODO: do we actually ever even care to stop on this node?
        (self.k.info(n)
        .loop(n.block_items, lambda stmt:
            self.k.visit(stmt)
            .expect(lambda flow:
                self.k
                .if_(isinstance(flow, Flow) and flow.type != 'Normal', lambda: self.k
                    # pass through to the looper that we should shortcircuit
                    .passthrough(lambda: (True, flow)))
                .else_(lambda: self.k.passthrough(lambda: (False, Flow('Normal'))))), shortcircuit=True)
        .expect(lambda flows:
            #self.k.passthrough(lambda: Flow('Normal', None) if isinstance(flows[-1], Val) else flows[-1])))
            self.k.passthrough(lambda: Flow('Normal', None) if not isinstance(flows[-1], Flow) else flows[-1])))

    def visit_Constant(self, n):
        # TODO: necessary to expand the type??
        (self.k.info(n)
        .side_effect('handle_constant', n.type, n.value))
        #if not is_string_type(n.type):
        #    self.k.passthrough(lambda: self.make_val([n.type], cast_to_python_val(self.make_val([n.type], n.value))))
        #else:
        #        self.handle_string_const(n.value)

    def visit_Continue(self, n):
        #assert continuations['continue'] is not None, 'Continue in invalid context'
        # TODO: put the loop in scope context
        (self.k.info(n).passthrough(lambda: Flow('Continue')))

    # name: the variable being declared
    # quals: list of qualifiers (const, volatile)
    # funcspec: list function specifiers (i.e. inline in C99)
    # storage: list of storage specifiers (extern, register, etc.)
    # type : declaration type (probably nested with all the modifiers)
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
                .expect(lambda val: self.k
                    .apply(lambda: self.state_change('declare', 'variable', n.name, type_, val.value))))
            .else_(lambda: self.k.apply(lambda: self.state_change('declare', 'variable', n.name, type_, None)))))
        # XXX XXX XXX *shouldn't* pass flow, becasue of structs
        #.passthrough(lambda: Flow('Normal')))

    def visit_Decl(self, n):
        # TODO: can n.type ever be None? only for funcdecl?
        if not n.type: return lambda type_: self.decl_helper(n, type_)
        else: self.decl_helper(n, None)


    def visit_DeclList(self, n):
        (self.k.info(n)
        .loop(n.decls, lambda decl:
            self.k.visit(decl)))
        # we want the decls to pass through. Like for a struct's decls
        #.expect(lambda decls: None)
        #.passthrough(lambda: Flow('Normal')))

    def visit_ExprList(self, n):
        (self.k.info(n)
        .loop(n.exprs, lambda expr: self.k.visit(expr)))

    # top-level node
    def visit_FileAST(self, n):
        # TODO: put global map placement in here??
        (self.k.info(n)
        .loop(n.ext, lambda ext: self.k.visit(ext))
        # don't really care about the return.
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
            .expect(lambda args: self.k.side_effect('call', 'memory', name.value, args))))

                #.if_(self.memory[name.value].type[0][0] == '(builtin)', lambda: self.k
                #    .apply(lambda: self.memory[name.value].array[0](args)))
                #.else_(lambda: self.k
                #    .kassert(lambda: self.memory[name.value].type[0][0] == '(user-defined)', 'blah3')
                #    # TODO: make these push_func_scope
                #    .apply(self.push_func_scope)
                #    .visit(self.memory[name.value].array[0])
                #    .apply(self.pop_func_scope)
                #    # TODO: check return type
                #    .expect(lambda flow:
                #        self.k
                #        .kassert(lambda: flow.type == 'Return', 'Didn\'t return from FuncCall')
                #        .passthrough(lambda: flow.value))))))

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

    def visit_FuncDef(self, n):
        # TODO: check against prior funcdecls?
        (self.k.info(n)
        .visit(n.decl)
        .expect(lambda val:
            self.k.side_effect('allocate', 'memory', n.decl.name, val.type, 1, [n.body], 'text'))
            #self.k.apply(lambda: self.memory_init(n.decl.name, val.type, 1, [n.body], 'text')))
        .passthrough(lambda: Flow('Normal')))


    def visit_ID(self, n):
        (self.k.info(n)
        .if_(self.context[-1] == 'lvalue', lambda: self.k
            .passthrough(lambda: lambda val: self.k.side_effect('update', 'variable', n.name, val)))
        .else_(lambda: self.k
            .passthrough(lambda: self.k.side_effect('get', 'variable', n.name))))

    def visit_IdentifierType(self, n):
        self.k.info(n).passthrough(lambda: n.names)

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

    def visit_ParamList(self, n):
        # TODO: can param.name be empty?
        # TODO: ellipsisparam should only go at the end
        (self.k.info(n)
        .loop(n.params, lambda param:
            self.k
            .if_(isinstance(param, c_ast.EllipsisParam), lambda:
                self.k.passthrough(lambda: ('...', None)))
            .else_(lambda: self.k
                .visit(param.type)
                .expect(lambda ptype: self.k.passthrough(lambda: (param.name, ptype))))))


    def visit_PtrDecl(self, n):
        (self.k.info(n)
        .visit(n.type)
        .expect(lambda type_: self.k.passthrough(lambda: type_ + ['*'])))

    def visit_Return(self, n):
        self.k.info(n)
        if n.expr: self.k.visit(n.expr)
        # TODO: we very explicitly want this to pass through to self.k.expect, not to the upper continuation
        else: self.k.passthrough(lambda: None)
        self.k.expect(lambda val: self.k.passthrough(lambda: Flow('Return', val)))

        # TODO: can we even write a return outside of a function?
        # TODO: we need to check this elsewhere
        #assert continuations['return'] is not None, 'Return in invalid context'

    def visit_Struct(self, n):
        # TODO: strong copy-paste between this and union
        # TODO: finish this
        (self.k.info(n)
        .loop(n.decls, lambda decl: self.k.visit(decl))
        .expect(lambda decls: self.k.side_effect('allocate', 'struct', n.name, decls)))

    # Like if, but returns the Val instead of the Flow.
    # TODO:shouldn't it disallow
    # some things inside of it?
    def visit_TernaryOp(self, n):
        (self.k.info(n)
        .visit(n.cond)
        .expect(lambda cond: self.k
            .if_(cond.value, lambda: self.k.visit(n.iftrue))
            .else_(lambda: self.k.visit(n.iffalse))))

    def visit_TypeDecl(self, n):
        self.k.info(n).visit(n.type)

    def visit_Typedef(self, n):
        # TODO: quals, storage, etc.
        (self.k.info(n).visit(n.type)
        .expect(lambda type_: self.k
            .apply(lambda: self.k.side_effect('declare', 'typedef', n.name, type_))))

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

    def visit_Union(self, n):
        # TODO: finish this
        (self.k.info(n)
        .loop(n.decls, lambda decl: self.k.visit(decl))
        .expect(lambda decls: self.k.side_effect('allocate', 'union', n.name, decls)))

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

