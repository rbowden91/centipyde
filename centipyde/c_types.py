import ctypes
import operator

from abc import ABCMeta
from typing import NewType, List, Optional, Tuple, Union, Type



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

# TODO: this isn't nearly completely enough. for example, upcast int to long long
def implicit_cast(self, val1, val2):
    #self.k.info(('implicit_cast',))
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
    if is_char_type(val.expanded_type):
        # TODO: check this over
        return bytes(val.value, 'latin-1')[1 if val.value.startswith("'") else 0]
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
        return [(type_[0], expand_type(type_[1], typedefs), type_[2], expand_type(type_[3], typedefs))]
    elif isinstance(type_, list):
        ret = []
        for t in type_:
            ret = ret + expand_type(t, typedefs)
        return ret
    elif type_ in typedefs:
        return typedefs[type_]
    else:
        # TODO: how to handle this? Could still pass a continuation, but only to help with this error?
        #my_assert(type_ in python_type_map or type_.startswith('[') or type_ == '*' or type_ == '...', 'Invalid type')
        return [type_]

# TODO: depends on the value? idk...
def is_char_type(type_):
    return len(type_) == 1 and type_[0] == 'char'

def is_float_type(type_):
    return len(type_) == 1 and type_[0] in float_types

def is_func_type(type_):
    #return type_[0].startswith('(')
    return isinstance(type_, tuple)

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


#unops = {
#   '+': operator.pos,
#   '-': operator.neg,
#   '~': operator.inv,
#   '!': operator.not_,
#}
#
#unops['sizeof'] = lambda _: assert_false("TODO")
## shouldn't call these directly, since they require accessing memory/locals
#for k in ['*', '&', 'p++', '++p', 'p--', '--p']:
#    unops[k] = lambda _: assert_false("Shouldn't call these unops")
#
#def boolean_and(lval):
#    return lambda rval: rval
#
#def boolean_or(lval):
#    return lambda rval: rval
#
## TODO: check results for overflow???
#binops = {
#    '+': operator.add,
#    '-': operator.sub,
#    '*': operator.mul,
#    # TODO: division by zero runtime error
#    #'/': lambda type_: (type_, (operator.truediv if is_float_type(type_) else operator.floordiv)),
#    # TODO: assert something about type_ being int,
#    '%': operator.mod,
#
#    '|': operator.or_,
#    '&': operator.and_,
#    '^': operator.xor,
#    '<<': operator.lshift,
#    # TODO: something with unsigned right shift??,
#    '>>': operator.rshift,
#
#    # TODO: any issues with falsiness?,
#    # TODO: what type does a boolean comparison return?,
#    # TODO: these are returning True instead of 1
#    '==': operator.eq,
#    '!=': operator.neq,
#    '<': operator.lt,
#    '<=': operator.le,
#    '>': operator.gt,
#    '>=': operator.ge,
#
#    # && and || are special, because they can short circuit. By the time we call this function, short-circuiting
#    # should already have been checked for
#    '&&': boolean_and,
#    '||': boolean_or
#}

#            .if_(n.op == '&&' and not lval.value, lambda: self.k
#                .passthrough(lambda: self.make_val(['_Bool'], 0)))
#            .elseif(n.op == '||' and lval.value, lambda: self.k
#                .passthrough(lambda: self.make_val(['_Bool'], 1)))
#            .else_(lambda: self.k
#                .visit(n.right)
#                .expect(lambda rval: self.k
#                    .apply(lambda: self.implicit_cast(lval, rval))
#                    .expect(lambda lval: self.k.expect(lambda rval: self.k
#                        .kassert(lambda: n.op != '%' or is_int_type(lval.type), lval.type)
#                        .kassert(lambda: n.op != '%' or rval != 0, "Can't mod by zero")
#                        .kassert(lambda: n.op != '/' or rval != 0, "Can't divide by zero")
#                        .passthrough(lambda: binops[n.op](lval.type))
#                        .expect(lambda typeval: self.k.passthrough(
#                            lambda: self.make_val(typeval[0], typeval[1](lval.value, rval.value))))))))))
