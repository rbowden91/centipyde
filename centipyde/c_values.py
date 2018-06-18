from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional

from .c_types import expand_type

class CInfo(object):
    def __init__(self, args):
        self.args = args

class CError(object):
    def __init__(self, message):
        self.message = message


# TODO: generics become a lot more powerful if we move away from "str" as the type for everything...
CTypeType = List[str]

class CType(object):
    # TODO: add actual like, c_int expansion here?
    __slots__ = ['type']
    def __init__(self, type_:CTypeType, expanded_type:CTypeType) -> None:
        self.type = type_
        self.expanded_type = expanded_type

    # def validate_val(self, val:CVal):

def make_ctype(type_:CTypeType, typedefs) -> CType:
    return CType(type_, expand_type(type_, typedefs))

# no such thing as "CTypedefVal"
# we don't store the name with these things, because theoretically they all end up as an assignment
# in memory anyway
class CAllocation(metaclass=ABCMeta): pass
class CTypeAllocation(CAllocation, metaclass=ABCMeta): pass
class CValAllocation(CAllocation):
    __slots__ = ['segment', 'type', 'name', 'len', 'array']
    def __init__(self, segment:str, type_:CType, name:str, len_:int, array:List[CVal]) -> None:
        self.segment = segment
        self.type = type_
        self.name = name
        self.len = len_
        self.array = array

#    def to_dict(self):
#        return {
#            'class': 'Memory',
#            'type': self.type,
#            'expanded_type': self.expanded_type,
#            'name': self.name,
#            'len': self.len,
#            'array': self.array,
#            'segment': self.segment
#        }


class CTypedefType(CTypeAllocation):
    def __init__(self, type_):
        self.type = type_

# This can't be a CTypeAllocation?
class CDeclType(object):
    __slots__ = ['name', 'type']
    def __init__(self, name, type_):
        self.name = name
        self.type = type_
        # TODO: also value, for ENUMs?

# TODO: opaque?
class CStructuredType(CTypeAllocation, meta=ABCMeta):
    __slots__ = ['decls']
    def __init__(self, decls : List[CDeclType]) -> None:
        self.decls = decls

class CStructType(CStructuredType): pass
class CUnionType(CStructuredType): pass
class CEnumType(CStructuredType): pass

#    def to_dict(self):
#        return {
#            'class': 'Address',
#            'base': self.base,
#            'offset': self.offset
#        }
#
#    def __str__(self):
#        return 'Address(' + str(self.base) + ', ' + str(self.offset) + ')'
#
#    # TODO: is __str__ necessary at this point? will str() automatically call repr if __str__ doesn't exist??
#    def __repr__(self):
#        return str(self)
#

#    def __str__(self):
#        return 'Val(' + str(self.type) + ', ' + str(self.value) + ')'
#
#    def __repr__(self):
#        return str(self)
#
#    def to_dict(self):
#        return {
#            'class': 'Val',
#            'type': self.type,
#            'value': self.val,
#        }
#
#class CTypeType(object):
    #pass


class CVal(metaclass=ABCMeta):
    pass

#class CArray(object):
#    def __init__(self, array):
#        self.array = array

# TODO: should this not be a val?
class CString(CVal):
    def __init__(self, string):
        self.string = string

# TODO: what about when they are opaque? decls=None?
# Should never deal with an address directly. It's either a pointer or a reference we're dealing with.
class CAddress(CVal, metaclass=ABCMeta):
    # TODO: some way of identifying when the thing this address is pointing to goes out of scope / is freed
    __slots__ = ['segment', 'scope', 'base', 'offset']

    # TODO: could use a bit more infrastructure to get rid of the Optional? Like, dependent on segment or something
    def __init__(self, segment:str, scope:Optional[Tuple[int,int]], base:str, offset:int) -> None:
        self.segment = segment
        self.scope = scope
        self.base = base
        self.offset = offset

class CReference(CAddress): pass # maybe more of a symbol than a reference?

class CPointer(CAddress): pass

class CFunction(object):
    # TODO: right way of doing this to enforce they're passed?
    def __init__(self, names, ret_type, params):
        self.names = names # in case the function has multiple names, like get_string v GetString
        self.ret_type = ret_type
        # TODO: check if params is valid, with ... only appearing at the end
        # XXX are these...types??
        self.params = params

    def call(self):
        assert False, "Function unimplemented!"

    # TODO: for like, getstring deallocations
    def __del__(self):
        pass

class CValue(object):
    __slots__ = ['type', 'val']
    def __init__(self, type_:CType, val:CVal) -> None:
        self.type = type_
        self.val = val
