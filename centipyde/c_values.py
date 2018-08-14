from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional, NewType, Union, Type

from .c_types import expand_type

class CInfo(object):
    def __init__(self, args):
        self.args = args

class CError(object):
    def __init__(self, message):
        self.message = message

# TODO: evidently, something like "unsigned x", where x is typedefed to int, isn't allowed

# TODO: too many casting rules...
# https://www.safaribooksonline.com/library/view/c-in-a/0596006977/ch04.html

# TODO: such a thing as unsigned _Bool?

Bool = NewType('Bool', int)
Char = NewType('Char', int)
UChar = NewType('UChar', int)
Short = NewType('Short', int)
UShort = NewType('UShort', int)
Int = NewType('Int', int)
UInt = NewType('UInt', int)
Long = NewType('Long', int)
ULong = NewType('ULong', int)
LongLong = NewType('LongLong', int)
ULongLong = NewType('ULongLong', int)
Integer = Union[Bool, Char, UChar, Short, UShort, Int, UInt, Long, ULong, LongLong, ULongLong]

Float = NewType('Float', float)
Double = NewType('Double', float)
LongDouble = NewType('LongDouble', float)
Floating = Union[Float, Double, LongDouble]

CType = Union[Type[Integer],
             Type[Floating],
             'PointerType',
             'ArrayType',
             'StructType',
             'UnionType',
             'EnumType',
             'FunctionType',
             'TypedefType']

ID = NewType('ID', str)
Decl = Tuple[CType, ID]
Param = Tuple[CType, Optional[ID]]

# The bool indicates whether the function is varargs or not (i.e. ...)
ParamList = Tuple[List[Param], bool]
EnumDecl = Tuple[ID, Optional[Int]]


class PointerType():
    __slots__ = ['type_']
    def __init__(self, type_:CType) -> None:
        self.type_ = type_

class ArrayType():
    __slots__ = ['length', 'type_']
    def __init__(self, length:Optional[Int], type_:CType) -> None:
        self.length = length
        self.type_ = type_

# TODO; should be abstract?
class StructuredType():
    __slots__ = ['decls']
    def __init__(self, decls : List[Decl]) -> None:
        self.decls = decls

StructType = NewType('StructType', StructuredType)
UnionType = NewType('UnionType', StructuredType)

class EnumType():
    __slots__ = ['decls']
    def __init__(self, decls : List[EnumDecl]) -> None:
        self.decls = decls

class FunctionType():
    __slots__ = ['return_', 'params']
    def __init__(self, return_ : CType, params : ParamList) -> None:
        self.return_ = return_
        self.params = params

class TypedefType():
    __slots__ = ['type_']
    def __init__(self, type_ : CType) -> None:
        self.type_ = type_

TypeVal = Union[StructType, UnionType, EnumType, FunctionType, TypedefType]
# TODO:
# Struct
# Union
# Enum
Val = Union[Integer, Floating, Address, Function, TypeVal]

# TODO: what about when they are opaque? decls=None?
class Address():
    # TODO: some way of identifying when the thing this address is pointing to goes out of scope / is freed
    __slots__ = ['segment', 'scope', 'base', 'offset']

    # TODO: could use a bit more infrastructure to get rid of the Optional? Like, dependent on segment or something
    def __init__(self, segment:str, scope:Optional[Tuple[int,int]], base:str, offset:int) -> None:
        self.segment = segment
        self.scope = scope
        self.base = base
        self.offset = offset

class Function(object):

    # args should already have been validated by the time we get here
    def call(self, **kwargs):
        assert False, "Function unimplemented!"

    # TODO: for like, getstring deallocations
    def __del__(self):
        pass

class Allocation():
    __slots__ = ['segment', 'type', 'len', 'items', 'symbol']
    def __init__(self, segment : str, type_ : CType, symbol : Optional[str], len_ : int, items : List[Val]) -> None:
        self.segment = segment
        self.type = type_
        self.symbol = symbol
        self.len = len_
        self.items = items



class CValue(object):
    __slots__ = ['type', 'val']
    def __init__(self, type_:CType, val:Val) -> None:
        self.type = type_
        self.val = val
