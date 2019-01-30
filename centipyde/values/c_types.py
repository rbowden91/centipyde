from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional, NewType, Union, Type

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

ID = NewType('ID', str)
Decl = Tuple[CType, ID]
Param = Tuple[CType, Optional[ID]]

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
# TODO: what about when they are opaque? decls=None?
class StructuredType():
    __slots__ = ['decls']
    def __init__(self, decls : List[Decl]) -> None:
        self.decls = decls

StructType = NewType('StructType', StructuredType)
UnionType = NewType('UnionType', StructuredType)

EnumDecl = Tuple[ID, Optional[Int]]
class EnumType():
    __slots__ = ['decls']
    def __init__(self, decls : List[EnumDecl]) -> None:
        self.decls = decls

# The bool indicates whether the function is varargs or not (i.e., ...)
ParamList = Tuple[List[Param], bool]
class FunctionType():
    __slots__ = ['return_', 'params']
    def __init__(self, return_ : CType, params : ParamList) -> None:
        self.return_ = return_
        self.params = params

class TypedefType():
    __slots__ = ['type_']
    def __init__(self, type_ : CType) -> None:
        self.type_ = type_

CType = Union[Type[Integer],
             Type[Floating],
             PointerType,
             ArrayType,
             StructType,
             UnionType,
             EnumType,
             FunctionType,
             TypedefType]
