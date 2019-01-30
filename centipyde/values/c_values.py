from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional, NewType, Union, Type

from .c_types import CType

# TODO:
class CStruct():
    pass

class CEnum():
    pass

class CUnion():
    pass

class CAddress(object):
    # TODO: some way of identifying when the thing this address is pointing to goes out of scope / is freed
    __slots__ = ['allocation', 'offset']

    # TODO: could use a bit more infrastructure to get rid of the Optional? Like, dependent on segment or something
    def __init__(self, allocation:ValAllocation, offset:int) -> None:
        self.allocation = allocation
        self.offset = offset

class CFunction(object):

    # args should already have been validated by the time we get here
    def call(self, **kwargs):
        assert False, "Function unimplemented!"

    # TODO: for like, getstring deallocations
    def __del__(self):
        pass

Val = Union[int, float, CStruct, CEnum, CUnion, CAddress, CFunction]

# this is also used for "constants"
# TODO: make this abstract?
class Allocation(object): pass

class TypeAllocation(Allocation):
    __slots__ = ['type']
    def __init__(self, type_ : CType) -> None:
        self.type = type_

class ValAllocation(Allocation):
    __slots__ = ['len', 'items']
    def __init__(self, len_ : int, items : List[Val]) -> None:
        self.len = len_
        self.items = items
