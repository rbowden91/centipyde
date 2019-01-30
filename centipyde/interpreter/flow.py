from enum import Enum, auto
from typing import Optional

from ..values.c_values import Allocation

# This only knows about "flows". No other values are really relevant
class FlowType(Enum):
    norm = auto()
    cont = auto()
    brk = auto()
    ret = auto()

class Flow(object):
    __slots__ = ['type', 'value']
    def __init__(self, type_:FlowType, value:Optional[Allocation] = None) -> None:
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

