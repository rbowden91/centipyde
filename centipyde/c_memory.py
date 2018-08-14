# TODO: look more into how C/clang handles unicode
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Tuple, Callable, Optional, Generic, TypeVar, Union, Optional, Type

from .c_values import *


# TODO: anything that modifies memory we might want to manually implement to check for aliasing or what not?
# TODO: malloc definitely needs to be done ourselves, right?
# TODO: oh, just do interposition for malloc, and that additionally helps us with GetString (and others). In fact, can
# we get where the memory was allocated?
# TODO: can you re-typedef something if the types don't conflict? In C11, evidently yes.
# TODO: can a typedef happen before the base type exists?? forward declarations?
# TODO: complain about unused/various warnings?

# segment, namespace, tag, (scope[0], scope[1])?, actual identifier name
#CIdentifier = Tuple[str, Optional[str], Optional[str], Optional[Tuple[int,int]], str]
# segment, name, tag
# TODO: NAME OF THE ALLOCATION VS REFERENCE TO IT
CIdentifier = Tuple[str, str, Optional[str]]

# TODO: make this no longer optional
CCallback = Callable[[], Optional[Val]]
CStateModifier = Union[CError, Val, Tuple[CCallback, CCallback]]

class CState(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None: pass

    @abstractmethod
    def allocate(self, name:CIdentifier, type_:CType, val:List[Val]) -> CStateModifier: pass

    @abstractmethod
    def update(self, name:CIdentifier, val:CValue) -> CStateModifier: pass

    @abstractmethod
    def get(self, name:CIdentifier) -> CStateModifier: pass

# how to apply it, how to undo it, error condition, and optional return value?
#T = TypeVar('T', bound=CValue)
#class CScope(Generic[T], CState):
class CScope(CState):
    def __init__(self) -> None:
        # Optional, because it might have been declared but not initialized
        # TODO: add in file scope
        self.scope : Dict['str', CValue] = {}

    # for use by subclasses
    def has(self, name:CIdentifier) -> bool:
        return name[1] in self.scope

    def do_set(self, name:str, val:CValue) -> None:
        self.scope[name] = val

    def do_delete(self, name:str) -> None:
        assert name in self.scope
        del self.scope[name]

    def do_get(self, name:str) -> CValue:
        val = self.scope[name]
        assert val is not None
        return val


    # TODO: technically, we can have multiple forward declarations of a struct, union, etc. (function decls too?)
    # TODO: this is also allowed for typedefs in C11, as long as the types don't conflict
    # TODO: also multiple weak declarations of globals?
    # TODO: something with typechecking and val?
    # TODO: put len_ back in so that we can have "sparse" arrays?
    def allocate(self, name_:CIdentifier, type_:CType, val:List[CVal]) -> CStateModifier[CValue]:
        if name_[1] in self.scope:
            # TODO: make this more explicit (add scopes on the way back up?)
            return CError("Identifier {} already exists in scope".format(name_[1]))
        # TODO: check that the type_ and vals actually align?
        # XXX XXX XXX
        # HMMM... this isn't true for ids/typedefs/variables
        new_val = CValue(type_, CMemoryAllocation(name_[0], type_, name_[1], len(val), val))

        forward = lambda: self.do_set(name_[1], new_val)
        reverse = lambda: self.do_delete(name_[1])
        return (forward, reverse)

    def get(self, name_:CIdentifier) -> CStateModifier[CValue]:
        name = name_[1]
        if name not in self.scope:
            # TODO: can we get the name of T?
            return CError("Identifier {} does not exist in scope".format(name))

        if self.scope[name].val is None:
            # TODO: this might be fine if, say, global
            return CError("Identifier {} is uninitialized in scope".format(name))

        return self.do_get(name)

    # TODO: shouldn't be able to update the type, only the actual value??
    def update(self, name_:CIdentifier, value:CValue) -> CStateModifier[CValue]:
        name = name_[1]
        if name not in self.scope:
            return CError("Identifier {} does not exist in scope".format(name))
        old_val = self.do_get(name)
        forward = lambda: self.do_set(name, value)
        reverse = lambda: self.do_set(name, old_val)
        return (forward, reverse)

# namespace for enums vs namespace as in static
class CTagScope(object):
    def __init__(self):
        self.enums = CScope[CEnumType]()
        self.unions = CScope[CUnionType]()
        self.structs = CScope[CStructType]()
        # TODO: use this to map to all the values that are referencing this variable. That way, when the variable goes
        # out of scope or something, we can mark the addresses as invalid, to protect against their use. (Circular
        # reference?)
        self.ids = CScope[Union[CTypedefType, CValAllocation]]()

    def get_by_tag(self, name:CIdentifier) -> CScope:
        assert name[2] is not None
        return getattr(self, name[2])

    # TODO: make the other ABC methods

# this is basically just the stack
# TODO: use a different T?
class CAutomaticScope(CState):
    def __init__(self) -> None:
        # first list is for regular function-level scoping, the second is for block-level
        self.scope : List[List[CTagScope]] = []

    def do_push_func_scope(self) -> CInfo:
        self.scope.append([])
        return CInfo([])

    def do_pop_func_scope(self) -> CInfo:
        self.scope.pop()
        return CInfo([])

    def do_push_scope(self) -> CInfo:
        # TODO: some type variable based on generic scope???
        self.scope[-1].append(CTagScope())
        return CInfo([])

    def do_pop_scope(self) -> CInfo:
        self.scope[-1].pop()
        return CInfo([])

    def allocate(self, name:CIdentifier, type_:CType, val:List[CVal]) -> CStateModifier[CValue]:
        return self.scope[-1][-1].get_by_tag(name).allocate(name, type_, val)

    def find_defining_scope(self, name:CIdentifier) -> Optional[int]:
        for i in range(len(self.scope)-1,-1,-1):
            if self.scope[-1][i].get_by_tag(name).has(name):
                return i
        return None

    def get(self, name:CIdentifier) -> CStateModifier[CValue]:
        scope = self.find_defining_scope(name)
        if scope is None:
            return CError("Identifier {} does not exist in scope".format(name))
        return self.scope[-1][scope].get_by_tag(name).get(name)

    def update(self, name:CIdentifier, val:CValue) -> CStateModifier[CValue]:
        scope = self.find_defining_scope(name)
        if scope is None:
            return CError("Identifier {} does not exist in scope".format(name))
        return self.scope[-1][scope].get_by_tag(name).update(name, val)

# TODO: some kind of merged threading? like, if only global state is changing, but the instructions are still in sync,
# aren't we good?? I guess this is kind of COW.
# TODO: mimic execute bit even????
class CMemory(CState):
    # TODO: static/file scope, extern, weak and strong refs, errno, symbols vs ids. extern can apparently make something
    # that is static actually global????? hmm... maybe that's C++-specific
    __slots__ = ['envp', 'argv', 'stack', 'heap', 'data', 'rodata', 'text', 'NULL']

    def __init__(self) -> None:
        # we need to handle what happens if something that is being pointed to "goes out of scope" / is freed / etc.
        # enums, unions, structs, and typedefs will all go in here as well, just because they are scoped, but they
        # shouldn't be considered as "taking up memory"
        #self.envp = CStaticSegment(),
        #self.argv = CStaticSegment(),
        #self.stack = CAutomaticIDSegment(),
        ## mmap
        #self.heap = CDynamicSegment(),
        ## for now, don't distinguish
        ##'uninitialized': CMemorySegment,
        ##'initialized': CMemorySegment,
        #self.data = CStaticNamespacedIDSegment(),
        ## for now, we only put string constants here, and not global/static consts
        #self.rodata = CRodataSegment(),
        ## TODO: this is a different definition of Namespaced
        #self.text = CReadonlyStaticNamespacedSegment(),
        #self.NULL = CNullSegment() # aka Unmapped Segment
        self.envp = CScope()
        self.argv = CScope()
        self.stack = CAutomaticScope()
        self.heap = CScope()
        self.data = CTagScope()
        self.rodata = CScope()
        self.text = CScope()
        self.NULL = CScope() # aka Unmapped Segment

    def allocate(self, name_:CIdentifier, type_:CType, val:List[Val]) -> CStateModifier[CValue]:
        return getattr(self, name_[0]).allocate(self, name_, type_, val)

    def get(self, name_:CIdentifier) -> CStateModifier[CValue]:
        return getattr(self, name_[0]).get(self, name_)

    # TODO: shouldn't be able to update the type, only the actual value??
    def update(self, name_:CIdentifier, value:CValue) -> CStateModifier[CValue]:
        return getattr(self, name_[0]).update(self, name_, value)
