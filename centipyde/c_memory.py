# TODO: add in file scope
# TODO: look more into how C/clang handles unicode
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Tuple, Callable, Optional, Generic, TypeVar, Union, Optional, Type

from .values.c_values import Allocation, ValAllocation, Val


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
#CSegmentModifier =  Union[CError, Val, Tuple[CCallback, CCallback]]
CSegmentModifier = Union[Val, Tuple[CCallback, CCallback]]

CScope = List[ValAllocation]
CNamedScope = Dict[str, Allocation]

# TODO: some kind of merged threading? like, if only global state is changing, but the instructions are still in sync,
# aren't we good?? I guess this is kind of COW.
# TODO: mimic execute bit even????
class CMemory(object):
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

        # TODO: only the stack, data, and text segments contain refs
        # TODO: function names appear in...global scope?
        # TODO: char [] vs char *
        self.envp : CScope = []
        self.argv : CScope = []
        self.stack : List[List[CNamedScope]] = []
        self.heap : CScope = []

        # static scope
        self.data : CNamedScope = {}
        self.rodata : CNamedScope = {}
        self.text : CNamedScope = {}
        self.NULL = None  # aka Unmapped Segment

    def do_push_func_scope(self) -> None:
        self.stack.append([])

    def do_pop_func_scope(self) -> None:
        self.stack.pop()

    def do_push_scope(self) -> None:
        self.stack[-1].append({})

    def do_pop_scope(self) -> None:
        self.stack[-1].pop()

    #def find_defining_scope(self, name:CIdentifier) -> Optional[int]:
    #    for i in range(len(self.scope)-1,-1,-1):
    #        if self.scope[-1][i].get_by_tag(name).has(name):
    #            return i
    #    return None

    #def get(self, name:CIdentifier) -> CSegmentModifier[CValue]:
    #    scope = self.find_defining_scope(name)
    #    if scope is None:
    #        return CError("Identifier {} does not exist in scope".format(name))
    #    return self.scope[-1][scope].get_by_tag(name).get(name)

    #def update(self, name:CIdentifier, val:CValue) -> CSegmentModifier[CValue]:
    #    scope = self.find_defining_scope(name)
    #    if scope is None:
    #        return CError("Identifier {} does not exist in scope".format(name))
    #    return self.scope[-1][scope].get_by_tag(name).update(name, val)

    #def allocate(self, name:CIdentifier, type_:CType, val:List[CVal]) -> CSegmentModifier[CValue]:
    #    return self.scope[-1][-1].get_by_tag(name).allocate(name, type_, val)

    # TODO: technically, we can have multiple forward declarations of a struct, union, etc. (function decls too?)
    # TODO: this is also allowed for typedefs in C11, as long as the types don't conflict
    # TODO: also multiple weak declarations of globals?
    # TODO: something with typechecking and val?
    # TODO: put len_ back in so that we can have "sparse" arrays?
    #def allocate(self, name_:CIdentifier, type_:CType, val:List[CVal]) -> CSegmentModifier[CValue]:
    #    if name_[1] in self.scope:
    #        # TODO: make this more explicit (add scopes on the way back up?)
    #        return CError("Identifier {} already exists in scope".format(name_[1]))
    #    # TODO: check that the type_ and vals actually align?
    #    # XXX XXX XXX
    #    # HMMM... this isn't true for ids/typedefs/variables
    #    new_val = CValue(type_, CMemoryAllocation(name_[0], type_, name_[1], len(val), val))

    #    forward = lambda: self.do_set(name_[1], new_val)
    #    reverse = lambda: self.do_delete(name_[1])
    #    return (forward, reverse)

    #def get(self, name_:CIdentifier) -> CSegmentModifier[CValue]:
    #    name = name_[1]
    #    if name not in self.scope:
    #        # TODO: can we get the name of T?
    #        return CError("Identifier {} does not exist in scope".format(name))

    #    if self.scope[name].val is None:
    #        # TODO: this might be fine if, say, global
    #        return CError("Identifier {} is uninitialized in scope".format(name))

    #    return self.do_get(name)

    ## TODO: shouldn't be able to update the type, only the actual value??
    #def update(self, name_:CIdentifier, value:CValue) -> CSegmentModifier[CValue]:
    #    name = name_[1]
    #    if name not in self.scope:
    #        return CError("Identifier {} does not exist in scope".format(name))
    #    old_val = self.do_get(name)
    #    forward = lambda: self.do_set(name, value)
    #    reverse = lambda: self.do_set(name, old_val)
    #    return (forward, reverse)


    #def identify(self, name_:CIdentifier) -> None:
    #    pass
    #    # first stack, then argv, then envp, then data?
