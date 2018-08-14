# TODO: look more into how C/clang handles unicode
from enum import Enum, auto
from typing import Dict, List, Tuple, Callable, Optional, Generic, TypeVar, Union, Optional, Type

from .cwrapper import LibCWrapper
from .c_values import *
from .interpret import Interpreter
from .c_memory import CMemory


# TODO: anything that modifies memory we might want to manually implement to check for aliasing or what not?
# TODO: malloc definitely needs to be done ourselves, right?
# TODO: oh, just do interposition for malloc, and that additionally helps us with GetString (and others). In fact, can
# we get where the memory was allocated?
# TODO: can you re-typedef something if the types don't conflict? In C11, evidently yes.
# TODO: can a typedef happen before the base type exists?? forward declarations?
# TODO: complain about unused/various warnings?

class CStds(Enum):
    #k_and_r = auto()
    #ansi = iso = c89 = c90 = auto()
    c99 = auto()
    c11 = auto()

class CRuntime(object):
    def __init__(self, interpreter : Interpreter, require_decls:bool = True) -> None:
        self.c_std : CStds = CStds.c99

        # values are tuples of type, num_elements, and array of elements
        # we might want something like this so that we can
        #self.identifiers = CIdentifiers()
        self.memory = CMemory()
        self.stdio = {
            'stdin': '',
            'stdout': '',
            'stderr': ''
        }
#CStdio()
        #self.filesystem = CFilesystem()

        self.libc = LibCWrapper()
        # self.libc.funcs
        self.interpreter = interpreter


#    def allocate_memory(self, name, type_, len_, array, segment):
#        expanded_type = expand_type(type_, self.typedefs)
#        self.memory[name] = Memory(type_, expanded_type, name, len_, array, segment)
#        (self.k
#        .info(('memory_init', self.memory[name])))
#        #.passthrough(self.memory[name]))
#
#    def declare_union(self, name, decls):
#        union = self.scope[-1]['unions'][name] = type(name, Union, {
#            '_fields_': []
#        })
#
#        for decl in decls:
#            union[name]._fields_.append(decl)
#        return CTypeDecl(name, union, union)
#
#    def remove_union(self, name):
#        pass
#
#    # TODO: bitfields?
#    # TODO: anonymous?
#    # TODO: name can be empty
#    def declare_struct(self, name, decls):
#        # TODO: can _fields_ be empty?
#        struct = self.scope[-1]['structs'][name] = type(name, Structure, {
#            '_fields_': []
#        })
#
#        # fields needs to come after, since it might reference self.structs[name]
#        for decl in decls:
#            # TODO: make sure everything in the decl is well-defined?
#            struct._fields_.append(decl)
#        return CTypeDecl(name, struct, struct)
#
#
#
#    # TODO: make sure it's not overriding an old typedef?
#    # TODO: preemptively expand the type
#    # TODO: ids interfere with typedefs
#    def declare_typedef(self, name, type_):
#        self.scope[-1]['typedefs'][name] = type_
#
#    def remove_typedef(self, name):
#        del self.scope[-1]['typedefs'][name]

#    def make_type(self, type_ : CTypeType) -> CType :
#        typedefs : Dict[str,CTypeType] = {}
#        expanded_type = expand_type(type_, typedefs)
#        return CType(type_, expanded_type)
#
#    def make_value(self, type_ : CTypeType, val : CVal) -> CValue:
#        # TODO: associate the expanded_type with types_ instead of with vals?
#        return CValue(self.make_type(type_), val)

    # executes a program from the main function
    # shouldn't call this multiple times, since memory might be screwed up (global values not reinitialized, etc.)
    # This is effectively our _start function. Note that, depending on which libc we are using, there might actually be
    # more stuff that is necessary to do, in which case maybe _start should be called directly. But not the case for
    # stdlibc

    def setup_array(self, segment, array):
        # TODO: this assumes no errors
        # TODO: more specifically, these are addresses
        segment = getattr(self.memory, array)
        new_array : List[CValue] = []
        for i in range(len(array)):
            arg = segment.allocate((array, array + '[' + str(i) + ']', None), self.make_type(['char']), [CString(array[i])])
            assert isinstance(arg, tuple)
            addr = arg[0]()
            assert addr is not None
            new_array.append(addr)

        new_array.append(self.make_value(['void','*'], CAddress('NULL', None, 'NULL', 0)))
        new_array2 = [arg.val for arg in new_array]
        array2 = segment.allocate((array, array, None), self.make_type(['char', '*']), new_array2)
        return array2
        #segment.finalize()

    # this is basically our elf loader, lol
    def setup_main(self, argv : List[bytearray] = [], envp : List[bytearray] = []):

        # TODO: need to reset global variables to initial values as well
        #for i in list(self.memory.keys()):
        #    if self.memory[i]['segment'] in ['heap', 'stack', 'argv']:
        #        del(self.memory[i])

        # TODO: validate argv

        # TODO: no matching pop_scope. Return could do that, but not all functions call
        # return. Make this an explicit call to a "FuncCall" node?

        # TODO: handle main being void


        argc = len(argv)
        new_argv = self.setup_array('argv', argv)
        new_envp = self.setup_array('envp', envp)

        # call function, automatically wrap into vals
        main = self.memory.get(('text', 'main', None))
        self.call_func(main, (argc, argv))

    def call_func(self, func, args):
        self.memory.stack.do_push_func_scope()
        self.memory.stack.do_push_scope()
        # call function, automatically wrap into vals
        self.call_func(func, args)

        self.memory.stack.do_pop_scope() # not particularly necessary
        self.memory.stack.do_pop_func_scope()
