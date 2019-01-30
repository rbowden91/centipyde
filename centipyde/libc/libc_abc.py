from typing import Dict
from enum import Enum, auto

from ctypes import *

# TODO: can only call at most 10 args? what about variable argument functions, like printf?
# TODO: anything that expects a null-terminated string, we should probably check for that ourselves
# TODO: anything that modifies memory we might want to manually implement to check for aliasing or what not?
# TODO: malloc definitely needs to be done ourselves, right?

class LibCWrapper(object):
    # TODO: also variables declared in pylibc?
    def __init__(self):
        self.funcs : Dict['str', CFunction] = PyLibC().funcs

class PyLibC(object):
    def __init__(self):
        from . import libc
        self.funcs : Dict['str', CFunction] = {}
        for name in libc:
            if isinstance(getattr(libc, CFunction)):
                instance = getattr(libc, CFunction)(env)
                if not isinstance(instance.names, set):
                    instance.names = set([instance.names])
                for name in instance.names:
                    self.funcs[names] = instance

        #for name in builtin_funcs:
        #    type_, func = builtin_funcs[name](self)
        #    val = self.make_val(type_, name)
        #    self.update_scope(name, val)
        #    self.memory_init(name, type_, 1,
        #            [(lambda func: lambda args: self.k.passthrough(lambda: func(args)))(func)], 'text')
    # TODO: call the destructure

class MuslLibC(object):
    pass
# TODO: does python check that c_char_p is nul terminated for us??
# this is musl-specific. bleck
#cwrapper.libc.__init_libc.argtypes = [POINTER(c_char_p), c_char_p]
#cwrapper.libc.__init_libc(byref(c_char_p(None)), b"./vigenere")
    #self.libc = cdll.LoadLibrary("clib/musl/lib/libc.so")
#cwrapper = LibCWrapper()
#cwrapper.libc.printf.argtypes = [c_char_p, c_char_p, c_int, c_double]
#cwrapper.libc.printf(b"String '%s', Int %d, Double %f\n", b"Hi", 10, 2.2)

class StdLibC(object):
    pass
    #self.libc = cdll.LoadLibrary("libc.so.6")
