from typing import Any
import operator

from .c_values import CAddress, CVal, CFunction


class GetString(CFunction):
    def __init__(self, env):
        self.env = env
        #self.allocation_list = []
        super.__init__(set('get_string', 'GetString'), ['string'], [('...', None)])

    def call(self, *args):
        # TODO: actually make the call to printf
        if len(args != 0):
            print(args)
        # TODO: make this more like GetString
        read = env.stdin.split('\n')#env.stdin.readline()
        # TODO: encoding??
        read = bytearray(read, 'latin-1') + bytearray([0])
        name = 'GetString ' + str(len(self.allocation_list))
        env.allocate_memory(name, ['char'], len(read), read, 'heap')
        self.allocation_list.append(name)
        return self.env.make_val(['string'], Address(name, 0))

    #def __del__(self):
    #    [env.deallocate_memory(name) for name in self.allocation_list]

class IsAlpha(CFunction):
    def __init__(self, env):
        super.__init__('isalpha', ['int'], [(['int'], 'c')])

    def call(self, c):
        return bytes([c.value]).decode('latin-1').isalpha()

class IsLower(CFunction):
    def __init__(self, env):
        super.__init__('islower', ['int'], [(['int'], 'c')])

    def call(self, c):
        return bytes([c.value]).decode('latin-1').islower()

class IsUpper(CFunction):
    def __init__(self, env):
        super.__init__('isupper', ['int'], [(['int'], 'c')])

    def call(self, c):
        return bytes([c.value]).decode('latin-1').isupper()

class ToUpper(CFunction):
    def __init__(self, env):
        super.__init__('toupper', ['int'], [(['int'], 'c')])

    def call(self, c):
        return bytes([c.value]).decode('latin-1').toupper().encode('latin-1')[0]

class ToLower(CFunction):
    def __init__(self, env):
        super.__init__('tolower', ['int'], [(['int'], 'c')])

    def call(self, c):
        return bytes([c.value]).decode('latin-1').tolower().encode('latin-1')[0]

def Strlen(CFunction):
    def __init__(self, env):
        # TODO: technically const
        # TODO: check for assertion failure on going over string length
        self.env = env
        super.__init__('strlen', ['size_t'], [(['*', 'char'], 's')])

    def call(self, s):
        # TODO: could iterate over memory til we hit \0
        arr, offset = args[0].value.base, args[0].value.offset
        # -1 to remove the \0
        if offset >= self.env.memory[arr].len:
            return CError("Pointer is past end of string")
        # TODO: check whether it is nul-terminated
        return self.env.memory[arr].len - offset - 1

class Printf(CFunction):
    def __init__(self, env):
        # TODO: technically const
        self.env = env
        super.__init__('printf', ['int'], [(['*', 'char'], 'fmt'), ('...', None)])

    def call(self, fmt, *args):
        # TODO: cast to python vals
        # also, typerrors can happen with the format string
        new_args = []
        for arg in args:
            if isinstance(arg.value, Address):
                # we're dealing with a pointer + offset
                arg = arg.value
                array = self.env.memory[arg.base]
                assert arg.offset < array.len
                array = array.array[arg.offset:]
                new_args.append(array.decode('latin-1'))
            else:
                new_args.append(arg.value)
        args = new_args

        if len(args) == 1:
            env.stdio.stdout += args[0]
        else:
            fmt = args[0]
            args = args[1:]
            # lol the python fmt % args format
            # This presumably doesn't work with strings
            env.stdio.stdout += operator.mod(fmt, tuple(args))
        # TODO: not the right return value
        return 1
