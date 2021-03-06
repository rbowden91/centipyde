import operator

from .values import Address, Val
from .exceptions import SegmentationFault

# TODO: strings should include \0...
def GetString(interpreter):
    counter = 0
    def helper(args):
        stdin = interpreter.get_stdin()
        nonlocal counter
        name = 'GetString ' + str(counter)
        counter += 1
        interpreter.memory_init(name, ['char'], len(stdin), stdin, 'heap')
        return interpreter.make_val(['string'], Address(name, 0))

    # TODO: technically const?
    type_ = [('(builtin)', ['string'], [], [])]
    return type_, helper

def get_string(interpreter):
    counter = 0
    def helper(args):
        stdin = interpreter.get_stdin()
        nonlocal counter
        name = 'GetString ' + str(counter)
        counter += 1
        interpreter.memory_init(name, ['char'], len(stdin), stdin, 'heap')
        return interpreter.make_val(['string'], Address(name, 0))

    # TODO: technically const?
    type_ = [('(builtin)', ['string'], [], [])]
    return type_, helper

def isalpha(interpreter):
    def helper(args):
        return interpreter.make_val(['int'], bytes([args[0].value]).decode('latin-1').isalpha())

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def islower(interpreter):
    def helper(args):
        return interpreter.make_val(['int'], bytes([args[0].value]).decode('latin-1').islower())

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def isupper(interpreter):
    def helper(args):
        return interpreter.make_val(['int'], bytes([args[0].value]).decode('latin-1').isupper())

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def printf(interpreter):
    def helper(args):
        # TODO: cast to python vals
        # also, typerrors can happen with the format string
        new_args = []
        for arg in args:
            if isinstance(arg.value, Address):
                # we're dealing with a pointer + offset
                arg = arg.value
                array = interpreter.memory[arg.base]
                assert arg.offset < array.len
                array = array.array[arg.offset:]
                new_args.append(array.decode('latin-1'))
            else:
                new_args.append(arg.value)
        args = new_args

        if len(args) == 1:
            stdout = args[0][1:-1]
        else:
            fmt = args[0]
            args = args[1:]
            # lol the python fmt % args format
            # 1:-1 gets rid of the double quotes
            # TODO: iterate over %s to look for \0. for now, blindly assume it's there
            args = [(string.strip('\x00') if isinstance(string, str) else string) for string in args]
            stdout = operator.mod(fmt[1:-1], tuple(args))
        interpreter.put_stdout(stdout)
        return interpreter.make_val(['int'], 1)

    # TODO: technically const?
    type_ = [('(builtin)', ['int'], [None], [['*', 'char'], ['...']])]
    return type_, helper

def strlen(interpreter):
    def helper(args):
        # TODO: could iterate over memory til we hit \0
        arr, offset = args[0].value.base, args[0].value.offset
        # -1 to remove the \0
        # TODO: myassert
        if offset >= interpreter.memory[arr].len:
            raise SegmentationFault()
        return interpreter.make_val(['size_t'], interpreter.memory[arr].len - offset - 1)

    # TODO: technically const?
    type_ = [('(builtin)', ['size_t'], [None], [['*', 'char']])]
    return type_, helper


def tolower(interpreter):
    def helper(args):
        return interpreter.make_val(['int'], bytes([args[0].value]).decode('latin-1').lower().encode('latin-1')[0])

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def toupper(interpreter):
    def helper(args):
        return interpreter.make_val(['int'], bytes([args[0].value]).decode('latin-1').upper().encode('latin-1')[0])

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper
