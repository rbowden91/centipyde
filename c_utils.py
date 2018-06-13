from interpret import Val, Address

# TODO: strings should include \0...
def GetString(interpreter):
    counter = 0
    def helper(args):
        stdin = interpreter.stdin.split('\n')
        interpreter.stdin = stdin[1:]
        stdin = bytearray(stdin[0], 'latin-1') + bytearray([0])
        nonlocal counter
        name = 'GetString ' + str(counter)
        counter += 1
        interpreter.memory_init(name, ['char'], len(stdin), stdin, 'heap')
        return Val(['string'], Address(name, 0))

    # TODO: technically const?
    type_ = [('(builtin)', ['string'], [], [])]
    return type_, helper

def isalpha(interpreter):
    def helper(args):
        return Val(['int'], bytes(args).decode('latin-1').isalpha())

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def islower(interpreter):
    def helper(args):
        return Val(['int'], bytes(args).decode('latin-1').islower())

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def isupper(interpreter):
    def helper(args):
        return Val(['int'], bytes(args).decode('latin-1').isupper())

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def printf(interpreter):
    def helper(args):
        new_args = []
        for arg in args:
            if isinstance(arg, Address):
                # we're dealing with a pointer + offset
                arg = arg.value
                array = interpreter.memory[arg.base]
                assert arg.offset < array['len']
                array = array['base'][arg.offset:]
                new_args.append(array.decode('latin-1'))
            else:
                new_args.append(arg.value)
        args = new_args

        if len(args) == 1:
            interpreter.stdout += args[0]
        else:
            fmt = args[0]
            args = args[1:]
            interpreter.stdout += args[0] % args[1:]
        return Val(['int'], 1)

    # TODO: technically const?
    type_ = [('(builtin)', ['int'], [None], [['*', 'char'], ['...']])]
    return type_, helper

def strlen(interpreter):
    def helper(args):
        # TODO: could iterate over memory til we hit \0
        arr, offset = args.value
        # -1 to remove the \0
        # TODO: myassert
        assert offset < interpreter.memory[args.value.base]['len'] - 1
        return Val(['size_t'], interpreter.memory[args.value.base]['len'] - args.value.offset - 1)

    # TODO: technically const?
    type_ = [('(builtin)', ['size_t'], [None], [['*', 'char']])]
    return type_, helper


def tolower(interpreter):
    def helper(args):
        return Val(['int'], bytes(args).decode('latin-1').lower().encode('latin-1')[0])

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper

def toupper(interpreter):
    def helper(args):
        return Val(['int'], bytes(args).decode('latin-1').upper().encode('latin-1')[0])

    type_ = [('(builtin)', ['int'], [None], [['int']])]
    return type_, helper
