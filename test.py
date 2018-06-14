import sys
from centipyde import interpret

if len(sys.argv) == 1:
    code = sys.stdin.readlines()
    code = '\n'.join(code)
    interpreter = interpret.init_interpreter(code, True)
elif len(sys.argv) == 2:
    interpreter = interpret.init_interpreter(sys.argv[1])
else:
    print('Takes just a single optional filename (and reads from stdin if not found)')
    sys.exit(1)

# some kind of JIT after the first execution?
interpreter.setup_main(['./vigenere', 'HELlO'], 'wOrld\n')
interpreter.run()
assert len(interpreter.k.passthroughs) == 1
ret = interpreter.k.get_passthrough(0)
print(ret)
print(ret.type, ret.value)
print(interpreter.stdout)
