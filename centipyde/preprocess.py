# Can't use generators, because then we can't readily reverse?
import os
import subprocess

from typing import Optional

# TODO: eventually handle this. pycparser doesn't play nicely with mypy yet
from pycparser import c_ast, c_parser # type: ignore

def preprocess_file(file_, is_code=False) -> Optional[bytes]:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    include_path = os.path.join(dir_path, 'clib/build/include')
    cpp_args = [r'cpp', r'-E', r'-g3', r'-gdwarf-2', r'-nostdinc', r'-I' + include_path,
                r'-D__attribute__(x)=', r'-D__builtin_va_list=int', r'-D_Noreturn=', r'-Dinline=', r'-D__volatile__=']
    cpp_args.append(file_ if not is_code else '-')

    # reading from stdin
    proc = subprocess.Popen(cpp_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, encoding='latin-1')
    stdout, stderr = proc.communicate(bytes(file_, 'latin-1') if is_code else None)
    if len(stderr) != 0:
        print('Uh oh! Stderr messages', proc.stderr)
        return None
    elif proc.returncode != 0:
        print('Uh oh! Nonzero error code')
        return None

    return stdout

# TODO: use getcwd for filename?
def parse_ast(file_, is_code=False) -> Optional[c_ast.Node]:
    parser = c_parser.CParser()
    # TODO: need to check for errors
    try:
        cfile = preprocess_file(file_, is_code)
        ast = parser.parse(cfile)
    except Exception as e:
        print('uh oh2', e)
        return None
    return ast
