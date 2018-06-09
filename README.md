# cinterpy
A Python3 interpreter for C code. Supports a subset of the x86\_64 Linux system calls that the standard C library uses.

Depends on pycparser (https://github.com/eliben/pycparser).

Currently supports most of C99, since that is what pycparser supports. Does not support C11 extensions.

Dependencies:
bash
clang
make
pycparser
python3.6
