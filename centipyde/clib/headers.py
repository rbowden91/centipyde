import os
import sys
from pycparser import c_ast, c_parser, preprocess_file

# TODO: musl just puts everything into a single libc. Seemingly, we have c, pthread, rt, m, dl, util, and xnet?

include_dir = u'build/include/'
os.chdir(include_dir)
cpp_args = [r'-I.', r'-nostdinc', r'-D__attribute__(x)=', r'-D__builtin_va_list=int',
            r'-D_Noreturn=', r'-Dinline=', r'-D__volatile__=']

class HeaderInfo(c_ast.NodeVisitor):
    def __init__(self):
        self.typedefs = {}
        self.func_decls = {}
        self.structs = {}
        self.enums = {}
        self.unions = {}
        self.global_vars = {}

    def save(self, n, dict_):
        header = str(n.coord).split(':')[0]
        if header.startswith('./'):
            header = header[2:]
        name = n.name
        if name not in dict_:
            dict_[name] = set()
        # TODO: make sure the types align if multiple headers define things?
        dict_[name].add(header)

    def visit_Typedef(self, n):
        self.save(n, self.typedefs)

    def visit_Decl(self, n):
        if isinstance(n.type, c_ast.FuncDecl):
            self.save(n, self.func_decls)
        elif isinstance(n.type, c_ast.Union):
            self.save(n.type, self.unions)
        elif isinstance(n.type, c_ast.Struct):
            self.save(n.type, self.structs)
        elif isinstance(n.type, c_ast.Enum):
            self.save(n.type, self.structs)
        elif 'extern' in n.storage:
            pass
        else:
            n.show(showcoord=True)
            assert False
    def visit_FuncDef(self, n):
        self.visit(n.decl)
        # we explicitly *do not* want to visit the function body



#dependencies = {
#}
#
#for root, dir_, files in os.walk(include_dir):
#    for header in files:
#        header = os.path.join(root, header)
#        stdout = preprocess_file(header, cpp_path='clang', cpp_args=[r'-M'] + cpp_args)
#        # the [2:] cuts off the "header.o:' and "header.h" entries
#        deps = [file_.strip() for file_ in stdout.split(' ')[2:]]
#        deps = [file_[len(include_dir):] for file_ in deps if file_.endswith('.h')]
#        # this is the only dependency loop, but seemingly the file doesn't actually depend on anything from it?
#        header = header[len(include_dir):]
#
#        if header == 'netinet/if_ether.h':
#            deps.remove('net/ethernet.h')
#
#        dependencies[header] = deps

macros = {}
typedefs = {}
structures = {}
func_decls = {}

dependencies_processed = set()

parser = c_parser.CParser()

header_info = HeaderInfo()
for root, dir_, files in os.walk(u'.'):
    for file_ in files:
        # cut out the ./
        header = os.path.join(root, file_)[2:]
        if header in ['sys/signal.h', 'sys/fcntl.h', 'sys/errno.h', 'sys/termios.h', 'sys/poll.h', 'wait.h']:
            # these are annoying files that just put up a warning telling you the correct header to use
            continue
        cfile = preprocess_file(header, cpp_path='clang', cpp_args=['-E'] + cpp_args)
        try:
            ast = parser.parse(cfile)
            header_info.visit(ast)
        except Exception as e:
            # This isn't necessarily a bad thing. Some of the header files are strictly meant to be included by other
            # files.
            if header not in ['bits/sem.h', 'bits/ipc.h', 'bits/stat.h', 'bits/statfs.h', 'bits/msg.h',
                              'bits/shm.h', 'bits/user.h', 'bits/termios.h', 'bits/link.h',
                              'bits/socket.h', 'bits/stdint.h', 'atomic_arch.h']:

                print("Couldn\'t parse file.", header, e)

        # now we want to collect macro definition information
        #defines = preprocess_file(header, cpp_path='clang', cpp_args=['-E', '-dM'] + cpp_args).split('\n')
        #for line in cfile.split('\n'):
        #    print(line)
        #    if line.startswith('#define'):
        #        print(line)

print(header_info.typedefs)
        #stdout = preprocess_file(header, cpp_path='clang', cpp_args=[r'-E', r'-dM'] + cpp_args)
        #defines = [define.split(' ')[1:] for define in defines.split('\n')]
        #for define in defines:
        #    if len(define) == 0: continue
        #    name = define[0]
        #    val = None if len(define) == 1 else ' '.join(define[1:])

        #    # make sure we don't overwrite another header files macros
        #    # TODO: do we have to worry about undefs?? Yikes
        #    if name not in macros:
        #        macros[name] = {
        #            'header': header,
        #            'val': val
        #        }

sys.exit(1)

while len(dependencies) > 0:
    found_one = False
    for header in list(dependencies.keys()):
        for dependency in dependencies[header]:
            if dependency not in dependencies_processed:
                break
        else:
            found_one = True
            del dependencies[header]
            dependencies_processed.add(header)
            defines = preprocess_file(header, cpp_path='clang',
                                     cpp_args=[r'-E', r'-dM'] + cpp_args)
            defines = [define.split(' ')[1:] for define in defines.split('\n')]
            for define in defines:
                if len(define) == 0: continue
                name = define[0]
                val = None if len(define) == 1 else ' '.join(define[1:])

                # make sure we don't overwrite another header files macros
                # TODO: do we have to worry about undefs?? Yikes
                if name not in macros:
                    macros[name] = {
                        'header': header,
                        'val': val
                    }

            cfile = preprocess_file(path, cpp_path='clang', cpp_args=['-E', '-P'] + cpp_args)
            try:
                ast = parser.parse(cfile)
            except Exception as e:
                print("Couldn\'t parse file.", path, e)
            header_info = HeaderInfo()
            header_info.visit(ast)
    assert found_one, dependencies.keys()

print(macros)
