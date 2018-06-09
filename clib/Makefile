# TODO: clean this file up
SHELL := /usr/bin/env bash

# we need to build some system headers for musl, but this can be swapped with other
# architectures that musl supports
# CS50IDE on Cloud9 is x86_64
ARCH = x86_64

CC = clang
CFLAGS = -E -P -nostdinc -I../../include -I../internal


all: cs50 musl
	export CFLAGS="$(CFLAGS)"; export CC="$(CC)"; ./preprocess.sh

cs50:
	mkdir -p build/{include,src/cs50}
	cp libcs50/src/cs50.h build/include/
	cp libcs50/src/cs50.c build/src/cs50/

musl:
	mkdir -p build/{include/bits,src}
	cd musl && $(MAKE) obj/include/bits/alltypes.h obj/include/bits/syscall.h obj/src/internal/version.h
	cp -r musl/arch/generic/bits/* build/include/bits/
	cp -r musl/arch/$(ARCH)/* build/include/
	cp -r musl/obj/include/bits build/include/
	cp -r musl/include/* build/include/
	cp -r musl/src/* build/src/
	cp -r musl/obj/src/internal/version.h build/src/internal/
	cp -r dummy_headers/* build/include/
	rm build/include/*.in build/include/bits/*.in
	rm build/src/malloc/DESIGN

clean:
	rm -rf musl/obj 2> /dev/null
	rm -rf build 2> /dev/null

.PHONY: all clean cs50 musl

# http://make.mad-scientist.net/managing-recipe-echoing/
$(V).SILENT:
