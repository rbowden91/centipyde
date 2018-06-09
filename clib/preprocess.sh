#!/usr/bin/env bash

set -e

CC=${CC:-clang}
CFLAGS=${CFLAGS:--E -P -nostdinc -I../../include -I../internal}

cd build/src
for i in *; do
    pushd "$i" > /dev/null
    for j in *; do
    	if [ -d "$j" ]; then
	    rm -rf "$j"
	else
	    $CC $CFLAGS "$j" > "$j".bak
	    mv "$j".bak "$j"
	fi
    done
    popd > /dev/null
done
