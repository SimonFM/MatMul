# ------------------------------------------------------------------------------
# Makefile for the MatMul matrix multiplication program.
#           
# Copyright (c) 2015 by its authors.
#
# This code is distributed under the BSD3 license. See AUTHORS, LICENSE.
# ------------------------------------------------------------------------------

# Uncomment to force gcc as the C compiler, instead of the system default.
# This might be necessary depending on the optimisations used.
# CC = gcc

prog_SDIR := src/
prog_ODIR := build/
prog_NAME := $(prog_ODIR)matmul
prog_SRCS := $(wildcard $(prog_SDIR)*.c)

define prog_HELP
@ echo 'SYNOPSIS'
@ echo '    make [target]'
@ echo ''
@ echo 'DESCRIPTION'
@ echo '    Makefile for the MatMul matrix multiplication program.'
@ echo ''
@ echo 'TARGETS'
@ echo '    all'
@ echo '    matmul      Generate the matrix multiplication executable of the same'
@ echo '                name in the build/ directory.'
@ echo ''
@ echo '    help        Print this help message.'
@ echo ''
@ echo '    clean'
@ echo '    distclean   Remove the generated build/ directory and all of its contents.'
endef

override CFLAGS += -std=gnu11 -march=native -O3

.PHONY: all help clean distclean

all: $(prog_NAME)

$(prog_NAME): $(prog_SRCS)
	@ mkdir $(prog_ODIR)
	$(LINK.c) $^ -o $@

help:
	$(prog_HELP)

clean:
	@- $(RM) -r $(prog_ODIR)

distclean: clean
