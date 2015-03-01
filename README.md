# MatMul

Efficient matrix multiplication.

### Introduction

The MatMul project was created by its authors (see `AUTHORS`) as part of their
coursework for a university module on concurrent systems. Its purpose is to
provide an efficient matrix multiplication routine of complex numbers in C using
various optimisation techniques. This core multiplication routine, `matmul()`,
can be found, unsurprisingly, in `src/matmul.c`.

The MatMul projet is distributed under the BSD3 license, but also includes
a test harness program, `harness.c`, written by the lecturer of the module.
As such, the harness is owned by the university, Trinity College Dublin. It has
been slightly modified by the authors of MatMul and is included purely for the
purpose of showing the MatMul program in action, as was intended for the module.

### Usage

The MatMul program can conveniently be built and run in one go using the `run`
convenience script, as either

    ./run <A nrows> <A ncols> <B nrows> <B ncols>

or

    ./run <size>

where the former dictates both dimensions for each matrix and the latter
multiplies two `size x size` matrices.

Try invoking `./run --help` to print a helpful description of its usage.

### Contents

The following is a brief description of the structure of the MatMul program.

* `Makefile` - Generates the executable harness program in the build/ directory.
* `run`      - Convenience script for automating the build and run process.

##### src/

This directory contains all the C code for the MatMul program.

* `matrix.h`  - Defines the complex unit stored in the matrices.
* `matmul.h`  - Defines macros, data- and prototypes used by the MatMul program.
* `matmul.c`  - Contains the efficient multiplication routine.
* `harness.c` - Creates random matrices and times the multiplication routine.

##### etc/

This directory contains a few files with useful information for the MatMul
project.

* `lab_description.txt` - The description of the university assignment.
* `pcm_info.txt`        - Describes the usage of Intel's PCM software.
* `sse_cheatsheet.txt`  - SSE intrinsics cheat sheet.
