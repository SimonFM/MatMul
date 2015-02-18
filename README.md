# MatMul

Parallel matrix multiplication.

### Usage

Run with

```
./run <A nrows> <A ncols> <B nrows> <B ncols>
```

Run in debug mode

```
./run -d <A nrows> <A ncols> <B nrows> <B ncols>
```

The `run` script invokes `make` with the corresponding flags. The `makefile`, in
turn, compiles `matMul.c` with a preprocessor option that defines `NCORES` as
the number of online cores on the system. This determines the maximum number of
pthreads spawned.
