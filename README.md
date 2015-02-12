# MatMul

Parallel matrix multiplication.

### Usage

Compile with

```
gcc -std=gnu11 matMul.c -o matMul
```

Run with

```
./matMul <A nrows> <A ncols> <B nrows> <B ncols>
```

##### Pthreads

Compile pthread branch with

```
gcc -std=gnu11 matMul.c -o matMul -lpthread
```
