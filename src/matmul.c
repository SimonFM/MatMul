/*
  MatMul - Efficient matrix multiplication.

  The recursive, block-oriented function mm() used herein is taken from
  Michael J. Quinn, "Parallel Programming in C with MPI and OpenMP",
  4th edition, Chapter 11.

  Copyright (c) 2015 by its authors.

  This code is distributed under the BSD3 license. See AUTHORS, LICENSE.
*/

#include <pthread.h>
#include "matmul.h"

/*
  The following globals are used by the recursive function.
*/
struct complex ** _A;           // Copy of pointer to matrix A
struct complex ** _B;           // Copy of pointer to matrix B
struct complex ** _C;           // Copy of pointer to matrix C
int _a_dim2;                    // Number of columns of A

const int THRESHOLD = 64;

/*
  Recursive, block-oriented matrix multiplication routine.
*/
void mm(int crow, int ccol, int arow, int acol, int brow, int bcol,
        int l, int m, int n) {

  int lhalf[3], mhalf[3], nhalf[3];
  struct complex * aptr, * bptr, * cptr;

  if ((m * n) > THRESHOLD) {

    // B doesn't fit in cache --- multiply blocks of A, B

    lhalf[0] = 0; lhalf[1] = l / 2; lhalf[2] = l - lhalf[1];
    mhalf[0] = 0; mhalf[1] = m / 2; mhalf[2] = m - mhalf[1];
    nhalf[0] = 0; nhalf[1] = n / 2; nhalf[2] = n - nhalf[1];
    for (int i = 0; (i < 2); i++) {
      for (int j = 0; (j < 2); j++) {
        for (int k = 0; (k < 2); k++) {
          mm(crow + lhalf[i], ccol + mhalf[j],
             arow + lhalf[i], acol + mhalf[k],
             brow + mhalf[k], bcol + nhalf[j],
             lhalf[i + 1], mhalf[k + 1], nhalf[j + 1]);
        }
      }
    }

  } else {

    // B fits in cache --- do standard multiply

    struct complex a, b;

    for (int i = 0; (i < l); i++) {
      for (int j = 0; (j < n); j++) {
        cptr = &_C[crow + i][ccol + j];
        aptr = &_A[arow + i][acol];
        bptr = &_B[brow][bcol + j];
        for (int k = 0; (k < m); k++) {
          a = *aptr; b = *bptr;
          (*cptr).real += ((a.real * b.real) - (a.imag * b.imag));
          (*cptr).imag += ((a.real * b.imag) + (a.imag * b.real));
          aptr ++;
          bptr += _a_dim2;
        }
      }
    }
  }
}

/*
  (Ostensibly) Efficient matrix multiplication routine.
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C,
            int a_dim1, int a_dim2, int b_dim2) {

  // Make global copies of parameters for the recursive function
  _A = A; _B = B; _C = C;
  _a_dim2 = a_dim2;

  // Initial call
  mm(0, 0, 0, 0, 0, 0, a_dim1, a_dim2, b_dim2);
}
