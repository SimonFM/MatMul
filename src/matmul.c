/*
  MatMul - Efficient matrix multiplication.

  Copyright (c) 2015 by its authors.

  This code is distributed under the BSD3 license. See AUTHORS, LICENSE.
*/

#include <x86intrin.h>
#include "matmul.h"

/*
  (Ostensibly) Efficient matrix multiplication routine.
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C,
            int a_dim1, int a_dim2, int b_dim2) {

  __m128 currentVectorReal, currentVectorImag;
  __m128 realVector1, realVector2;
  __m128 imagVector1, imagVector2;
  __m128 realByReal, imagByImag;
  __m128 realByImag, imagByReal;
  __m128 result1, result2, final;

  const int remainder = a_dim2 % 4;
  const int s = a_dim2 - remainder;

  struct complex a0, a1, a2, a3;
  struct complex b0, b1, b2, b3;
  struct complex sum;

  float vectorAsArray[4];

  for (int i = 0; (i < a_dim1); i++) {
    for(int j = 0; (j < b_dim2); j++) {

      sum = (struct complex){0.0, 0.0};

      for(int k = 0; (k < s); k += 4) {
        a0 = A[i][k];
        a1 = A[i][k + 1];
        a2 = A[i][k + 2];
        a3 = A[i][k + 3];
        b0 = B[k][j];
        b1 = B[k + 1][j];
        b2 = B[k + 2][j];
        b3 = B[k + 3][j];
        realVector1 = _mm_setr_ps(a0.real, a1.real,
                                  a2.real, a3.real);
        realVector2 = _mm_setr_ps(b0.real, b1.real,
                                  b2.real, b3.real);

        imagVector1 = _mm_setr_ps(a0.imag, a1.imag,
                                  a2.imag, a3.imag);
        imagVector2 = _mm_setr_ps(b0.imag, b1.imag,
                                  b2.imag, b3.imag);

        result1 = _mm_mul_ps(realVector1, realVector2);
        result2 = _mm_mul_ps(imagVector1, imagVector2);

        final = _mm_sub_ps(result1, result2);
        _mm_store_ps(vectorAsArray, final);
        sum.real = sum.real + vectorAsArray[0] + vectorAsArray[1] +
                              vectorAsArray[2] + vectorAsArray[3];

        result1 = _mm_mul_ps(realVector1, imagVector2);
        result2 = _mm_mul_ps(imagVector1, realVector2);

        final = _mm_add_ps(result2, result1);
        _mm_store_ps(vectorAsArray, final);
        sum.imag = sum.imag + vectorAsArray[0] + vectorAsArray[1] +
                              vectorAsArray[2] + vectorAsArray[3];
      }

      // Handle multiplication for any remainder row/column elements
      if(remainder > 0) {
        switch(remainder) {
        case 1:
          realVector1 = _mm_setr_ps(A[i][s].real, 0, 0, 0);
          realVector2 = _mm_setr_ps(B[s][j].real, 0, 0, 0);

          imagVector1 = _mm_setr_ps(A[i][s].imag, 0, 0, 0);
          imagVector2 = _mm_setr_ps(B[s][j].imag, 0, 0, 0);
          break;

        case 2:
          realVector1 = _mm_setr_ps(A[i][s].real, A[i][s + 1].real, 0, 0);
          realVector2 = _mm_setr_ps(B[s][j].real, B[s + 1][j].real, 0, 0);

          imagVector1 = _mm_setr_ps(A[i][s].imag, A[i][s + 1].imag, 0, 0);
          imagVector2 = _mm_setr_ps(B[s][j].imag, B[s + 1][j].imag, 0, 0);
          break;

        case 3:
          realVector1 = _mm_setr_ps(A[i][s].real,     A[i][s + 1].real,
                                    A[i][s + 2].real, 0);
          realVector2 = _mm_setr_ps(B[s][j].real,     B[s + 1][j].real,
                                    B[s + 2][j].real, 0);

          imagVector1 = _mm_setr_ps(A[i][s].imag,     A[i][s + 1].imag,
                                    A[i][s + 2].imag, 0);
          imagVector2 = _mm_setr_ps(B[s][j].imag,     B[s + 1][j].imag,
                                    B[s + 2][j].imag, 0);
          break;
        }

        result1 = _mm_mul_ps(realVector1, realVector2);
        result2 = _mm_mul_ps(imagVector1, imagVector2);

        final = _mm_sub_ps(result1, result2);
        _mm_store_ps(vectorAsArray, final);
        sum.real = sum.real + vectorAsArray[0] + vectorAsArray[1] +
                              vectorAsArray[2] + vectorAsArray[3];

        result1 = _mm_mul_ps(realVector1, imagVector2);
        result2 = _mm_mul_ps(imagVector1, realVector2);

        final = _mm_add_ps(result2, result1);
        _mm_store_ps(vectorAsArray, final);
        sum.imag = sum.imag + vectorAsArray[0] + vectorAsArray[1] +
                              vectorAsArray[2] + vectorAsArray[3];
      } // end if

      C[i][j] = sum;
    }

  }
}
