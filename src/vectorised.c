/*
  Test and timing harness program for developing a dense matrix multiplication
  routine for the CS3014 module.

  Authors:  Basil L. Contovounesios
            Ben Lynch
            Simon Markham
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <xmmintrin.h>
#include <pthread.h>

/*
  Uncomment the following to use sysconf(_SC_NPROCESSORS_ONLN) to determine
  number of online cores on the system.
*/
// #include <unistd.h>

/*
  The following two definitions of DEBUGGING control whether or not debugging
  information is written out. Defining DEBUG, e.g. via preprocessor options
  at compilation, puts the program into debugging mode.
*/
#ifdef DEBUG
#define DEBUGGING(_x) _x
#else
#define DEBUGGING(_x)
#endif

/*
  The following definition reflects the number of online cores on the system
  and determines the maximum number of pthreads created. It defaults to 64,
  which is the number of cores on the target machine stoker. It is intended to
  be defined at compilation via a preprocessor option if run on a different
  target.
*/
#ifndef NCORES
#define NCORES 64
#endif

/*
  Complex number unit stored in matrices.
*/
struct complex {
  float real;
  float imag;
};

/*
  Write matrix to stdout.
*/
void write_out(struct complex ** a, int dim1, int dim2) {
  for (int i = 0; (i < dim1); i++) {
    for (int j = 0; (j < dim2 - 1); j++) {
      printf("%f + %fi ", a[i][j].real, a[i][j].imag);
    }
    printf("%f + %fi\n", a[i][dim2 - 1].real, a[i][dim2 - 1].imag);
  }
}

/*
  Create new empty matrix.
*/
struct complex ** new_empty_matrix(int dim1, int dim2) {

  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);

  for (int i = 0; (i < dim1); i++) {
    result[i] = &new_matrix[i * dim2];
  }

  return result;
}

/*
  Free matrix.
*/
void free_matrix(struct complex ** matrix) {
  free (matrix[0]); // Free the contents
  free (matrix);    // Free the header
}

/*
  Take a copy of the matrix and return in a newly allocated matrix.
*/
struct complex ** copy_matrix(struct complex ** source_matrix,
                              int dim1, int dim2) {

  struct complex ** result = new_empty_matrix(dim1, dim2);

  for (int i = 0; (i < dim1); i++) {
    for (int j = 0; (j < dim2); j++) {
      result[i][j] = source_matrix[i][j];
    }
  }

  return result;
}

/*
  Create a matrix and fill it with random numbers.
*/
struct complex ** gen_random_matrix(int dim1, int dim2) {

  const int random_range = 512;
  struct complex ** result;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  // Use the microsecond part of the current time as a pseudo-random seed
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  // Fill the matrix with random numbers
  for (int i = 0; (i < dim1); i++) {
    for (int j = 0; (j < dim2); j++) {
      // Evenly generate values in the range [0, random_range - 1)
      result[i][j].real = (float)(random() % random_range);
      result[i][j].imag = (float)(random() % random_range);

      // At no loss of precision, negate the values sometimes so the range is
      // now (-(random_range - 1), random_range - 1)
      if (random() & 1) result[i][j].real = -result[i][j].real;
      if (random() & 1) result[i][j].imag = -result[i][j].imag;
    }
  }

  return result;
}

/*
  Check the sum of absolute differences is within reasonable epsilon.
*/
void check_result(struct complex ** result, struct complex ** control,
                  int dim1, int dim2) {

  double diff = 0.0;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  for (int i = 0; (i < dim1); i++) {
    for (int j = 0; (j < dim2); j++) {
      diff = abs(control[i][j].real - result[i][j].real);
      sum_abs_diff += diff;
      diff = abs(control[i][j].imag - result[i][j].imag);
      sum_abs_diff += diff;
    }
  }

  if (sum_abs_diff > EPSILON) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
            sum_abs_diff, EPSILON);
  }
}

/*
  Multiply matrix A times matrix B and put result in matrix C.
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C,
            int a_dim1, int a_dim2, int b_dim2) {

  struct complex sum;

  for (int i = 0; (i < a_dim1); i++) {
    for(int j = 0; (j < b_dim2); j++) {
      sum = (struct complex){0.0, 0.0};
      for (int k = 0; (k < a_dim2); k++) {
        // The following code does: sum += A[i][k] * B[k][j];
        sum.real = sum.real + (A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag);
        sum.imag = sum.imag + (A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real);
      }
      C[i][j] = sum;
    }
  }
}

/*
  The fast version of matmul() written by the team.
*/
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C,
                 int a_dim1, int a_dim2, int b_dim2) {

  struct complex sum;
  __m128 currentVectorReal, currentVectorImag;
  __m128 realVector1, realVector2;
  __m128 imagVector1, imagVector2;
  __m128 realByReal, imagByImag;
  __m128 realByImag, imagByReal;
  __m128 result1, result2;
  __m128 final;

  const int remainder = a_dim2 % 4;
  const int s = a_dim2 - remainder;

  struct complex a;
  struct complex a1;
  struct complex a2;
  struct complex a3;
  struct complex b;
  struct complex b1;
  struct complex b2;
  struct complex b3;

  float vectorAsArray[4];

  for (int i = 0; (i < a_dim1); i++) {
    for( int j = 0; (j < b_dim2); j++) {

      sum = (struct complex){0.0, 0.0};

      for(int k = 0; (k < s); k += 4) {
        a = A[i][k];
        a1 = A[i][k + 1];
        a2 = A[i][k + 2];
        a3 = A[i][k + 3];
        b = B[k][j];
        b1 = B[k + 1][j];
        b2 = B[k + 2][j];
        b3 = B[k + 3][j];
        realVector1 = _mm_setr_ps(a.real,  a1.real,
                                  a2.real, a3.real);
        realVector2 = _mm_setr_ps(b.real,     b1.real,
                                  b2.real, b3.real);

        imagVector1 = _mm_setr_ps(a.imag,  a1.imag,
                                  a2.imag, a3.imag);
        imagVector2 = _mm_setr_ps(b.imag,  b1.imag,
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

        // the following code does: sum += A[i][k] * B[k][j];
        // sum.real += A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
        // sum.imag += A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
      }

      // Handles the multiplication for any remainder row/column elements
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

/*
  Returns the difference, in microseconds, between the two given times.
*/
long long time_diff(struct timeval * start, struct timeval *end) {
  return ((end->tv_sec - start->tv_sec) * 1000000L) +
         (end->tv_usec - start->tv_usec);
}

/*
  Main harness.
*/
int main(int argc, char ** argv) {

  struct complex ** A, ** B, ** C;
  struct complex ** ctrl_matrix;
  long long ctrl_time, mult_time;
  int a_dim1, a_dim2, b_dim1, b_dim2;
  struct timeval time0, time1, time2;
  double speedup;

  if (argc != 5) {
    fputs("Usage: matMul <A nrows> <A ncols> <B nrows> <B ncols>\n", stderr);
    exit(1);
  } else  {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  // Check the matrix sizes are compatible
  if (a_dim2 != b_dim1) {
    fprintf(stderr, "FATAL number of columns of A (%d) does not "
                    "match number of rows of B (%d)\n", a_dim2, b_dim1);
    exit(1);
  }

  // Allocate the matrices
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  C = new_empty_matrix(a_dim1, b_dim2);
  ctrl_matrix = new_empty_matrix(a_dim1, b_dim2);

  DEBUGGING({
    puts("Matrix A:");
    write_out(A, a_dim1, a_dim2);
    puts("\nMatrix B:");
    write_out(B, b_dim1, b_dim2);
    puts("");
  })

  // Record control start time
  gettimeofday(&time0, NULL);

  // Use a simple matmul routine to produce control result
  matmul(A, B, ctrl_matrix, a_dim1, a_dim2, b_dim2);

  DEBUGGING( {
    puts("Resultant matrix:");
    write_out(ctrl_matrix, a_dim1, b_dim2);
  } )

  // Record start time
  gettimeofday(&time1, NULL);

  // Perform matrix multiplication
  team_matmul(A, B, C, a_dim1, a_dim2, b_dim2);

  // Record finishing time
  gettimeofday(&time2, NULL);

  // Compute elapsed times and speedup factor
  ctrl_time = time_diff(&time0, &time1);
  mult_time = time_diff(&time1, &time2);
  speedup   = (float)ctrl_time / mult_time;

  printf("Control time: %lld μs\n", ctrl_time);
  printf("Matmul  time: %lld μs\n", mult_time);

  if ((mult_time > 0) && (ctrl_time > 0)) {
    printf("Speedup:      %.2fx\n", speedup);
  }

  // Now check that team_matmul() gives the same answer as the control
  check_result(C, ctrl_matrix, a_dim1, b_dim2);

  DEBUGGING({
    puts("Resultant matrix:");
    write_out(C, a_dim1, b_dim2);
  })

  // Free all matrices
  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
  free_matrix(ctrl_matrix);

  return EXIT_SUCCESS;
}
