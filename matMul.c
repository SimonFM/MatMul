/*
  Test and timing harness program for developing a dense matrix multiplication
  routine for the CS3014 module
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <xmmintrin.h>
#include <pthread.h>
#include <unistd.h>     // sysconf()

/*
  The following two definitions of DEBUGGING control whether or not debugging
  information is written out. To put the program into debugging mode, uncomment
  the following line:
*/

// #define DEBUGGING(_x) _x 

/*
  To stop the printing of debugging information, use the following line: 
*/

#define DEBUGGING(_x)

// Unit stored in matrices
struct complex {
  float real;
  float imag;
};

// Matrix indices passed to pthread slave function
struct thread_args {
  int i0;
  int i1;
  int j0;
  int j1;
};

/*
  The following globals are used by the pthread slave function
*/
struct complex ** _A;           // Copy of pointer to matrix A
struct complex ** _B;           // Copy of pointer to matrix B
struct complex ** _C;           // Copy of pointer to matrix C
int _a_dim1, _a_dim2, _b_dim2;  // Number of columns in A

const int NCORES = 64;

/*
  Write matrix to stdout
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
  Create new empty matrix 
*/
struct complex ** new_empty_matrix(int dim1, int dim2) {

  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);

  for (int i = 0; (i < dim1); i++) {
    result[i] = &new_matrix[i * dim2];
  }

  return result;
}

void free_matrix(struct complex ** matrix) {
  free (matrix[0]); // Free the contents
  free (matrix);    // Free the header
}

/*
  Take a copy of the matrix and return in a newly allocated matrix
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
  Create a matrix and fill it with random numbers 
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
  Check the sum of absolute differences is within reasonable epsilon
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
  Multiply matrix A times matrix B and put result in matrix C
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C, 
            int a_dim1, int a_dim2, int b_dim2) {

  struct complex sum;

  for (int i = 0; (i < a_dim1); i++) {
    for(int j = 0; (j < b_dim2); j++) {
      sum = (struct complex){0.0, 0.0};
      for (int k = 0; (k < a_dim2); k++) {
        // The following code does: sum += A[i][k] * B[k][j];
        sum.real += A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
        sum.imag += A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
      }
      C[i][j] = sum;
    }
  }
}

/*
  Pthread slave function. Each one carries out one dot product calculation.
*/
void * dotProd(void * args) {

  struct thread_args * arg = (struct thread_args *)args;
  struct complex sum = {0.0, 0.0};

  for (int i = arg->i0; (i < _a_dim1) && (i < arg->i1); i++) {
    for (int j = arg->j0; (j < _b_dim2) && (j < arg->j1); j++) {
      for (int k = 0; (k < _a_dim2); k++) {
        // The following code does: sum += A[i][k] * B[k][j];
        sum.real += (_A[i][k].real * _B[k][j].real) -
                    (_A[i][k].imag * _B[k][j].imag);
        sum.imag += (_A[i][k].real * _B[k][j].imag) +
                    (_A[i][k].imag * _B[k][j].real);
      }
      _C[i][j] = sum;
    }
  }

  pthread_exit(NULL);
}

/*
  The fast version of matmul written by the team
*/
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C,
                 int a_dim1, int a_dim2, int b_dim2) {

  int rc;
  int i0 = 0, i1 = 0, j0 = 0, j1 = 0;

  const long ncores = sysconf(_SC_NPROCESSORS_ONLN);
  const int DELTA_I = (a_dim1 / ncores) + (a_dim1 % ncores != 0);
  const int DELTA_J = (b_dim2 / ncores) + (b_dim2 % ncores != 0);
  // const int nthreads = (a_dim1 / DELTA_I

  pthread_t * threads = malloc(sizeof(*threads) * ncores);
  struct thread_args * arg_array = malloc(sizeof(*arg_array) * ncores);

  printf("DI = %d\n", DELTA_I);
  printf("DJ = %d\n", DELTA_J);

  for (int i = 0; ((i1 >= a_dim1) || (j1 >= b_dim2)) && (i < ncores); i++) {
    i0 = i * DELTA_I; i1 = i0 + DELTA_I;
    j0 = i * DELTA_J; j1 = j0 + DELTA_J;
    printf("thread %d\n", i);
    printf("i0 = %d\n", i0);
    printf("i1 = %d\n", i1);
    printf("j0 = %d\n", j0);
    printf("j1 = %d\n", j1);
    arg_array[i] = (struct thread_args){i0, i1, j0, j1};
    rc = pthread_create(&threads[i], NULL, dotProd, (void *) &arg_array[i]);
    if (rc) {
      fprintf(stderr, "ERROR return code from pthread_create(): %d\n", rc);
      fprintf(stderr, "Thread number: %d\n", i);
      exit(-1);
    }
  }
  for (int i = 0; (i < ncores); i++) {
    pthread_join(threads[i], NULL);
  }
}

struct t_args {
  int i0;
  int i1;
};

void * dProd(void * args) {

  struct t_args * arg = (struct t_args *) args;
  struct complex sum;
  const int i0 = arg->i0;
  const int i1 = arg->i1;
  // printf("dProd%d i0 = %d\n", i0, i0);
  // printf("dProd%d i1 = %d\n", i0, i1);

  for (int i = i0; (i < i1) && (i < _a_dim1); i++) {
    // printf("dProd%d i = %d\n", i0, i);
    for (int j = 0; (j < _b_dim2); j++) {
      sum = (struct complex){0.0, 0.0};
      // printf("dProd%d j = %d\n", i0, j);
      for (int k = 0; (k < _a_dim2); k++) {
        // printf("dProd%d k = %d\n", i0, k);
        // The following code does: sum += A[i][k] * B[k][j];
        sum.real += (_A[i][k].real * _B[k][j].real) -
                    (_A[i][k].imag * _B[k][j].imag);
        sum.imag += (_A[i][k].real * _B[k][j].imag) +
                    (_A[i][k].imag * _B[k][j].real);
      }
      _C[i][j] = sum;
    }
  }

  pthread_exit(NULL);
}

void tmatmul(struct complex ** A, struct complex ** B, struct complex ** C,
             int a_dim1, int a_dim2, int b_dim2) {

  if (a_dim1 < NCORES) {
    struct complex sum;

    for (int i = 0; (i < a_dim1); i++) {
      for(int j = 0; (j < b_dim2); j++) {
        sum = (struct complex){0.0, 0.0};
        for (int k = 0; (k < a_dim2); k++) {
          // The following code does: sum += A[i][k] * B[k][j];
          sum.real += A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
          sum.imag += A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
        }
        C[i][j] = sum;
      }
    }
    return;
  }

  int rc;
  const int DELTA = (a_dim1 / NCORES) + (a_dim1 % NCORES != 0);
  // printf("DELTA = %d\n", DELTA);

  pthread_t  * threads = malloc(sizeof(*threads) * NCORES);
  struct t_args * args = malloc(sizeof(*args)    * NCORES);

  for (int i = 0; (i < NCORES); i++) {
    // printf("Thread %d\n", i);
    // printf("i0 = %d\n", i * DELTA);
    // printf("i1 = %d\n", (i + 1) * DELTA);
    args[i] = (struct t_args){i * DELTA, (i + 1) * DELTA};
    rc = pthread_create(&threads[i], NULL, dProd, (void *) &args[i]);
    if (rc) {
      fprintf(stderr, "ERROR return code from pthread_create(): %d\n", rc);
      fprintf(stderr, "Thread number: %d\n", i);
      exit(-1);
    }
  }
  for (int i = 0; (i < NCORES); i++) {
    pthread_join(threads[i], NULL);
  }
}

/*
  Returns the difference, in microseconds, between the two given times
*/
long long time_diff(struct timeval * start, struct timeval *end) {
  return ((end->tv_sec - start->tv_sec) * 1000000L) +
         (end->tv_usec - start->tv_usec);
}

/*
  Main harness
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

  _a_dim1 = a_dim1; _a_dim2 = a_dim2; _b_dim2 = b_dim2;
  _A = A; _B = B; _C = C;

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
  tmatmul(A, B, C, a_dim1, a_dim2, b_dim2);

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

  DEBUGGING( {
    puts("Resultant matrix:");
    write_out(C, a_dim1, b_dim2);
  } )

  // Free all matrices
  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
  free_matrix(ctrl_matrix);

  return EXIT_SUCCESS;
}
