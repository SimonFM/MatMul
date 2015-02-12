/*
  Test and timing harness program for developing a dense matrix
  multiplication routine for the CS3014 module
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
  The following two definitions of DEBUGGING control whether or not
  debugging information is written out. To put the program into
  debugging mode, uncomment the following line: 
*/

// #define DEBUGGING(_x) _x 

/*
  To stop the printing of debugging information, use the following line: 
*/

#define DEBUGGING(_x)

// Unit stored in matrices
struct complex 
{
  float real;
  float imag;
};

// Matrix indices passed to pthread slave function
struct thread_args
{
  int i;
  int j;
};

/*
  The following globals are used by the pthread slave function
*/
struct complex ** GA;   // Copy of pointer to matrix A
struct complex ** GB;   // Copy of pointer to matrix B
struct complex ** GC;   // Copy of pointer to matrix C
int A_DIM2;             // Number of columns in A

/*
  The sheer size of the following two matrices should be indicative of the
  impracticality and naivety of the current parallelisation effort, which is a
  proof of concept. YOU'VE BEEN WARNED!
*/
// Array of pthreads, one for each element in C
pthread_t ** THREADS;
// Array of arguments to pthread slave, one for each element in C
struct thread_args ** ARG_ARRAY;

/*
  Write matrix to stdout
*/
void write_out(struct complex ** a, int dim1, int dim2)
{
  for ( int i = 0; i < dim1; i++ )
  {
    for ( int j = 0; j < dim2 - 1; j++ )
    {
      printf("%f + %fi ", a[i][j].real, a[i][j].imag);
    }
    printf("%f + %fi\n", a[i][dim2-1].real, a[i][dim2-1].imag);
  }
}

/*
  Create new empty matrix 
*/
struct complex ** new_empty_matrix(int dim1, int dim2)
{
  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);

  for ( int i = 0; i < dim1; i++ )
  {
    result[i] = &new_matrix[i * dim2];
  }

  return result;
}

/*
  Take a copy of the matrix and return in a newly allocated matrix
*/
struct complex ** copy_matrix(struct complex ** source_matrix, int dim1, int dim2)
{
  struct complex ** result = new_empty_matrix(dim1, dim2);

  for ( int i = 0; i < dim1; i++ )
  {
    for ( int j = 0; j < dim2; j++ )
    {
      result[i][j] = source_matrix[i][j];
    }
  }
  return result;
}

/*
  Create a matrix and fill it with random numbers 
*/
struct complex ** gen_random_matrix(int dim1, int dim2)
{
  struct complex ** result;
  struct timeval seedtime;
  int seed;
  long long upper, lower;

  result = new_empty_matrix(dim1, dim2);

  /*
    Use the microsecond part of the current time as a pseudo-random seed
  */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /*
    Fill the matrix with random numbers 
  */
  for ( int i = 0; i < dim1; i++ )
  {
    for ( int j = 0; j < dim2; j++ )
    {
      upper = random();
      lower = random();
      result[i][j].real = (float)((upper << 32) | lower);
      upper = random();
      lower = random();
      result[i][j].imag = (float)((upper << 32) | lower);
    }
  }

  return result;
}

/*
  Check the sum of absolute differences is within reasonable epsilon
*/
void check_result(struct complex ** result, struct complex ** control, int dim1, int dim2)
{
  double diff = 0.0;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  for ( int i = 0; i < dim1; i++ )
  {
    for ( int j = 0; j < dim2; j++ )
    {
      diff = abs(control[i][j].real - result[i][j].real);
      sum_abs_diff += diff;
      diff = abs(control[i][j].imag - result[i][j].imag);
      sum_abs_diff += diff;
    }
  }

  if ( sum_abs_diff > EPSILON )
  {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
            sum_abs_diff, EPSILON);
  }
}

/*
  Multiply matrix A times matrix B and put result in matrix C
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C, 
            int a_dim1, int a_dim2, int b_dim2)
{
  struct complex sum;

  for ( int i = 0; i < a_dim1; i++ )
  {
    for( int j = 0; j < b_dim2; j++ )
    {
      sum.real = 0.0;
      sum.imag = 0.0;
      for ( int k = 0; k < a_dim2; k++ )
      {
        // the following code does: sum += A[i][k] * B[k][j];
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
void * dotProd(void * args)
{
  struct thread_args * arg = (struct thread_args *)args;
  struct complex sum = {0.0, 0.0};
  int i = arg->i, j = arg->j;

  for ( int k = 0; k < A_DIM2; k++ )
  {
    // the following code does: sum += A[i][k] * B[k][j];
    sum.real += (GA[i][k].real * GB[k][j].real) -
                (GA[i][k].imag * GB[k][j].imag);
    sum.imag += (GA[i][k].real * GB[k][j].imag) +
                (GA[i][k].imag * GB[k][j].real);
  }
  GC[i][j] = sum;

  pthread_exit(NULL);
}

/*
  The fast version of matmul written by the team
*/
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C,
                 int a_dim1, int a_dim2, int b_dim2)
{
  int rc;   // pthread_create() return code

  // Global copies for slave function
  GA = A; GB = B; GC = C;
  A_DIM2 = a_dim2;

  for ( int i = 0; i < a_dim1; i++ )
  {
    for( int j = 0; j < b_dim2; j++ )
    {
      // Assign each dot product calculation to a new pthread... GENIUS!
      ARG_ARRAY[i][j].i = i;
      ARG_ARRAY[i][j].j = j;
      rc = pthread_create(&THREADS[i][j], NULL, dotProd,
                          (void *) &ARG_ARRAY[i][j]);
      if (rc) {
        printf("ERROR return code from pthread_create(): %d\n", rc);
        printf("Thread number: %d\n", i * j);
        exit(-1);
      }
    }
  }
}

int main(int argc, char ** argv)
{
  struct complex ** A, ** B, ** C;
  struct complex ** control_matrix;
  long long mul_time;
  int a_dim1, a_dim2, b_dim1, b_dim2;
  struct timeval start_time;
  struct timeval stop_time;

  if ( argc != 5 ) 
  {
    fprintf(stderr, "Usage: matMul <A nrows> <A ncols> <B nrows> <B ncols>\n");
    exit(1);
  }
  else 
  {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  /*
    Check the matrix sizes are compatible
  */
  if ( a_dim2 != b_dim1 )
  {
    fprintf(stderr,
            "FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
            a_dim2, b_dim1);
    exit(1);
  }

  /*
    Allocate the matrices
  */
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  C = new_empty_matrix(a_dim1, b_dim2);
  control_matrix = new_empty_matrix(a_dim1, b_dim2);

  // Allocate pthread array
  THREADS = malloc(sizeof(*THREADS) * a_dim1);
  for ( int i = 0; i < a_dim1; i++ )
  {
    THREADS[i] = malloc(sizeof(**THREADS) * b_dim2);
  }
  // Allocate slave function parameter array
  ARG_ARRAY = malloc(sizeof(*ARG_ARRAY) * a_dim1);
  for ( int i = 0; i < a_dim1; i++ )
  {
    ARG_ARRAY[i] = malloc(sizeof(**ARG_ARRAY) * b_dim2);
  }

  DEBUGGING(puts("First matrix:"));
  DEBUGGING(write_out(A, a_dim1, a_dim2));
  DEBUGGING(puts("Second matrix:"));
  DEBUGGING(write_out(B, b_dim1, b_dim2));

  /*
    Record starting time of naive matrix multiplication
  */
  gettimeofday(&start_time, NULL);

  /*
    Use a simple matmul routine to produce control result 
  */
  matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

  /*
    Record finishing time of naive matrix multiplication
  */
  gettimeofday(&stop_time, NULL);
  mul_time = ((stop_time.tv_sec - start_time.tv_sec) * 1000000L) +
             (stop_time.tv_usec - start_time.tv_usec);
  printf("matmul time:      %lld microseconds\n", mul_time);

  /*
    Record starting time 
  */
  gettimeofday(&start_time, NULL);

  /*
    Perform matrix multiplication
  */
  team_matmul(A, B, C, a_dim1, a_dim2, b_dim2);

  /*
    Record finishing time 
  */
  gettimeofday(&stop_time, NULL);
  mul_time = ((stop_time.tv_sec - start_time.tv_sec) * 1000000L) +
             (stop_time.tv_usec - start_time.tv_usec);
  printf("team_matmul time: %lld microseconds\n", mul_time);

  DEBUGGING(puts("Resultant matrix:"));
  DEBUGGING(write_out(C, a_dim1, b_dim2));

  /*
    Now check that the team's matmul routine gives the same answer
    as the known working version 
  */
  check_result(C, control_matrix, a_dim1, b_dim2);

  return EXIT_SUCCESS;
}
