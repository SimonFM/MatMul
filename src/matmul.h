/*
  Header defining the prototype for the core multiplication routine of the
  MatMul program.

  Copyright (c) 2015 by its authors.

  This code is distributed under the BSD3 license. See AUTHORS, LICENSE.
*/

#ifndef MATMUL_H
# define MATMUL_H

# ifndef COMPLEX_H
#  include "complex.h"
# endif

/*
  (Ostensibly) Efficient matrix multiplication routine.
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C,
            int a_dim1, int a_dim2, int b_dim2);

#endif // MATMUL_H
