#ifndef MATHSTRUCTS_H
#define MATHSTRUCTS_H

typedef double *rowt;
typedef rowt *matrixcontents;

typedef struct matrixt {
    matrixcontents contents;
    int rows;
    int cols;
};
typedef struct matrixt *matrixt;

extern matrixcontents matrix_makeContents(int rows, int cols);
extern matrixt matrix_make(int rows, int cols);

extern void matrix_zeros(matrixt m);
extern void matrix_rands(matrixt m);
extern void matrix_free(matrixt m);
extern void matrix_setContents(matrixt matrix, double *new);
extern void matrix_add(matrixt A, matrixt B);
extern void matrix_subtract(matrixt A, matrixt B);
extern matrixt matrix_scalarMult(matrixt A, double s);
extern void matrix_matMult(matrixt A, matrixt B, matrixt C);
extern void matrix_add_vector(matrixt A, matrixt B);
extern void matrix_elemMult(matrixt A, matrixt B, matrixt C);
extern void matrix_dot_vector(matrixt A, matrixt B);
extern matrixt matrix_sum_rows(matrixt B);
extern matrixt matrix_transposeOf(matrixt B);
extern matrixt matrix_copyOf(matrixt B);
extern matrixt submatrix(int a, int b, int c, int d, matrixt m);
extern void matrix_print(matrixt matrix);

#endif