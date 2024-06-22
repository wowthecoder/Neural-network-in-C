#ifndef R4T_H
#define R4T_H

#include "math_structs.h"

typedef matrixt *r4trowt;
typedef r4trowt *r4tcontents;

typedef struct r4t {
    r4tcontents contents;
    int rows;
    int cols;
    int elemrows;
    int elemcols;
};
typedef struct r4t *r4t;

extern r4tcontents r4t_makeContents(int rows, int cols, int elemrows, int elemcols);
extern r4t r4t_make(int rows, int cols, int elemrows, int elemcols);
extern void r4t_zeros(r4t m);
extern void r4t_rands(r4t m);
extern void r4t_free(r4t m);
extern void r4t_add(r4t A, r4t B);
extern void r4t_elemMult(r4t A, r4t B, r4t C);
extern void r4t_subtract(r4t A, r4t B);
extern r4t r4t_scalarMult(r4t A, double s);
extern void r4t_MultDotCorr(r4t A, r4t B, r4t C);
extern r4t r4t_transposeOf(r4t B);
extern r4t r4t_copyOf(r4t B);

#endif