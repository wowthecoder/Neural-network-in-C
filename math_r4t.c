/*
This file is for rank 4 tensor (r4t) operations - a 'matrix of matrices'
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "math_r4t.h"
#include "conv.c"



/*
 * @brief make r4t contents - a [row x cols] matrix of [elemrows x elemcols] matrices
*/
r4tcontents r4t_makeContents(int rows, int cols, int elemrows, int elemcols){
    r4tcontents m = malloc(rows * sizeof(r4trowt));
    if(m == NULL) return NULL;
    m[0] = malloc(rows*cols*sizeof(matrixt));
    if(m[0] == NULL) {
        free(m);
        return NULL;
    }
    for(int i = 1; i < rows; i++) {
        m[i] = m[i-1] + cols;
    }
    return m;
}

/*
 * @brief make the r4t
*/
r4t r4t_make(int rows, int cols, int elemrows, int elemcols){
    r4t m = malloc(sizeof(struct r4t));
    if(m == NULL) return NULL;
    m->contents = r4t_makeContents(rows, cols, elemrows, elemcols);
    m->rows = rows;
    m->cols = cols;
    m->elemrows = elemrows;
    m->elemcols = elemcols;
    return m;
}

/*
 * @brief initialisation as zero r4t
*/
void r4t_zeros(r4t m){
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            matrix_zeros(m->contents[i][j]);
        }
    }
}


/*
 * @brief initialisation as random float r4t
*/
void r4t_rands(r4t m){
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            matrix_rands(m->contents[i][j]);
        }
    }
}


/*
 * @brief free r4t contents
*/
void r4t_free(r4t m){
    free(m->contents[0]);
    free(m->contents);
    free(m);
}


/*
 * @brief set contents of r4t to itself plus another (A += B)
 * Both matrices have same dimensions
*/
void r4t_add(r4t A, r4t B){
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_add(A->contents[i][j], B->contents[i][j]);
        }
    }
}


/*
 * @brief returns element-wise multiplication of B and C
 * A, B and C must have the same dimensions
*/
void r4t_elemMult(r4t A, r4t B, r4t C) {
    assert(A->rows == B->rows && A->cols == B->cols);
    assert(B->rows == C->rows && B->cols == C->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            matrix_elemMult(A->contents[i][j], B->contents[i][j], C->contents[i][j]);
        }
    }
}


/*
 * @brief set contents of r4t to itself minus another
*/
void r4t_subtract(r4t A, r4t B){
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_subtract(A->contents[i][j], B->contents[i][j]);
        }
    }
}

/*
 * @brief set contents of r4t to a scalar multiple of itself
*/
r4t r4t_scalarMult(r4t A, double s){
    r4t res = r4t_make(A->rows, A->cols, A->elemrows, A->elemcols);
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            res->contents[i][j] = matrix_scalarMult(A->contents[i][j], s);
        }
    }
    return res;
}

/*
 * @brief set contents of r4t to one multiplied by another with 
 cross corellation at the element level (A = B.|* C) where B is 
 the kernal matrix
*/
void r4t_MultDotCorr(r4t A, r4t B, r4t C){
    assert(A->rows == B->rows && A->cols == C->cols);
    for(int i = 0; i < B->rows; i++) {
        for(int j = 0; j < C->cols; j++) {
            matrixt total = matrix_make(A->elemrows, A->elemcols);
            for(int k = 0; k < B->cols; k++) {
                matrix_add(total, corr2d(C->contents[k][j], B->contents[i][k]));
            }
            A->contents[i][j] = total;
        }
    }
}

/*
 * @brief set contents of r4t to transpose of another
*/
r4t r4t_transposeOf(r4t B){
    r4t A = r4t_make(B->cols, B->rows, B->elemrows, B->elemcols);
    for(int i = 0; i < B->rows; i++) {
        for(int j = 0; j < B->cols; j++) {
            A->contents[j][i] = B->contents[i][j];
        }
    }
    return A;
}


/*
 * @brief set contents of matrix to copy of another
*/
r4t r4t_copyOf(r4t B){
    r4t A = r4t_make(B->cols, B->rows, B->elemrows, B->elemcols);
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            A->contents[i][j] = matrix_copyOf(B->contents[i][j]);
        }
    }
    return A;
}
