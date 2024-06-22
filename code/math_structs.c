#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "math_structs.h"

/*
to be implemented: 

Matrix Struct; DONE

make a matrix; DONE
initialisation as zero matrix; DONE
initialisation as random float matrix; DONE
free matrix; DONE
setcontents; DONE
add; DONE
subtract; DONE
scalar mult; DONE
matrix mult; DONE
transpose; DONE
content copy; DONE
visual tools; DONE

flatten?; NOPE
*/


/*
 * @brief make matrix contents
*/
matrixcontents matrix_makeContents(int rows, int cols){
    matrixcontents m = malloc(rows * sizeof(rowt));
    if(m == NULL) return NULL;
    m[0] = malloc(rows*cols*sizeof(double));
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
 * @brief make the matrix
*/
matrixt matrix_make(int rows, int cols){
    matrixt m = malloc(sizeof(struct matrixt));
    if(m == NULL) return NULL;
    m->contents = matrix_makeContents(rows, cols);
    m->rows = rows;
    m->cols = cols;
    return m;
}

/*
 * @brief initialisation as zero matrix
*/
void matrix_zeros(matrixt m){
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            m->contents[i][j] = 0;
        }
    }
}


/*
 * @brief initialisation as random float matrix
*/
void matrix_rands(matrixt m){
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            m->contents[i][j] = (double)rand()/RAND_MAX - 0.5;
        }
    }
}


/*
 * @brief free matrix contents
*/
void matrix_free(matrixt m){
    free(m->contents[0]);
    free(m->contents);
    free(m);
}


/*
 * @brief set contents of matrix
*/
void matrix_setContents(matrixt A, double *new){
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            A->contents[i][j] = *(new+i)+j;
        }
    }
}

/*
 * @brief set contents of matrix to itself plus another (A += B)
 * Both matrices have same dimensions
*/
void matrix_add(matrixt A, matrixt B){
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            A->contents[i][j] += B->contents[i][j];
        }
    }
}

/*
 * @brief set contents of matrix A to itself plus vector B (A += B)
 * Basically all elements in a row in A add the sole element in same row in B
*/
void matrix_add_vector(matrixt A, matrixt B) {
    assert(A->rows == B->rows);
    assert(B->cols == 1);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->contents[i][j] += B->contents[i][0];
        }
    }
}

/*
 * @brief returns element-wise multiplication of B and C
 * A, B and C must have the same dimensions
*/
void matrix_elemMult(matrixt A, matrixt B, matrixt C) {
    assert(A->rows == B->rows && A->cols == B->cols);
    assert(B->rows == C->rows && B->cols == C->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->contents[i][j] = B->contents[i][j] * C->contents[i][j];
        }
    }
}

/*
 * @brief A is a matrix, B is a vector. Result matrix has same dimensions as A. 
 * The columns of res are the columns of A dot B.
*/
void matrix_dot_vector(matrixt A, matrixt B) {
    assert(A->rows == B->rows && B->cols == 1);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->contents[i][j] *= B->contents[i][0];
        }
    }
}

/*
 * @brief res is a vector, B is a matrix. Sets the elements of res to the sum of each row in B.
*/
matrixt matrix_sum_rows(matrixt B) {
    matrixt res = matrix_make(B->rows, 1);
    for (int i = 0; i < B->rows; i++) {
        double sum = 0;
        for (int j = 0; j < B->cols; j++) {
            sum += B->contents[i][j];
        }
        res->contents[i][0] = sum;
    }
    return res;
}

/*
 * @brief set contents of matrix to itself minus another
*/
void matrix_subtract(matrixt A, matrixt B){
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            A->contents[i][j] -= B->contents[i][j];
        }
    }
}

/*
 * @brief set contents of matrix to a scalar multiple of itself
*/
matrixt matrix_scalarMult(matrixt A, double s){
    matrixt res = matrix_make(A->rows, A->cols);
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            res->contents[i][j] = A->contents[i][j] * s;
        }
    }
    return res;
}

/*
 * @brief set contents of matrix to one multiplied by another (A = BxC)
*/
void matrix_matMult(matrixt A, matrixt B, matrixt C){
    assert(A->rows == B->rows && A->cols == C->cols);
    for(int i = 0; i < B->rows; i++) {
        for(int j = 0; j < C->cols; j++) {
            double total = 0;
            for(int k = 0; k < B->cols; k++) {
                total += B->contents[i][k] * C->contents[k][j];
            }
            A->contents[i][j] = total;
        }
    }
}

/*
 * @brief set contents of matrix to transpose of another
*/
matrixt matrix_transposeOf(matrixt B){
    matrixt A = matrix_make(B->cols, B->rows);
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
matrixt matrix_copyOf(matrixt B){
    matrixt A = matrix_make(B->rows, B->cols);
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            A->contents[i][j] = B->contents[i][j];
        }
    }
    return A;
}

/*
 * @brief print matrix
*/
void matrix_print(matrixt m){
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            printf( "[%08.5lf]", m->contents[i][j] );
        }
        putchar('\n');
    }
}

matrixt submatrix(int a, int b, int c, int d, matrixt m) {// gets the submatrix of m at [a,b], [c,d]
    matrixt res = matrix_make(b-a+1,d-c+1);
    for(int i = 0; i < res->cols; i++){
        for(int j = 0; j < res->rows; j++){
            res->contents[i][j]= m->contents[i+a][j+c];
        }
    }
    return res;
}

/*
 * TEST MAIN - WORKS!
*/
// int main(){
//     matrixt ma = matrix_make(2, 2);
//     matrixt mb = matrix_make(2, 3);
//     matrixt mc = matrix_make(3, 2);
//     if( ma==NULL || mb==NULL || mc==NULL ) {
//         fprintf( stderr, "Unable to allocate matrices!\n" );
//         return 1;
//     }
//     matrix_rands( mb );
//     matrix_rands( mc );
//     matrix_matMult(ma, mb, mc );
//     matrixt mbT = matrix_transposeOf( mb );
//     printf( "\n\nA = B.C = \n" ); matrix_print( ma );
//     printf( "\n\nB = \n" ); matrix_print( mb );
//     printf( "\n\nC = \n" ); matrix_print( mc );
//     printf( "\n\nBT = \n" ); matrix_print( mbT );
//     matrix_free( mc );
//     matrix_free( mb );
//     matrix_free( ma );
//     return 0;
// }

/*
 * SOME MORE TESTS - WORKS!
*/
// int main(){
//     matrixt mb = matrix_make(2, 3);
//     matrixt mc = matrix_make(2, 1);
//     if(mb==NULL || mc==NULL ) {
//         fprintf( stderr, "Unable to allocate matrices!\n" );
//         return 1;
//     }
//     matrix_rands( mb );
//     mc = matrix_copyOf(mb);
//     printf( "\n\nA = \n" ); matrix_print( mb );
//     printf( "\n\nB = A = \n" ); matrix_print( mc );
//     double arr[] = {{1.0,2.0,3.0},{4.0,5.0,6.0}};
//     matrix_setContents(mc, arr);
//     printf( "\n\nnewB = \n" ); matrix_print( mc );
//     printf( "\n\nA = \n" ); matrix_print( mb );

//     return 0;
// }

