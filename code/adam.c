#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "adam.h"

// returns true if it fails
adam_optimizer adam_create(int num_neurons, int num_inputs, int batch_size, double beta1, double beta2, double epsilon) {
    adam_optimizer adam = malloc(sizeof(struct adam_optimizer));
    if (adam == NULL) {
        fprintf(stderr, "Error creating adam");
        return NULL;
    }
    adam->m_weights = matrix_make(num_neurons, num_inputs);
    adam->m_bias = matrix_make(num_neurons, batch_size);
    matrix_zeros(adam->m_weights);
    matrix_zeros(adam->m_bias);
    adam->mhat_weights = matrix_make(num_neurons, num_inputs);
    adam->mhat_bias = matrix_make(num_neurons, batch_size);
    
    adam->s_weights = matrix_make(num_neurons, num_inputs);
    adam->s_bias = matrix_make(num_neurons, batch_size);
    matrix_zeros(adam->s_weights);
    matrix_zeros(adam->s_bias);
    adam->shat_weights = matrix_make(num_neurons, num_inputs);
    adam->shat_bias = matrix_make(num_neurons, batch_size);

    if (adam->m_weights == NULL || adam->m_bias == NULL || adam->mhat_bias == NULL || adam->mhat_weights == NULL ||
        adam->s_weights == NULL || adam->s_bias == NULL || adam->shat_bias == NULL || adam->shat_weights == NULL) 
    {
        fprintf(stderr, "Error creating adam");
        return NULL;
    }
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->epsilon = epsilon;
    return adam;
}

static void optimize(matrixt m, matrixt mhat, matrixt s, matrixt shat, matrixt deltas, matrixt inputs_T, double beta1, double beta2, int epoch) {
    matrixt g;
    if (inputs_T != NULL) {
        g = matrix_make(deltas->rows, inputs_T->cols);
        matrix_matMult(g, deltas, inputs_T);
    } else {
        g = matrix_copyOf(deltas);
    }

    // Equation 1
    matrix_scalarMult2(m, beta1);
    matrixt beta1_g = matrix_scalarMult(g, 1 - beta1);
    matrix_subtract(m, beta1_g);

    // Equation 2
    matrixt beta2_g = matrix_scalarMult(g, 1 - beta2);
    matrixt beta2_s = matrix_scalarMult(s, beta2);
    matrix_elemMult(s, beta2_g, g);
    matrix_add(s, beta2_s);

    // Equation 3
    matrix_free(mhat); // free old value
    double x = 1.0 / (1 - pow(beta1, epoch));
    mhat = matrix_scalarMult(m, x);

    // Equation 4
    matrix_free(shat); // free old value
    double y = 1.0 / (1 - pow(beta2, epoch));
    shat = matrix_scalarMult(s, y);

    matrix_free(beta1_g);
    matrix_free(beta2_g);
    matrix_free(beta2_s);
    matrix_free(g);
}

// the input matrix is already transposed
void adam_optimize(adam_optimizer adam, matrixt deltas, matrixt inputs_T, int epoch) {
    optimize(adam->m_weights, adam->mhat_weights, adam->s_weights, adam->shat_weights, deltas, inputs_T, adam->beta1, adam->beta2, epoch);
    optimize(adam->m_bias, adam->mhat_bias, adam->s_bias, adam->shat_bias, deltas, NULL, adam->beta1, adam->beta2, epoch);
    // printf("m_weights dimension is %d x %d \n", adam->m_weights->rows, adam->m_weights->cols);
}

// Needed in equation 5: computes m_hat / sqrt(s_hat + epsilon)
matrixt compute_change(matrixt m_hat, matrixt s_hat, double epsilon, double l_rate) {
    assert(m_hat->rows == s_hat->rows && m_hat->cols == s_hat->cols);
    matrixt res = matrix_make(s_hat->rows, s_hat->cols);
    for (int i = 0; i < res->rows; i++) {
        for (int j = 0; j < res->cols; j++) {
            res->contents[i][j] = l_rate * m_hat->contents[i][j] / sqrt(s_hat->contents[i][j] + epsilon);
        }
    }
    return res;
} 

void adam_free(adam_optimizer adam) {
    matrix_free(adam->m_weights);
    matrix_free(adam->m_bias);
    matrix_free(adam->mhat_weights);
    matrix_free(adam->mhat_bias);
    matrix_free(adam->s_weights);
    matrix_free(adam->s_bias);
    matrix_free(adam->shat_weights);
    matrix_free(adam->shat_bias);
    free(adam);
}