#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

static void optimize(matrixt m, matrixt mhat, matrixt s, matrixt shat, matrixt deltas, matrixt inputs, int epoch) {
    matrixt g = matrix_copyOf(deltas);
    if (inputs != NULL) {
        matrix_elemMult(g, g, inputs);
    }

    // Equation 1
    matrixt beta1_m = matrix_scalarMult(m, adam->beta1);
    matrixt beta1_g = matrix_scalarMult(g, 1 - adam->beta1);
    matrix_subtract(beta1_m, beta1_g);
    matrix_free(m); // free old values
    m = beta1_m; // do not free beta1_m because adam->momentum is pointing to it

    // Equation 2
    matrixt beta2_g = matrix_scalarMult(g, 1 - adam->beta2);
    matrixt beta2_s = matrix_scalarMult(s, adam->beta2);
    matrix_elemMult(s, beta2_g, g);
    matrix_add(s, beta2_s);

    // Equation 3
    matrix_free(mhat); // free old value
    double x = 1.0 / (1 - pow(adam->beta1, epoch));
    mhat = matrix_scalarMult(m, x);

    // Equation 4
    matrix_free(shat); // free old value
    double y = 1.0 / (1 - pow(adam->beta2, epoch));
    shat = matrix_scalarMult(s, y);

    matrix_free(beta1_g);
    matrix_free(beta2_g);
    matrix_free(beta2_s);
}

// the input matrix is already transposed
void adam_optimize(adam_optimizer adam, matrixt g, matrixt inputs, int epoch) {
    optimize(adam->m_weights, adam->mhat_weights, adam->s_weights, adam->shat_weights, deltas, inputs, epoch);
    optimize(adam->m_bias, adam->mhat_bias, adam->s_bias, adam->shat_bias, deltas, NULL, epoch);
}

// Needed in equation 5: computes m_hat / sqrt(s_hat + epsilon)
matrixt compute_change(matrixt m_hat, matrixt s_hat, double epsilon, double l_rate) {
    matrixt res = matrix_make(s_hat->rows, s_hat->cols);
    for (int i = 0; i < res->rows; i++) {
        for (int j = 0; j < res->cols; j++) {
            res->contents[i][j] = l_rate * m_hat->contents[i][j] / sqrt(s_hat->contents[i][j] + epsilon);
        }
    }
    return res;
} 

// m is either the weight matrix or bias vector
// lr is already divided by batch size
void adam_update(adam_optimizer adam, matrixt m, bool weights, double l_rate) {
    matrixt inv_sqrt_shat = sqrt_epsilon(adam->s_hat, adam->epsilon);
    matrixt change = matrix_make(num_outputs, batch_size);
    matrix_elemMult(change, adam->m_hat, inv_sqrt_shat);
    matrixt lr_changes = matrix_scalarMult(change, l_rate);
    matrixt sum_changes = matrix_sum_rows(lr_changes);
    if (weights) {
        matrix_matMult(weight_changes, lr_changes, inputs_T);
    } else {
        matrix_add(m, sum_changes);
    }

    matrix_free(inv_sqrt_shat);
    matrix_free(change);
    matrix_free(lr_changes);
    matrix_free(sum_changes);
}



void adam_free(adam_optimizer adam) {
    matrix_free(adam->m_weights);
    matrix_free(adam->m_bias);
    matrix_free(adam->mhat_weights);
    matrix_free(adam->mhat_bias);
    matrix_free(adam->s_weights);
    matrix_free(adam->s_bias);
    matrix_free(adam->shat_weights);
    matrix_free(adam->shat_bias)
    free(adam);
}