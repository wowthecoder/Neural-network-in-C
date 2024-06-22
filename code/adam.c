#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "adam.h"

// returns true if it fails
adam_optimizer adam_create(int num_neurons, int batch_size, double beta1, double beta2, double epsilon) {
    adam_optimizer adam = malloc(sizeof(struct adam_optimizer));
    if (adam == NULL) {
        fprintf(stderr, "Error creating adam");
        return NULL;
    }
    adam->momentum = matrix_make(num_neurons, batch_size);
    matrix_zeros(adam->momentum);
    adam->m_hat = matrix_make(num_neurons, batch_size);
    adam->s = matrix_make(num_neurons, batch_size);
    matrix_zeros(adam->s);
    adam->s_hat = matrix_make(num_neurons, batch_size);
    if (adam->momentum == NULL || adam->m_hat == NULL || adam->s == NULL || adam->s_hat == NULL) {
        fprintf(stderr, "Error creating adam");
        return NULL;
    }
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->epsilon = epsilon;
    return adam;
}

void adam_optimize(adam_optimizer adam, matrixt deltas, int epoch) {
    // Equation 1
    matrixt beta1_m = matrix_scalarMult(adam->momentum, adam->beta1);
    matrixt beta1_deltas = matrix_scalarMult(deltas, 1 - adam->beta1);
    matrix_subtract(beta1_m, beta1_deltas);
    matrix_free(adam->momentum); // free old values
    adam->momentum = beta1_m; // do not free beta1_m because adam->momentum is pointing to it

    // Equation 2
    matrixt beta2_deltas = matrix_scalarMult(deltas, 1 - adam->beta2);
    matrixt beta2_s = matrix_scalarMult(adam->s, adam->beta2);
    matrix_elemMult(adam->s, beta2_deltas, deltas);
    matrix_add(adam->s, beta2_s);

    // Equation 3
    matrix_free(adam->m_hat); // free old value
    double x = 1.0 / (1 - pow(adam->beta1, epoch));
    adam->m_hat = matrix_scalarMult(adam->momentum, x);

    // Equation 4
    matrix_free(adam->s_hat); // free old value
    double y = 1.0 / (1 - pow(adam->beta2, epoch));
    adam->s_hat = matrix_scalarMult(adam->s, y);

    matrix_free(beta1_deltas);
    matrix_free(beta2_deltas);
    matrix_free(beta2_s);
}

// Needed in equation 5: computes 1 / sqrt(s_hat + epsilon)
matrixt sqrt_epsilon(matrixt s_hat, double epsilon) {
    matrixt res = matrix_make(s_hat->rows, s_hat->cols);
    for (int i = 0; i < res->rows; i++) {
        for (int j = 0; j < res->cols; j++) {
            res->contents[i][j] = 1.0 / sqrt(s_hat->contents[i][j] + epsilon);
        }
    }
    return res;
} 

void adam_free(adam_optimizer adam) {
    matrix_free(adam->momentum);
    matrix_free(adam->m_hat);
    matrix_free(adam->s);
    matrix_free(adam->s_hat);
    free(adam);
}