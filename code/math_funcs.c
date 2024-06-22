#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "math_funcs.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*
 * @brief takes matrix M, returns tanh(M)
*/
matrixt tanh_reg(matrixt m){
    matrixt res = matrix_make(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            res->contents[i][j] = tanh(m->contents[i][j]);
        }   
    }
    return res;
}

/*
 * @brief takes matrix M, returns tanh'(M)
*/
matrixt tanh_der(matrixt m){
    matrixt res = matrix_make(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            int val = tanh(m->contents[i][j]);
            res->contents[i][j] = 1.0 - (val * val);
        }   
    }
    return res;
}

matrixt relu(matrixt m) {
    matrixt res = matrix_make(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            res->contents[i][j] = MAX(0, m->contents[i][j]);
        }   
    }
    return res;
}

matrixt relu_der(matrixt m) {
    matrixt res = matrix_make(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            if (m->contents[i][j] >= 0) {
                res->contents[i][j] = 1;
            } else {
                res->contents[i][j] = 0;
            }
        }   
    }
    return res;
}

static double sigmoidd(double x) {
    return 1.0 / (1.0 + exp(-x));
}

matrixt sigmoid(matrixt m) {
    matrixt res = matrix_make(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            res->contents[i][j] = sigmoidd(m->contents[i][j]);
        }   
    }
    return res;
}

matrixt sigmoid_der(matrixt m) {
    matrixt res = matrix_make(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            double x = sigmoidd(m->contents[i][j]);
            res->contents[i][j] = x * (1 - x);
        }   
    }
    return res;
}

/*
 * Softmax will only be used in the output layer.
 * The dimension of m will be num_classes * batch_size
*/ 
matrixt softmax(matrixt m) {
    matrixt res = matrix_make(m->rows, m->cols);
    double sum_exps[m->cols];
    for (int c = 0; c < m->cols; c++) {
        double sum_col = 0;
        for (int r = 0; r < m->rows; r++) {
            sum_col += exp(m->contents[r][c]);
        }
        sum_exps[c] = sum_col;
    }
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            res->contents[i][j] = exp(m->contents[i][j]) / sum_exps[j];
        }   
    }
    return res;
}

/*
 * returns a matrix of 1s because we simplified the delta calculation with chain rule
*/ 
matrixt softmax_der(matrixt m) {
    matrixt res = matrix_make(m->rows, m->cols);
    for (int i = 0; i < res->rows; i++) {
        for (int j = 0; j < res->cols; j++) {
            res->contents[i][j] = 1;
        }
    }
    return res;
}

/* p is the probability that a neuron is dropped. m is a num_neurons x 1 matrix representing the mask. */
void dropout(matrixt m, double p) {
    if (p == 0) {
        return;
    }
    for (int i = 0; i < m->rows; i++) {
        double x = (double)rand() / RAND_MAX; // x is a double between 0 and 1 inclusive
        if (x < p) { // drop this neuron
            m->contents[i][0] = 0;
        } else {
            m->contents[i][0] = 1;
        }
    }
}

/* 
  * Exponential scheduling of learning rate, n: n(t) = n0 x 0.1^(t/s) 
  * Learning rate decreases every s epochs
*/
double exp_lr_scheduler(double lr0, double s, int epoch) {
    return lr0 * pow(0.1, (epoch / s));
}

/*
 * @brief functions takes two matrices and works out the Mean Sum of Squared Errors loss value
 * The dimensions are 1 * batch_size
*/
double MSE(matrixt targets, matrixt preds, int num_samples){
    assert(targets->rows == 1);
    assert(targets->rows == preds->rows && targets->cols == preds->cols);
    double total = 0;
    for (int i = 0; i < num_samples; i++) {
        double val = targets->contents[0][i] - preds->contents[0][i];
        total += val * val;
    }
    return total / num_samples;
}

/*
 * @brief takes observed and predicted, returns dE/dO matrix
 * We use MSE = 1/2n * sumof((y_pred - y_true) ^ 2) 
 * Error of 1 sample is 1/2 * (y_pred - y_true) ^ 2
*/
matrixt MSE_der(matrixt targets, matrixt preds){
    assert(targets->rows == preds->rows && targets->cols == preds->cols);
    matrixt err = matrix_make(preds->rows, preds->cols);
    for(int i = 0; i < preds->rows; i++){
        // removed 2.0 because constant factor doesnt matter
        for (int j = 0; j < preds->cols; j++) {
            err->contents[i][j] = (preds->contents[i][j] - targets->contents[i][j]);
        }
    }
    return err;
}

/* For each col (sample), calculate the cross entropy loss. Then sum for all samples. */
/* Used to calculate mean loss across all samples later */
double sum_cross_entropy(matrixt targets, matrixt preds) {
    assert(targets->rows == preds->rows && targets->cols == preds->cols);
    double total = 0;
    for (int i = 0; i < targets->cols; i++) {
        double sum_outputs = 0;
        for (int j = 0; j < targets->rows; j++) {
            sum_outputs += targets->contents[j][i] * log(preds->contents[j][i]);
        }
        total += -sum_outputs;
    }
    return total;
}

/* Count the number of correct predictions. */
double correct_count(matrixt targets, matrixt preds) {
    assert(targets->rows == preds->rows && targets->cols == preds->cols);
    double count = 0;
    for (int i = 0; i < targets->cols; i++) {
        int pred_class = 0, target_class = 0;
        double pred_maxprob = 0, target_maxprob = 0;
        for (int j = 0; j < targets->rows; j++) {
            if (preds->contents[j][i] > pred_maxprob) {
                pred_maxprob = preds->contents[j][i];
                pred_class = j;
            }
            if (targets->contents[j][i] > target_maxprob) {
                target_maxprob = targets->contents[j][i];
                target_class = j;
            }
        }
        if (pred_class == target_class) {
            count++;
        }
    }
    return count;
}
