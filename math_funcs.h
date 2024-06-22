#include "math_structs.h"
#include "math_r4t.h"

typedef matrixt (*activation_func)(matrixt);
typedef matrixt (*loss_der)(matrixt, matrixt);
// typedef double (*loss_func)(matrixt, matrixt, int);

/* activation functions */
matrixt tanh_reg(matrixt m);
matrixt tanh_der(matrixt m);
matrixt relu(matrixt m);
matrixt relu_der(matrixt m);
matrixt sigmoid(matrixt m);
matrixt sigmoid_der(matrixt m);
matrixt softmax(matrixt m);
matrixt softmax_der(matrixt m);

/* dropout */
void dropout(matrixt m, double p);

/* learning rate scheduler */
double exp_lr_scheduler(double lr0, double s, int epoch);

r4t r4t_tanh_reg(r4t m);

/* loss functions */
double MSE(matrixt targets, matrixt preds, int num_samples);
matrixt MSE_der(matrixt targets, matrixt preds);
double sum_cross_entropy(matrixt targets, matrixt preds);
double correct_count(matrixt targets, matrixt preds);