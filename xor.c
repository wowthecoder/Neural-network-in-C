#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ann.h"
#include "math_funcs.h"
#include "math_structs.h"
#include "helper.h"

#define NUM_EPOCHS 25000
#define BATCH_SIZE 4

/* Creates and trains a simple ann for XOR. */
int main()
{
  srand(time(NULL));
  printf("Big data machine learning.\n\n");
  printf("--------------------------\n");

  /* Intializes random number generator */
  srand(42);

  /* Here is some BIG DATA to train, XOR function. */
  const double inputs[4][2] = {{0, 0},
                               {0, 1},
                               {1, 0},
                               {1, 1}};
  const double targets[] = {0, 1, 1, 0};

  /* Convert inputs and targets to matrixt data type */
  matrixt inputs_m = matrix_make(2, 4);
  matrixt targets_m = matrix_make(1, 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      inputs_m->contents[j][i] = inputs[i][j];
    }
    targets_m->contents[0][i] = targets[i];
  }

  printf("PART I - Creating a layer.\n\n");
  printf("Trying to layer_create.\n");
  dense_layer *first_l = layer_create();
  if (!first_l) {
    printf("Couldn't create the first layer :(\n");
    return EXIT_FAILURE;
  }
  printf("Running layer_init.\n");
  if (layer_init(first_l, 2, 1, NULL, NULL, 0.0, NULL)) {
    printf("Couldn't layer_init first layer...\n");
    return EXIT_FAILURE;
  }
  printf("Here are some of the properties:\n");
  printf("  num_outputs: %i\n", first_l->num_outputs);
  printf("   num_inputs: %i\n", first_l->num_inputs);
  printf("   outputs[0]: %f\n", first_l->outputs->contents[0][0]);
  printf("   outputs[1]: %f\n", first_l->outputs->contents[1][0]);

  printf("\nCreating second layer.\n");
  dense_layer *second_l = layer_create();
  if (!second_l) {
    printf("Couldn't create the second layer :(\n");
    return EXIT_FAILURE;
  }
  printf("Running layer_init on second layer.\n");
  if (layer_init(second_l, 1, 1, &sigmoid, &sigmoid_der, 0.0, first_l)) {
    printf("Couldn't layer_init second layer...\n");
    return EXIT_FAILURE;
  }
  printf("Here are some of the properties:\n");
  printf("  num_outputs: %i\n", second_l->num_outputs);
  printf("   num_inputs: %i\n", second_l->num_inputs);
  printf("   weights[0]: %f\n", second_l->weights->contents[0][0]);
  printf("   weights[1]: %f\n", second_l->weights->contents[0][1]);
  printf("    biases[0]: %f\n", second_l->biases->contents[0][0]);
  printf("   outputs[0]: %f\n", second_l->outputs->contents[0][0]);

  printf("\nComputing second layer outputs:\n");
  layer_compute_outputs(second_l, false);
  printf("Here is the new output:\n");
  printf("   outputs[0]: %f\n", second_l->outputs->contents[0][0]);
  
  printf("\nFreeing both layers.\n");
  layer_free(second_l);
  layer_free(first_l);

  /* Create neural network. */
  printf("\n--------------------------\n");
  printf("PART II - Creating a neural network.\n");
  printf("2 inputs, 2 hidden neurons and 1 output.\n\n");
  printf(" * - * \\ \n");
  printf("         * - \n");
  printf(" * - * / \n\n");
  int layer_outputs[] = {2, 2, 1}; // number of neurons in each layer
  activation_func acts[] = {NULL, &sigmoid, &sigmoid};
  activation_func act_ders[] = {NULL, &sigmoid_der, &sigmoid_der};
  double dropout_probs[] = {0.0, 0.0, 0.0};
  ann_t *xor_ann = ann_create(3, 4, &MSE_der, acts, act_ders, dropout_probs, layer_outputs);
  if (!xor_ann) {
    printf("Couldn't create the neural network :(\n");
    return EXIT_FAILURE;
  }

  /* Initialise weights to random. */
  printf("Initialising network with random weights...\n");

  /* Print hidden layer weights, biases and outputs. */
  printf("The current state of the hidden layer:\n");
  for(int i=0; i < layer_outputs[1]; ++i) {
    for(int j=0; j < layer_outputs[0]; ++j)
      printf("  weights[%i][%i]: %f\n", i, j, xor_ann->input_layer->next->weights->contents[i][j]);
  }
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  biases[%i]: %f\n", i, xor_ann->input_layer->next->biases->contents[i][0]);
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  outputs[%i]: %f\n", i, xor_ann->input_layer->next->outputs->contents[i][0]);

  // Added myself for debugging
  printf("Current state of output layer:\n");
  printf("  weight[0]: %f\n", xor_ann->output_layer->weights->contents[0][0]);
  printf("  weight[1]: %f\n", xor_ann->output_layer->weights->contents[0][1]);

  /* Dummy run to see random network output. */
  printf("Current random outputs of the network:\n");
  ann_predict_batch(xor_ann, inputs_m, false);
  for(int i = 0; i < 4; ++i) {
    printf("  [%1.f, %1.f] -> %f\n", inputs[i][0], inputs[i][1], xor_ann->output_layer->outputs->contents[0][i]);
  }

  /* Train the network. */
  printf("\nTraining the network...\n");
  double epochs[NUM_EPOCHS];
  double training_losses[NUM_EPOCHS];
  for(int i = 0; i < NUM_EPOCHS; ++i) {
    /* This is an epoch, running through the entire data. */
    /* We try training on entire batch at once instead of one by one */
    ann_train(xor_ann, inputs_m, targets_m, 5.0, i+1);
    epochs[i] = i + 1;
    training_losses[i] = MSE(targets_m, xor_ann->output_layer->outputs, 4);
    // We shuffle the data each epoch to prevent converging into a cycle
    for (int i = 0; i < inputs_m -> cols; i++) {
        int cswap = rand() % inputs_m -> cols; // column to swap with
        if(i==cswap) continue; // no switch
        // temp vars
        double fst = inputs_m->contents[0][i];
        double snd = inputs_m->contents[1][i];
        double tg = inputs_m -> targets_m -> contents[0][i];
        // replace current column with cswap stuff
        for (int j = 0; j < 2; j++) {
          inputs_m->contents[j][i] = inputs_m->contents[j][cswap];
        }
        targets_m->contents[0][i] = targets_m->contents[0][cswap];
        // replace cswap stuff with saved temp
        inputs_m->contents[0][cswap] = fst;
        inputs_m->contents[1][cswap] = snd;
        targets_m->contents[0][cswap] = tg;
    }
  }
}

  /* Plot the graph of training loss over time */
  plot_metrics(epochs, training_losses, NULL, NUM_EPOCHS, "Training MSE over time", "MSE", "xor.png");

  /* Print hidden layer weights, biases and outputs. */
  printf("The current state of the hidden layer:\n");
  for(int i=0; i < layer_outputs[1]; ++i) {
    for(int j=0; j < layer_outputs[0]; ++j)
      printf("  weights[%i][%i]: %f\n", i, j, xor_ann->input_layer->next->weights->contents[i][j]);
  }
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  biases[%i]: %f\n", i, xor_ann->input_layer->next->biases->contents[i][0]);
  for(int i=0; i < layer_outputs[1]; ++i)
    printf("  outputs[%i]: %f\n", i, xor_ann->input_layer->next->outputs->contents[i][0]);

  /* Let's see the results. */
  printf("\nAfter training magic happened the outputs are:\n");
  ann_predict_batch(xor_ann, inputs_m, false);
  for(int i = 0; i < 4; ++i) {
    printf("  [%1.f, %1.f] -> %f\n", inputs[i][0], inputs[i][1], xor_ann->output_layer->outputs->contents[0][i]);
  }

  /* Time to clean up. */
  ann_free(xor_ann);
  matrix_free(inputs_m);
  matrix_free(targets_m);

  return EXIT_SUCCESS;
}
