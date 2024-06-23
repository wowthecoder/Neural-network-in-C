#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdbool.h>

#include "ann.h"
#include "math_funcs.h"
#include "math_structs.h"
#include "helper.h"

// set appropriate path for data
#define TRAIN_IMAGE "../data/train-images.idx3-ubyte"
#define TRAIN_LABEL "../data/train-labels.idx1-ubyte"
#define TEST_IMAGE "../data/test-images.idx3-ubyte"
#define TEST_LABEL "../data/test-labels.idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

#define NUM_EPOCHS 20
#define BATCH_SIZE 16
#define NUM_CLASSES 10

unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];
int width[MAX_NUM_OF_IMAGES], height[MAX_NUM_OF_IMAGES];

int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];
unsigned char train_label_char[NUM_TRAIN][1];
unsigned char test_label_char[NUM_TEST][1];

double train_image[NUM_TRAIN][SIZE];
double test_image[NUM_TEST][SIZE];
int  train_label[NUM_TRAIN];
int test_label[NUM_TEST];


void FlipLong(unsigned char * ptr)
{
    // register unsigned char val;
    unsigned char val;
    
    // Swap 1st and 4th bytes
    val = *(ptr);
    *(ptr) = *(ptr+3);
    *(ptr+3) = val;
    
    // Swap 2nd and 3rd bytes
    ptr += 1;
    val = *(ptr);
    *(ptr) = *(ptr+1);
    *(ptr+1) = val;
}


void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char data_char[][arr_n], int info_arr[])
{
    int i, j, k, fd;
    unsigned char *ptr;

    if ((fd = open(file_path, O_RDONLY)) == -1) {
        fprintf(stderr, "couldn't open image file");
        exit(-1);
    }
    
    read(fd, info_arr, len_info * sizeof(int));
    
    // read-in information about size of data
    for (i=0; i<len_info; i++) { 
        ptr = (unsigned char *)(info_arr + i);
        FlipLong(ptr);
        ptr = ptr + sizeof(int);
    }
    
    // read-in mnist numbers (pixels|labels)
    for (i=0; i<num_data; i++) {
        read(fd, data_char[i], arr_n * sizeof(unsigned char));   
    }

    close(fd);
}


void image_char2double(int num_data, unsigned char data_image_char[][SIZE], double data_image[][SIZE])
{
    int i, j;
    for (i=0; i<num_data; i++)
        for (j=0; j<SIZE; j++)
            data_image[i][j]  = (double)data_image_char[i][j] / 255.0;
}


void label_char2int(int num_data, unsigned char data_label_char[][1], int data_label[])
{
    int i;
    for (i=0; i<num_data; i++)
        data_label[i]  = (int)data_label_char[i][0];
}


void load_mnist()
{
    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, LEN_INFO_IMAGE, SIZE, train_image_char, info_image);
    image_char2double(NUM_TRAIN, train_image_char, train_image);

    read_mnist_char(TEST_IMAGE, NUM_TEST, LEN_INFO_IMAGE, SIZE, test_image_char, info_image);
    image_char2double(NUM_TEST, test_image_char, test_image);
    
    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, LEN_INFO_LABEL, 1, train_label_char, info_label);
    label_char2int(NUM_TRAIN, train_label_char, train_label);
    
    read_mnist_char(TEST_LABEL, NUM_TEST, LEN_INFO_LABEL, 1, test_label_char, info_label);
    label_char2int(NUM_TEST, test_label_char, test_label);
}

int main() {
    load_mnist();

    /* Create a DNN with 2 hidden layers, each with 30 neurons. 4 layers total. */
    printf("Creating a neural network.\n");
    printf("784 inputs, 100 hidden neurons, 100 hidden neurons and 10 output classes.\n\n");
    int layer_outputs[] = {784, 100, 100, 10}; // number of neurons in each layer
    activation_func acts[] = {NULL, &relu, &relu, &softmax};
    activation_func act_ders[] = {NULL, &relu_der, &relu_der, &softmax_der};
    double dropout_probs[] = {0.0, 0.0, 0.4, 0.0};
    // passing MSE_der here because error of output layer for softmax+cross entropy loss just turns out to be MSE_der 
    ann_t *mnist_ann = ann_create(4, BATCH_SIZE, &MSE_der, acts, act_ders, dropout_probs, layer_outputs);
    if (!mnist_ann) {
        printf("Couldn't create the neural network :(\n");
        return EXIT_FAILURE;
    }

    /* Training by batches */
    printf("Training the network...\n");
    double epochs[NUM_EPOCHS];
    double training_losses[NUM_EPOCHS];
    double val_losses[NUM_EPOCHS];
    double training_accuracy[NUM_EPOCHS];
    double val_accuracy[NUM_EPOCHS];
    int num_train_batches = NUM_TRAIN / BATCH_SIZE; // 60000 / 16 = 3750
    int num_val_batches = NUM_TEST / BATCH_SIZE; // 10000 / 16 = 625
    /* Setup the batches first */
    matrixt train_img_batches[num_train_batches];
    matrixt train_label_batches[num_train_batches];
    matrixt val_img_batches[num_val_batches];
    matrixt val_label_batches[num_val_batches];
    for (int i = 0; i < num_train_batches; i++) {
        matrixt batch_img = matrix_make(SIZE, BATCH_SIZE);
        matrixt batch_labels = matrix_make(NUM_CLASSES, BATCH_SIZE);
        matrix_zeros(batch_labels);
        for (int j = 0; j < BATCH_SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                batch_img->contents[k][j] = train_image[i * BATCH_SIZE + j][k];
            }
            batch_labels->contents[train_label[i * BATCH_SIZE + j]][j] = 1;
        }
        train_img_batches[i] = batch_img;
        train_label_batches[i] = batch_labels;
    }
    for (int i = 0; i < num_val_batches; i++) {
        matrixt batch_img = matrix_make(SIZE, BATCH_SIZE);
        matrixt batch_labels = matrix_make(NUM_CLASSES, BATCH_SIZE);
        matrix_zeros(batch_labels);
        for (int j = 0; j < BATCH_SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                batch_img->contents[k][j] = test_image[i * BATCH_SIZE + j][k];
            }
            batch_labels->contents[test_label[i * BATCH_SIZE + j]][j] = 1;
        }
        val_img_batches[i] = batch_img;
        val_label_batches[i] = batch_labels;
    }

    /* Actual training YAY */
    for(int i = 0; i < NUM_EPOCHS; ++i) {
        double sum_train_entropy = 0, sum_val_entropy = 0;
        double num_correct_train = 0, num_correct_val = 0;
        // double lr = exp_lr_scheduler(0.1, 10, i);
        /* We try training on entire batch at once instead of one by one */
        for (int j = 0; j < num_train_batches; j++) {
            ann_train(mnist_ann, train_img_batches[j], train_label_batches[j], 0.1, i+1);
            sum_train_entropy += sum_cross_entropy(train_label_batches[j], mnist_ann->output_layer->outputs);
            num_correct_train += correct_count(train_label_batches[j], mnist_ann->output_layer->outputs);
        }
        for (int j = 0; j < num_val_batches; j++) {
            ann_predict_batch(mnist_ann, val_img_batches[j], false);
            sum_val_entropy += sum_cross_entropy(val_label_batches[j], mnist_ann->output_layer->outputs);
            num_correct_val += correct_count(val_label_batches[j], mnist_ann->output_layer->outputs);
        }
        epochs[i] = i + 1;
        training_losses[i] = sum_train_entropy / NUM_TRAIN;
        val_losses[i] = sum_val_entropy / NUM_TEST;
        training_accuracy[i] = num_correct_train / NUM_TRAIN;
        val_accuracy[i] = num_correct_val / NUM_TEST;
        printf("Completed epoch %d, training_loss = %lf, val_loss = %lf, training_accuracy = %lf, val_accuracy = %lf \n", i+1, training_losses[i], val_losses[i], training_accuracy[i], val_accuracy[i]);
    }

    /* Plot the graph of training and validation loss over time */
    plot_metrics(epochs, training_losses, val_losses, NUM_EPOCHS, "Training and Validation loss over time", "Categorical Cross Entropy Loss", "mnist_loss.png");

    /* Plot the graph of training and validation accuracy over time */
    plot_metrics(epochs, training_accuracy, val_accuracy, NUM_EPOCHS, "Training and Validation accuracy over time", "Accuracy", "mnist_accuracy.png");


    /* Time to clean up. */
    ann_free(mnist_ann);
    for (int i = 0; i < num_train_batches; i++) {
        matrix_free(train_img_batches[i]);
        matrix_free(train_label_batches[i]);
    }
    for (int i = 0; i < num_val_batches; i++) {
        matrix_free(val_img_batches[i]);
        matrix_free(val_label_batches[i]);
    }

    return EXIT_SUCCESS;

}