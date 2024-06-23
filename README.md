# Neural-network-in-C
This is my extension for the C group project in Imperial College First year Computing.
## 📁  File structure
```
├── README.md                           // Readme
├── code
│   ├── Makefile                        // For easier compilation, type "make mnist" in cmd                    
│   ├── adam.c                          // code for Adam optimizer
│   ├── adam.h                          // header file for Adam optimizer                                                                            
│   ├── ann.c                           // code for neural network (create, predict, train)
│   ├── ann.h                           // header file for ann.c
│   ├── cnnlayer.c                      // code for Convolutional layers
│   ├── cnnlayer.h                      // header file for cnnlayer.c
│   ├── conv.c                          // operations for convolutional layers
│   ├── data.temp                       // temp file to store training data for graph plotting
│   ├── flattenlayer.c                  // flatten layer for CNN
│   ├── flattenlayer.h                  // header file for flatten layer
│   ├── helper.c                        // code for graph plotting
│   ├── helper.h                        // header file for helper.c
│   ├── layer.c                         // code for fully connected layers
│   ├── layer.h                         // header file for layer.c
│   ├── math_funcs.c                    // math functions, e.g. activation and loss functions
│   ├── math_funcs.h                    // header file for math_funcs.c
│   ├── math_r4t.c                      // operations on the r4t struct 
│   ├── math_r4t.h                      // header file for math_r4t.c
│   ├── math_structs.c                  // matrix operations
│   ├── math_structs.h                  // header file for math_structs.c
│   ├── mnist.c                         // Main program -- tests the DNN on the MNIST dataset
│   ├── tensor.c                        // implementation of Pytorch Tensors
│   ├── tensor.h                        // header file for tensor.c
│   ├── tensor_math.c                   // math operations on tensors
│   └── xor.c                           // testing program to test sigmoidal MLP to learn the XOR function. Use "make xor" to build.
├── data                                // Image data downloaded from official MNIST website
│   ├── test-images.idx3-ubyte          
│   ├── test-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
└── results
    ├── mnist_accuracy_100x100.png      // Accuracy graph for 2 hidden layers (100+100), both dropout 0.4, mini-batch GD
    ├── mnist_accuracy_100x100_adam.png // Accuracy graph for 2 hidden layers (100+100), only final hidden layer dropout 0.4, Adam optimizer (BEST)
    ├── mnist_accuracy_30x30.png        // Accuracy graph for 2 hidden layers (30+30), no dropout, mini-batch GD (INITIAL)
    ├── mnist_accuracy_60x60.png        // Accuracy graph for 2 hidden layers (60+60), both dropout 0.4, mini-batch GD
    ├── mnist_loss_100x100.png          // Same descriptions as above, but for loss graph (Mean Cross Entropy loss of the whole training set in each epoch)
    ├── mnist_loss_100x100_adam.png
    ├── mnist_loss_30x30.png
    ├── mnist_loss_60x60.png
    └── xor.png                         // Loss graph (MSE) for XOR network (1 hidden layer, 2 neurons)
```
## 🔢 Math and Logic
Only the Deep Neural Network (with Fully Connected Layers) is tested with MNIST. Convolutional network is partially complete.
### Network architecture
### Backpropagation
### Mini-batch Gradient Descent
### Dropout 
### Adam Optimizer
### Training and Graph plotting
First load the byte input files of the MNIST images, then split them into batches of 16. Then trains the network for 20 epochs. \
Finally plots the graph using `gnuplot`.
## 🔨 Build and Run
Prerequisites:
- Linux / WSL (does not work on Windows 🤷)
- gnuplot (install via `sudo apt install gnuplot` in command line) \
If you encounter 404 Not Found errors while installing `gnuplot`, run `sudo apt-get update` then `sudo apt install gnuplot --fix-missing`. \
Once you have the prerequisites, type `cd code` followed by `make mnist` to generate the object files and `mnist` executable. Then run the program by typing `./mnist`. \
The output images will be produced in the `code` directory.

## 📊 Results
The validation set is not included in training and is a true reflection of how well the model is doing. \
Here's the results of some different network architectures that I tried:
1. 2 hidden layers (100+100), only final hidden layer dropout 0.4, Adam optimizer (BEST) \
Best network, highest validation accuracy is 97.40%.
2. 2 hidden layers (30+30), no dropout, mini-batch GD (INITIAL) \
Surprisingly the 2nd best network, the best amongst the mini-batch GD networks. Reached highest validation accuracy of 95.59%
3. 2 hidden layers (60+60), both dropout 0.4, mini-batch GD \
Highest validation accuracy 95.10%
5. 2 hidden layers (100+100), both dropout 0.4, mini-batch GD \
Surprisingly worse than 60+60, validation accuracy was only about 94+% \
<br>
From the graphs of 3 and 4, the validation accuracy/loss is consistently better than the training accuracy/loss, which indicates underfitting.

## 🌟 Credit/Acknowledgment
Credits to my group members Jeffrey Chang and Sam Shariatmadari for writing the code for the convolutional layers.

## 🚀 Future Improvements 
 - Put more neurons in the hidden layers. I did not do that because my potato machine will probably explode.
 - Try to use GPU for matrix operations as it provides a significant speedup.
 - (Specific for image classification) Perform some data augmentation to generate more training examples, such as rotating and resizing the images.
