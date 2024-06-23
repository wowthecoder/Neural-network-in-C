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
**4 Layers, batch size 16**:
1. Input layer: 784 outputs (28 x 28 px input image) 
2. Hidden layer: 60 neurons, RELU activation, dropout probability 0.2
3. Hidden layer: 60 neurons, RELU activation, dropout probability 0.2
4. Output layer: 10 neurons (to represent 10 classes), softmax activation

Number of training examples: 60000 (3750 batches) \
Number of validation examples: 10000 (625 batches) 

### Backpropagation
I started by implementing Stochastic Gradient Descent - update for every sample
![image](https://github.com/wowthecoder/Neural-network-in-C/assets/82577844/264817b5-7a52-43c3-bd91-b43eefb95ff0)

### Mini-batch Gradient Descent
The idea is that instead of beginning with a single input vector, $x$, we can begin with a matrix $X=[𝑥_1 𝑥_2 … 𝑥_𝑚]$ whose columns are the vectors in the mini-batch. We forward-propagate by multiplying by the weight matrices, adding a suitable matrix for the bias terms.

Setup: Change the dimensions of the delta matrix from (num_neurons x 1) to (num_neurons * batch_size) \[later denote as n x b], and calculate the deltas for each sample in a batch. \
Change the learning rate $\eta$ to $\eta / b$. \
<img src="https://github.com/wowthecoder/Neural-network-in-C/assets/82577844/6d0dcc51-b58e-4aa6-b69e-75f14c444cde" width="550">
<img src="https://github.com/wowthecoder/Neural-network-in-C/assets/82577844/f7b0eefd-23ff-4730-a92f-ce79cd4e66a6" width="400">

Does not actually provide a significant speedup because our matrix library is not optimised like NumPy. Fun to implement nonetheless.
### Dropout 
Dropout is a regularisation technique to reduce overfitting. It simply means that during training, we randomly omit some of the neurons in the layers with probability $p$, effectively only training a sub-network.
This forces the neurons to be more "independent" and not be reliant on certain inputs. The resulting network is essentially an ensemble of all the trained sub-networks, which generalises better to new data. \
<br>
Used a technique called inverted dropout to avoid modifying the outputs during testing phase. Here $q = 1 - p$
![image](https://github.com/wowthecoder/Neural-network-in-C/assets/82577844/9c1634cf-7115-4b99-9b17-3179bf5e2640)
![image](https://github.com/wowthecoder/Neural-network-in-C/assets/82577844/595539ec-4ce5-4a0d-8c18-896652ce37e7)


NOTE: When computing the delta matrices in backpropagation, the deltas are multiplied by the dropout mask to filter out the neurons that are dropped.

### Adam Optimizer
This was the missing piece that boosted the accuracy by a lot. Adam optimizer has a higher convergence speed than SGD, hence more optimal weights and biases are found. 
![image](https://github.com/wowthecoder/Neural-network-in-C/assets/82577844/0e62003b-4c5a-46ca-93d2-198df3c59437)

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
