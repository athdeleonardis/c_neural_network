# c_neural_network

Simple neural network written in c.

Uses a custom matrix library to store data and do computations.

Currently the neural network library itself only allows for creating, loading, evaluating and training models.

The neural networks currently use mean-squared error and sigmoid activation functions.
The use of mean-squared error is fundamental to the current implementation, however activation functions \
can be set layer by layer.

# Requirements

make: Used for project compilation.
gcc: Used by make for c file compilation.
A Linux terminal: To run make.

# Build

Run make in the base director of the project.

The object files will be output into the folder 'obj', and the compiled test programs in 'build/tests'.

# Usage

All files in the 'src' directory are necessary to use the neural network.
'neural_network.h' contains the base of the library, the neural network struct and functions to create and delete neural networks.
'neural_network_file.h' contains the functions necessary to load/save neural networks from/to files.
'neural_network_train.h' contains the functions necessary to train a neural network on input and output data.

# License

[MIT](https://choosealicense.com/licenses/mit/)
