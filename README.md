# c_neural_network

A library for simple feed-forward neural networks written in C.

The library currently has the following features:
- A custom matrix library, contained in 'src/matrix.h' and 'src/matrix.c'.
- A feed-forward neural network struct, 'neural_network_t', contained in 'src/neural_network.h' and 'src/neural_network.c'.
- Computing the output of neural networks against inputs, two separate implementations contained in 'src/neural_network.h' and 'src/neural_network_train.h'.
- Activation functions that can be set layer-by-layer, currently implemented 'sigmoid', 'relu' and 'leaky relu' in the files 'src/activation_function.h' and 'src/activation_function.c'.
- Saving and loading of the neural network's structure or structure & weights & biases, contained in the files 'src/neural_network_file.h' and 'src/neural_network_file.c'.
- Training of the neural network against inputs and expected outputs, contained in 'neural_network_train.h' and 'neural_network_train.c'.

## Build

You can generate your build system of choice using CMake. E.g.

`cmake -S . -B build`

# Examples

For examples on how the library is used, you can look through
- The test files located in the folder 'test'
- The example apps located in the foler 'app'
  | The app 'mnist' can be used to train and test neural network files against the MNIST dataset,
  | a popular example dataset in AI, where the inputs are 28x28 pixel images of handwritten digits, and outputs are the digit drawn in the image.
  | The training and testing datasets contain 60,000 and 10,000 cases respectively.
  | Read about the mnist dataset and it's format here:
  | https://yann.lecun.com/exdb/mnist/

## License

[MIT](./LICENSE.md)