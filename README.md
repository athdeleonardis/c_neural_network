# c_neural_network

Simple neural network written in c.

Uses a custom matrix library to store data and do computations.

Currently the neural network library itself only allows for creating, loading, evaluating and training models.

The neural networks currently use mean-squared error and sigmoid activation functions.
The use of mean-squared error is fundamental to the current implementation, however activation functions \
can be set layer by layer.

# Build

In the project's root directory, run 'make'.

The object files will output into the folder 'obj/'. \
Tests will output into the folder 'build/test/'. \
Apps will output into the folder 'build/app/'. \

# License

[MIT](./LICENSE.md)