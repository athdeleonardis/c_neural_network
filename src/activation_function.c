#include "activation_function.h"

#include "error.h"

#include <math.h>
#include <string.h>

//
// Auxilliary Definitions
//

/**
 * Maps from (-inf,+inf) to (-1,1). Similarly shaped to a tangent function.
*/
double sigmoid(double);
double sigmoid_derivative(double);
double relu(double);
double relu_derivative(double);
double leaky_relu(double);
double leaky_relu_derivative(double);

//
// 'neural_network_activation_functions.h' Implementations
//

// TODO: Parsing tree
activation_function_t activation_function_get(const char *name) {
    // Chain of if-elses hooray
    if (strcmp(name, "sigmoid") == 0) {
        activation_function_t af = { "sigmoid", sigmoid, sigmoid_derivative };
        return af;
    }
    if (strcmp(name, "relu") == 0) {
        activation_function_t af = { "relu", relu, relu_derivative };
        return af;
    }
    if (strcmp(name, "leaky_relu") == 0) {
        activation_function_t af = { "leaky_relu", leaky_relu, leaky_relu_derivative };
        return af;
    }
    make_error("Activation function does not exist");
}

void activation_function_copy(activation_function_t src, activation_function_t *dest) {
    strcpy(dest->name, src.name);
    dest->function = src.function;
    dest->derivative = src.derivative;
}

//
// Auxilliary Implementations
//

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double relu(double x) {
    if (x > 0)
        return x;
    return 0;
}

double relu_derivative(double x) {
    if (x > 0)
        return 1;
    return 0;
}

double leaky_relu(double x) {
    if (x > 0)
        return x;
    return 0.5 * x;    
}

double leaky_relu_derivative(double x) {
    if (x > 0)
        return 1;
    return 0.5;
}
