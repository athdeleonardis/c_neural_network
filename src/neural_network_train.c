#include "neural_network_train.h"

#include <stdio.h>
#include <stdlib.h>

//
// Auxilliary Definitions
//

typedef struct {
    matrix_t **layer_outputs;
    matrix_t **layer_output_derivatives;
    matrix_t **layer_errors;
} neural_network_eval_t;

neural_network_eval_t new_eval(int hidden_layer_count);
void del_eval(neural_network_eval_t eval, int hidden_layer_count);
void get_eval(neural_network_t *, matrix_t *input, neural_network_eval_t);
void get_error(neural_network_t *, matrix_t *output, neural_network_eval_t);
void apply_eval(neural_network_t *, matrix_t *input, matrix_t *output, neural_network_eval_t, double p);

//
// 'neural_network_train.h' Implementations
//

void neural_network_train_case(neural_network_t *nn, matrix_t *input, matrix_t *output, double p) {
    neural_network_eval_t eval = new_eval(nn->hidden_layer_count);
    get_eval(nn, input, eval);
    get_error(nn, output, eval);
    apply_eval(nn, input, output, eval, p);
    del_eval(eval, nn->hidden_layer_count);
}

//
// Auxilliary Implementations
//

/**
 * For each layer of the neural network, create a datastructure to store the outputs at each layer,
 * the derivatives of each output, and the errors of each output.
*/
neural_network_eval_t new_eval(int hidden_layer_count) {
    neural_network_eval_t eval = {};
    eval.layer_output_derivatives = (matrix_t **)malloc((hidden_layer_count + 1)*sizeof(matrix_t));
    eval.layer_outputs = (matrix_t **)malloc((hidden_layer_count + 1)*sizeof(matrix_t));
    eval.layer_errors = (matrix_t **)malloc((hidden_layer_count + 1)*sizeof(matrix_t));
    return eval;
}

void del_eval(neural_network_eval_t eval, int hidden_layer_count) {
    for (int i = 0; i < hidden_layer_count + 1; i++) {
        matrix_delete(eval.layer_output_derivatives[i]);
        matrix_delete(eval.layer_outputs[i]);
        matrix_delete(eval.layer_errors[i]);
    }
    free(eval.layer_output_derivatives);
    free(eval.layer_outputs);
    free(eval.layer_errors);
}

/**
 * Get the outputs at each layer, and their respective derivatives.
*/
void get_eval(neural_network_t *nn, matrix_t *input, neural_network_eval_t eval) {
    matrix_t *prev_outputs;
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        if (i)
            prev_outputs = eval.layer_outputs[i-1];
        else
            prev_outputs = input;
        
        eval.layer_outputs[i] = matrix_multiply_add(nn->layers[i].weights, prev_outputs, nn->layers[i].biases);
        eval.layer_output_derivatives[i] = matrix_transpose(eval.layer_outputs[i]);

        matrix_apply_function_i(eval.layer_outputs[i], nn->layers[i].activation_function.function);
        matrix_apply_function_i(eval.layer_output_derivatives[i], nn->layers[i].activation_function.derivative);
    }
}

/**
 * Get the error of each every output at each layer.
*/
void get_error(neural_network_t *nn, matrix_t *output, neural_network_eval_t eval) {
    // Output layer error
    eval.layer_errors[nn->hidden_layer_count] = matrix_copy(eval.layer_outputs[nn->hidden_layer_count]);
    for (int i = 0; i < nn->output_size; i++) {
        eval.layer_errors[nn->hidden_layer_count]->data[i] -= output->data[i];
    }
    matrix_multiply_scalar_i(eval.layer_errors[nn->hidden_layer_count], eval.layer_output_derivatives[nn->hidden_layer_count]);

    // Hidden layer error, propogated backwards via the activation function derivative
    for (int i = nn->hidden_layer_count; i > 0; i--) {
        eval.layer_errors[i-1] = matrix_multiply(eval.layer_errors[i], nn->layers[i].weights);
        matrix_multiply_scalar_i(eval.layer_errors[i-1], eval.layer_output_derivatives[i-1]);
    }
}

/**
 * Apply the error and output values to each weight of the neural network.
*/
void apply_eval(neural_network_t *nn, matrix_t *input, matrix_t *output, neural_network_eval_t eval, double p) {
    for (int k = 0; k < nn->hidden_layer_count + 1; k++) {
        matrix_t *weights = nn->layers[k].weights;
        matrix_t *biases = nn->layers[k].biases;
        for (int j = 0; j < weights->rows; j++) {
            for (int i = 0; i < weights->cols; i++) {
                double output;
                if (k)
                    output = matrix_get(eval.layer_outputs[k-1], 0, i);
                else
                    output = matrix_get(input, 0, i);
                double error = matrix_get(eval.layer_errors[k], j, 0);
                matrix_set(weights, i, j,
                    matrix_get(weights, i, j) - p * output * error
                );
            }

            matrix_set(biases, 0, j,
                matrix_get(biases, 0, j) - p * matrix_get(eval.layer_errors[k], j, 0)
            );
        }
    }
}
