#include "neural_network_train.h"

#include <stdio.h>
#include <stdlib.h>

//
// Auxilliary Definitions
//

typedef struct {
    matrix_t **activation_derivative;
    matrix_t **activation_function;
    matrix_t **errors;
} neural_network_eval_t;

neural_network_eval_t new_eval(int hidden_layer_count);
void del_eval(neural_network_eval_t eval, int hidden_layer_count);
void get_eval(neural_network_t *, matrix_t *input, neural_network_eval_t);
void get_error(neural_network_t *, matrix_t *output, neural_network_eval_t);
void apply_eval(neural_network_t *, matrix_t *input, matrix_t *output, neural_network_eval_t, double p);

//
// Auxilliary Implementations
//

neural_network_eval_t new_eval(int hidden_layer_count) {
    neural_network_eval_t eval = {};
    eval.activation_derivative = (matrix_t **)malloc((hidden_layer_count + 1)*sizeof(matrix_t));
    eval.activation_function = (matrix_t **)malloc((hidden_layer_count + 1)*sizeof(matrix_t));
    eval.errors = (matrix_t **)malloc((hidden_layer_count + 1)*sizeof(matrix_t));
    return eval;
}

void del_eval(neural_network_eval_t eval, int hidden_layer_count) {
    for (int i = 0; i < hidden_layer_count + 1; i++) {
        matrix_delete(eval.activation_derivative[i]);
        matrix_delete(eval.activation_function[i]);
        matrix_delete(eval.errors[i]);
    }
    free(eval.activation_derivative);
    free(eval.activation_function);
    free(eval.errors);
}

void get_eval(neural_network_t *nn, matrix_t *input, neural_network_eval_t eval) {
    matrix_t *prev_outputs;
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        if (i)
            prev_outputs = eval.activation_function[i-1];
        else
            prev_outputs = input;
        
        eval.activation_function[i] = matrix_multiply_add(nn->layers[i].weights, prev_outputs, nn->layers[i].biases);
        eval.activation_derivative[i] = matrix_transpose(eval.activation_function[i]);

        matrix_apply_function_i(eval.activation_function[i], nn->layers[i].activation_function.function);
        matrix_apply_function_i(eval.activation_derivative[i], nn->layers[i].activation_function.derivative);
    }
}

void get_error(neural_network_t *nn, matrix_t *output, neural_network_eval_t eval) {
    // Output layer error
    eval.errors[nn->hidden_layer_count] = matrix_copy(eval.activation_function[nn->hidden_layer_count]);
    for (int i = 0; i < nn->output_size; i++) {
        eval.errors[nn->hidden_layer_count]->data[i] -= output->data[i];
    }
    matrix_multiply_scalar_i(eval.errors[nn->hidden_layer_count], eval.activation_derivative[nn->hidden_layer_count]);

    // Hidden layer error
    for (int i = nn->hidden_layer_count; i > 0; i--) {
        eval.errors[i-1] = matrix_multiply(eval.errors[i], nn->layers[i].weights);
        matrix_multiply_scalar_i(eval.errors[i-1], eval.activation_derivative[i-1]);
    }
}

void apply_eval(neural_network_t *nn, matrix_t *input, matrix_t *output, neural_network_eval_t eval, double p) {
    for (int k = 0; k < nn->hidden_layer_count + 1; k++) {
        matrix_t *weights = nn->layers[k].weights;
        matrix_t *biases = nn->layers[k].biases;
        for (int j = 0; j < weights->rows; j++) {
            for (int i = 0; i < weights->cols; i++) {
                double output;
                if (k)
                    output = matrix_get(eval.activation_function[k-1], 0, i);
                else
                    output = matrix_get(input, 0, i);
                double error = matrix_get(eval.errors[k], j, 0);
                matrix_set(weights, i, j,
                    matrix_get(weights, i, j) - p * output * error
                );
            }

            matrix_set(biases, 0, j,
                matrix_get(biases, 0, j) - p * matrix_get(eval.errors[k], j, 0)
            );
        }
    }
}

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
