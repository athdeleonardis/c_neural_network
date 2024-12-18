#include "neural_network_train.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>

//
// 'neural_network_train.c' definitions
//

void check_input_size(neural_network_t *nn, matrix_t *mat);
void check_output_size(neural_network_t *nn, matrix_t *output);
void neural_network_evaluation_layer_initialize(neural_network_evaluation_layer_t *eval_layer, double *data, int array_size, int *offset);

//
// 'neural_network_train.h' implementations
//

void neural_network_train_case(neural_network_t *nn, matrix_t *input, matrix_t *output, double p) {
    check_input_size(nn, input);
    check_output_size(nn, output);

    neural_network_evaluation_t eval;
    neural_network_evaluation_initialize(nn, &eval);
    neural_network_evaluation_outputs(nn, input, eval);
    neural_network_evaluation_errors(nn, output, eval);
    neural_network_evaluation_apply(nn, input, eval, p);
    neural_network_evaluation_delete(eval);
}

//
// 'neural_network_train.c' implementations
//

void check_input_size(neural_network_t *nn, matrix_t *input) {
    cnd_make_error(input->cols != 1 && input->rows != nn->input_size, "Input matrix size incompatible with neural network.\n");
}

void check_output_size(neural_network_t *nn, matrix_t *output) {
    cnd_make_error(output->cols != 1 && output->rows != nn->output_size, "Output matrix size incompatible with neural network.\n");
}

void neural_network_evaluation_initialize(neural_network_t *nn, neural_network_evaluation_t *eval) {
    // For each output layer of the neural network, allocate three arrays, outputs derivatives and errors.
    int data_length = nn->output_size;
    for (int i = 0; i < nn->hidden_layer_count; i++) {
        data_length += nn->hidden_layer_sizes[i];
    }
    data_length *= 3;
    eval->all_data = (double *)malloc(data_length * sizeof(double));
    eval->layers = (neural_network_evaluation_layer_t *)malloc((nn->hidden_layer_count + 1) * sizeof(neural_network_evaluation_layer_t));

    // Partition the array 'all_data' into each 'neural_network_evaluation_layer_t's matrix elements.
    int i = 0;
    int data_offset = 0;
    for (; i < nn->hidden_layer_count; i++) {
        neural_network_evaluation_layer_initialize(&eval->layers[i], eval->all_data, nn->hidden_layer_sizes[i], &data_offset);
    }
    neural_network_evaluation_layer_initialize(&eval->layers[i], eval->all_data, nn->output_size, &data_offset);
}

void neural_network_evaluation_layer_initialize(neural_network_evaluation_layer_t *eval_layer, double *data, int array_size, int *offset) {
    matrix_initialize_from_array(&eval_layer->outputs, 1, array_size, data, offset);
    matrix_initialize_from_array(&eval_layer->derivatives, array_size, 1, data, offset);
    matrix_initialize_from_array(&eval_layer->errors, array_size, 1, data, offset);
}

void neural_network_evaluation_delete(neural_network_evaluation_t eval) {
    free(eval.all_data);
    free(eval.layers);
}

void neural_network_evaluation_outputs(neural_network_t *nn, matrix_t *input, neural_network_evaluation_t eval) {
    matrix_t *prev_outputs;
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        if (i)
            prev_outputs = &eval.layers[i-1].outputs;
        else
            prev_outputs = input;

        matrix_multiply_o(&nn->layers[i].weights, prev_outputs, &eval.layers[i].outputs);
        matrix_add_i(&eval.layers[i].outputs, &nn->layers[i].biases);
        matrix_transpose_o(&eval.layers[i].outputs, &eval.layers[i].derivatives);

        matrix_apply_function_i(&eval.layers[i].outputs, nn->layers[i].activation_function.function);
        matrix_apply_function_i(&eval.layers[i].derivatives, nn->layers[i].activation_function.derivative);
    }
}

void neural_network_evaluation_errors(neural_network_t *nn, matrix_t *output, neural_network_evaluation_t eval) {
    int final_layer = nn->hidden_layer_count;
    // Output layer error
    matrix_transpose_o(&eval.layers[final_layer].outputs, &eval.layers[final_layer].errors);
    // Need to subtract expected output from actual output, but their dimensions are the tranpose of eachother.
    for (int i = 0; i < eval.layers[final_layer].errors.cols; i++) {
        eval.layers[final_layer].errors.data[i] -= output->data[i];
    }
    matrix_multiply_scalar_i(&eval.layers[final_layer].errors, &eval.layers[final_layer].derivatives);

    // Hidden layer error, propogated backwards via the activation function derivative
    for (int i = nn->hidden_layer_count; i > 0; i--) {
        matrix_multiply_o(&eval.layers[i].errors, &nn->layers[i].weights, &eval.layers[i-1].errors);
        matrix_multiply_scalar_i(&eval.layers[i-1].errors, &eval.layers[i-1].derivatives);
    }
}

void neural_network_evaluation_apply(neural_network_t *nn, matrix_t *input, neural_network_evaluation_t eval, double p) {
    for (int k = 0; k < nn->hidden_layer_count + 1; k++) {
        matrix_t *weights = &nn->layers[k].weights;
        matrix_t *biases = &nn->layers[k].biases;
        for (int j = 0; j < weights->rows; j++) {
            for (int i = 0; i < weights->cols; i++) {
                double output;
                if (k)
                    output = matrix_get(&eval.layers[k-1].outputs, 0, i);
                else
                    output = matrix_get(input, 0, i);
                double error = matrix_get(&eval.layers[k].errors, j, 0);
                matrix_set(weights, i, j,
                    matrix_get(weights, i, j) - p * output * error
                );
            }

            matrix_set(biases, 0, j,
                matrix_get(biases, 0, j) - p * matrix_get(&eval.layers[k].errors, j, 0)
            );
        }
    }
}
