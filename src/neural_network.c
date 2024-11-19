#include "neural_network.h"

#include "random.h"

#include <stdio.h>
#include <stdlib.h>

//
// 'neural_network.c' definitions
//

void neural_network_layers_create(neural_network_t *, char **activation_functions);

//
// 'neural_network.c' implementations
//

void neural_network_layers_create(neural_network_t *nn, char **activation_functions) {
    nn->layers = (layer_t *)malloc((nn->hidden_layer_count + 1) * sizeof(layer_t));

    int cols = nn->input_size;
    int rows;
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        if (i != nn->hidden_layer_count)
            rows = nn->hidden_layer_sizes[i];
        else
            rows = nn->output_size;
        
        matrix_create_i(&nn->layers[i].weights, cols, rows);
        matrix_create_i(&nn->layers[i].biases, 1, rows);
        activation_function_copy(activation_function_get(activation_functions[i]), &nn->layers[i].activation_function);

        cols = rows;
    }
}

//
// 'neural_network.h' implementations
//

neural_network_t *neural_network_create(int input_size, int output_size, int hidden_layer_count, int *hidden_layer_sizes, char **activation_functions) {
    neural_network_t *nn = (neural_network_t *)malloc(sizeof(neural_network_t));
    nn->input_size = input_size;
    nn->output_size = output_size;
    nn->hidden_layer_count = hidden_layer_count;
    nn->hidden_layer_sizes = (int *)malloc(nn->hidden_layer_count * sizeof(int));
    for (int i = 0; i < nn->hidden_layer_count; i++)
        nn->hidden_layer_sizes[i] = hidden_layer_sizes[i];
    
    neural_network_layers_create(nn, activation_functions);
    return nn;
}

void neural_network_layers_from_array(neural_network_t *nn, double *data, char **activation_function_names) {
    int offset = 0;
    int cols = nn->input_size;
    int rows;
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        if (i != nn->hidden_layer_count)
            rows = nn->hidden_layer_sizes[i];
        else
            rows = nn->output_size;
        
        matrix_initialize_from_array(&nn->layers[i].weights, cols, rows, data, &offset);
        matrix_initialize_from_array(&nn->layers[i].biases, 1, rows, data, &offset);
        activation_function_copy(activation_function_get(activation_function_names[i]), &nn->layers[i].activation_function);

        cols = rows;
    }
}

void neural_network_delete(neural_network_t *nn) {
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        layer_t *layer = &nn->layers[i];
        free(layer->weights.data);
        free(layer->biases.data);
    }
    free(nn->hidden_layer_sizes);
    free(nn->layers);
    free(nn);
}

void neural_network_print(neural_network_t *nn) {
    printf("Neural network:\nInput size: %d\nOutput size: %d\nNumber of hidden layers: %d\nHidden layer sizes: ", nn->input_size, nn->output_size, nn->hidden_layer_count);
    for (int i = 0; i < nn->hidden_layer_count; i++)
        printf("%d ", nn->hidden_layer_sizes[i]);
    printf("\n");
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        printf("Layer %d:\nActivation Function: %s\nWeights: ", i, nn->layers[i].activation_function.name);
        matrix_print(&nn->layers[i].weights);
        printf("Biases: ");
        matrix_print(&nn->layers[i].biases);
    }
}

void neural_network_layers_randomize(neural_network_t *nn) {
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        layer_t layer = nn->layers[i];
        for (int j = 0; j < layer.weights.cols * layer.weights.rows; j++)
            layer.weights.data[j] = random_double_between((double)-1, (double)1);
        for (int j = 0; j < layer.biases.cols * layer.biases.rows; j++)
            layer.biases.data[j] = random_double_between((double)-1, (double)1);
    }
}

void neural_network_evaluate(neural_network_t *nn, int n_cases, matrix_t *inputs, matrix_t *outputs) {
    for (int i = 0; i < n_cases; i++) {
        matrix_t *output_i = inputs+i;
        for (int j = 0; j < nn->hidden_layer_count + 1; j++) {
            matrix_t *old = output_i;
            output_i = matrix_multiply_add(&nn->layers[j].weights, output_i, &nn->layers[j].biases);
            matrix_apply_function_i(output_i, nn->layers[j].activation_function.function);
            if (j)
                matrix_delete(old);
        }
        matrix_copy_o(output_i, outputs+i);
        matrix_delete(output_i);
    }
}
