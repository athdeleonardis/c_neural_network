#include "neural_network.h"

#include "random.h"

#include <stdio.h>
#include <stdlib.h>

//
// Auxilliary Definitions
//

void neural_network_layers_create(neural_network_t *);

//
// Auxilliary Implementations
//

void neural_network_layers_create(neural_network_t *nn) {
    nn->layers = (layer_t *)malloc((nn->hidden_layer_count + 1) * sizeof(layer_t));

    int cols = nn->input_size;
    int rows;
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        if (i != nn->hidden_layer_count)
            rows = nn->hidden_layer_sizes[i];
        else
            rows = nn->output_size;
        
        nn->layers[i].weights = matrix_create(cols, rows);
        nn->layers[i].biases = matrix_create(1, rows);

        cols = rows;
    }
}

//
// "neural_network.h" Implementations
//

neural_network_t *neural_network_create(int input_size, int output_size, int hidden_layer_count, int *hidden_layer_sizes) {
    neural_network_t *nn = (neural_network_t *)malloc(sizeof(neural_network_t));
    nn->input_size = input_size;
    nn->output_size = output_size;
    nn->hidden_layer_count = hidden_layer_count;
    nn->hidden_layer_sizes = (int *)malloc(nn->hidden_layer_count * sizeof(int));
    for (int i = 0; i < nn->hidden_layer_count; i++)
        nn->hidden_layer_sizes[i] = hidden_layer_sizes[i];
    
    neural_network_layers_create(nn);
    return nn;
}

void neural_network_delete(neural_network_t *nn) {
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        layer_t *layer = &nn->layers[i];
        matrix_delete(layer->weights);
        matrix_delete(layer->biases);
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
        printf("Layer %d:\nWeights: ", i);
        matrix_print(nn->layers[i].weights);
        printf("Biases: ");
        matrix_print(nn->layers[i].biases);
    }
}

void neural_network_layers_randomize(neural_network_t *nn) {
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        layer_t layer = nn->layers[i];
        for (int j = 0; j < layer.weights->cols * layer.weights->rows; j++)
            layer.weights->data[j] = random_double_between((double)-1, (double)1);
        for (int j = 0; j < layer.biases->cols * layer.biases->rows; j++)
            layer.biases->data[j] = random_double_between((double)-1, (double)1);
    }
}