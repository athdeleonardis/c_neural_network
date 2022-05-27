#include "neural_network_file.h"

#include "load_file.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>

//
// Auxilliary Definitions
//

neural_network_t *neural_network_load(FILE *, int is_dynamic);
void neural_network_save(neural_network_t *, FILE *, int is_dynamic);
void neural_network_layers_load(neural_network_t *, FILE *file);
void neural_network_layers_save(neural_network_t *, FILE *file);

//
// Auxilliary Implementations
//

void neural_network_layers_save(neural_network_t *nn, FILE *file) {
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        layer_t layer = nn->layers[i];
        fwrite(layer.weights->data, sizeof(double), layer.weights->cols * layer.weights->rows, file);
        fwrite(layer.biases->data, sizeof(double), layer.biases->cols * layer.biases->rows, file);
    }
}

void neural_network_layers_load(neural_network_t *nn, FILE *file) {
    for (int i = 0; i < nn->hidden_layer_count+1; i++) {
        layer_t layer = nn->layers[i];
        fread(layer.weights->data, sizeof(double), layer.weights->cols * layer.weights->rows, file);
        fread(layer.biases->data, sizeof(double), layer.biases->cols * layer.biases->rows, file);
    }
}

void neural_network_save(neural_network_t *nn, FILE *file, int is_dynamic) {
    char c;
    if (is_dynamic)
        c = 'd';
    else
        c = 's';
    fwrite(&c, sizeof(char), 1, file);
    fwrite(&nn->input_size, sizeof(int), 1, file);
    fwrite(&nn->output_size, sizeof(int), 1, file);
    fwrite(&nn->hidden_layer_count, sizeof(int), 1, file);
    fwrite(nn->hidden_layer_sizes, sizeof(int), nn->hidden_layer_count, file);
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        fwrite(nn->layers[i].activation_function.name, sizeof(nn->layers[i].activation_function.name), 1, file);
    }
}

neural_network_t *neural_network_load(FILE *file, int is_dynamic) {
    // static vs dynamic model save
    char type;
    fread(&type, sizeof(char), 1, file);
    cnd_make_error(is_dynamic && type != 'd', "Attempting to load dynamic model from static savefile");

    // input_size output_size hidden_layer_count
    int buffer[3];
    fread(buffer, sizeof(buffer), 1, file);

    // hidden_layer_sizes
    int *hidden_layer_sizes = (int *)malloc(buffer[2] * sizeof(int));
    fread(hidden_layer_sizes, sizeof(int), buffer[2], file);

    char **activation_function_names = (char **)malloc(sizeof(char *));
    for (int i = 0; i < buffer[2] + 1; i++) {
        activation_function_names[i] = (char *)malloc(ACTIVATION_FUNCTION_NAME_SIZE * sizeof(char));
        fread(activation_function_names[i], sizeof(char), ACTIVATION_FUNCTION_NAME_SIZE, file);
    }

    neural_network_t *nn = neural_network_create(buffer[0], buffer[1], buffer[2], hidden_layer_sizes, activation_function_names);

    free(hidden_layer_sizes);
    for (int i = 0; i < buffer[2]; i++)
        free(activation_function_names[i]);
    free(activation_function_names);

    return nn;
}

//
// 'neural_network_files.h' Implementations
//

neural_network_t *neural_network_load_static(const char *filename) {
    FILE *file = load_file(filename);
    neural_network_t *nn = neural_network_load(file, 0);
    fclose(file);
    return nn;
}

neural_network_t *neural_network_load_dynamic(const char *filename) {
    FILE *file = load_file(filename);
    neural_network_t *nn = neural_network_load(file, 1);
    neural_network_layers_load(nn, file);
    fclose(file);
    return nn;
}

void neural_network_save_static(neural_network_t *nn, const char *filename) {
    FILE *file = fopen(filename, "wb");
    neural_network_save(nn, file, 0);
    fclose(file);
}

void neural_network_save_dynamic(neural_network_t *nn, const char *filename) {
    FILE *file = fopen(filename, "wb");
    neural_network_save(nn, file, 1);
    neural_network_layers_save(nn, file);
    fclose(file);
}
