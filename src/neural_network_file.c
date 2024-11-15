#include "neural_network_file.h"

#include "file_load.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>

#define NEURAL_NETWORK_FILE_TYPE_STATIC 's'
#define NEURAL_NETWORK_FILE_TYPE_DYNAMIC 'd'

//
// 'neural_network_file.c' definitions
//

void neural_network_save_internal_type(FILE *file, char type);
void neural_network_save_internal_structure(FILE *file, neural_network_t *nn);
void neural_network_save_internal_layers(FILE *file, neural_network_t *nn);

void neural_network_load_internal_type(FILE *file, char type);
neural_network_t *neural_network_load_internal_structure(FILE *file);
void neural_network_load_internal_layers(FILE *file, neural_network_t *nn);

//
// 'neural_network_file.h' implementations
//

void neural_network_save_static(neural_network_t *nn, const char *filename) {
    FILE *file = fopen(filename, "wb");
    neural_network_save_internal_type(file, NEURAL_NETWORK_FILE_TYPE_STATIC);
    neural_network_save_internal_structure(file, nn);
    fclose(file);
}

void neural_network_save_dynamic(neural_network_t *nn, const char *filename) {
    FILE *file = fopen(filename, "wb");
    neural_network_save_internal_type(file, NEURAL_NETWORK_FILE_TYPE_DYNAMIC);
    neural_network_save_internal_structure(file, nn);
    neural_network_save_internal_layers(file, nn);
    if (fclose(file))
        printf("Error when closing file?\n");
}

neural_network_t *neural_network_load_static(const char *filename) {
    FILE *file = file_load(filename);
    neural_network_load_internal_type(file, NEURAL_NETWORK_FILE_TYPE_STATIC);
    neural_network_t *nn = neural_network_load_internal_structure(file);
    fclose(file);
    return nn;
}

neural_network_t *neural_network_load_dynamic(const char *filename) {
    FILE *file = file_load(filename);
    neural_network_load_internal_type(file, NEURAL_NETWORK_FILE_TYPE_DYNAMIC);
    neural_network_t *nn = neural_network_load_internal_structure(file);
    neural_network_load_internal_layers(file, nn);
    fclose(file);
    return nn;
}

//
// 'neural_network_file.c' implementations
//

void neural_network_save_internal_type(FILE *file, char type) {
    fwrite(&type, sizeof(char), 1, file);
}

void neural_network_save_internal_structure(FILE *file, neural_network_t *nn) {
    fwrite(&nn->input_size, sizeof(int), 1, file);
    fwrite(&nn->output_size, sizeof(int), 1, file);
    fwrite(&nn->hidden_layer_count, sizeof(int), 1, file);
    fwrite(nn->hidden_layer_sizes, sizeof(int), nn->hidden_layer_count, file);
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        fwrite(nn->layers[i].activation_function.name, sizeof(nn->layers[i].activation_function.name), 1, file);
    }
}

void neural_network_save_internal_layers(FILE *file, neural_network_t *nn) {
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        layer_t layer = nn->layers[i];
        fwrite(layer.weights->data, sizeof(double), layer.weights->cols * layer.weights->rows, file);
        fwrite(layer.biases->data, sizeof(double), layer.biases->cols * layer.biases->rows, file);
    }
}

void neural_network_load_internal_type(FILE *file, char type) {
    // static vs dynamic model save
    char file_type;
    fread(&file_type, sizeof(char), 1, file);
    if (type == NEURAL_NETWORK_FILE_TYPE_STATIC)
        return;
    cnd_make_error(file_type != type, "Attempting to load dynamic model from static model savefile");
}

neural_network_t *neural_network_load_internal_structure(FILE *file) {
    int buffer[3] = { 0, 0, 0 };
    fread(buffer, sizeof(int), 3, file);
    int input_size = buffer[0];
    int output_size = buffer[1];
    int hidden_layer_count = buffer[2];

    // hidden_layer_sizes
    int *hidden_layer_sizes = (int *)malloc(hidden_layer_count * sizeof(int));
    fread(hidden_layer_sizes, sizeof(int), hidden_layer_count, file);

    char **activation_function_names = (char **)malloc(hidden_layer_count * sizeof(char *));
    for (int i = 0; i < hidden_layer_count + 1; i++) {
        activation_function_names[i] = (char *)malloc(ACTIVATION_FUNCTION_NAME_SIZE * sizeof(char));
        fread(activation_function_names[i], sizeof(char), ACTIVATION_FUNCTION_NAME_SIZE, file);
    }

    neural_network_t *nn = neural_network_create(input_size, output_size, hidden_layer_count, hidden_layer_sizes, activation_function_names);

    free(hidden_layer_sizes);
    for (int i = 0; i < hidden_layer_count + 1; i++)
        free(activation_function_names[i]);
    free(activation_function_names);

    return nn;
}

void neural_network_load_internal_layers(FILE *file, neural_network_t *nn) {
    for (int i = 0; i < nn->hidden_layer_count+1; i++) {
        layer_t layer = nn->layers[i];
        fread(layer.weights->data, sizeof(double), layer.weights->cols * layer.weights->rows, file);
        fread(layer.biases->data, sizeof(double), layer.biases->cols * layer.biases->rows, file);
    }
}
