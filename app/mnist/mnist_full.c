#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#include "mnist.h"
#include "mnist_full.h"
#include "../../src/neural_network.h"
#include "../../src/neural_network_train.h"
#include "../../src/neural_network_file.h"

//
// 'mnist_full.c' definitions
//

#define NN_INPUT_SIZE INPUT_SIZE
#define NN_OUTPUT_SIZE OUTPUT_SIZE
#define NN_HIDDEN_LAYER_COUNT 1
#define NN_HIDDEN_LAYER_SIZE_1 32
#define NN_HIDDEN_LAYER_SIZES { NN_HIDDEN_LAYER_SIZE_1 }
#define NN_ACTIVATION_FUNCTIONS { "sigmoid", "sigmoid" }
#define NN_LAYER_DATA_SIZE ((NN_INPUT_SIZE + 1) * NN_HIDDEN_LAYER_SIZE_1 + (NN_HIDDEN_LAYER_SIZE_1 + 1) * NN_OUTPUT_SIZE)

#define BATCH_SIZE 128
#define TRAINING_PARAMETER_INITIAL 0.01
#define TRAINING_PARAMETER_FINAL 0.001

#define N_THREADS 2

typedef struct {
    double *inputs_data;
    matrix_t *inputs;
    unsigned char *outputs;
    matrix_t *output_map;
    neural_network_evaluation_t *eval;
} storage_t;

neural_network_t *neural_network_init();
double training_parameter_calc(double p_high, double p_low, int cases_correct, int total_cases);
void train_all_cases(neural_network_t *nn, mnist_handle_t *mh, storage_t storage, double training_parameter);
int evaluate_all_cases(neural_network_t *nn, mnist_handle_t *mh, storage_t storage);
void log_start(const char *filename);
void log_append(const char *filename, char *str);
void log_append_time(const char *filename, char *string_buffer, const char *label, clock_t start, clock_t end);

//
// 'mnist_full.h' implementations
//

void mnist_full() {
    //
    // Setup
    //

    // For console and file logging
    const char *log_file_name = "logs/mnist.txt";
    char string_buffer[64];

    // Initialize the MNIST file handle
    unsigned char input_data_unformated[BATCH_SIZE * INPUT_SIZE];
    mnist_handle_t mnist_handle_training = mnist_handle_init(MNIST_N_CASES_TRAINING, BATCH_SIZE, input_data_unformated);
    mnist_handle_t mnist_handle_testing = mnist_handle_init(MNIST_N_CASES_TESTING, BATCH_SIZE, input_data_unformated);
    mnist_images_load(MNIST_DATASET_TRAINING_IMAGES, &mnist_handle_training);
    mnist_labels_load(MNIST_DATASET_TRAINING_LABELS, &mnist_handle_training);
    mnist_images_load(MNIST_DATASET_TESTING_IMAGES, &mnist_handle_testing);
    mnist_labels_load(MNIST_DATASET_TESTING_LABELS, &mnist_handle_testing);

    // Storage space to send mnist_handle image data to.
    double inputs_data[BATCH_SIZE * INPUT_SIZE];
    matrix_t inputs[BATCH_SIZE];
    matrix_initialize_multiple_from_array(inputs, BATCH_SIZE, 1, INPUT_SIZE, inputs_data);

    // A map from digit to neural network output.
    double output_map_data[OUTPUT_DATA_SIZE];
    mnist_initialize_output_data(output_map_data);
    matrix_t output_map[OUTPUT_SIZE];
    mnist_initialize_outputs(output_map, output_map_data);

    unsigned char outputs[BATCH_SIZE];

    // Set up a randomized neural network.
    int hidden_layer_sizes[NN_HIDDEN_LAYER_COUNT+1] = NN_HIDDEN_LAYER_SIZES;
    char *activation_function_names[NN_HIDDEN_LAYER_COUNT+1] = NN_ACTIVATION_FUNCTIONS;
    layer_t neural_network_layers[NN_HIDDEN_LAYER_COUNT+1];
    double neural_network_layer_data[NN_LAYER_DATA_SIZE];
    neural_network_t neural_network = {
        .input_size=NN_INPUT_SIZE,
        .output_size=NN_OUTPUT_SIZE,
        .hidden_layer_count=NN_HIDDEN_LAYER_COUNT,
        .hidden_layer_sizes=hidden_layer_sizes,
        .layers=neural_network_layers
    };
    neural_network_layers_from_array(&neural_network, neural_network_layer_data, activation_function_names);
    neural_network_layers_randomize(&neural_network);

    neural_network_evaluation_t eval;
    neural_network_evaluation_initialize(&neural_network, &eval);

    storage_t storage = {};
    storage.inputs_data = inputs_data;
    storage.inputs = inputs;
    storage.outputs = outputs;
    storage.output_map = output_map;
    storage.eval = &eval;

    //
    // Training and evaluating
    //

    log_start(log_file_name);

    int best_epoch = 0;
    int max_num_correct = 0;
    clock_t start_total;
    for (int i = 0; 1; i++) {
        sprintf(string_buffer, "-- Epoch %02d --\n", i);
        log_append(log_file_name, string_buffer);

        // Training
        clock_t start_epoch = clock();
        clock_t start = start_epoch;
        if (i) {
            double training_parameter = training_parameter_calc(TRAINING_PARAMETER_INITIAL, TRAINING_PARAMETER_FINAL, max_num_correct, mnist_handle_testing.num_cases);
            train_all_cases(&neural_network, &mnist_handle_training, storage, training_parameter);
            log_append(log_file_name, "Trained all cases.\n");
        }
        else {
            log_append(log_file_name, "No training.\n");
        }
        log_append_time(log_file_name, string_buffer, "Time taken", start, clock());

        // Test against training data
        start = clock();
        int training_cases_correct = evaluate_all_cases(&neural_network, &mnist_handle_training, storage);
        sprintf(string_buffer, "Training dataset evaluation: %d / %d, %.01f%%\n", training_cases_correct, mnist_handle_training.num_cases, (double)100 * training_cases_correct / mnist_handle_training.num_cases);
        log_append(log_file_name, string_buffer);
        log_append_time(log_file_name, string_buffer, "Time taken", start, clock());

        // Test against testing data
        start = clock();
        int testing_cases_correct = evaluate_all_cases(&neural_network, &mnist_handle_testing, storage);
        sprintf(string_buffer, "Testing dataset evaluation: %d / %d, %.01f%%\n", testing_cases_correct, mnist_handle_testing.num_cases, (double)100 * testing_cases_correct / mnist_handle_testing.num_cases);
        log_append(log_file_name, string_buffer);
        clock_t end = clock();
        log_append_time(log_file_name, string_buffer, "Time taken", start, end);
        log_append_time(log_file_name, string_buffer, "Epoch time taken", start_epoch, end);
        log_append_time(log_file_name, string_buffer, "Total time taken", start_total, end);

        if (testing_cases_correct > max_num_correct) {
            sprintf(string_buffer, "New best epoch. Saving neural network.\n");
            log_append(log_file_name, string_buffer);
            max_num_correct = testing_cases_correct;
            best_epoch = i;
            neural_network_save_dynamic(&neural_network, "models/mnist.model.dynamic");
        }
        else {
            sprintf(string_buffer, "Epoch performed worse than last. Exiting.\n");
            log_append(log_file_name, string_buffer);
            break;
        }
    }

    neural_network_evaluation_delete(eval);
    mnist_handle_close(&mnist_handle_training);
    mnist_handle_close(&mnist_handle_testing);
}

/**
 * As more cases become correct, lower the training parameter.
*/
double training_parameter_calc(double p_high, double p_low, int cases_correct, int total_cases) {
    double proportion_correct = (double)cases_correct / total_cases;
    double lerp_factor = proportion_correct * proportion_correct;
    return p_low * lerp_factor + p_high * (1 - lerp_factor);
}

void train_all_cases(neural_network_t *nn, mnist_handle_t *mh, storage_t storage, double training_parameter) {
    mnist_reset(mh);
    int num_cases;
    while (num_cases = mnist_load_batch(mh, storage.inputs_data, storage.outputs)) {
        for (int i = 0; i < num_cases; i++) {
            unsigned char label = storage.outputs[i];
            matrix_t *output = &storage.output_map[label];
            //neural_network_train_case(nn, mh->inputs+i, output, training_parameter);
            neural_network_evaluation_outputs(nn, storage.inputs+i, *storage.eval);
            neural_network_evaluation_errors(nn, output, *storage.eval);
            neural_network_evaluation_apply(nn, storage.inputs+i, *storage.eval, training_parameter);
        }
        printf("Trained: %d / %d\r", mh->index, mh->num_cases);
        fflush(stdout);
    }
}

/**
 * @return The number of cases correctly classified.
*/
int evaluate_all_cases(neural_network_t *nn, mnist_handle_t *mh, storage_t storage) {
    int num_cases_correct = 0;
    mnist_reset(mh);
    int num_cases;
    while (num_cases = mnist_load_batch(mh, storage.inputs_data, storage.outputs)) {
        //neural_network_evaluate(nn, num_cases, mh->inputs, outputs_calculated);
        for (int i = 0; i < num_cases; i++) {
            neural_network_evaluation_outputs(nn, storage.inputs+i, *storage.eval);
            unsigned char label = storage.outputs[i];
            unsigned char label_calculated = mnist_output_to_number(&storage.eval->layers[nn->hidden_layer_count].outputs);
            num_cases_correct += label == label_calculated;
        }
        printf("Tested: %d / %d\r", mh->index, mh->num_cases);
        fflush(stdout);
    }
    return num_cases_correct;
}

void log_start(const char *filename) {
    FILE *file = fopen(filename, "w");
    fprintf(file, "Training neural network on the MNIST training dataset.\n");
    fclose(file);
}

void log_append(const char *filename, char *str) {
    FILE *file = fopen(filename, "a");
    printf("%s", str);
    fprintf(file, "%s", str);
    fclose(file);
}

void log_append_time(const char *filename, char *string_buffer, const char *label, clock_t start, clock_t end) {
    int milliseconds = (end - start) * 1000 / CLOCKS_PER_SEC;
    int seconds  = (milliseconds / 1000) % 60;
    int minutes = milliseconds / 1000 / 60;
    milliseconds = milliseconds % 1000;
    FILE *file = fopen(filename, "a");
    sprintf(string_buffer, "%s: %dm %d.%ds\n", label, minutes, seconds, milliseconds);
    fprintf(file, "%s", string_buffer);
    printf("%s", string_buffer);
    fclose(file);
}
