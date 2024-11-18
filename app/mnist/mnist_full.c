#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

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

#define TRAINING_PARAMETER_INITIAL 0.01
#define TRAINING_PARAMETER_FINAL 0.001

neural_network_t *neural_network_init();
double training_parameter_calc(double p_high, double p_low, int cases_correct, int total_cases);
void train_all_cases(neural_network_t *nn, mnist_handle_t *mh, matrix_t *outputs, double training_parameter);
int evaluate_all_cases(neural_network_t *nn, mnist_handle_t *mh, matrix_t *outputs_calculated);
void log_start(const char *filename);
void log_append(const char *filename, char *str);
void log_append_time(const char *filename, char *string_buffer, clock_t start, clock_t end);
void log_append_total_time(const char *filename, char *string_buffer, clock_t start, clock_t end);

//
// 'mnist_full.h' implementations
//

void mnist_full() {
    const char *log_file_name = "logs/mnist.txt";

    char string_buffer[64];
    double input_data[BATCH_COUNT * INPUT_SIZE];
    mnist_handle_t mnist_handle_training = mnist_handle_init(MNIST_N_CASES_TRAINING, input_data);
    mnist_handle_t mnist_handle_testing = mnist_handle_init(MNIST_N_CASES_TESTING, input_data);
    mnist_images_load(MNIST_DATASET_TRAINING_IMAGES, &mnist_handle_training);
    mnist_labels_load(MNIST_DATASET_TRAINING_LABELS, &mnist_handle_training);
    mnist_images_load(MNIST_DATASET_TESTING_IMAGES, &mnist_handle_testing);
    mnist_labels_load(MNIST_DATASET_TESTING_LABELS, &mnist_handle_testing);

    double outputs_calculated_data[BATCH_COUNT * OUTPUT_SIZE];
    matrix_t outputs_calculated[BATCH_COUNT];
    int outputs_calculated_data_offset = 0;
    for (int i = 0; i < BATCH_COUNT; i++) {
        matrix_initialize_from_array(outputs_calculated+i, 1, OUTPUT_SIZE, outputs_calculated_data, &outputs_calculated_data_offset);
    }

    double output_data[OUTPUT_DATA_SIZE];
    mnist_initialize_output_data(output_data);
    matrix_t outputs[OUTPUT_SIZE];
    mnist_initialize_outputs(outputs, output_data);

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

    log_start(log_file_name);

    int best_epoch = 0;
    int max_num_correct = 0;
    clock_t start_total;
    for (int i = 0; 1; i++) {
        sprintf(string_buffer, "-- Epoch %02d --\n", i);
        log_append(log_file_name, string_buffer);

        clock_t start = clock();
        if (i) {
            double training_parameter = training_parameter_calc(TRAINING_PARAMETER_INITIAL, TRAINING_PARAMETER_FINAL, max_num_correct, mnist_handle_testing.num_cases);
            train_all_cases(&neural_network, &mnist_handle_training, outputs, training_parameter);
            log_append(log_file_name, "Trained all cases.\n");
        }
        else {
            log_append(log_file_name, "No training.\n");
        }

        int training_cases_correct = evaluate_all_cases(&neural_network, &mnist_handle_training, outputs_calculated);
        sprintf(string_buffer, "Training dataset evaluation: %d / %d, %.01f%%\n", training_cases_correct, mnist_handle_training.num_cases, (double)100 * training_cases_correct / mnist_handle_training.num_cases);
        log_append(log_file_name, string_buffer);
        int testing_cases_correct = evaluate_all_cases(&neural_network, &mnist_handle_testing, outputs_calculated);
        sprintf(string_buffer, "Testing dataset evaluation: %d / %d, %.01f%%\n", testing_cases_correct, mnist_handle_testing.num_cases, (double)100 * testing_cases_correct / mnist_handle_testing.num_cases);
        log_append(log_file_name, string_buffer);
        clock_t end = clock();
        log_append_time(log_file_name, string_buffer, start, end);
        log_append_total_time(log_file_name, string_buffer, start_total, end);

        if (testing_cases_correct > max_num_correct) {
            sprintf(string_buffer, "New best epoch. Saving neural network.\n");
            log_append(log_file_name, string_buffer);
            max_num_correct = testing_cases_correct;
            best_epoch = i;
            neural_network_save_dynamic(&neural_network, "models/mnist.model.dynamic");
        }
        else {
            sprintf(string_buffer, "Epoch performed worst than last. Exiting.\n");
            log_append(log_file_name, string_buffer);
            break;
        }
    }

    mnist_handle_close(&mnist_handle_training);
    mnist_handle_close(&mnist_handle_testing);
}

/**
 * As more cases become correct, lower the training parameter.
*/
double training_parameter_calc(double p_high, double p_low, int cases_correct, int total_cases) {
    double proportion_correct = (double)cases_correct / total_cases;
    double lerp_factor = sqrt(proportion_correct);
    return p_low * lerp_factor + p_high * (1 - lerp_factor);
}

void train_all_cases(neural_network_t *nn, mnist_handle_t *mh, matrix_t *outputs, double training_parameter) {
    mnist_reset(mh);
    while (mnist_has_batch(mh)) {
        int num_cases = mnist_load_batch(mh);
        for (int i = 0; i < num_cases; i++) {
            unsigned char label = mh->output_data[i];
            matrix_t *output = &outputs[label];
            neural_network_train_case(nn, mh->inputs+i, output, training_parameter);
        }
        printf("Trained: %d / %d\r", mh->index, mh->num_cases);
        fflush(stdout);
    }
}

/**
 * @return The number of cases correctly classified.
*/
int evaluate_all_cases(neural_network_t *nn, mnist_handle_t *mh, matrix_t *outputs_calculated) {
    int num_cases_correct = 0;
    mnist_reset(mh);
    while (mnist_has_batch(mh)) {
        int num_cases = mnist_load_batch(mh);
        neural_network_evaluate(nn, num_cases, mh->inputs, outputs_calculated);
        for (int i = 0; i < num_cases; i++) {
            unsigned char label = mh->output_data[i];
            unsigned char label_calculated = mnist_output_to_number(outputs_calculated+i);
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

void log_append_time(const char *filename, char *string_buffer, clock_t start, clock_t end) {
    int milliseconds = (end - start) * 1000 / CLOCKS_PER_SEC;
    int seconds  = (milliseconds / 1000) % 60;
    int minutes = milliseconds / 1000 / 60;
    milliseconds = milliseconds % 1000;
    FILE *file = fopen(filename, "a");
    sprintf(string_buffer, "Epoch time taken: %dm %d.%ds\n", minutes, seconds, milliseconds);
    fprintf(file, "%s", string_buffer);
    printf("%s", string_buffer);
    fclose(file);
}

void log_append_total_time(const char *filename, char *string_buffer, clock_t start, clock_t end) {
    int milliseconds = (end - start) * 1000 / CLOCKS_PER_SEC;
    int seconds  = (milliseconds / 1000) % 60;
    int minutes = milliseconds / 1000 / 60;
    milliseconds = milliseconds % 1000;
    FILE *file = fopen(filename, "a");
    sprintf(string_buffer, "Total time taken: %dm %d.%ds\n", minutes, seconds, milliseconds);
    fprintf(file, "%s", string_buffer);
    printf("%s", string_buffer);
    fclose(file);
}
