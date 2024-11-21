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
#include "../../src/error.h"

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

#define BATCH_SIZE 100
#define TRAINING_PARAMETER_INITIAL 0.01
#define TRAINING_PARAMETER_FINAL 0.001

#define N_THREADS 4

typedef struct {
    neural_network_t *neural_network;
    mnist_handle_t *mnist_handle;
    matrix_t *output_map;
    pthread_mutex_t *mutex;
    double *inputs_data;
    matrix_t *inputs;
    unsigned char *outputs;
    neural_network_evaluation_t *evaluations;
} storage_t;

typedef struct {
    storage_t storage;
    int thread_num;
    int *num_cases_correct;
} evaluation_storage_t;

neural_network_t *neural_network_init();
double training_parameter_calc(double p_high, double p_low, int cases_correct, int total_cases);
void train_all_cases(neural_network_t *nn, mnist_handle_t *mh, storage_t storage, double training_parameter);
int evaluate_all_cases(storage_t storage);
/**
 * @param eval_storage_ptr Intended to be passed an 'evaluation_storage_t *'.
 */
void *evaluate_all_cases_thread(void *eval_storage_ptr);
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
    unsigned char input_data_unformated[INPUT_SIZE * BATCH_SIZE];
    mnist_handle_t mnist_handle_training = mnist_handle_init(MNIST_N_CASES_TRAINING, BATCH_SIZE, input_data_unformated);
    mnist_handle_t mnist_handle_testing = mnist_handle_init(MNIST_N_CASES_TESTING, BATCH_SIZE, input_data_unformated);
    mnist_images_load(MNIST_DATASET_TRAINING_IMAGES, &mnist_handle_training);
    mnist_labels_load(MNIST_DATASET_TRAINING_LABELS, &mnist_handle_training);
    mnist_images_load(MNIST_DATASET_TESTING_IMAGES, &mnist_handle_testing);
    mnist_labels_load(MNIST_DATASET_TESTING_LABELS, &mnist_handle_testing);

    // Storage space to send mnist_handle image data to.
    double inputs_data[INPUT_SIZE * BATCH_SIZE * N_THREADS];
    matrix_t inputs[BATCH_SIZE * N_THREADS];
    matrix_initialize_multiple_from_array(inputs, BATCH_SIZE * N_THREADS, 1, INPUT_SIZE, inputs_data);

    // A map from digit to neural network output.
    double output_map_data[OUTPUT_DATA_SIZE];
    mnist_initialize_output_data(output_map_data);
    matrix_t output_map[OUTPUT_SIZE];
    mnist_initialize_outputs(output_map, output_map_data);

    unsigned char outputs[BATCH_SIZE * N_THREADS];

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

    neural_network_evaluation_t evaluations[N_THREADS];
    for (int i = 0; i < N_THREADS; i++) {
        neural_network_evaluation_initialize(&neural_network, &evaluations[i]);
    }

    pthread_mutex_t mutex;
    if (pthread_mutex_init(&mutex, NULL))
        make_error("Failed to initialize mutex.\n");

    storage_t storage = {
        .neural_network=&neural_network,
        .mnist_handle=NULL,
        .output_map=output_map,
        .mutex=&mutex,
        .inputs_data=inputs_data,
        .inputs=inputs,
        .outputs=outputs,
        .evaluations=evaluations
    };

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
        storage.mnist_handle = &mnist_handle_training;
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
        int training_cases_correct = evaluate_all_cases(storage);
        sprintf(string_buffer, "Training dataset evaluation: %d / %d, %.01f%%\n", training_cases_correct, mnist_handle_training.num_cases, (double)100 * training_cases_correct / mnist_handle_training.num_cases);
        log_append(log_file_name, string_buffer);
        log_append_time(log_file_name, string_buffer, "Time taken", start, clock());

        // Test against testing data
        storage.mnist_handle = &mnist_handle_testing;
        start = clock();
        int testing_cases_correct = evaluate_all_cases(storage);
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

    for (int i = 0; i < N_THREADS; i++) {
        neural_network_evaluation_delete(evaluations[i]);
    }
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
            neural_network_evaluation_outputs(nn, storage.inputs+i, *storage.evaluations);
            neural_network_evaluation_errors(nn, output, *storage.evaluations);
            neural_network_evaluation_apply(nn, storage.inputs+i, *storage.evaluations, training_parameter);
        }
        printf("Trained: %d / %d\r", mh->index, mh->num_cases);
        fflush(stdout);
    }
}

/**
 * @return The number of cases correctly classified.
*/
int evaluate_all_cases(storage_t storage) {
    int num_cases_correct = 0;
    mnist_reset(storage.mnist_handle);

    pthread_t thread_ids[N_THREADS];
    evaluation_storage_t evaluation_storages[N_THREADS];
    int thread_num_correct[N_THREADS];
    for (int i = 0; i < N_THREADS; i++) {
        storage_t *thread_storage = &evaluation_storages[i].storage;
        thread_storage->neural_network=storage.neural_network;
        thread_storage->mnist_handle=storage.mnist_handle;
        thread_storage->output_map=storage.output_map;
        thread_storage->mutex=storage.mutex;
        thread_storage->inputs_data=storage.inputs_data + i*INPUT_SIZE*BATCH_SIZE;
        thread_storage->inputs=storage.inputs + i*BATCH_SIZE;
        thread_storage->outputs=storage.outputs + i*BATCH_SIZE;
        thread_storage->evaluations=storage.evaluations + i;
        evaluation_storages[i].thread_num = i;
        evaluation_storages[i].num_cases_correct = &thread_num_correct[i];
        if (pthread_create(&thread_ids[i], NULL, evaluate_all_cases_thread, &evaluation_storages[i]))
            make_error("Couldn't create thread.\n");
    }
    int num_threads_complete = 0;
    while (num_threads_complete < N_THREADS) {
        pthread_join(thread_ids[num_threads_complete], NULL);
        printf("Joined with thread %d      \n", num_threads_complete);
        num_cases_correct += thread_num_correct[num_threads_complete];
        num_threads_complete++;
    }
    printf("                              \r");
    return num_cases_correct;
}

void *evaluate_all_cases_thread(void *eval_storage_ptr) {
    evaluation_storage_t *eval_storage = (evaluation_storage_t *)(eval_storage_ptr);
    storage_t *storage = &eval_storage->storage;
    int batch_size;
    int *num_cases_correct = eval_storage->num_cases_correct;
    *num_cases_correct = 0;
    while (1) {
        pthread_mutex_lock(storage->mutex);
        batch_size = mnist_load_batch(storage->mnist_handle, storage->inputs_data, storage->outputs);
        pthread_mutex_unlock(storage->mutex);
        if (!batch_size)
            return NULL;
        for (int i = 0; i < batch_size; i++) {
            neural_network_evaluation_outputs(storage->neural_network, storage->inputs+i, *storage->evaluations);
            unsigned char label = storage->outputs[i];
            unsigned char label_calculated = mnist_output_to_number(&storage->evaluations->layers[storage->neural_network->hidden_layer_count].outputs);
            *num_cases_correct += label == label_calculated;
        }
        if (eval_storage->thread_num == 0) {
            printf("Tested: %d / %d\r", storage->mnist_handle->index, storage->mnist_handle->num_cases);
            fflush(stdout);
        }
    }
    return NULL;
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
