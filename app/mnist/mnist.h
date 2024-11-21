#include <stdio.h>
#include <stdint.h>
#include "../../src/matrix.h"

//
// 'mnist.h' definitions
//

#define MNIST_N_CASES_TRAINING 60000
#define MNIST_N_CASES_TESTING 10000
#define MNIST_DATASET_TRAINING_IMAGES "datasets/mnist/train-images.idx3-ubyte"
#define MNIST_DATASET_TRAINING_LABELS "datasets/mnist/train-labels.idx1-ubyte"
#define MNIST_DATASET_TESTING_IMAGES "datasets/mnist/t10k-images.idx3-ubyte"
#define MNIST_DATASET_TESTING_LABELS "datasets/mnist/t10k-labels.idx1-ubyte"

#define IMAGE_WIDTH 28
#define INPUT_SIZE IMAGE_WIDTH * IMAGE_WIDTH
#define OUTPUT_SIZE 10
#define OUTPUT_DATA_SIZE OUTPUT_SIZE * OUTPUT_SIZE

typedef struct {
    FILE *inputs_file;
    FILE *outputs_file;
    int32_t index;
    int32_t num_cases;
    unsigned char *input_data_buffer;
    int batch_size;
} mnist_handle_t;

mnist_handle_t mnist_handle_init(int32_t num_cases, int batch_size, unsigned char *input_data_buffer);
void mnist_handle_close(mnist_handle_t *handle);
void mnist_images_load(const char *filename, mnist_handle_t *handle);
void mnist_labels_load(const char *filename, mnist_handle_t *handle);
int mnist_load_batch(mnist_handle_t *handle, double *inputs, unsigned char *outputs);
void mnist_reset(mnist_handle_t *handle);
void mnist_initialize_output_data(double *data);
void mnist_initialize_outputs(matrix_t *outputs, double *data);
unsigned char mnist_output_to_number(matrix_t *output);
