/**
 * This is an application that does the following:
 * 1. Loads the mnist training and testing datasets.
 * 2. Trains a randomly generated neural network on the mnist training dataset.
 * 3. Writes the evaluation of the neural network on the training and testing datasets to a file.
 * 4. Saves the neural network to a file.
*/

/**
 * Read about the mnist dataset and it's format here:
 * https://yann.lecun.com/exdb/mnist/
*/

#include <stdint.h>
#include <stdio.h>
#include "../src/file_load.h"
#include "../src/error.h"
#include "../src/matrix.h"

#define MAGIC_NUMBER_1 2049
#define MAGIC_NUMBER_2 2051
#define TRAINING_DATA_COUNT 60000
#define TEST_DATA_COUNT 10000
#define IMAGE_WIDTH 28

#define INPUT_SIZE IMAGE_WIDTH * IMAGE_WIDTH
#define BATCH_COUNT 100
#define BATCH_INPUT_SIZE BATCH_COUNT * INPUT_SIZE
#define OUTPUT_DATA_SIZE 10 * 10
#define READ_BUFFER_SIZE 128
#define PIXEL_MAX 255.0

typedef struct {
    FILE *inputs_file;
    FILE *outputs_file;
    int32_t num_cases;
    int32_t index;
    unsigned char input_data_unformatted[BATCH_INPUT_SIZE];
    double input_data[BATCH_INPUT_SIZE];
    matrix_t inputs[BATCH_COUNT];
    unsigned char output_data[BATCH_COUNT];
} mnist_handle_t;

int32_t flip_endianness(int32_t value);
mnist_handle_t mnist_handle_init(int32_t num_cases);
void mnist_handle_close(mnist_handle_t *handle);
void mnist_images_load(const char *filename, mnist_handle_t *handle);
void mnist_labels_load(const char *filename, mnist_handle_t *handle);
void mnist_load_batch(mnist_handle_t *handle);
void initialize_output_data(double *data);
void initialize_outputs(matrix_t *outputs, double *data);


int main(int argc, char *argv) {
    mnist_handle_t mnist_handle = mnist_handle_init(TRAINING_DATA_COUNT);
    mnist_images_load("datasets/mnist/train-images.idx3-ubyte", &mnist_handle);
    mnist_labels_load("datasets/mnist/train-labels.idx1-ubyte", &mnist_handle);

    double output_data[OUTPUT_DATA_SIZE];
    initialize_output_data(output_data);
    matrix_t outputs[10];
    initialize_outputs(outputs, output_data);

    mnist_load_batch(&mnist_handle);

    for (int i = 0; i < BATCH_COUNT; i++) {
        printf("%d ", mnist_handle.output_data[i]);
    }
    printf("\n");

    mnist_handle_close(&mnist_handle);
}

int32_t flip_endianness(int32_t value)
{
    return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) | ((value & 0x00FF0000) >> 8) | ((value & 0xFF000000) >> 24); 
} 

mnist_handle_t mnist_handle_init(int32_t num_cases) {
    mnist_handle_t handle = {};
    handle.num_cases = num_cases;
    for (int i = 0; i < BATCH_COUNT; i++) {
        handle.inputs[i].cols = 1;
        handle.inputs[i].rows = 10;
        handle.inputs[i].data = handle.input_data + 10 * i;
    }
    return handle;
}

void mnist_handle_close(mnist_handle_t *handle) {
    fclose(handle->inputs_file);
    fclose(handle->outputs_file);
    handle->inputs_file = NULL;
    handle->outputs_file = NULL;
}

void mnist_images_load(const char *filename, mnist_handle_t *handle) {
    handle->inputs_file = file_load(filename);

    // Checks
    {
        int32_t file_magic_number;
        fread(&file_magic_number, sizeof(int32_t), 1, handle->inputs_file);
        file_magic_number = flip_endianness(file_magic_number);
        cnd_make_error(file_magic_number != MAGIC_NUMBER_2, "Magic number does not match.\n");
    }
    {
        int32_t file_num_cases;
        fread(&file_num_cases, sizeof(int32_t), 1, handle->inputs_file);
        file_num_cases = flip_endianness(file_num_cases);
        cnd_make_error(file_num_cases != handle->num_cases, "Number of cases does not match.\n");
    }
    for (int i = 0; i < 2; i++) {
        int32_t file_image_width;
        fread(&file_image_width, sizeof(int32_t), 1, handle->inputs_file);
        file_image_width = flip_endianness(file_image_width);
        cnd_make_error(file_image_width != IMAGE_WIDTH, "File's listed image width does not match.\n");
    }
}

void mnist_labels_load(const char *filename, mnist_handle_t *handle) {
    handle->outputs_file = file_load(filename);

    // Checks
    {
        int32_t file_magic_number;
        fread(&file_magic_number, sizeof(int32_t), 1, handle->outputs_file);
        file_magic_number = flip_endianness(file_magic_number);
        cnd_make_error(file_magic_number != MAGIC_NUMBER_1, "Magic number does not match.\n");
    }
    {
        int32_t file_num_cases;
        fread(&file_num_cases, sizeof(int32_t), 1, handle->outputs_file);
        file_num_cases = flip_endianness(file_num_cases);
        cnd_make_error(file_num_cases != handle->num_cases, "Number of cases does not match.\n");
    }
}

void mnist_load_batch(mnist_handle_t *handle) {
    // How many cases to read.
    int num_cases = handle->num_cases - handle->index;
    if (num_cases > BATCH_COUNT)
        num_cases = BATCH_COUNT;
    handle->index += num_cases;

    fread(handle->input_data, sizeof(unsigned char), BATCH_INPUT_SIZE, handle->inputs_file);
    fread(handle->output_data, sizeof(unsigned char), BATCH_COUNT, handle->outputs_file);

    for (int i = 0; i < BATCH_INPUT_SIZE; i++) {
        handle->input_data[i] = handle->input_data_unformatted[i] / PIXEL_MAX;
    }
}

void initialize_output_data(double *data) {
    // E.g. label 0 is represented by [1 0 0 0 0 0 0 0 0 0]
    // E.g. label 3 is represented by [0 0 0 1 0 0 0 0 0 0]
    for (int i = 0; i < OUTPUT_DATA_SIZE; i++) {
        data[i] = 0;
        if (i % 10 == (i / 10) % 10)
            data[i] = 1;
    }
}

void initialize_outputs(matrix_t *outputs, double *data) {
    for (int i = 0; i < 10; i++) {
        outputs[i].cols = 1;
        outputs[i].rows = 10;
        outputs[i].data = data + 10 * i;
    }
}
