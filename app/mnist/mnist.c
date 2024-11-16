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
#include "../../src/file_load.h"
#include "../../src/matrix.h"
#include "../../src/neural_network.h"
#include "mnist.h"

#define MAGIC_NUMBER_1 2049
#define MAGIC_NUMBER_2 2051
#define TEST_DATA_COUNT 10000

#define READ_BUFFER_SIZE 128
#define PIXEL_MAX 255.0

//
// 'mnist.c' definitions
//

int32_t flip_endianness(int32_t value);

//
// 'mnist.c' implementations
//

int32_t flip_endianness(int32_t value)
{
    return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) | ((value & 0x00FF0000) >> 8) | ((value & 0xFF000000) >> 24); 
} 

//
// 'mnist.h' implementations
//

mnist_handle_t mnist_handle_init(int32_t num_cases) {
    mnist_handle_t handle = {};
    handle.num_cases = num_cases;
    for (int i = 0; i < BATCH_COUNT; i++) {
        handle.inputs[i].cols = 1;
        handle.inputs[i].rows = INPUT_SIZE;
        handle.inputs[i].data = handle.input_data + INPUT_SIZE * i;
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

int mnist_has_batch(mnist_handle_t *handle) {
    return handle->index < handle->num_cases;
}

/**
 * Returns the number of cases in the loaded batch.
*/
int mnist_load_batch(mnist_handle_t *handle) {
    // How many cases to read.
    int num_cases = handle->num_cases - handle->index;
    if (num_cases > BATCH_COUNT)
        num_cases = BATCH_COUNT;
    handle->index += num_cases;

    fread(handle->input_data, sizeof(unsigned char), BATCH_INPUT_SIZE, handle->inputs_file);
    fread(handle->output_data, sizeof(unsigned char), BATCH_COUNT, handle->outputs_file);

    for (int i = 0; i < num_cases; i++) {
        handle->input_data[i] = handle->input_data_unformatted[i] / PIXEL_MAX;
    }
    return num_cases;
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

/**
 * Find the index of the inputted array which has the highest value.
*/
unsigned char output_to_number(matrix_t *output) {
    unsigned char number = 0;
    double highest_value = output->data[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        double new_value = output->data[i];
        if (new_value > highest_value) {
            highest_value = new_value;
            number = i;
        }
    }
    return number;
}
