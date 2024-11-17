#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "mnist_train.h"
#include "mnist_test.h"
#include "mnist_full.h"
#include "../../src/random.h"
#include "../../src/error.h"

#define MODE_TRAIN 1
#define MODE_TEST 2
#define MODE_FULL 3

typedef struct {
    int mode;
    const char *model_filename;
    int epochs;
    int do_overwrite;
} cmd_args_t;

void read_args(cmd_args_t *cmd_args, int argc, char *argv[], int *argi);
void check_args(cmd_args_t cmd_args);
int arg_matches(const char *arg, const char *arg1, const char *arg2);

int main(int argc, char *argv[]) {
    cmd_args_t cmd_args = {};
    cmd_args.epochs = 1;
    int argi = 1;
    while (argi < argc) {
        read_args(&cmd_args, argc, argv, &argi);
    }
    check_args(cmd_args);

    random_init();

    switch (cmd_args.mode) {
        case MODE_TRAIN: {
            mnist_train(cmd_args.model_filename, cmd_args.epochs, cmd_args.do_overwrite);
            return 0;
        }
        case MODE_TEST: {
            cnd_make_error(cmd_args.model_filename == NULL, "Model file not specified. Use '--help' for more information.\n");
            mnist_test(cmd_args.model_filename);
            return 0;
        }
        case MODE_FULL: {
            mnist_full();
            return 0;
        }
    }
}

void read_args(cmd_args_t *cmd_args, int argc, char *argv[], int *argi) {
    const char *arg = argv[*argi];
    *argi += 1;
    if (arg_matches(arg, "--help", "-h")) {
        printf("Available commands:\n--help | -h : Display all valid commands, or help information on used commands.\n--mode | -m : Always required. Set the mode to either 'train', 'test' or 'full'.\n--load-file | -l : Required for mode 'test'. Load a neural network from a dynamic model file.\n--epochs | -i : The number of times all test cases are iterated over in training. Default value is 1.\n--overwrite | -o : During training, saving the neural network after each iteration overwrites the previous save.\n");
        exit(EXIT_SUCCESS);
        return;
    }
    if (arg_matches(arg, "--mode", "-h")) {
        cnd_make_error(cmd_args->mode, "Mode already chosen.\n");
        cnd_make_error(*argi == argc, "Expected another argument. Use '--mode --help' to find out more.\n");
        arg = argv[*argi];
        *argi += 1;
        if (arg_matches(arg, "--help", "-h")) {
            printf("Available modes: 'train', 'test', 'full'.\nExample usage: --mode train --load-file models/example.model.dynamic\n");
            exit(EXIT_SUCCESS);
            return;
        }
        if (strcmp(arg, "train") == 0) {
            cmd_args->mode = MODE_TRAIN;
            return;
        }
        if (strcmp(arg, "test") == 0) {
            cmd_args->mode = MODE_TEST;
            return;
        }
        if (strcmp(arg, "full") == 0) {
            cmd_args->mode = MODE_FULL;
            return;
        }
        make_error("Invalid mode selected. Use '--mode --help' to see valid arguments.\n");
    }
    if (arg_matches(arg, "--load-file", "-l")) {
        cnd_make_error(cmd_args->model_filename != NULL, "Model filename already specified.\n");
        cnd_make_error(*argi == argc, "Expected another argument. Use '--load-file --help' to find out more.\n");
        arg = argv[*argi];
        *argi += 1;
        if (arg_matches(arg, "--help", "-h")) {
            printf("Input the filename of your model.\nExample usage: --file-name mnist.model.dynamic\n");
            exit(EXIT_SUCCESS);
            return;
        }
        cmd_args->model_filename = arg;
        return;
    }
    if (arg_matches(arg, "--epochs", "-e")) {
        cnd_make_error(*argi == argc, "Expected another argument. Use '--epochs --help' to find out more.\n");
        arg = argv[*argi];
        *argi += 1;
        if (arg_matches(arg, "--help", "-h")) {
            printf("The number of times each test case is iterated over during training.\nExample use: --mode train --iterations 5\n");
        }
        cmd_args->epochs = atoi(arg);
        cnd_make_error(cmd_args->epochs <= 0, "Inputted string for iterations is not a valid number.\n");
        return;
    }
    if (arg_matches(arg, "--overwrite", "-o")) {
        cmd_args->do_overwrite = 1;
        return;
    }
    printf("Argument not recognized: '%s'.\nUse '--help' for a list of all valid arguments.\n", arg);
    exit(EXIT_FAILURE);
}

void check_args(cmd_args_t cmd_args) {
    cnd_make_error(cmd_args.mode == 0, "Mode not selected. Use '--help' to print a list of all valid arguments.\n");
}

int arg_matches(const char *arg, const char *arg1, const char *arg2) {
    return strcmp(arg, arg1) == 0 || strcmp(arg, arg2) == 0;
}
