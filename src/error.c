#include "error.h"

#include <stdio.h>
#include <stdlib.h>

void make_error(const char *message) {
    printf("Error: %s\n", message);
    exit(EXIT_FAILURE);
}

void cnd_make_error(int check, const char *message) {
    if (check) {
        printf("Error: %s\n", message);
        exit(EXIT_FAILURE);
    }
}