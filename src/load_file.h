#ifndef LOAD_FILE
#define LOAD_FILE

#include "error.h"

#include <stdio.h>

FILE *load_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    cnd_make_error(file == NULL, "File does not exist");
    return file;
}

#endif