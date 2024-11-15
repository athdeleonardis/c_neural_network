#include "file_load.h"

FILE *file_load(const char *filename) {
    FILE *file = fopen(filename, "rb");
    cnd_make_error(file == NULL, "File does not exist");
    return file;
}
