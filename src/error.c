#include "error.h"

#include <stdio.h>
#include <stdlib.h>

//
// 'error.h' implementations
//

void make_error(const char *message) {
    printf("Error: %s\n", message);
    exit(EXIT_FAILURE);
}

/**
 * Conditionally throw an error if 'check' is not zero.
 * @param check The boolean value. If this is not zero, an error will be thrown.
 * @param message The message to be printed if the check fails.
*/
void cnd_make_error(int check, const char *message) {
    if (check) {
        make_error(message);
    }
}
