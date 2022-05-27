#include "random.h"

#include <stdlib.h>
#include <time.h>

//
// 'random.h' Implementations
//

void random_init() {
    srand(time(NULL));
}

double random_double_between(double min, double max) {
    return min + ((double) rand()) / RAND_MAX * (max - min);
}

int random_int_between(int min_inclusive, int max_non_inclusive) {
    return min_inclusive + rand() % (max_non_inclusive - min_inclusive);
}
