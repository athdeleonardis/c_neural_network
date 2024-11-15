#include "random.h"

#include <stdlib.h>
#include <time.h>

//
// 'random.h' Implementations
//

void random_init() {
    srand(time(NULL));
}

void random_init_seeded(int seed) {
    srand(seed);
}

double random_double_between(double min, double max) {
    return min + ((double) rand()) / RAND_MAX * (max - min);
}

int random_int_between(int min_inclusive, int max_non_inclusive) {
    return min_inclusive + rand() % (max_non_inclusive - min_inclusive);
}
