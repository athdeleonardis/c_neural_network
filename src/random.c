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
