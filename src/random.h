#ifndef RANDOM
#define RANDOM

//
// 'random.h' Definitions
//

void random_init();
void random_init_seeded(int seed);
double random_double_between(double min, double max);
int random_int_between(int min, int max);

#endif
