#ifndef NEURAL_NETWORK_FILE
#define NEURAL_NETWORK_FILE

#include "neural_network.h"

neural_network_t *neural_network_load_static(const char *filename);
neural_network_t *neural_network_load_dynamic(const char *filename);
void neural_network_save_static(neural_network_t *, const char *filename);
void neural_network_save_dynamic(neural_network_t *, const char *filename);

#endif
