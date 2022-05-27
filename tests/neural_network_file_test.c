#include "../src/neural_network.h"
#include "../src/neural_network_file.h"
#include "../src/random.h"

#include <stdio.h>

int main() {
    random_init();

    printf("Step 1: Create neural network\n");
    int hidden_layer_sizes[] = {8, 4};
    neural_network_t *nn1 = neural_network_create(2, 2, 2, hidden_layer_sizes);
    neural_network_layers_randomize(nn1);
    neural_network_print(nn1);

    printf("\nStep 2: Save neural network\n");
    neural_network_save_dynamic(nn1, "models/test.model.dynamic");

    printf("\nStep 3: Delete neural network\n");
    neural_network_delete(nn1);

    printf("\nStep 4: Load neural network\n");
    neural_network_t *nn2 = neural_network_load_dynamic("models/test.model.dynamic");
    neural_network_print(nn2);

    neural_network_delete(nn2);
}
