add_library(c_neural_network_lib STATIC activation_function.c error.c file_load.c matrix.c neural_network_file.c neural_network_train.c neural_network.c random.c)
target_link_libraries(c_neural_network_lib m)
