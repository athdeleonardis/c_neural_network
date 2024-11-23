add_executable(test_matrix $<TARGET_OBJECTS:c_neural_network> test_matrix.c)
add_executable(test_neural_network_evaluate $<TARGET_OBJECTS:c_neural_network> test_neural_network_evaluate.c)
add_executable(test_neural_network_file $<TARGET_OBJECTS:c_neural_network> test_neural_network_file.c)
add_executable(test_neural_network_train $<TARGET_OBJECTS:c_neural_network> test_neural_network_train.c)
