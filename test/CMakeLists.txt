set(TESTS test_matrix test_neural_network_evaluate test_neural_network_file test_neural_network_train)

foreach (T IN LISTS TESTS)
    add_executable(${T} ${T}.c)
    target_link_libraries(${T} c_neural_network_lib)
endforeach()
