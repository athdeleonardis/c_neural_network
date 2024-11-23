add_executable(mnist $<TARGET_OBJECTS:c_neural_network> main.c mnist_full.c mnist_test.c mnist_train.c mnist.c thread_wrapper.c)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(mnist PRIVATE Threads::Threads)
