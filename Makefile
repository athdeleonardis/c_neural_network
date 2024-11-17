EXE=c_neural_network
CC=gcc
CFLAGS=--std=c99 -lm
SRC_DIR=src
OBJ_DIR=obj
TST_DIR=test
BLD_DIR=build
APP_DIR=app
OBJS = random.o error.o file_load.o matrix.o neural_network.o neural_network_file.o neural_network_train.o activation_function.o
TST_NAMES = matrix neural_network_file neural_network_evaluate neural_network_train
APP_NAMES = mnist
DEPS=$(addprefix $(OBJ_DIR)/, $(OBJS))
TSTS=$(addprefix $(BLD_DIR)/$(TST_DIR)/test_, $(TST_NAMES))
APPS=$(addprefix $(BLD_DIR)/$(APP_DIR)/, $(APP_NAMES))
MNIST_DEPS=$(addprefix $(APP_DIR)/mnist/, main.c mnist.c mnist.h mnist_train.c mnist_train.h mnist_test.c mnist_test.h)

.PHONY: all prebuild obj test clean app

all: prebuild obj test app

prebuild:
	@mkdir -p $(OBJ_DIR) $(BLD_DIR) $(BLD_DIR)/$(TST_DIR) $(BLD_DIR)/$(APP_DIR)

obj: prebuild $(DEPS)

test: prebuild $(TSTS)

app: prebuild $(APPS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/%.h
	$(CC) -c -o $@ $< $(CFLAGS)

$(BLD_DIR)/$(TST_DIR)/%: $(TST_DIR)/%.c $(DEPS)
	$(CC) -o $@ $^ $(CFLAGS)

$(BLD_DIR)/$(APP_DIR)/mnist: $(MNIST_DEPS) $(DEPS)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f -r $(OBJ_DIR) $(BLD_DIR)
