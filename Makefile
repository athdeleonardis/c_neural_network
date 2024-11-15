EXE=c_neural_network
CC=gcc
CFLAGS=--std=c99 -lm
SRC_DIR=src
OBJ_DIR=obj
TST_DIR=test
BLD_DIR=build
OBJS = random.o error.o file_load.o matrix.o neural_network.o neural_network_file.o neural_network_train.o activation_function.o
TST_NAMES = matrix neural_network_file neural_network_evaluate neural_network_train
TSTS=$(addprefix $(BLD_DIR)/$(TST_DIR)/test_, $(TST_NAMES))
DEPS=$(addprefix $(OBJ_DIR)/, $(OBJS))

.PHONY: all prebuild test clean

all: prebuild test

prebuild:
	@mkdir -p $(OBJ_DIR) $(BLD_DIR) $(BLD_DIR)/$(TST_DIR)

test: prebuild $(TSTS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/%.h
	$(CC) -c -o $@ $< $(CFLAGS)

$(BLD_DIR)/$(TST_DIR)/%: $(TST_DIR)/%.c $(DEPS)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f -r $(OBJ_DIR) $(BLD_DIR)
