EXE=c_neural_network
CC=gcc
CFLAGS=--std=c99
SRC_DIR=src
OBJ_DIR=obj
TST_DIR=tests
BLD_DIR=build
OBJS = error.o matrix.o neural_network.o neural_network_file.o load_file.o random.o
TSTS = $(addprefix $(BLD_DIR)/$(TST_DIR)/, matrix_test neural_network_file_test neural_network_evaluate_test)
DEPS=$(addprefix $(OBJ_DIR)/, $(OBJS))

.PHONY: all prebuild build tests clean

all: prebuild build tests

prebuild:
	@mkdir -p $(OBJ_DIR) $(BLD_DIR) $(BLD_DIR)/$(TST_DIR)

tests: prebuild $(TSTS)

build: $(SRC_DIR)/main.c $(DEPS)
	$(CC) -o $(BLD_DIR)/$(EXE) $^ $(CFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/%.h
	$(CC) -c -o $@ $< $(CFLAGS)

$(BLD_DIR)/$(TST_DIR)/%: $(TST_DIR)/%.c $(DEPS)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f -r $(OBJ_DIR) $(BLD_DIR)