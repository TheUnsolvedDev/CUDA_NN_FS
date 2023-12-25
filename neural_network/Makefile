CC := gcc
NVCC := nvcc
CFLAGS := -Wall -Wextra
CUDAFLAGS := -arch=sm_75 #-lcudart

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

SRC := $(wildcard $(SRC_DIR)/*.c)
CUDA_SRC := $(wildcard $(SRC_DIR)/*.cu)
OBJ := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC))
CUDA_OBJ := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.cu.o, $(CUDA_SRC))
TARGET := $(BIN_DIR)/program

DEPS := $(wildcard $(SRC_DIR)/*.h)
CUDA_DEPS := $(wildcard $(SRC_DIR)/*.cuh)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ) $(CUDA_OBJ)
	$(NVCC) $(CUDAFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(DEPS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu $(CUDA_DEPS)
	@mkdir -p $(@D)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)/*.o $(OBJ_DIR)/*.cu.o $(TARGET)
