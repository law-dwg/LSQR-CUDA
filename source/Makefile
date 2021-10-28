TARGET := lsqr
BUILD_DIR := ./build

# Main
SRC_MAIN := ./gpu/main.cu
OBJ_MAIN := $(BUILD_DIR)/$(SRC_MAIN).o

# GPU code
SRC_GPU := ./gpu
NVCC := nvcc
NVCC_FLAGS = -std=c++17 -lcublas -lcusparse -lcusolver
NVCC_COMPUTE_FLAGS = -arch=sm_50
NVCC_OPT_FLAGS = -Xptxas -O3 -use_fast_math
NVCC_BE_QUIET = -Wno-deprecated-gpu-targets
NVCC_DEBUG_FLAGS = -G
SOURCES_CU =	$(SRC_GPU)/vectorCUDA.cu \
				$(SRC_GPU)/vectorCUBLAS.cu \
				$(SRC_GPU)/matrixCUSPARSE.cu \
				$(SRC_GPU)/matrixCUDA.cu \
				$(SRC_GPU)/kernels.cu \
				$(SRC_GPU)/utils.cu

OBJS_CU := $(SOURCES_CU:%=$(BUILD_DIR)/%.o)

# CPU code
SRC_CPU := ./cpu
CC = g++
CC_OPT_FLAGS = -O3
CC_DEBUG_FLAGS = -g 
SOURCES_CPP = 	$(SRC_CPU)/utils.cpp \
				$(SRC_CPU)/lsqr.cpp \
				$(SRC_CPU)/vectorCPU.cpp
OBJS_CPP := $(SOURCES_CPP:%=$(BUILD_DIR)/%.o)

# Debug
DEBUG_TARGET := lsqr_debug
DEBUG_BUILD_DIR := ./build_debug
DEBUG_OBJS_CU := $(SOURCES_CU:%=$(DEBUG_BUILD_DIR)/%.o)
DEBUG_OBJS_CPP := $(SOURCES_CPP:%=$(DEBUG_BUILD_DIR)/%.o)

# Main build
$(BUILD_DIR)/$(TARGET): $(OBJS_CPP) $(OBJS_CU) $(OBJ_MAIN)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_COMPUTE_FLAGS) $(NVCC_BE_QUIET) $(CC_OPT_FLAGS) $(NVCC_OPT_FLAGS) $(OBJ_MAIN) $(OBJS_CU) $(OBJS_CPP) -o $@

$(OBJ_MAIN): $(SRC_MAIN)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_COMPUTE_FLAGS) $(NVCC_BE_QUIET) $(NVCC_OPT_FLAGS) -dc $< -o $@

$(BUILD_DIR)/%.cu.o: %.cu %.cuh
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_COMPUTE_FLAGS) $(NVCC_BE_QUIET) $(NVCC_OPT_FLAGS) -dc $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp %.hpp
	mkdir -p $(dir $@)
	$(CC) $(CC_OPT_FLAGS) -c $< -o $@

# PHONYS
.PHONY: debug all clean clean_mats run $(BUILD_DIR)/$(TARGET) format

debug: $(DEBUG_BUILD_DIR)/$(DEBUG_TARGET)
$(DEBUG_BUILD_DIR)/$(DEBUG_TARGET): $(DEBUG_OBJS_CPP) $(DEBUG_OBJS_CU)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_COMPUTE_FLAGS) $(NVCC_DEBUG_FLAGS) -g $(OBJS_CPP) $(OBJS_CU) -o $@

$(DEBUG_BUILD_DIR)/%.cu.o: %.cu %.cuh
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_COMPUTE_FLAGS) $(NVCC_DEBUG_FLAGS) -dc $< -o $@

$(DEBUG_BUILD_DIR)/%.cpp.o: %.cpp %.hpp
	mkdir -p $(dir $@)
	$(CC) $(CC_DEBUG_FLAGS) -c $< -o $@

all: format clean clean_mats $(BUILD_DIR)/$(TARGET) run

clean:
	rm -r $(BUILD_DIR)

clean_mats:
	rm input/*.mat input/*.vec output/*.vec

clean_debug:
	rm -r $(DEBUG_BUILD_DIR)

run: $(BUILD_DIR)/$(TARGET)
	$(BUILD_DIR)/$(TARGET)

format:
	clang-format -i --verbose --style={'ColumnLimit: 150'} $(SRC_GPU)/*.* $(SRC_CPU)/*.* test/*/*.cu*