TARGET := lsqr
BUILD_DIR := ./build

# GPU code
SRC_GPU := ./gpu
NVCC := nvcc
NVCC_FLAGS = -std=c++17 -lcublas -lcusparse -lcusolver
NVCC_COMPUTE_FLAGS = -arch=sm_50
NVCC_OPT_FLAGS = -Xptxas -O3 -use_fast_math
NVCC_BE_QUIET = -Wno-deprecated-gpu-targets
NVCC_DEBUG_FLAGS = -G
SOURCES_CU =	$(SRC_GPU)/lsqr.cu \
				$(SRC_GPU)/matVec_gpu.cu \
				$(SRC_GPU)/matVec_cublas.cu \
				$(SRC_GPU)/MatrixCUSPARSE.cu \
				$(SRC_GPU)/MatrixCSR.cu \
				$(SRC_GPU)/utils.cu

OBJS_CU := $(SOURCES_CU:%=$(BUILD_DIR)/%.o)

# CPU code
SRC_CPU := ./cpu
CC = g++
CC_OPT_FLAGS = -O3
CC_DEBUG_FLAGS = -g 
SOURCES_CPP = 	$(SRC_CPU)/matrixBuilder.cpp \
				$(SRC_CPU)/lsqr.cpp \
				$(SRC_CPU)/matVec_cpu.cpp
OBJS_CPP := $(SOURCES_CPP:%=$(BUILD_DIR)/%.o)

# Debug
DEBUG_TARGET := lsqr_debug
DEBUG_BUILD_DIR := ./build_debug
DEBUG_OBJS_CU := $(SOURCES_CU:%=$(DEBUG_BUILD_DIR)/%.o)
DEBUG_OBJS_CPP := $(SOURCES_CPP:%=$(DEBUG_BUILD_DIR)/%.o)

# Main build
$(BUILD_DIR)/$(TARGET): $(OBJS_CPP) $(OBJS_CU)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_COMPUTE_FLAGS) $(NVCC_BE_QUIET) $(CC_OPT_FLAGS) $(NVCC_OPT_FLAGS) $(OBJS_CU) $(OBJS_CPP) -o $@

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
	rm input/*.txt output/*.txt

clean_debug:
	rm -r $(DEBUG_BUILD_DIR)

run: $(BUILD_DIR)/$(TARGET)
	$(BUILD_DIR)/$(TARGET)

format:
	clang-format -i --verbose --style={'ColumnLimit: 150'} $(SRC_GPU)/*.* $(SRC_CPU)/*.* test/*/*.cu*