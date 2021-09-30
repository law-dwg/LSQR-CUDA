TARGET := lsqr
BUILD_DIR := ./build
SRC_GPU := ./gpu
SRC_CPU := ./cpu
NVCC := nvcc
CC = g++
NVCC_FLAGS = -std=c++17 -arch=sm_50 -lcublas
NVCC_OPT_FLAGS = -Xptxas -O3
CC_OPT_FLAGS = -O3
NVCC_DEBUG_FLAGS = -g -G
L_FLAGS = -lm
SOURCES_CU =	$(SRC_GPU)/lsqr_gpu.cu \
				$(SRC_GPU)/matVec_gpu.cu \
				lsqr.cu

SOURCES_CPP = 	$(SRC_CPU)/lsqr_cpu.cpp \
				$(SRC_CPU)/matVec_cpu.cpp \
				matrixBuilder.cpp
OBJS_CU := $(SOURCES_CU:%=$(BUILD_DIR)/%.o)
OBJS_CPP := $(SOURCES_CPP:%=$(BUILD_DIR)/%.o)


$(BUILD_DIR)/$(TARGET): $(OBJS_CPP) $(OBJS_CU)
 
$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OPT_FLAGS) -dc $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(NVCC) $(CC_OPT_FLAGS) -c $< -o $@

.PHONY: clean
clean:
    rm -r $(BUILD_DIR)
