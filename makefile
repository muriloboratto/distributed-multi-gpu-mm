CC = mpic++
NVCC = nvcc

CUDA_DIR = $(CUDA_HOME)
NCCL_DIR = $(NCCL_HOME)

CUDA_LIB = -L$(CUDA_DIR)/lib64
CUDA_INC = -I$(CUDA_DIR)/include \
           -I$(CUDA_DIR)/samples/common/inc

NCCL_LIB = -L$(NCCL_DIR)/lib
NCCL_INC = -I$(NCCL_DIR)/include

C_FLAGS = -O2 -Wall -g
LD_LIBS = -lcudart -lnccl -lm -fopenmp

CUDA_FLAGS = -Wno-deprecated-gpu-targets
SIMD_FLAGS = -fno-tree-slp-vectorize -fno-tree-vect-loop-version

EXE = mmb
OBJ_C = $(EXE).o

CUFILE = mulmat_kernel
OBJ_CUDA = $(CUFILE).o

$(EXE): $(OBJ_C) $(OBJ_CUDA)
    $(CC) $(OBJ_C) $(OBJ_CUDA) -o $(EXE) \
        $(CUDA_LIB) $(NCCL_LIB) $(LD_LIBS)

$(OBJ_C): $(EXE).c
    $(CC) $(C_FLAGS) $(SIMD_FLAGS) \
        $(CUDA_INC) $(NCCL_INC) \
        -c $< -fopenmp

$(OBJ_CUDA): $(CUFILE).cu
    $(NVCC) $(CUDA_FLAGS) $(CUDA_INC) $(NCCL_INC) -c $<

clean:
    rm -f *.o $(EXE)