ifeq ($(dbg), 1)
	DBGFLAGS=-g -G
endif

# # sm_61 is the architecture of my MX150 GPU
brandes: brandes.cu utils.cuh utils.cu kernels.cuh kernels.cu errors.h
	nvcc $(DBGFLAGS) -std=c++14 -arch sm_61 -o brandes brandes.cu utils.cu kernels.cu

clean:
	rm -f brandes *.o