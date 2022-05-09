ifeq ($(dbg), 1)
	DBGFLAGS=-g -G
endif

# brandes: brandes.cu brandes_host.cuh brandes_device.cuh brandes_host.cu brandes_device.cu errors.h
# # TODO -DNDEBUG
# # sm_61 is the architecture of my MX150 GPU
# 	nvcc $(DBGFLAGS) -std=c++14 -arch sm_61 -o brandes brandes.cu brandes_host.cu brandes_device.cu

newbrandes: brandes.cu utils.cuh utils.cu kernels.cuh kernels.cu errors.h
	nvcc $(DBGFLAGS) -std=c++14 -arch sm_61 -o brandes brandes.cu utils.cu kernels.cu

clean:
	rm -f brandes *.o