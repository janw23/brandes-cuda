brandes: brandes.cu brandes_host.cuh brandes_device.cuh brandes_host.cu brandes_device.cuh errors.h
# TODO -DNDEBUG
# sm_61 is the architecture of my MX150 GPU
# -g and -G are for debugging
	nvcc -g -G -std=c++14 -arch sm_61 -o brandes brandes.cu brandes_host.cu brandes_device.cu

clean:
	rm -f brandes *.o