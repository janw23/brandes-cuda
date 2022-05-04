brandes: brandes.cu brandes_host.cuh brandes_device.cuh brandes_host.cu brandes_device.cuh errors.h
# TODO -DNDEBUG
	nvcc -std=c++14 -o brandes brandes.cu brandes_host.cu brandes_device.cu

clean:
	rm -f brandes *.o