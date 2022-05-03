brandes: brandes.cu
# TODO -DNDEBUG
	nvcc -std=c++14 -o brandes brandes.cu

clean:
	rm -f brandes *.o