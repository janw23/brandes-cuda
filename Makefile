brandes: brandes.cu
	nvcc -o brandes brandes.cu

clean:
	rm -f brandes *.o