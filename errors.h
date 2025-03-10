// This file is from the source provided by the university.

#ifndef __ERRORS_H__
#define __ERRORS_H__
#include <stdio.h>

static void HandleError(cudaError_t err,
						const char *file,
						int line) {
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString(err), 
				file, line );
		exit( EXIT_FAILURE );
	}
}

#define HANDLE_ERROR( err) (HandleError( err, __FILE__, __LINE__ ))



#endif // __ERRORS_H__
