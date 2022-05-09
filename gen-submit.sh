rm -rf submit
mkdir submit

for f in brandes.cu utils.cuh utils.cu kernels.cuh kernels.cu errors.h Makefile ../tests/gowalla.txt
  do cp -r $f submit/
done