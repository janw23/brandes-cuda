rm -r submit
mkdir submit

for f in brandes.cu brandes_device.cu brandes_device.cuh brandes_host.cu brandes_host.cuh errors.h Makefile ../tests/gowalla.txt
  do cp -r $f submit/
done