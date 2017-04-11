#!/bin/bash

rm -f *.o *.obj *.mexw64 *.mexmaci64

INC='-I/Applications/MATLAB_R2016b.app//toolbox/vision/builtins/src/ocvcg/opencv/include/'
LIB='-L/Applications/MATLAB_R2016b.app/bin/maci64/ -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab'

mex im2colstep.c
mex $INC -c MxArray.cpp
mex $INC -c calcIIF.cpp
mex $INC -c resize.cpp
mex $INC $LIB calcIIF.o MxArray.o
mex $INC $LIB resize.o MxArray.o
