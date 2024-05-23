# CustomOP Compilation

This is an example for compiling the custom rms norm op, this should be used when we see the deviation in outputs or out of range fp16 values due to rms norm kernel. This will handle the out of range issues without affecting accuracy and perf.

Note: Minimum CMAKE version is 3. (g++ compiler and c++11)

## Steps to compile the custom op
**Note**: Provide specific SRC_FILE path accordingly, below example is for custom rms norm
```
mkdir build && cd build
cmake .. -D CMAKE_CXX_FLAGS="-Wall" -D SRC_FILE=../CustomRMSNorm/src/customrmsnorm_functions.cpp -D CUSTOM_LIB=customrmsnorm_lib
make all
cd ..
```

## Move the custom op shared library to the customop src directory
**Note**: Provide specific build so path accordingly, below example is for custom rms norm
```
mv build/libcustomrmsnorm_lib.so CustomRMSNorm/src/
```