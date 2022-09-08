## Compiling

For prime pXX
```bash

cd pXX

mkdir build

cd build

cmake ..

make clean

make config

make
```
Before compiling, the GPU's architecture needs to be specified by setting ```CMAKE_CUDA_ARCHITECTURES``` in the ```CMakeLists.txt```.  
  

## Implemented algorithms

- vow-gcs: van Oorschot & Wiener (vOW) Golden Collision Search (GCS)

  

## Requirements

- CUDA Toolkit
- cmake version >= 3.8
- python3

  

## Execution

```bash
./vowgcs
```
