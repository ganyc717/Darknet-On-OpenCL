Darknet on OpenCL
==========

## Darknet
Darknet is an open source neural network framework written in C and CUDA.<br> 
It is fast, easy to install, and supports CPU and GPU computation.<br>
You can find the origin project [here](https://github.com/pjreddie/darknet).<br>
## Darknet on OpenCL
As the origin darknet project is written in CUDA, this project is to port<br>
the darknet to OpenCL. Also, darknet is assumed to run on Linux and used <br>
some POSIX C APIs, This project rewrite this with standard C++ libraries.<br>
So that it could also run on Windows.<br>
## Dependency
`OpenCL`<br>
Make sure you have OpenCL installed, and set environment variables OPENCL_SDK <br>
point to your OpenCL installed path.<br>
`clBLAS`<br>
clBLAS is equivalent to cuBLAS, you can find the it [here](https://github.com/clMathLibraries/clBLAS)<br>
If your platform is Windows on x64, I have already prepared the clBLAS <br>
binary library. You can find it here.
## Build
`Windows`<br>
This project is prepared with Visual Studio 2017, just open darknet_cl.sln<br>
and build it.<br>
`Linux`<br>
May provide Makefile later...<br>
## Usage
Once you compiled and generate darknet_cl.exe, it has the same usage as darknet,
you can find it [here](https://pjreddie.com/darknet/).<br>
## Attention
This project didn't build the DarkGo into the darknet_cl, maybe support it later.<br>
