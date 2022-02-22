# CUDA-Solar-System-Simulation

The program is run from a Python script. It uses a dynamic library from Linux, namely .so (shared object). The Makefile allows you to create this library. The project.cu file contains functions that use the GPU for calculations.
Python has a ctypes structure corresponding to the one in C++/CUDA to send data in the right form.

The data for the calculations was manually downloaded from https://ssd.jpl.nasa.gov/horizons/app.html#/ as of 29/12/2021.
