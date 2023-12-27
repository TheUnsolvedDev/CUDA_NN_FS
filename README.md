# Neural Network Implementation in CUDA and C

This repository contains an implementation of a neural network from scratch using CUDA and C. The goal of this project is to leverage the parallel computing capabilities of CUDA to accelerate the training and inference processes of a neural network.

## Introduction
Neural networks are powerful models for machine learning and artificial intelligence tasks. This project focuses on building a basic neural network using the C programming language with CUDA extensions for parallelism. CUDA allows us to harness the computational power of NVIDIA GPUs, speeding up the training and inference phases of the neural network.

For running this make sure GCC and NVCC installed in your system while doing it my GCC version is 11.4.0 and NVCC version is 11.8. **USE THE RUN FILE FOR INSTALLING AND ADD THE PATH TO THE BASHRC FILE**

## Prerequisites
To run this project, you need the following dependencies:
- CUDA-enabled GPU
- CUDA Toolkit installed
- C Compiler (e.g., GCC)

## Getting Started
1. Clone the repository: `https://github.com/TheUnsolvedDev/CUDA_NN_FS`
2. Navigate to the project directory: `cd neural_network`
3. Build the project: `make`

## Project Structure
```bash
neural_network
├── bin
│   └── program
├── document.md
├── generate.py
├── include
│   ├── activations.cuh
│   ├── benchmark.h
│   ├── initializers.h
│   ├── losses.cuh
│   ├── operations.cuh
│   ├── optimizer.h
│   └── utils.h
├── linear_data.csv
├── Makefile
├── obj
│   ├── activations.cu.o
│   ├── benchmark.o
│   ├── initializers.o
│   ├── losses.cu.o
│   ├── main.o
│   ├── operations.cu.o
│   ├── optimizers.o
│   └── utils.o
├── run.sh
├── src
│   ├── activations.cu
│   ├── benchmark.c
│   ├── initializers.c
│   ├── losses.cu
│   ├── main.c
│   ├── operations.cu
│   ├── optimizers.c
│   ├── output
│   └── utils.c
└── valgrind-out.txt

```
