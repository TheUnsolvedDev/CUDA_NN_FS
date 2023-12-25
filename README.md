# Neural Network Implementation in CUDA and C

This repository contains an implementation of a neural network from scratch using CUDA and C. The goal of this project is to leverage the parallel computing capabilities of CUDA to accelerate the training and inference processes of a neural network.

## Introduction
Neural networks are powerful models for machine learning and artificial intelligence tasks. This project focuses on building a basic neural network using the C programming language with CUDA extensions for parallelism. CUDA allows us to harness the computational power of NVIDIA GPUs, speeding up the training and inference phases of the neural network.

## Prerequisites
To run this project, you need the following dependencies:
- CUDA-enabled GPU
- CUDA Toolkit installed
- C Compiler (e.g., GCC)

## Getting Started
1. Clone the repository: `https://github.com/TheUnsolvedDev/IIT_Madras_days`
2. Navigate to the project directory: `cd neural_network`
3. Build the project: `make`

## Project Structure
```bash
neural_network
├── bin
│   └── program
├── document.md
├── Makefile
├── obj
│   ├── activations.cu.o
│   ├── initializers.o
│   ├── main.o
│   ├── matmul.cu.o
│   └── utils.o
├── run.sh
├── src
│   ├── activations.cu
│   ├── activations.cuh
│   ├── initializers.c
│   ├── initializers.h
│   ├── main.c
│   ├── matmul.cu
│   ├── matmul.cuh
│   ├── utils.c
│   └── utils.h
└── valgrind-out.txt
```
