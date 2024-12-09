#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"
#include "activations_cpu.h"

float sigmoid(float x)
{
    return 1/(1+exp(-x));
}

float sigmoid_derivative(float x)
{
    return sigmoid(x)*(1-sigmoid(x));
}

tensor1d sigmoid1d_activation(tensor1d t1d)
{
    for (int i = 0;i<t1d.size;i++)
    {
        t1d.data[i] = sigmoid(t1d.data[i]);
    }
    return t1d;
}

tensor1d sigmoid1d_activation_derivative(tensor1d t1d)
{
    for (int i = 0;i<t1d.size;i++)
    {
        t1d.data[i] = sigmoid_derivative(t1d.data[i]);
    }
    return t1d;
}

tensor2d sigmoid2d_activation(tensor2d t2d)
{
    for (int i = 0;i<t2d.row_size;i++)
    {
        for (int j = 0;j<t2d.col_size;j++)
        {
            t2d.data[i][j] = sigmoid(t2d.data[i][j]);
        }
    }
    return t2d;
}

tensor2d sigmoid2d_activation_derivative(tensor2d t2d)
{
    for (int i = 0;i<t2d.row_size;i++)
    {
        for (int j = 0;j<t2d.col_size;j++)
        {
            t2d.data[i][j] = sigmoid_derivative(t2d.data[i][j]);
        }
    }
    return t2d;
}

float relu(float x)
{
    if(x>0)
    {
        return x;
    }
    else
    {
        return 0;
    }
}

float relu_derivative(float x)
{
    if(x>0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

tensor1d relu1d_activation(tensor1d t1d)
{
    for (int i = 0;i<t1d.size;i++)
    {
        t1d.data[i] = relu(t1d.data[i]);
    }
    return t1d;
}

tensor1d relu1d_activation_derivative(tensor1d t1d)
{
    for (int i = 0;i<t1d.size;i++)
    {
        t1d.data[i] = relu_derivative(t1d.data[i]);
    }
    return t1d;
}

tensor2d relu2d_activation(tensor2d t2d)
{
    for (int i = 0;i<t2d.row_size;i++)
    {
        for (int j = 0;j<t2d.col_size;j++)
        {
            t2d.data[i][j] = relu(t2d.data[i][j]);
        }
    }
    return t2d;
}

tensor2d relu2d_activation_derivative(tensor2d t2d)
{
    for (int i = 0;i<t2d.row_size;i++)
    {
        for (int j = 0;j<t2d.col_size;j++)
        {
            t2d.data[i][j] = relu_derivative(t2d.data[i][j]);
        }
    }
    return t2d;
}

float leaky_relu(float x)
{
    if(x>0)
    {
        return x;
    }
    else
    {
        return 0.01*x;
    }
}

float leaky_relu_derivative(float x)
{
    if(x>0)
    {
        return 1;
    }
    else
    {
        return 0.01;
    }
}

tensor1d leaky_relu1d_activation(tensor1d t1d)
{
    for (int i = 0;i<t1d.size;i++)
    {
        t1d.data[i] = leaky_relu(t1d.data[i]);
    }
    return t1d;
}

tensor1d leaky_relu1d_activation_derivative(tensor1d t1d)
{
    for (int i = 0;i<t1d.size;i++)
    {
        t1d.data[i] = leaky_relu_derivative(t1d.data[i]);
    }
    return t1d;
}

tensor2d leaky_relu2d_activation(tensor2d t2d)
{
    for (int i = 0;i<t2d.row_size;i++)
    {
        for (int j = 0;j<t2d.col_size;j++)
        {
            t2d.data[i][j] = leaky_relu(t2d.data[i][j]);
        }
    }
    return t2d;
}

tensor2d leaky_relu2d_activation_derivative(tensor2d t2d)
{
    for (int i = 0;i<t2d.row_size;i++)
    {
        for (int j = 0;j<t2d.col_size;j++)
        {
            t2d.data[i][j] = leaky_relu_derivative(t2d.data[i][j]);
        }
    }
    return t2d;
}

float tanh_(float x)
{
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

float tanh_derivative(float x)
{
    return 1-pow(tanh_(x),2);
}

tensor1d tanh1d_activation(tensor1d t1d)
{
    for (int i = 0;i<t1d.size;i++)
    {
        t1d.data[i] = tanh_(t1d.data[i]);
    }
    return t1d;
}

tensor1d tanh1d_activation_derivative(tensor1d t1d)
{
    for (int i = 0;i<t1d.size;i++)
    {
        t1d.data[i] = tanh_derivative(t1d.data[i]);
    }
    return t1d;
}

tensor2d tanh2d_activation(tensor2d t2d)
{
    for (int i = 0;i<t2d.row_size;i++)
    {
        for (int j = 0;j<t2d.col_size;j++)
        {
            t2d.data[i][j] = tanh_(t2d.data[i][j]);
        }
    }
    return t2d;
}

tensor2d tanh2d_activation_derivative(tensor2d t2d)
{
    for (int i = 0;i<t2d.row_size;i++)
    {
        for (int j = 0;j<t2d.col_size;j++)
        {
            t2d.data[i][j] = tanh_derivative(t2d.data[i][j]);
        }
    }
    return t2d;
}

