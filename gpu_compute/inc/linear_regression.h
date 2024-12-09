#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

float calculate_mse(tensor1d* predictions, tensor1d* targets);
void linear_regression(tensor2d* X, tensor1d* y, int iterations, float learning_rate);

#endif // !LINEAR_REGRESSION_H