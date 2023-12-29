/*
 * @Author: Shuvrajeet Das
 * @Date: 2023-12-28 17:08:13
 * @Last Modified by: shuvrajeet
 * @Last Modified time: 2023-12-28 17:13:07
 */
#ifndef METRICS_H_
#define METRICS_H_

#include "initializers.h"

float r2_score(tensor y_true, tensor y_pred);
float binary_accuracy(tensor y_true, tensor y_pred);

#endif // METRICS_H_