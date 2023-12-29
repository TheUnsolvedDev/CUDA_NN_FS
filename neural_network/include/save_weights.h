/*
 * @Author: Shuvrajeet Das
 * @Date: 2023-12-28 13:45:30
 * @Last Modified by:   shuvrajeet
 * @Last Modified time: 2023-12-28 13:45:30
 */
#ifndef SAVE_WEIGHTS_H_
#define SAVE_WEIGHTS_H_

#include "initializers.h"

void dump_tensor_to_file(tensor *var, const char *filename);
tensor *read_tensor_from_file(const char *filename);

#endif // SAVE_WEIGHTS_H_