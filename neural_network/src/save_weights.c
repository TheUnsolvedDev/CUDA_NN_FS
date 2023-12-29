/*
 * @Author: Shuvrajeet Das
 * @Date: 2023-12-28 13:46:52
 * @Last Modified by:   shuvrajeet
 * @Last Modified time: 2023-12-28 13:46:52
 */

#include <stdio.h>
#include <stdlib.h>

#include "initializers.h"

void dump_tensor_to_file(tensor *var, const char *filename)
{
    FILE *file = fopen(filename, "wb");

    if (file == NULL)
    {
        printf("Error: Unable to open file %s for writing.\n", filename);
        return;
    }

    fwrite(&(var->size[0]), sizeof(int), 1, file);
    fwrite(&(var->size[1]), sizeof(int), 1, file);

    for (int i = 0; i < var->size[0]; i++)
    {
        fwrite(var->matrix[i], sizeof(float), var->size[1], file);
    }

    fclose(file);
}

tensor *read_tensor_from_file(const char *filename)
{
    FILE *file = fopen(filename, "rb");

    if (file == NULL)
    {
        printf("Error: Unable to open file %s for reading.\n", filename);
        return NULL;
    }

    tensor *result = (tensor *)malloc(sizeof(tensor));

    fread(&(result->size[0]), sizeof(int), 1, file);
    fread(&(result->size[1]), sizeof(int), 1, file);

    result->matrix = (float **)malloc(result->size[0] * sizeof(float *));
    for (int i = 0; i < result->size[0]; i++)
    {
        result->matrix[i] = (float *)malloc(result->size[1] * sizeof(float));
    }

    for (int i = 0; i < result->size[0]; i++)
    {
        fread(result->matrix[i], sizeof(float), result->size[1], file);
    }

    fclose(file);
    return result;
}