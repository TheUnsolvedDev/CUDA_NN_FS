/*
 * @Author: Shuvrajeet Das 
 * @Date: 2023-12-28 13:44:46 
 * @Last Modified by:   shuvrajeet 
 * @Last Modified time: 2023-12-28 13:44:46 
 */
#ifndef BENCHMARK_H_
#define BENCHMARK_H_

void cpu_test(int m, int n, int k);
void gpu_test(int m, int n, int k);
void time_test(int iterations, int m, int n, int k);

#endif