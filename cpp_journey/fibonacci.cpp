#include <iostream>
#include <cmath>
#include <bits/stdc++.h>

int fibonacci(int n)
{
    if (n == 1) return 0;
    else if (n == 2) return 1;
    else return fibonacci(n-1) + fibonacci(n-2);
}

int main() 
{
    std::cout << "Enter the n-th number to find fib(n): ";
    int value;
    std::cin >> value;
    
    std::cout << "fib(" << value << ") = " << fibonacci(value) << std::endl;
    
    return 0;
}