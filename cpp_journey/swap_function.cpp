#include <iostream>
#include <cmath>
#include <bits/stdc++.h>

void swap(int *x, int *y)
{
    int temp = *x;
    *x = *y;
    *y = temp;
}

int main() 
{
    std::cout << "Enter two numbers to swap: (a b)" << std::endl;
    int number1, number2;
    std::cin >> number1 >> number2;
    
    std::cout << "Before swap: a = " << number1 << ", b = " << number2 << std::endl;
    swap(&number1, &number2);
    std::cout << "After swap: a = " << number1 << ", b = " << number2 << std::endl;
    return 0;
}