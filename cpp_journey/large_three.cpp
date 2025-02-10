#include <iostream>
#include <cmath>
#include <bits/stdc++.h>

int main() 
{
    std::cout << "Enter three numbers: (a b c)" << std::endl;
    int number1, number2, number3;
    std::cin >> number1 >> number2 >> number3;
    
    int largest = std::max(number1, std::max(number2, number3));
    int smallest = std::min(number1, std::min(number2, number3));
    
    std::cout << "Largest number: " << largest << std::endl;
    std::cout << "Smallest number: " << smallest << std::endl;    
    return 0;
}