#include <iostream>
#include <cmath>
#include <bits/stdc++.h>

bool is_prime(int n)
{
    if (n == 0) return false;
    if (n == 1) return true;
    if (n == 2) return true;
    
    int counter = 2;
    while (counter <= sqrt(n)) 
    {
        if (n % counter == 0) return false;
        counter++;
    }
    return true;
}

int main() 
{
    std::cout << "Enter a number to check if it's prime or not: ";
    int value;
    std::cin >> value;
    
    if (is_prime(value)) std::cout << value << " is a prime number." << std::endl;
    else std::cout << value << " is not a prime number." << std::endl;
    return 0;
}