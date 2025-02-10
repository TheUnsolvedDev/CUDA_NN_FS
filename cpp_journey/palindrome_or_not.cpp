#include <iostream>
#include <cmath>
#include <bits/stdc++.h>

bool is_palindrome(int n)
{
    int new_number = 0;
    int old_number = n;
    
    while (old_number != 0)
    {
        int remainder = old_number % 10;
        new_number = new_number * 10 + remainder;
        old_number /= 10;
    }
    
    if(new_number == n) return true;
    else return false;
}


int main() 
{
    std::cout << "Enter a number to check if it's palindrome or not: ";
    int value;
    std::cin >> value;
    
    if (is_palindrome(value)) std::cout << value << " is a palindrome number." << std::endl;
    else std::cout << value << " is not a palindrome number." << std::endl;
    return 0;
}