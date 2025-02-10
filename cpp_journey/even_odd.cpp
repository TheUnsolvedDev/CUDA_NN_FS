#include <iostream>
#include <cmath>
#include <bits/stdc++.h>

int main()
{
    std::cout << "Enter a number to see if its odd or even: " ;
    int value;
    std::cin >> value;
    
    if (value % 2 == 0) std::cout << "It's even." << std::endl;
    else std::cout << "It's odd." << std::endl;
    return 0;
}