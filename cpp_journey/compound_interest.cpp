#include <iostream>
#include <bits/stdc++.h>
#include <cmath>

int main() {
    double principle, rate, time;
    
    std::cout << "Enter Principle amount: ";
    std::cin >> principle;
    std::cout << "Rate: ";
    std::cin >> rate;
    std::cout << "Time: ";
    std::cin >> time;
    
    double amount = principle * pow((1 + rate / 100), time);
    double compound_interest = amount - principle;
    
    std::cout << "Compound interest: " << compound_interest << std::endl;
    
    return 0;
}
