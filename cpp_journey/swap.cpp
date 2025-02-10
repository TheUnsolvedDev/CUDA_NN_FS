#include <iostream>

int main()
{
    std::cout << "Enter two numbers: ";
    int a,b;
    std::cin >> a >> b;
    std::cout << "Before swap: a = " << a << ", b = " << b << std::endl;
    a = a + b;
    b = a - b;
    a = a - b;
    
    std::cout << "After swap: a = " << a << ", b = " << b << std::endl;
    
    return 0;
}