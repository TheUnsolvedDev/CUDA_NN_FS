#include <iostream>
#include <cmath>
#include <bits/stdc++.h>

bool check_equal(int *array1, int  *array2, int size1, int size2)
{   
    if (size1 != size2) return false;
    std::sort(array1, array1+size1);
    std::sort(array2, array2+size2);
    
    for (int i = 0; i < size1; i++)
    {
        if (array1[i] != array2[i]) return false;
    }
    
    return true;
}

int main() 
{
    int array1[] = {1,2,3,4,5,6,7,8};
    int array2[] = {1,2,3,4,8,7,6,5};
    
    int size_array1 = sizeof(array1)/sizeof(int);
    int size_array2 = sizeof(array2)/sizeof(int);
    std::cout << size_array1 << " " << size_array2 << std::endl;
    
    if (check_equal(array1, array2, size_array1, size_array2)) std::cout << "Arrays are equal." << std::endl;
    else std::cout << "Arrays are not equal." << std::endl;
    
    return 0;
}