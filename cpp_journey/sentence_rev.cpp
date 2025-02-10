#include <iostream>
#include <cmath>
#include <bits/stdc++.h>

void reverse(char *str, int size, int idx)
{
    if (idx == size) return;
    char temp_value = str[idx];
    reverse(str, size, idx+1);
    std::cout << temp_value;
}

int main() 
{
    char a[] = "A quick brown fox jumps over the lazy dog.";
    int size = sizeof(a)/sizeof(a[0]);
    
    reverse(a, size, 0);
    std::cout << "" << std::endl;
    return 0;
}