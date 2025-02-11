#include <iostream>
#include <cmath>
#include <vector>
#include <bits/stdc++.h>

void size_vec(std::vector<int> vec)
{
    std::cout << "The size of the given vector is:" << vec.size() << std::endl;
}

void cap_vec(std::vector<int> vec)
{
    std::cout << "The capacity of the vector is:" << vec.capacity() << std::endl;
}

int main()
{
    std::vector<int> vec;
    size_vec(vec);
    cap_vec(vec);
    
    vec.push_back(1);
    vec.push_back(2); // size increment twice
    size_vec(vec);
    cap_vec(vec);
    
    vec.push_back(3);
    size_vec(vec);
    cap_vec(vec);
    
    vec.push_back(4);
    vec.emplace_back(5); // emplace back and push back are quite same
    
    for (int val : vec)
    {
        std::cout << val << " ";
    }
    std::cout << "" << std::endl;
    
    vec.pop_back();
    
    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << "" << std::endl;
    
    std::vector<int> vec1 = {1,2,3,4,5,6,7,8,9,10};
    std::cout << "Size of the vector is: " << vec1.size() << std::endl;
    
    std::cout << "Front:" << vec1.front() << std::endl;
    std::cout << "Back:" << vec1.back() << std::endl;
    
    
    std::vector<int> vec2(7,10);
    std::cout << "Size of the vector is: " << vec2.size() << std::endl;
    for (int val : vec2)
    {
        std::cout << val << " ";
    }
    std::cout << "" << std::endl;
    
    std::cout << "Vector 1 begin:" << *(vec1.begin()) << std::endl;
    std::cout << "Vector 1 end:" << *(vec1.end()) << std::endl;
    
    vec1.insert(vec1.begin() + 3, 100);
    for (int val : vec1)
    {
        std::cout << val << " ";
    }
    std::cout << "" << std::endl;
    
    std::cout << "Printing through iterator:-----" << std::endl;
    std::vector<int>::iterator it;
    for(it = vec1.begin(); it != vec1.end(); it++)
    {
        std::cout << *(it) << " ";
    }
    std::cout << "" << std::endl;
    
    std::cout << "Printing using auto" << std::endl;
    for (auto it = vec1.begin(); it != vec1.end(); it+=1)
    {
        std::cout << *(it) << " ";
    }
    std::cout << "" << std::endl;
    return 0;
}