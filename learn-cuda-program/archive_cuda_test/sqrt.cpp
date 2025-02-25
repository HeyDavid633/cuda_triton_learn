#include <iostream>
#include <cmath>
#include <cstdlib>
using namespace std;

int main()
{
    int head_size = 4;
    float score = 8.0;
    score = sqrtf(static_cast<float>(head_size));

    score = score / sqrtf(static_cast<float>(head_size));

    cout<<"score: "<<score<<endl;
}
