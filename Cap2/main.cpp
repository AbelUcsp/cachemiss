#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <time.h>
#include<stdio.h> 
using namespace std;
#define max 1000 

//int MAX = 300;
double A[max][max], x[max], y[max];


void UnoFor(int MAX){
    for (int i = 0; i < MAX; i++)
        for (int j = 0; j < MAX; j++)
            y[i] += A[i][j]*x[j];
}


void DosFor(int MAX){
    for (int j = 0; j < MAX; j++)
        for (int i = 0; i < MAX; i++)
            y[i] += A[i][j]*x[j];
}



int main()
{
    UnoFor(max);
    //DosFor(max);


    return 0;
}
