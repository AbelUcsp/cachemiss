#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <time.h>
using namespace std;

void FillMTRIX(int **M, int size){
    srand (time(NULL));
	for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            M[i][j] = 1 + rand() % (10-1);
}

void Multiplication3FOR(int **A, int **B,  int **SOL, int size){
    FillMTRIX(A, size);
    FillMTRIX(B, size);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j){
            SOL[i][j] = 0;
            for (int k = 0; k < size; ++k)
                SOL[i][j] += A[i][k] * B[k][j];
        }
}

void MultiplicationBlock(int **A, int **B,  int **SOL, int size){
    FillMTRIX(A, size);
    FillMTRIX(B, size);

    int s = 4;
    int N = size;
    for(int jj=0;jj<N;jj+= s){
        for(int kk=0;kk<N;kk+= s){
                for(int i=0;i<N;i++){
                        for(int j = jj; j<((jj+s)>N?N:(jj+s)); j++){
                                int temp = 0;
                                for(int k = kk; k<((kk+s)>N?N:(kk+s)); k++){
                                        temp += A[i][k]*B[k][j];
                                }
                                SOL[i][j] += temp;
                        }
                }
        }
    }

}



void PrintM(int** A, int n){
	for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++)
            cout << A[i][j] << " ";
        cout << "\n";
	}
}

int main()
{

	int size_ = 500;
	int **A = new int*[size_];
	int **B = new int*[size_];
	int **SOL = new int*[size_];
	int **SOL2 = new int*[size_];
	for(int i = 0; i < size_; ++i){
		A[i] = new int[size_];
		B[i] = new int[size_];
		SOL[i] = new int[size_];
		SOL2[i] = new int[size_];
    }


    //Multiplication3FOR(A,B,SOL,size_);
    //PrintM(SOL, size_);
    //cout << "\n\n\n\n\n";
    MultiplicationBlock(A,B,SOL2,size_);
    //PrintM(SOL2, size_);



}
