
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include<stdio.h>
#include "matrixUtils.h"

using namespace std;

void executeCPU(string X, string Y);
void getAlignments(string X, string Y, Matrix matrix);
void executeGPU(string X, string Y);

	typedef struct {
	char *content;
	int length;
} CUDAstring;

int GAP = -1, MISMATCH = -1, MATCH = 1;

int main() {
	// Size of vectors
	string X("GGTTGACTAAATTTGCTATCATATTATTTGGGCCCGGGGGGGGGGGGAATTTTTCCCCGGGGGTCAG");
	string Y("TGTTACGGATGGGGGAAGCAGTTTTTTCAGCAGTTTCAGCATGCATCAGCTTTCAGCATCGTCAGTCA");
	for(int i = 0;i < 10;i++)
	X = X + X;
	for(int i = 0;i < 10;i++)
        Y = Y + Y;
	X = X.substr(0,10000);
 	Y = Y.substr(0,10000);
	struct timeval start, end;

	gettimeofday(&start, NULL);
	executeCPU(X, Y);
	gettimeofday(&end, NULL);
	cout << "CPU calculation ended in "
			<< (end.tv_sec - start.tv_sec) * 1000
					+ (end.tv_usec - start.tv_usec) / 1000 << endl;

	gettimeofday(&start, NULL);
	executeGPU(X, Y);
	gettimeofday(&end, NULL);
	cout << "GPU calculation ended in "
			<< (end.tv_sec - start.tv_sec) * 1000
					+ (end.tv_usec - start.tv_usec) / 1000 << endl;

	return 0;
}


__global__ void needlemanKernel(Matrix matrix, CUDAstring X, CUDAstring Y,
		int dia,int MATCH, int MISMATCH, int GAP,int count,int *d_x_location,int *d_y_location) 

	int xs = X.length + 1;
	int ys = Y.length + 1;
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if( index < count) 
	{
		// printf("index %d \n ",index);

		int j = d_y_location[index];
		int i = d_x_location[index];	
		// printf("i=%d,  j = %d \n ",i,j);
			int matchVal, yGapVal, xGapVal;
	// Match/mismatch
	if (Y.content[i - 1] == X.content[j - 1]) {
		matchVal = matrix.elements[(i - 1) * xs + j - 1] + MATCH;
	} else {
		matchVal = matrix.elements[(i - 1) * xs + j - 1] + MISMATCH;
	}
	// X Gap
	xGapVal = matrix.elements[(i - 1) * xs + j] + GAP;
	// Y Gap
	yGapVal = matrix.elements[(i) * xs + j - 1] + GAP;

	matrix.elements[i * xs + j] =
			(matchVal > xGapVal ? matchVal : xGapVal) > yGapVal ?
					(matchVal > xGapVal ? matchVal : xGapVal) : yGapVal;
	}

}

void executeGPU(string X, string Y) {

	struct timeval start, end;
	int xs = X.size() + 1;
	int ys = Y.size() + 1;
	size_t size = xs * ys * sizeof(int);

	Matrix matrix = mallocMatrix(xs, ys);

	Matrix d_matrix;
	d_matrix.width = matrix.width;
	d_matrix.height = matrix.height;

	CUDAstring d_X;
	d_X.length = X.size();
	char x_char[X.size()];
	strcpy(x_char, X.c_str());

	CUDAstring d_Y;
	d_Y.length = Y.size();
	char y_char[Y.size()];
	strcpy(y_char, Y.c_str());

	for (int i = 0; i < xs; i++) {
		matrix.elements[i] = GAP * i;
	}

	for (int j = 0; j < ys; j++) {
		matrix.elements[j * xs] = GAP * j;
	}

	cudaMalloc(&d_matrix.elements, size);
	cudaMalloc(&d_X.content, X.size() * sizeof(char));
	cudaMalloc(&d_Y.content, Y.size() * sizeof(char));

	cudaMemcpy(d_matrix.elements, matrix.elements, size,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_X.content, x_char, X.size() * sizeof(char),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y.content, y_char, Y.size() * sizeof(char),
			cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(1024);
	dim3 numBlocks((xs * ys + 1024) / threadsPerBlock.x);

	for (int dia = 3; dia < xs + ys ; dia++)
	 {
		int* x_location;
		int* y_location;
		int* d_x_location;
		int* d_y_location;
		int len = min(matrix.width,matrix.height);

		x_location = (int*)malloc(len* sizeof(int));
		y_location = (int*)malloc(len* sizeof(int));
		cudaMalloc((void**)&d_x_location, len* sizeof(int));
		cudaMalloc((void**)&d_y_location, len* sizeof(int));
		
		int count = 0;
		// cout<<"dia" << dia<<" xs ,ys  "<<xs<<"  "<<ys<< "  \n "<<endl;
		for(int row = 1; row < min(dia,xs); row ++)
				{
					int col = dia - 1 - row;
					if(row >= xs | col >= ys | col <= 0 )
						continue;
					y_location[count] = col;
					x_location[count] = row;

					count += 1;
					// cout<<"matrix count "<<count<<" row "<<row<<" col "<<col<<" array "<< x_location[0]<<endl;
				}
		cudaMemcpy(d_x_location, x_location, len* sizeof(int),
			cudaMemcpyHostToDevice);
		cudaMemcpy(d_y_location, y_location, len * sizeof(int),
			cudaMemcpyHostToDevice);
		gettimeofday(&start, NULL);
		needlemanKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_X, d_Y,dia,
				MATCH, MISMATCH, GAP,count,d_x_location,d_y_location);
				cudaDeviceSynchronize();

				gettimeofday(&end, NULL);

			}
	cudaMemcpy(matrix.elements, d_matrix.elements, size,
			cudaMemcpyDeviceToHost);

	cudaFree(d_matrix.elements);
	cudaFree(d_X.content);
	cudaFree(d_Y.content);

	for (int i = 0; i < ys; i++) {
		for (int j = 0; j < xs; j++) {
			cout << matrix.elements[i * xs + j] << "\t";
		}
		cout << endl;
	}

	// getAlignments(X, Y, matrix);
}



void getAlignments(string X, string Y, Matrix matrix) {
	int xs = X.size() + 1;
	int ys = Y.size() + 1;

	int i = ys - 1;
	int j = xs - 1;
	bool hadPath = true;

	vector < stack<int*> > stList = vector<stack<int*> >();
	stack<int*> st = stack<int*>();

	int *arr = (int*) malloc(2 * sizeof(int));
	arr[0] = i;
	arr[1] = j;
	st.push(arr);
	stList.push_back(st);

	while (hadPath) {
		hadPath = false;
		vector < stack<int*> > temp = stList;
		stList = vector<stack<int*> >();

		for (int ii = 0; ii < temp.size(); ++ii) {
			int* pointer = temp[ii].top();
			i = pointer[0];
			j = pointer[1];

			if (i - 1 >= 0
					&& readMatrix(matrix, i - 1, j) + GAP
							== readMatrix(matrix, i, j)) {
				stack<int*> newSt = stack<int*>(temp[ii]);
				arr = (int*) malloc(2 * sizeof(int));
				arr[0] = i - 1;
				arr[1] = j;
				newSt.push(arr);
				stList.push_back(newSt);

				hadPath = true;
			}

			if (j - 1 >= 0
					&& readMatrix(matrix, i, j - 1) + GAP
							== readMatrix(matrix, i, j)) {
				stack<int*> newSt = stack<int*>(temp[ii]);
				arr = (int*) malloc(2 * sizeof(int));
				arr[0] = i;
				arr[1] = j - 1;
				newSt.push(arr);
				stList.push_back(newSt);

				while (newSt.size() > 0) {
					newSt.pop();
				}

				hadPath = true;
			}

			if (i - 1 >= 0 && j - 1 >= 0 && Y[i - 1] == X[j - 1]) {
				if (readMatrix(matrix, i - 1, j - 1) + MATCH
						== readMatrix(matrix, i, j)) {
					stack<int*> newSt = stack<int*>(temp[ii]);
					arr = (int*) malloc(2 * sizeof(int));
					arr[0] = i - 1;
					arr[1] = j - 1;
					newSt.push(arr);
					stList.push_back(newSt);
					hadPath = true;
				}

			} else if (i - 1 >= 0 && j - 1 >= 0 && Y[i - 1] != X[j - 1]) {
				if (readMatrix(matrix, i - 1, j - 1) + MISMATCH
						== readMatrix(matrix, i, j)) {
					stack<int*> newSt = stack<int*>(temp[ii]);
					arr = (int*) malloc(2 * sizeof(int));
					arr[0] = i - 1;
					arr[1] = j - 1;
					newSt.push(arr);
					stList.push_back(newSt);
					hadPath = true;
				}
			}
		}
		if (stList.size() == 0) {
			stList = temp;
		}
	}

	for (int ii = 0; ii < stList.size(); ++ii) {
		stack<int*> stack = stList[ii];
		vector<int*> path = vector<int*>();

		while (!stack.empty()) {
			int* arr = stack.top();
			path.push_back(arr);
			stack.pop();
		}

		string xSeq = "";
		string matchSeq = "";
		string ySeq = "";

		for (int k = 1; k < path.size(); k++) {
			int i0 = path[k - 1][0];
			int j0 = path[k - 1][1];
			int i1 = path[k][0];
			int j1 = path[k][1];

			// if a match move
			if (i1 == i0 + 1 && j1 == j0 + 1) {
				xSeq += X[j0];
				ySeq += Y[i0];

				matchSeq += X[j0] == Y[i0] ? "|" : " ";
			}
			// if X gap
			else if (i1 == i0 + 1 && j1 == j0) {
				xSeq += "-";
				ySeq += Y[i1];
				matchSeq += " ";
			}
			// if Y gap
			else if (i1 == i0 && j1 == j0 + 1) {
				xSeq += X[i1];
				ySeq += "-";
				matchSeq += " ";
			}

		}
	}

}

void executeCPU(string X, string Y) {
	int xs = X.size() + 1;
	int ys = Y.size() + 1;

	Matrix matrix = mallocMatrix(xs, ys);

	for (int i = 0; i < xs; i++) {
		matrix.elements[i] = GAP * i;
	}

	for (int j = 0; j < ys; j++) {
		matrix.elements[j * xs] = GAP * j;
	}
 
	for (int i = 1; i < ys; i++) {
		for (int j = 1; j < xs; j++) {
			int matchVal, yGapVal, xGapVal;
			// Match/mismatch
			if (Y[i - 1] == X[j - 1]) {
				matchVal = matrix.elements[(i - 1) * xs + j - 1] + MATCH;
			} else {
				matchVal = matrix.elements[(i - 1) * xs + j - 1] + MISMATCH;
			}
			// X Gap
			xGapVal = matrix.elements[(i - 1) * xs + j] + GAP;
			// Y Gap
			yGapVal = matrix.elements[(i) * xs + j - 1] + GAP;

			matrix.elements[i * xs + j] =
					(matchVal > xGapVal ? matchVal : xGapVal) > yGapVal ?
							(matchVal > xGapVal ? matchVal : xGapVal) : yGapVal;
		}
	}

	for (int i = 0; i < ys; i++) {
		for (int j = 0; j < xs; j++) {
			cout << matrix.elements[i * xs + j] << "\t";
		}
		cout << endl;
	}

	getAlignments(X, Y, matrix);
}

