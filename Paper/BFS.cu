%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_NODES 5

typedef struct
{
	int start;     // indice del primer nodo adyacente en conjunto de aristas	
	int length;    // Número de nodos adyacentes 
} Node;

__global__ void CUDA_BFS_KERNEL(Node *Va, int *Ea, bool *Fa, bool *Xa, int *Ca,bool *done)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id > NUM_NODES)
		*done = false;


	if (Fa[id] == true && Xa[id] == false)
	{
		printf("%d ", id); //para imprimir los vértices que aparecen en el orden de BFS	
		Fa[id] = false;
		Xa[id] = true;
		__syncthreads();  //sincronización de nivel/frontera
		int k = 0;
		int i;
		int start = Va[id].start;
		int end = start + Va[id].length;
		for (int i = start; i < end; i++) 
		{
			int nid = Ea[i];

			if (Xa[nid] == false)
			{
				Ca[nid] = Ca[id] + 1;
				Fa[nid] = true;
				*done = false;
			}

		}

	}

}

//La frontera BFS corresponde a todos los nodos que se procesan en el nivel actual.


int main()
{



 clock_t begin = clock();

	 Node node[NUM_NODES];
	
	
	//int edgesSize = 2 * NUM_NODES;
	int edges[NUM_NODES];

	node[0].start = 0;
	node[0].length = 2;

	node[1].start = 2;
	node[1].length = 1;

	node[2].start = 3;
	node[2].length = 1;

	node[3].start = 4;
	node[3].length = 1;

	node[4].start = 5;
	node[4].length = 0;

	edges[0] = 1;
	edges[1] = 2;	
	edges[2] = 4;
	edges[3] = 3;
	edges[4] = 4;

	bool frontier[NUM_NODES] = { false };
	bool visited[NUM_NODES] = { false };
	int cost[NUM_NODES] = { 0 };

	int source = 0;
	frontier[source] = true;

	Node* Va;     //conjunto de vertices
	cudaMalloc((void**)&Va, sizeof(Node)*NUM_NODES);
	cudaMemcpy(Va, node, sizeof(Node)*NUM_NODES, cudaMemcpyHostToDevice);

	int* Ea;    //conjunto de aristas
	cudaMalloc((void**)&Ea, sizeof(Node)*NUM_NODES);
	cudaMemcpy(Ea, edges, sizeof(Node)*NUM_NODES, cudaMemcpyHostToDevice);

	bool* Fa;   //frontera
	cudaMalloc((void**)&Fa, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Fa, frontier, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

	bool* Xa;   //visited
	cudaMalloc((void**)&Xa, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Xa, visited, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

	int* Ca;    //cost
	cudaMalloc((void**)&Ca, sizeof(int)*NUM_NODES);
	cudaMemcpy(Ca, cost, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);

	

	int num_blks = 1;
	int threads = 5;



	bool done;
	bool* d_done;
	cudaMalloc((void**)&d_done, sizeof(bool));
	printf("\n\n");
	int count = 0;

	printf("Order: \n\n");
	do {
		count++;
		done = true;
		cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
		CUDA_BFS_KERNEL <<<num_blks, threads >>>(Va, Ea, Fa, Xa, Ca,d_done);
		cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost);

	} while (!done);



	printf("Parallel code took cution.\n");

	cudaMemcpy(cost, Ca, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost);
	

  clock_t end = clock();
	double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
	printf("Parallel code took %.3f sec for execution.\n", elapsed_time);


/*
	printf("\nCost: ");
	for (int i = 0; i<NUM_NODES; i++)
		printf( "%d    ", cost[i]);
	printf("\n");
 */
}