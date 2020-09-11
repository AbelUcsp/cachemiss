#include <stdio.h>  
#include <stdlib.h>  
#include <mpi.h>  
  

      
void Get_dims(int* m_p, int* local_m_p, int* n_p, int* local_n_p,  
      int my_rank, int comm_sz, MPI_Comm comm);  
void Allocate_arrays(double** local_A_pp, double** local_x_pp,   
      double** local_y_pp, int local_m, int n, int local_n,   
      MPI_Comm comm);  
void Read_matrix(char prompt[], double local_A[], int m, int local_m,   
      int n, int my_rank, MPI_Comm comm);  
void Read_vector(char prompt[], double local_vec[], int n, int local_n,   
      int my_rank, MPI_Comm comm);  
void Print_matrix(char title[], double local_A[], int m, int local_m,   
      int n, int my_rank, MPI_Comm comm);  
void Print_vector(char title[], double local_vec[], int n,  
      int local_n, int my_rank, MPI_Comm comm);  
void Mat_vect_mult(double local_A[], double local_x[],   
      double local_y[], int local_m, int n, int local_n,   
      MPI_Comm comm);  
  
  

int main(void) {  
      
   srand(time(NULL));
   double* local_A;  
   double* local_x;  
   double* local_y;  
   int m, local_m, n, local_n;  
   int my_rank, comm_sz;  
   MPI_Comm comm;  
  
   MPI_Init(NULL, NULL);  
   comm = MPI_COMM_WORLD;  
   MPI_Comm_size(comm, &comm_sz);  
   MPI_Comm_rank(comm, &my_rank);  
  
  

   Get_dims(&m, &local_m, &n, &local_n, my_rank, comm_sz, comm);  
   Allocate_arrays(&local_A, &local_x, &local_y, local_m, n, local_n, comm);  
   Read_matrix("A", local_A, m, local_m, n, my_rank, comm);  
   Read_vector("x", local_x, n, local_n, my_rank, comm);  

   float inicio, final;
   MPI_Barrier(comm);
   inicio = MPI_Wtime(); 
   Mat_vect_mult(local_A, local_x, local_y, local_m, n, local_n, comm);  
  
   final = MPI_Wtime(); 
   printf("TIempo: %f\n\n", final-inicio);   

   ///Print_vector("y", local_y, m, local_m, my_rank, comm);  
   free(local_A);  
   free(local_x);  
   free(local_y);  
   MPI_Finalize();  
   return 0;  
}  /* main */  
  

void Get_dims(  
      int*      m_p        /* out */,   
      int*      local_m_p  /* out */,  
      int*      n_p        /* out */,  
      int*      local_n_p  /* out */,  
      int       my_rank    /* in  */,  
      int       comm_sz    /* in  */,  
      MPI_Comm  comm       /* in  */) {  
   int local_ok = 1;  
  
   if (my_rank == 0) {  
      printf("numero filas\n");  
      scanf("%d", m_p);  
      printf("numero  columnas\n");  
      scanf("%d", n_p);  
   }  

// MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
   MPI_Bcast(m_p, 1, MPI_INT, 0, comm);  
   MPI_Bcast(n_p, 1, MPI_INT, 0, comm);  
   if (*m_p <= 0 || *n_p <= 0 || *m_p % comm_sz != 0  || *n_p % comm_sz != 0) 
         local_ok = 0;  


   *local_m_p = *m_p/comm_sz;  
   *local_n_p = *n_p/comm_sz;  
}  /* Get_dims */  
  

void Allocate_arrays(  
      double**  local_A_pp  /* out */,   
      double**  local_x_pp  /* out */,   
      double**  local_y_pp  /* out */,   
      int       local_m     /* in  */,   
      int       n           /* in  */,     
      int       local_n     /* in  */,   
      MPI_Comm  comm        /* in  */) {  
  
   int local_ok = 1;  
  
   *local_A_pp = malloc(local_m*n*sizeof(double));  
   *local_x_pp = malloc(local_n*sizeof(double));  
   *local_y_pp = malloc(local_m*sizeof(double));  
  
   if (*local_A_pp == NULL || local_x_pp == NULL ||  
         local_y_pp == NULL) local_ok = 0;  
         
}  


void Read_matrix(  
      char      prompt[]   /* in  */,   
      double    local_A[]  /* out */,   
      int       m          /* in  */,   
      int       local_m    /* in  */,   
      int       n          /* in  */,  
      int       my_rank    /* in  */,  
      MPI_Comm  comm       /* in  */) {  
   double* A = NULL;  
   int local_ok = 1;  
   int i, j;  
  
   if (my_rank == 0) {  
      A = malloc(m*n*sizeof(double));  
      if (A == NULL) local_ok = 0;  


      for (i = 0; i < m; i++)  
         for (j = 0; j < n; j++)  
            A[i*n+j] = rand () % 10 + 15;
      MPI_Scatter(A, local_m*n, MPI_DOUBLE,   
            local_A, local_m*n, MPI_DOUBLE, 0, comm);  
      free(A);  
   } else {  
            
      MPI_Scatter(A, local_m*n, MPI_DOUBLE,   
            local_A, local_m*n, MPI_DOUBLE, 0, comm);  
   }  
}  


void Read_vector(  
      char      prompt[]     /* in  */,   
      double    local_vec[]  /* out */,   
      int       n            /* in  */,  
      int       local_n      /* in  */,  
      int       my_rank      /* in  */,  
      MPI_Comm  comm         /* in  */) {  
   double* vec = NULL;  
   int i, local_ok = 1;  
  
   if (my_rank == 0) {  
      vec = malloc(n*sizeof(double));  
      if (vec == NULL) local_ok = 0;  

            
      for (i = 0; i < n; i++)  
            vec[i] = rand () % 1 + 15;
       

      MPI_Scatter(vec, local_n, MPI_DOUBLE,  
            local_vec, local_n, MPI_DOUBLE, 0, comm);  
      free(vec);  
   } else {  
            
      MPI_Scatter(vec, local_n, MPI_DOUBLE,  
            local_vec, local_n, MPI_DOUBLE, 0, comm);  
   }  
}  /* Read_vector */  
  


void Print_matrix(  
      char      title[]    /* in */,  
      double    local_A[]  /* in */,   
      int       m          /* in */,   
      int       local_m    /* in */,   
      int       n          /* in */,  
      int       my_rank    /* in */,  
      MPI_Comm  comm       /* in */) {  
   double* A = NULL;  
   int i, j, local_ok = 1;  
  
   if (my_rank == 0) {  
      A = malloc(m*n*sizeof(double));  
      if (A == NULL) local_ok = 0;  
            
      MPI_Gather(local_A, local_m*n, MPI_DOUBLE,  
            A, local_m*n, MPI_DOUBLE, 0, comm);  
      printf("\nThe matrix %s\n", title);  

      for (i = 0; i < m; i++) {  
         for (j = 0; j < n; j++)  
            printf("%f ", A[i*n+j]);  
         printf("\n");  
      }  
      printf("\n");  
      free(A);  
   } else {  
         
      MPI_Gather(local_A, local_m*n, MPI_DOUBLE,  
            A, local_m*n, MPI_DOUBLE, 0, comm);  
   }  
}  /* Print_matrix */  
  

void Print_vector(  
      char      title[]     /* in */,   
      double    local_vec[] /* in */,   
      int       n           /* in */,  
      int       local_n     /* in */,  
      int       my_rank     /* in */,  
      MPI_Comm  comm        /* in */) {  
   double* vec = NULL;  
   int i, local_ok = 1;  
  
   if (my_rank == 0) {  
      vec = malloc(n*sizeof(double));  
      if (vec == NULL) local_ok = 0;  
            
      MPI_Gather(local_vec, local_n, MPI_DOUBLE,  
            vec, local_n, MPI_DOUBLE, 0, comm);  
      printf("\nThe vector %s\n", title);  
      for (i = 0; i < n; i++)  
         printf("%f ", vec[i]);  
      printf("\n");  
      free(vec);  
   }  else {  
            
      MPI_Gather(local_vec, local_n, MPI_DOUBLE,  
            vec, local_n, MPI_DOUBLE, 0, comm);  
   }  
}  /* Print_vector */  
  

void Mat_vect_mult(  
      double    local_A[]  /* in  */,   
      double    local_x[]  /* in  */,   
      double    local_y[]  /* out */,  
      int       local_m    /* in  */,   
      int       n          /* in  */,  
      int       local_n    /* in  */,  
      MPI_Comm  comm       /* in  */) {  
   double* x;  
   int local_i, j;  
   int local_ok = 1;  
  
   x = malloc(n*sizeof(double));  
   if (x == NULL) local_ok = 0;  

         
   MPI_Allgather(local_x, local_n, MPI_DOUBLE,  
         x, local_n, MPI_DOUBLE, comm);  
  
   for (local_i = 0; local_i < local_m; local_i++) {  
      local_y[local_i] = 0.0;  
      for (j = 0; j < n; j++)  
         local_y[local_i] += local_A[local_i*n+j]*x[j];  
   }  
   free(x);  
}  