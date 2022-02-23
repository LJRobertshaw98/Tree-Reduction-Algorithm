#include <mpi.h>
#include <stdio.h>
#include "reduce.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

void operators(const int *sendbuf, int *recvbuf, int count, MPI_Op op){
  if(op == MPI_SUM){
    for(int c=0; c<count; c++){
      recvbuf[c] += sendbuf[c];
    }
  }
  else if(op == MPI_PROD){
    for(int c=0; c<count; c++){
      recvbuf[c] *= sendbuf[c];
    }
  } 
  else if(op == MPI_MIN){
    for(int c=0; c<count; c++){
      if(sendbuf[c] < recvbuf[c]){
        recvbuf[c] = sendbuf[c];
      }
    }
  }
  else if(op == MPI_MAX){
    for(int c=0; c<count; c++){
      if(sendbuf[c] > recvbuf[c]){
        recvbuf[c] = sendbuf[c];
      }
    }
  }
  else{
    printf("Operator invalid!\n");
  }
}

int tree_allreduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm){ 

  // Create variables for rank and size
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Define MPI_INT size for use later when copying from recvbuf to sendbuf  
  int MPIIntSize;
  MPI_Type_size(MPI_INT, &MPIIntSize);

  int vrank = rank;
  int vsize = size;

  int layer = 1;

  if(vsize==1){  // If size is 1, so only process is rank 0, then move data from sendbuf to recvbuf. No need to broadcast.
    MPI_Sendrecv(sendbuf, count, MPI_INT, vrank, 0, recvbuf, count, MPI_INT, vrank, 0, comm, MPI_STATUS_IGNORE);
  }
  else{
    // While size is greater than 1 there is 2 or more ranks to operate on
    while(vsize > 1){  // While loop active until size=1 when only process remaining is rank 0
      if(vrank < vsize){  // Filters out odd ranks which are always bigger than size after sending their data to their left even recvbuffer
        if( (vrank % 2) != 0 ){ // If rank is odd
          MPI_Ssend(sendbuf, count, MPI_INT, (vrank-1)*layer, vrank*layer, comm); // Send contents of the even sendbuf to recvbuf, use tag to ensure correct delivery
          vrank *= vsize;  // multiplying odd ranks by sizes ensures they are always > or = size when the if(rank < size) comes from next while iteration
        }
        else{  // If rank is even
          // For an even rank, the values for the even number is stored in sendbuf, and the values of the odd rank is stored in recvbuf.
          MPI_Recv(recvbuf, count, MPI_INT, (vrank+1)*layer, (vrank+1)*layer, comm, MPI_STATUS_IGNORE);  // Receive contents of sendbuf from rank+1 into recvbuf
          operators(sendbuf, recvbuf, count, op);  // Call above operators function to do the operations, recvbuf is modified to contain result
          memcpy(sendbuf, recvbuf, count * MPIIntSize);  // Copy the recvbuffer to the sendbuffer, so the send buffer keeps data from last iteration
          vrank /= 2;  // Half the rank so for next iteration of while loop rank 0 --> rank 0, rank 2 --> rank 1, rank 4 --> rank 2, etc...
        }
      }
      vsize /= 2;  // Half the size to reflect the processes contracting pairwise
      layer *= 2;  // Multiply layer by 2 so the send destinations are always correct
    }
  }

  // Broadcast result
  if(rank==0){
    for(int i=1; i<size; i++){
      MPI_Ssend(sendbuf, count, MPI_INT, i, 0, comm);
    }
  }
  else{
    MPI_Recv(recvbuf, count, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
  }

  return 0;
}
