#include<stdio.h>
#include<stdbool.h>
#include<stdlib.h>

#include"cuda.h"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"cuda_common.cuh"

__global__ void shannon_entropy(bool* TEP, int steps, int number_of_nodes, int counterB,
    int* histogram, int attributes, int rules_size, int transient_offset){
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float division = (float)1/(float)attributes + 0.00001;

    int size_of_batch = rules_size*attributes;
    int rule_offset = rules_size*counterB*attributes;
    int rule_idx = blockIdx.x;
    int iterations = number_of_nodes/(blockDim.x + 0.0001)+1;
    

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x * iter + threadIdx.x;
            if(node_idx < number_of_nodes){ 
                int sum = 0;
                for(int i=transient_offset;i<steps;i++){
                    sum += TEP[node_idx+number_of_nodes*i+rule_idx*number_of_nodes*steps];
                }
                float p1 = (float) sum/(float)steps;
                float p0 = (float)(steps-sum) / (float)steps;
                float value = (-1) * (p1 * log2f(p1+0.001) + p0 * log2f(p0+0.001));
                int index = (int) (value/division);
                atomicAdd(&histogram[index+attributes*rule_idx+counterB*size_of_batch],1);
        }
        __syncthreads();
    }
}

//requires rules_size*steps to calculate. To avoid concurrency, every thread will calculate the population
// a timestep
__global__ void population(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes, float* population){
    //population -> size attributes*rules_size

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int rule_offset = rules_size*counterB*attributes;
    int resolution = steps/attributes;
    int step_idx = gid % steps;
    int rule_idx = gid / number_of_nodes;
    int node_idx = gid % number_of_nodes;

    

    if(gid < number_of_nodes*rules_size){
            for(int i=0;i<attributes;i++){
                float increment = (float)TEP[rule_idx*number_of_nodes*steps+ (i+1)*number_of_nodes*resolution + node_idx]/(float)number_of_nodes;
                atomicAdd(&population[rule_offset+attributes*rule_idx+i], increment);
            }
        __syncthreads();
    }
    
}

__global__ void density_histogram(float* dtep, int steps, int number_of_nodes, int counterB,
int rules_size, int* histogram, int attributes, int transient_offset, int max_size){


    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float division = (float)1/(float)attributes + 0.00001;

    int size_of_batch = rules_size*(max_size+1);
    int rule_offset = rules_size*counterB*(max_size+1);
    int rule_idx = blockIdx.x;
    int iterations = number_of_nodes/(blockDim.x+0.0001)+1;

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x * iter + threadIdx.x;
        if(node_idx < number_of_nodes){ 
               for(int i=transient_offset;i<steps;i++){
                   int index = (int) (dtep[node_idx+number_of_nodes*i+number_of_nodes*steps*rule_idx]/division);
                   atomicAdd(&histogram[index+(max_size+1)*rule_idx+counterB*size_of_batch],1);
               }
        }
    }

    __syncthreads();

}

__global__ void density_histogram_selection(float* dtep, int steps, int number_of_nodes,
int rules_size, int* histogram, int attributes, int transient_offset){


    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float division = (float)1/(float)attributes + 0.00001;

    int rule_idx = blockIdx.x;
    int iterations = number_of_nodes/(blockDim.x+0.0001)+1;

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x * iter + threadIdx.x;
        if(node_idx < number_of_nodes){ 
               for(int i=transient_offset;i<steps;i++){
                   int index = (int) (dtep[node_idx+number_of_nodes*i+number_of_nodes*steps*rule_idx]/division);
                   atomicAdd(&histogram[index+attributes*rule_idx],1);
               }
        }
    }

    __syncthreads();

}

__global__ void density_state_histogram(float* dtep, bool* TEP, int steps, int number_of_nodes,
int rules_size, int* histogram, int attributes, int transient_offset){


    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float division = (float)2/(float)attributes + 0.00001;

    int size_of_batch = rules_size*attributes;

    int rule_idx = blockIdx.x;
    int iterations = number_of_nodes/(blockDim.x + 0.0001)+1;

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x * iter + threadIdx.x;
        if(node_idx < number_of_nodes){ 
               for(int i=transient_offset;i<steps;i++){
                   float value = dtep[node_idx+number_of_nodes*i+number_of_nodes*steps*rule_idx]*(2*TEP[node_idx+number_of_nodes*i+number_of_nodes*steps*rule_idx]-1);
                   int index = (int) ((value+1)/division);
                   atomicAdd(&histogram[index+attributes*rule_idx],1);
               }
        }
    }

    __syncthreads();

}


__global__ void energy_histogram(float* dtep, int steps, int number_of_nodes, int counterB,
int rules_size, int* histogram, int attributes, int transient_offset, int max_size){


    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float division = (float)1/(float)attributes + 0.00001;

    int size_of_batch = rules_size*(max_size+1);
    int rule_offset = rules_size*counterB*(max_size+1);
    int rule_idx = blockIdx.x;
    int iterations = number_of_nodes/(blockDim.x + 0.0001)+1;

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x * iter + threadIdx.x;
        if(node_idx < number_of_nodes){ 
                float energy=0;
                for(int i=transient_offset;i<steps;i++){
                  energy+=pow(dtep[node_idx+number_of_nodes*i+number_of_nodes*steps*rule_idx], 2)/(steps-transient_offset);
                   
               }
               int index = (int) (energy/division);
               atomicAdd(&histogram[index+(max_size+1)*rule_idx+counterB*size_of_batch],1);
        }
    }

    if(threadIdx.x == 0){
        histogram[max_size+(max_size+1)*rules_size*rule_idx] = attributes;
    }

    __syncthreads();

}

__global__ void mean_histogram(float* dtep, int steps, int number_of_nodes, int counterB,
int rules_size, int* histogram, int attributes, int transient_offset, int max_size){


    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float division = (float)1/(float)attributes + 0.00001;

    int size_of_batch = rules_size*(max_size+1);
    int rule_offset = rules_size*counterB*(max_size+1);
    int rule_idx = blockIdx.x;
    int iterations = number_of_nodes/(blockDim.x + 0.0001)+1;

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x * iter + threadIdx.x;
        if(node_idx < number_of_nodes){ 
                float sum=0;
               for(int i=transient_offset;i<steps;i++){
                    sum+=dtep[node_idx+number_of_nodes*i+number_of_nodes*steps*rule_idx]/(steps-transient_offset);
                   
               }
               int index = (int) (sum/division);
               atomicAdd(&histogram[index+(max_size+1)*rule_idx+counterB*size_of_batch],1);
        }
    }

    if(threadIdx.x == 0){
        histogram[max_size+(max_size+1)*rules_size*rule_idx] = attributes;
    }

    __syncthreads();

}

__global__ void contrast_histogram(float* dtep, int steps, int number_of_nodes, int counterB,
int rules_size, int* histogram, int attributes, int transient_offset, int max_size){


    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float division = (float)1/(float)attributes + 0.00001;

    int size_of_batch = rules_size*(max_size+1);
    int rule_offset = rules_size*counterB*(max_size+1);
    int rule_idx = blockIdx.x;
    int iterations = number_of_nodes/(blockDim.x + 0.0001)+1;

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x * iter + threadIdx.x;
        if(node_idx < number_of_nodes){ 
                float sum=0;
               for(int i=transient_offset;i<steps;i++){
                   sum+=dtep[node_idx+number_of_nodes*i+number_of_nodes*steps*rule_idx]*(i-transient_offset)*(i-transient_offset); //p(i)*i^2
                   
               }
               int index = (int) (sum/division);
               atomicAdd(&histogram[index+(max_size+1)*rule_idx+counterB*size_of_batch],1);
        }
    }

    if(threadIdx.x == 0){
        histogram[max_size+(max_size+1)*rules_size*rule_idx] = attributes;
    }

    __syncthreads();

}



// calculate word entropy histogram, this histogram has 20 indexes. Larger word length considered is 20 and every index in the
// histogram has 2 bins.
__global__ void word_entropy_histogram(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes, int max_word_length,  int* histogram, int transient_offset){
   
    int gid = blockIdx.x*blockDim.x+threadIdx.x;

    int size_of_batch = rules_size*attributes;
    int rule_offset = rules_size*counterB*attributes;
    int iterations = number_of_nodes/(blockDim.x+0.0001)+1;
    int rule_idx = blockIdx.x;


    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x*iter+threadIdx.x;
        if(node_idx < number_of_nodes){
            int word_size=0;
            for(int i=transient_offset;i<steps;i++){
                if(TEP[node_idx + i*number_of_nodes + rule_idx*number_of_nodes*steps] == false){
                    if(word_size > 0 && word_size <=max_word_length){
                        int index = word_size/(max_word_length/attributes+0.001);
                        atomicAdd(&histogram[index+rule_idx*attributes+rule_offset], 1);
                    }
                    word_size=0;
                }
                else{
                    word_size+=1;
                }
            }
        }
        __syncthreads();
    }
    

}

//pick the sum_of_states array to make the two point correlation. We pick values for r between 1 and 20, we paralelize it for each thread to calculate every step
__global__ void tp_correlation(bool* TEP, int steps, int number_of_nodes, int counterB, int rules_size, int attributes, float* correlation){
    
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    int size_of_batch = rules_size*counterB*attributes;
    int step_idx = gid % steps;
    int rule_idx = gid / steps;

    if(gid < rules_size*steps){
        for(int r = 1; r<= attributes;r++){
            int counter=0;
            int prod = 0;
            int sum_m = 0;
            int sum_r = 0;
            for(int i=0;i<number_of_nodes-r;i++){
                int value_m;
                int value_r;
                if(TEP[i + step_idx*number_of_nodes + rule_idx*number_of_nodes*steps]){
                    value_m = 1;
                }
                else{
                    value_m = -1;
                }
                if(TEP[i+r + step_idx*number_of_nodes + rule_idx*number_of_nodes*steps]){
                    value_r = 1;
                }
                else{
                    value_r = -1;
                }

                sum_m+=value_m;
                sum_r+=value_r;
                prod+=value_m*value_r;
                counter++;
            }
            float mean_prod = (float) prod/(float)counter;
            float mean_m = (float) sum_m/(float)counter;
            float mean_r = (float)sum_r/(float)counter;

            float value = (mean_prod - mean_m*mean_r)/steps;

            atomicAdd(&correlation[r-1 + rule_idx*attributes + size_of_batch], value);    
        }     
        __syncthreads();  
    }
}

void lz_complexity_cpu(bool* TEP, int steps, int number_of_nodes, int rules_size, int* lz, int transient_offset){

    for(int rule_idx=0; rule_idx<rules_size;rule_idx++){
        for(int node_idx=0;node_idx<number_of_nodes;node_idx++){
            
            int* seq = (int*) malloc(sizeof(int)*steps);
            int* sizes = (int*) malloc(sizeof(int)*steps);
            if(TEP[node_idx + rule_idx*number_of_nodes*steps]){
                seq[0] = 1;
            }
            else{
                seq[0] = 0;
            }
            sizes[0] = 1;
            int complexity=1;
            int size=1;
            int index=transient_offset;
            int length = steps;
            while(index + size < length){
                long decimal=0;
                for(int i=0;i<size;i++){
                    if(TEP[node_idx+number_of_nodes*(index+i)+rules_size*number_of_nodes*steps]){
                        int multiplier =1;       
                        for(int m=0;m<i;m++){
                            multiplier*=2;
                        }
                        decimal+=multiplier;
                    }    
                }
                //printf(" %d %d %d %d\n", index, size, decimal, complexity);
                int previous_size=size;
                for(int i=0;i<complexity;i++){
                    if(seq[i] == decimal && sizes[i] == size){
                        size++;
                        break;
                    }
                }
                if(previous_size == size){
                    seq[complexity] = decimal;
                    sizes[complexity] = size;
                    complexity++;
                    index+=size;
                    size=1;
                    
                }
                
            }
            lz[node_idx + rule_idx*number_of_nodes] = complexity;

            free(seq);
            free(sizes);
         }
    }
}

__global__ void lz_test(bool* TEP, int steps, int number_of_nodes, int rules_size, int* lz){

    int gid = blockDim.x*blockIdx.x + threadIdx.x;

    int iterations = number_of_nodes/(blockDim.x+0.0001)+1;
    int rule_idx = blockIdx.x;

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x*iter+threadIdx.x;
        if(node_idx < number_of_nodes){

            node* seq = new node;
            seq->next = NULL;
            seq->vertex = TEP[node_idx+rules_size*number_of_nodes*steps];
            int complexity=1;
            int size=1;
            int index=1;
            int length = steps;
            while(index < length){
                long decimal=0;
                for(int i=0;i<size;i++){
                    decimal+= pow(2,(size+1))*TEP[node_idx+number_of_nodes*(index+i)+rules_size*number_of_nodes*steps];
                }
                printf(" %d %d %ld %d", index, size, decimal, complexity);
                node* p = new node;
                p = seq->next;
                int previous_size=size;
                while(p){
                    if(p->vertex == decimal){
                        size++;
                        break;
                    }
                    p = p->next;
                }
                if(previous_size == size){
                    complexity++;
                    index++;
                    size=1;
                    p->next=NULL;
                    p->vertex = decimal;
                }
            }
            lz[node_idx + rule_idx*number_of_nodes] = complexity;

        }
        
    }
    __syncthreads();
}


__global__ void lz_complexity_2(bool* TEP, int steps, int number_of_nodes, int rules_size, int* lz){

    int gid = blockDim.x*blockIdx.x + threadIdx.x;

    int iterations = number_of_nodes/(blockDim.x+0.0001)+1;
    int rule_idx = blockIdx.x;

    for(int iter=0;iter<iterations;iter++){
        int node_idx = blockDim.x*iter+threadIdx.x;
        if(node_idx < number_of_nodes){

            int u = 0;
            int v = 1;
            int w = 1;
            int v_max = 1;
            int length = steps;
            int complexity = 1;
            while(1){
                 if(TEP[node_idx+(u+v-1)*number_of_nodes+rule_idx*number_of_nodes*steps] == TEP[node_idx+(w+v-1)*number_of_nodes+rule_idx*number_of_nodes*steps]){
                     v+=1;
                     if(w+v>=length){
                         complexity++;
                         break;
                     }
                }
                 else{
                    if(v > v_max){
                        v_max = v;
                    }
                    u++;
                    if(u==w){
                        complexity++;
                        w += v_max;
                        if(w > length){
                            break;
                        }
                        else{
                            u=0;
                            v=1;
                            v_max = 1;
                        }
                    }
                    else{
                        v=1;
                    }
                }
            }
            lz[node_idx + number_of_nodes*rule_idx] = complexity;
        }
    }
    __syncthreads();

}