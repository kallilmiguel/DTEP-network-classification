#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include<dirent.h>
#include"lib/graph.h"


#include "cuda.h"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"lib/cuda_common.cuh"
#include"lib/tep.cuh"

int main(int argc, char* argv[]){

    char* graph_name = argv[1];
    const char* data_path = argv[2];
    const char* rule_path = argv[3];
    const char* output_path = argv[4];
    const char* seed_path = argv[5];

    printf("%s\n", graph_name);
    printf("%s\n", data_path);
    printf("%s\n", rule_path);
    printf("%s\n", output_path);
    printf("%s\n", seed_path);

    int MAX_NODES = 12000;
    int steps=350;

    FILE* fseed = fopen(seed_path, "r");
    bool* init_state = (bool*) malloc(sizeof(bool)*MAX_NODES);
    for(int i=0;i<MAX_NODES;i++){
        fscanf(fseed, "%d\n", &init_state[i]);
    }
    
    printf("Iniciando grafo: %s\n", graph_name);

    char *file_path = (char*) malloc(sizeof(char)*200);
    strcpy(file_path,data_path);
    strcat(file_path, graph_name);
    
    Graph* G = read_adjList(file_path);
    
    printf("Nodes: %d, edges: %d\n", G->numVertices, G->numTransitions);
    //generate_dteps_selected(G, steps, dir->d_name, init_state, rule_path, output_path);
    generate_dteps_selected_statistics(G, steps, graph_name, init_state, rule_path, output_path);

    free(file_path);
    //freeGraph(G);
    free(G);

    //free(init_state);


    return 0;
}
