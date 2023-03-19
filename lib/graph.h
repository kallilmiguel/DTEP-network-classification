#include<stdio.h>
#include<stdlib.h>
#include<dirent.h>
#include<string.h>
#include<math.h>
#include<ctype.h>

typedef struct node{
    int vertex;
    struct node* next;
}node;

typedef struct{
    int numVertices;
    int numTransitions;
    node** adjLists;
    int* degrees;
}Graph;

typedef struct{
    int* neighbors;
    int* indexes;
}arr_adjList;

node* createNode(int v);
Graph* createGraph(int vertices);
void addEdge(Graph* graph, int source, int destiny);
const char *get_filename_ext(const char *filename);
Graph* read_adjList(char* path);
void printGraph(Graph* G, int n_of_iterations);
arr_adjList* convert_adjList_to_array(Graph* G);


//Create a node
node* createNode(int v){
    node *newNode = (node*) malloc(sizeof(node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

//Create a graph
Graph* createGraph(int vertices){
    Graph *graph = (Graph*) malloc(sizeof(Graph));
    graph->numVertices = vertices;
    graph->numTransitions = 0;

    graph->adjLists = (node**) malloc(vertices * sizeof(node*));
    graph->degrees = (int*) malloc(vertices*sizeof(int));

    int i;
    for(i=0; i<vertices;i++){
        graph->adjLists[i] = NULL;
        graph->degrees[i] = 0;
    }

    return graph;
}

bool hasEdge(Graph* G, int source, int destiny){
    node* p = G->adjLists[source];
    while(p){
        if(p->vertex == destiny){
            return true;
        }
        p = p->next;
    }
    return false;
}

//Add edge (in bidirectional graph)
void addEdge(Graph* graph, int source, int destiny){
    // if(!hasEdge(graph, source, destiny)){
        //add edge from s to d
        node* newNode = createNode(destiny);
        newNode->next = graph->adjLists[source];
        graph->adjLists[source] = newNode;
        graph->degrees[source]++;
        
        //add edge from d to s
        // node* newNode2 = createNode(source);
        // newNode2->next = graph->adjLists[destiny];
        // graph->adjLists[destiny] = newNode2;

        graph->numTransitions++;
    // }
}

void freeGraph(Graph* graph){
    for(int i=0;i<graph->numVertices;i++){
        node*p = graph->adjLists[i];
        while(p){
            node *a = p;
            p = p->next;
            free(a);
        }
    }
    free(graph);
    return;
}

//print the graph
void printGraph(Graph* graph, int n_of_iterations)
{
    int v;
    for (v = 0; v < n_of_iterations; v++) {
        node* temp = graph->adjLists[v];
        printf("\nVertex %d\n ", v);
        while (temp) {
            printf("%d -> ", temp->vertex);
            temp = temp->next;
        }
        printf("\n");
    }
}

Graph* read_adjList(char* path){

    
    FILE *f =fopen(path, "r");

    int counter = 0;
    int lines_until_vertex_read=0;

    while(1){
        char c = fgetc(f);
        if(isdigit(c)){
            break;
        }
        else{
            while(fgetc(f) != '\n');
            lines_until_vertex_read++;
        }
    }

    rewind(f);

    while(1){
        char c = fgetc(f);
        if(c == EOF){
            break;
        }
        else if(c == '\n'){
            counter++;
        }
        else if(c == '#'){
            while(c != '\n' && c != EOF){
                c = fgetc(f);
            }
        }
    }

    Graph *G = createGraph(counter-lines_until_vertex_read);

    rewind(f);

    char* str = (char*) malloc(sizeof(char)*10000);

    for(int i=0;i<lines_until_vertex_read;i++){
        fgets(str, 10000, f);
    }

    while(fgets(str,10000, f)!= NULL){
        int vertex;
        if(str[0] != '#'){
            sscanf(str,"%d ", &vertex);
            int edge;
            int char_offset;
            int size = strlen(str);
            int previous=-1;
            if(vertex >1){
                char_offset = 1+ceil(log10((double)vertex+0.001));
            }
            else{
                char_offset = 2;
            }
            //printf("\n %d ->", vertex);
            while(sscanf(&str[char_offset], "%d \n", &edge)!= NULL && char_offset< size){
                if(previous == edge){
                    break;
                }
                else if(edge > 1){
                    char_offset += 1+ceil(log10((double)edge+0.001)); 
                }
                else{
                    char_offset +=2;
                }
                previous = edge;
                addEdge(G, vertex, edge);
                addEdge(G, edge, vertex);
            }
        }
    }

    fclose(f);
    free(str);
    return G;
}

arr_adjList* convert_adjList_to_array(Graph* G){

    int number_of_nodes = G->numVertices;

    int transition_counter=0;
    
    for(int i=0;i<number_of_nodes;i++){
        node *p = G->adjLists[i];
        while(p){
            transition_counter++;
            p = p->next;
        }
    }

    arr_adjList* adjList;
    adjList = (arr_adjList*) malloc(sizeof(arr_adjList));

    adjList->indexes = (int*)malloc(sizeof(int)*number_of_nodes);
    adjList->neighbors = (int*)malloc(sizeof(int)*transition_counter);

    for(int i=0; i<transition_counter;i++){
        node *p = G->adjLists[i];
        while(p){
            adjList->neighbors[i] = p->vertex;
            p = p->next;
            i++;
        }
    }

    return adjList;
}

const char *get_filename_ext(const char *filename){
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}

bool* set_max_degree_true(Graph* G){
    
    int num_nodes = G->numVertices;
    
    bool* init_state = (bool*) malloc(sizeof(bool)*num_nodes);
    
    int max = 0;
    for(int i=0;i<num_nodes;i++){
        if (G->degrees[i] > max){
            max = G->degrees[i];
        }
    }
    for(int i=0;i<num_nodes;i++){
        if (G->degrees[i] == max){
            init_state[i] = true;
        }
        else{
            init_state[i] = false;
        }
    }
    return init_state;
}

bool* set_max_degree_false(Graph* G){
    
    int num_nodes = G->numVertices;
    
    bool* init_state = (bool*) malloc(sizeof(bool)*num_nodes);
    
    int max = 0;
    for(int i=0;i<num_nodes;i++){
        if (G->degrees[i] > max){
            max = G->degrees[i];
        }
    }
    for(int i=0;i<num_nodes;i++){
        if (G->degrees[i] == max){
            init_state[i] = false;
        }
        else{
            init_state[i] = true;
        }
    }
    return init_state;
}

bool* set_above_mean_degree_true(Graph* G){
    
    int num_nodes = G->numVertices;
    
    bool* init_state = (bool*) malloc(sizeof(bool)*num_nodes);
    
    float mean;
    float sum = 0;

    for(int i=0;i<num_nodes;i++){
        sum += G->degrees[i];
    }
    mean = (float) sum / (float) num_nodes;
    for(int i=0;i<num_nodes;i++){
        if (G->degrees[i] >= mean){
            init_state[i] = true;
        }
        else{
            init_state[i] = false;
        }
    }
    return init_state;
}

bool* set_below_mean_degree_true(Graph* G){
    
    int num_nodes = G->numVertices;
    
    bool* init_state = (bool*) malloc(sizeof(bool)*num_nodes);
    
    float mean;
    float sum = 0;

    for(int i=0;i<num_nodes;i++){
        sum += G->degrees[i];
    }
    mean = (float) sum / (float) num_nodes;
    for(int i=0;i<num_nodes;i++){
        if (G->degrees[i] >= mean){
            init_state[i] = false;
        }
        else{
            init_state[i] = true;
        }
    }
    return init_state;
}

bool* set_mean_degree_true(Graph* G){
    
    int num_nodes = G->numVertices;
    
    bool* init_state = (bool*) malloc(sizeof(bool)*num_nodes);
    
    float mean;
    float sum = 0;

    for(int i=0;i<num_nodes;i++){
        sum += G->degrees[i];
    }
    mean = (float) sum / (float) num_nodes;
    for(int i=0;i<num_nodes;i++){
        if (G->degrees[i] == mean){
            init_state[i] = true;
        }
        else{
            init_state[i] = false;
        }
    }
    return init_state;
}

bool* set_mean_degree_false(Graph* G){
    
    int num_nodes = G->numVertices;
    
    bool* init_state = (bool*) malloc(sizeof(bool)*num_nodes);
    
    float mean;
    float sum = 0;

    for(int i=0;i<num_nodes;i++){
        sum += G->degrees[i];
    }
    mean = (float) sum / (float) num_nodes;
    for(int i=0;i<num_nodes;i++){
        if (G->degrees[i] == mean){
            init_state[i] = false;
        }
        else{
            init_state[i] = true;
        }
    }
    return init_state;
}

bool* set_init_state(Graph* G, const char* policy){
    
    bool* init_state;

    printf("%s\n", policy);
    if(strcmp(policy, "max_true") == 0){
        printf("Set max degree true\n");
        init_state = set_max_degree_true(G);
    }
    else if(strcmp(policy, "max_false") == 0){
        printf("Set max degree false\n");
        init_state = set_max_degree_false(G);
    }
    else if(strcmp(policy, "mean_true") == 0){
        printf("Set mean degree true\n");
        init_state = set_mean_degree_true(G);
    }
    else if(strcmp(policy, "mean_false") == 0){
        printf("Set mean degree false\n");
        init_state = set_mean_degree_false(G);
    }
    else if(strcmp(policy, "above_mean_true") == 0){
        printf("Set above mean degree true\n");
        init_state = set_above_mean_degree_true(G);
    }
    else if(strcmp(policy, "below_mean_true") == 0){
        printf("Set below mean degree true\n");
        init_state = set_below_mean_degree_true(G);
    }
    else{
        printf("Generate random seed\n");
        init_state = (bool*) malloc(sizeof(bool)*G->numVertices);
        for(int i=0;i<G->numVertices;i++){
            init_state[i] = rand() & 1;
        }
    }

    return init_state;
}
