#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

void save_csv_int(char* filename, const char* output_folder, int* array, int values_per_line, int number_of_lines){
    FILE *fout;

    char* pout = (char*) malloc(sizeof(char)*100);
    char* file_wo_ext = (char*)malloc(sizeof(char)*100);
    char* ext = ".csv";

    snprintf(file_wo_ext, strlen(filename)-3, "%s", filename);
    strcat(file_wo_ext, ext);
    strcpy(pout, output_folder);
    strcat(pout, file_wo_ext);

    fout = fopen(pout, "w");
    printf("Save name: %s\n", pout);

    for(int i=0;i<values_per_line*number_of_lines;i++){
        if((i+1)%values_per_line == 0){
            fprintf(fout, "%d\n", array[i]);
        }
        else{
            fprintf(fout, "%d,", array[i]);
        }
    }

    fclose(fout);
    free(pout);
    free(file_wo_ext);
}

void save_csv_TEPs(const char* filename, const char* output_folder, bool* TEP, int* degree, int values_per_line, int number_of_lines, int transient_offset){
    FILE *fout;

    char* pout = (char*) malloc(sizeof(char)*100);
    char* file_wo_ext = (char*)malloc(sizeof(char)*100);
    char* ext = ".csv";

    snprintf(file_wo_ext, strlen(filename)-3, "%s", filename);
    strcat(file_wo_ext, ext);
    strcpy(pout, output_folder);
    strcat(pout, file_wo_ext);

    fout = fopen(pout, "w");
    printf("Save name: %s\n", pout);

    for(int i=0;i<values_per_line;i++){
        if((i+1)%values_per_line == 0){
            fprintf(fout, "%d\n", degree[i]);
        }
        else{
            fprintf(fout, "%d,", degree[i]);
        }
    }

    for(int i=transient_offset*values_per_line;i<values_per_line*number_of_lines;i++){
        if((i+1)%values_per_line == 0){
            fprintf(fout, "%d\n", TEP[i]);
        }
        else{
            fprintf(fout, "%d,", TEP[i]);
        }
    }

    fclose(fout);
    free(pout);
    free(file_wo_ext);
}

void save_csv_dTEPs(const char* filename, const char* output_folder, float* TEP, int* degree, int values_per_line, int number_of_lines, int transient_offset){
    FILE *fout;

    char* pout = (char*) malloc(sizeof(char)*100);
    char* file_wo_ext = (char*)malloc(sizeof(char)*100);
    char* ext = ".csv";

    snprintf(file_wo_ext, strlen(filename)-3, "%s", filename);
    strcat(file_wo_ext, ext);
    strcpy(pout, output_folder);
    strcat(pout, file_wo_ext);

    fout = fopen(pout, "w");
    printf("Save name: %s\n", pout);

    for(int i=0;i<values_per_line;i++){
        if((i+1)%values_per_line == 0){
            fprintf(fout, "%d\n", degree[i]);
        }
        else{
            fprintf(fout, "%d,", degree[i]);
        }
    }

    for(int i=transient_offset*values_per_line;i<values_per_line*number_of_lines;i++){
        if((i+1)%values_per_line == 0){
            fprintf(fout, "%.4f\n", TEP[i]);
        }
        else{
            fprintf(fout, "%.4f,", TEP[i]);
        }
    }

    fclose(fout);
    free(pout);
    free(file_wo_ext);
}

void save_csv_float(const char* filename, const char* output_folder, float* array, int values_per_line, int number_of_lines){
    FILE *fout;

    char* pout = (char*) malloc(sizeof(char)*100);
    char* file_wo_ext = (char*)malloc(sizeof(char)*100);
    char* ext = ".csv";

    snprintf(file_wo_ext, strlen(filename)-3, "%s", filename);
    strcat(file_wo_ext, ext);
    strcpy(pout, output_folder);
    strcat(pout, file_wo_ext);

    fout = fopen(pout, "w");
    printf("Save name: %s\n", pout);

    for(int i=0;i<values_per_line*number_of_lines;i++){
        if((i+1)%values_per_line == 0){
            fprintf(fout, "%.4f\n", array[i]);
        }
        else{
            fprintf(fout, "%.4f,", array[i]);
        }
    }

    fclose(fout);
    free(pout);
    free(file_wo_ext);
}

void save_csv_bool(char* filename, const char* output_folder, bool* array, int values_per_line, int number_of_lines){
    FILE *fout;

    char* pout = (char*) malloc(sizeof(char)*100);
    char* file_wo_ext = (char*)malloc(sizeof(char)*100);
    char* ext = ".csv";

    snprintf(file_wo_ext, strlen(filename)-3, "%s", filename);
    strcat(file_wo_ext, ext);
    strcpy(pout, output_folder);
    strcat(pout, file_wo_ext);

    fout = fopen(pout, "w");
    printf("Save name: %s\n", pout);

    for(int i=0;i<values_per_line*number_of_lines;i++){
        if((i+1)%values_per_line == 0){
            fprintf(fout, "%d\n", array[i]);
        }
        else{
            fprintf(fout, "%d,", array[i]);
        }
    }

    fclose(fout);
    free(pout);
    free(file_wo_ext);
}
