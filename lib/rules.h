#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

#define NB_SIZE 8

typedef struct {
    bool rule[NB_SIZE + 1];
}rules;

typedef struct{
    bool bRule[NB_SIZE+1];
    bool sRule[NB_SIZE+1];
}lifelike;


lifelike* getSelectedRules(const char* path_to_rule, int* nr){
    FILE *ruleFile;
    
    ruleFile = fopen(path_to_rule, "r");

    char* str = (char*) malloc(sizeof(char)*30);
    int number_of_rules = 0;            
    while(fgets(str, 30, ruleFile)!= NULL){
        number_of_rules++;
    }
    lifelike* selectedRules = (lifelike*)malloc(sizeof(lifelike)*number_of_rules);

    rewind(ruleFile);
    for(int i=0;i<number_of_rules;i++){
        for (int j = 0; j < NB_SIZE + 1; j++) {
            selectedRules[i].bRule[j] = false;
            selectedRules[i].sRule[j] = false;
        }
        fgets(str, 30, ruleFile);    
        char readingRule;
        int counter=0;
        while(1){
            char c = str[counter];
            if(c == '\0'){
                break;
            }
            else if(c == 'B' || c == 'S'){
                readingRule = c;
            }
            else{
                if(readingRule == 'B'){
                    selectedRules[i].bRule[(int)c - 48] = true; 
                }
                else if(readingRule == 'S'){
                    selectedRules[i].sRule[(int)c - 48] = true; 
                }
               
            }
            counter++;
        }
    }
    *nr = number_of_rules;
    return selectedRules;
}

rules* getAllRules() {
    int number_of_rules = 1;
    for (int i = 0; i < NB_SIZE + 1; i++) {
        number_of_rules *= 2;
    }

    rules* allRules = (rules*)malloc(sizeof(rules) * number_of_rules);

    int counter = 0;

    FILE* ruleFile;
    const char* rulePath = "data/rules/allRules.txt";

    ruleFile = fopen(rulePath, "r");
    
    for (int i = 0; i < NB_SIZE + 1; i++) {
        allRules[counter].rule[i] = false;
    }
    counter++;
    
    while (1) {

        char c = fgetc(ruleFile);
        if (c == EOF) {
            break;
        }
        else if (c == '\n') {
            counter++;
            for (int i = 0; i < NB_SIZE + 1; i++) {
                allRules[counter].rule[i] = false;
            }
        }
        else if (c != ',' && c != ' ') {
            allRules[counter].rule[(int)c - 48] = true;
        }
    }
    return allRules;
}