#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "linelib.h"

char str_center_node[MAX_STRING], str_attribute_nodes[MAX_STRING], str_attribute_edges[MAX_STRING], node_file[MAX_STRING], hin_file[MAX_STRING], output_file[MAX_STRING];

std::vector<int> center_nodes, attribute_nodes, attribute_edges;

int binary = 0, num_threads = 1, vector_size = 100, negative = 5;
long long samples = 1, edge_count_actual;
real alpha = 0.025, starting_alpha;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

line_node attribute, center;
line_hin star_hin;
std::vector<line_trainer> trainer_array;

double func_rand_num()
{
	return gsl_rng_uniform(gsl_r);
}

void *TrainModelThread(void *id) 
{
	long long edge_count = 0, last_edge_count = 0;
	unsigned long long next_random = (long long)id;
	real *error_vec = (real *)calloc(vector_size, sizeof(real));

	while (1)
	{
		if (edge_count > samples / num_threads + 2) break;

		if (edge_count - last_edge_count>10000)
		{
			edge_count_actual += edge_count - last_edge_count;
			last_edge_count = edge_count;
			printf("%cAlpha: %f Progress: %.3lf%%", 13, alpha, (real)edge_count_actual / (real)(samples + 1) * 100);
			fflush(stdout);
			alpha = starting_alpha * (1 - edge_count_actual / (real)(samples + 1));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}
        for (int i=0; i<(int)trainer_array.size(); i++) {trainer_array[i].train_sample(alpha, error_vec, func_rand_num, next_random);}

		edge_count += 3;
	}
	free(error_vec);
	pthread_exit(NULL);
}

void TrainModel() {
	long a;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	starting_alpha = alpha;

	attribute.init(node_file, attribute_nodes, vector_size);
	center.init(node_file, center_nodes, vector_size);
	star_hin.init(hin_file, &center, &attribute);

	printf("Learning embedding considering attribute node types: %s; attribute edge types: %s\n", str_attribute_nodes, str_attribute_edges);

   for (int i=0; i<(int)trainer_array.size(); i++) {trainer_array[i].init(attribute_edges[i], &star_hin, negative);}

	clock_t start = clock();
	printf("Training process:\n");
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
   
   center.output(output_file, binary);
	attribute.output(output_file, binary);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

void parse_para() {
    int edge_type_count=0;
    std::string substri;
    std::stringstream temp0(str_center_node), temp1(str_attribute_nodes), temp2(str_attribute_edges);
    while (std::getline(temp0, substri, ',')) {
        center_nodes.push_back(atoi(substri.c_str()));
    }
    while (std::getline(temp1, substri, ',')) {
        attribute_nodes.push_back(atoi(substri.c_str()));
    }
    while (std::getline(temp2, substri, ',')) {
        attribute_edges.push_back(atoi(substri.c_str()));
        edge_type_count++;
    }
    trainer_array.resize(edge_type_count);
}

void write_para() {
    remove(output_file);
    FILE *writef = fopen(output_file,"w");
    fprintf(writef, "size=%d, ", vector_size);
    fprintf(writef, "negative=%d, ", negative);
    fprintf(writef, "samples=%.2lfM, ", samples / 1000000.0);
    fprintf(writef, "alpha=%f", alpha);
    fprintf(writef, "\n");
    
    fprintf(writef, "Attribute node types: %s; Attribute edge types: %s\n", str_attribute_nodes, str_attribute_edges);
    
    fclose(writef);
}


int main(int argc, char **argv) {

    int i;
	if ((i = ArgPos((char *)"-center", argc, argv)) > 0) strcpy(str_center_node, argv[i + 1]);  
	if ((i = ArgPos((char *)"-attribute", argc, argv)) > 0) strcpy(str_attribute_nodes, argv[i + 1]);      
	if ((i = ArgPos((char *)"-edges", argc, argv)) > 0) strcpy(str_attribute_edges, argv[i + 1]);  
	if ((i = ArgPos((char *)"-node", argc, argv)) > 0) strcpy(node_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-hin", argc, argv)) > 0) strcpy(hin_file, argv[i + 1]);    
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = atoi(argv[i + 1])*(long long)(1000000);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    

	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);
    
   parse_para();
   write_para();
   
	TrainModel();
	return 0;
}