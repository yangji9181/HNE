#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "linelib.h"
#include "ransampl.h"

#define MAX_PATH_LENGTH 100

char node_file[MAX_STRING], link_file[MAX_STRING], path_file[MAX_STRING], output_file[MAX_STRING];
int binary = 0, num_threads = 1, vector_size = 100, negative = 5, iters = 10, epoch, num_paths = 0;
long long samples = 1, edge_count_actual;
real alpha = 0.025, starting_alpha;

int mode, model = 2;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

line_node node0, node1;
line_link *plink;
line_map *cur_map;
std::vector<line_map *> pmap;
std::map<std::string, int> tp2flag;
double *path_weights;

ransampl_ws *smp_path;

double func_rand_num()
{
    return gsl_rng_uniform(gsl_r);
}

void *training_thread(void *id)
{
    long long edge_count = 0, last_edge_count = 0;
    unsigned long long next_random = (long long)id;
    real *error_vec = (real *)calloc(vector_size, sizeof(real));
    real *error_p = (real *)calloc(vector_size, sizeof(real));
    real *error_q = (real *)calloc(vector_size, sizeof(real));
    int *node_lst = (int *)malloc(MAX_PATH_LENGTH * sizeof(int));
    int path_id = 0;
    
    while (1)
    {
        //judge for exit
        if (edge_count > samples / num_threads + 2) break;
        
        if (edge_count - last_edge_count > 1000)
        {
            edge_count_actual += edge_count - last_edge_count;
            last_edge_count = edge_count;
            printf("%cEpoch: %d/%d Alpha: %f Progress: %.3lf%%", 13, epoch + 1, iters, alpha, (real)edge_count_actual / (real)(samples + 1) * 100);
            fflush(stdout);
            alpha = starting_alpha * (1 - edge_count_actual / (real)(samples * iters+ 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        path_id = (int)(ransampl_draw(smp_path, gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r)));
        
        plink[path_id].train_path(node_lst, error_vec, error_p, error_q, alpha, func_rand_num, next_random, model, mode);
        edge_count++;
    }
    free(node_lst);
    free(error_vec);
    free(error_p);
    free(error_q);
    pthread_exit(NULL);
}

void TrainModel() {
    long a;
    FILE *fi;
    char cpath[MAX_STRING];
    std::string path, tp;
    int path_length;
    double cur_weight;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    starting_alpha = alpha;
    
    printf("Parameters: \n");
    printf("-----------------------------------\n");
    printf("Model: %d\n", model);
    printf("Vector Size: %d\n", vector_size);
    printf("Iterations: %d\n", iters);
    printf("Samples Per Iteration: %.2lfM\n", samples / 1000000.0);
    printf("Negative Samples: %d\n", negative);
    printf("Inital Learning Rate: %f\n", alpha);
    printf("-----------------------------------\n");
    
    fi = fopen(path_file, "rb");
    num_paths = 0;
    while (fscanf(fi, "%s %lf", cpath, &cur_weight) == 2)
    {
        printf("%s\t%lf\n", cpath, cur_weight);
        num_paths++;
    }
    printf("-----------------------------------\n");
    fclose(fi);
    
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    
    node0.init(node_file, vector_size);
    node0.init_hin(link_file);
    
    path_weights = (double *)malloc(num_paths * sizeof(double));
    plink = new line_link [num_paths];
    
    fi = fopen(path_file, "rb");
    pmap.clear();
    for (int k = 0; k != num_paths; k++)
    {
        fscanf(fi, "%s %lf", cpath, &cur_weight);
        path = cpath;
        path_length = strlen(cpath);
        for (int i = 0; i < path_length; i++) for (int j = i + 1; j < path_length; j++)
        {
            tp = path.substr(i, j - i + 1);
            if (path[i] > path[j]) for (int k = i; k <= j; k++)
                tp[k - i] = path[j + i - k];
            if (tp2flag[tp] == 1) continue;
            
            tp2flag[tp] = 1;
            cur_map = new line_map;
            cur_map->init(tp, vector_size);
            pmap.push_back(cur_map);
        }
        
        plink[k].init(path, &node0, &node0, negative);
        path_weights[k] = cur_weight;
    }
    fclose(fi);
    
    for (int k = 0; k != num_paths; k++) plink[k].init_map(pmap);
    smp_path = ransampl_alloc(num_paths);
    ransampl_set(smp_path, path_weights);
    
    clock_t start = clock();
    printf("Training:");
    for (epoch = 0; epoch != iters; epoch++)
    {
        edge_count_actual = samples * epoch;
        mode = 0;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    }
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    
    node0.output(output_file, binary);
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

void write_para() {
    FILE *writef = fopen(output_file,"w");
    fprintf(writef, "size=%d, ", vector_size);
    fprintf(writef, "negative=%d, ", negative);
    fprintf(writef,"samples=%.2lfM, ", samples / 1000000.0);
    fprintf(writef,"iters=%d, ", iters);
    fprintf(writef,"alpha=%f", alpha);
    fprintf(writef, "\n");
    fclose(writef);
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("metapath2vec-ESim\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-node <file>\n");
        printf("\t\tA dictionary of all nodes\n");
        printf("\t-link <file>\n");
        printf("\t\tAll links between nodes. Links are directed.\n");
        printf("\t-path <int>\n");
        printf("\t\tAll meta-paths. One path per line.\n");
        printf("\t-output <int>\n");
        printf("\t\tThe output file.\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million\n");
        printf("\t-iters <int>\n");
        printf("\t\tSet the number of interations.\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./esim -node node.txt -link link.txt -path path.txt -output vec.emb -binary 1 -size 100 -negative 5 -samples 5 -iters 20 -threads 12\n\n");
        return 0;
    }
    output_file[0] = 0;
    if ((i = ArgPos((char *)"-node", argc, argv)) > 0) strcpy(node_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-link", argc, argv)) > 0) strcpy(link_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-path", argc, argv)) > 0) strcpy(path_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-emb", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = (long long)(atof(argv[i + 1])*1000000);
    if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) iters = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    write_para();
    TrainModel();
    return 0;
}