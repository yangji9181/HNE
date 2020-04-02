#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <Eigen/Dense>
#include "ransampl.h"
#include <iostream>

#define MAX_STRING 500
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
const int neg_table_size = 1e8;
const int hash_table_size = 30000000;

typedef float real;

typedef Eigen::Matrix< real, Eigen::Dynamic,
Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >
BLPMatrix;

typedef Eigen::Matrix< real, 1, Eigen::Dynamic,
Eigen::RowMajor | Eigen::AutoAlign >
BLPVector;

struct struct_node {
    char *word;
    char type;
};

class line_node
{
protected:
    struct struct_node *node;
    int node_size, node_max_size, vector_size;
    char node_file[MAX_STRING], link_file[MAX_STRING];
    int *node_hash;
    real *_vec;
    Eigen::Map<BLPMatrix> vec;
    std::vector<int> *hin;
    long long hin_size;
    
    int *edge_u, *edge_v;
    
    int get_hash(char *word);
    int add_node(char *word, char type);
public:
    line_node();
    ~line_node();
    
    friend class line_link;
    
    void init(char *file_name, int vector_dim);
    void init_hin(char *file_name);
    int search(char *word);
    void output(char *file_name, int binary);
};

class line_map
{
protected:
    std::string type;
    int vector_size;
    real *_S;
    real *_P, *_Q;
    real bias;
    Eigen::Map<BLPMatrix> S;
    Eigen::Map<BLPVector> P, Q;
public:
    line_map();
    ~line_map();
    
    friend class line_link;
    
    void init(std::string map_type, int vector_dim);
};

class line_link
{
protected:
    line_node *node_u, *node_v;
    real *expTable;
    int neg_samples;
    double **dp_cnt, **dp_cnt_fd;
    int **neg_table;
    ransampl_ws *smp_init;
    double *smp_init_weight;
    int *smp_init_index;
    ransampl_ws ***smp_dp;
    double ***smp_dp_weight;
    int ***smp_dp_index;
    std::vector<line_map *> pmap;
    std::map<std::string, int> tp2mid;
    
    std::string path;
    int path_size;
public:
    line_link();
    ~line_link();
    
    void sample_path(int *node_lst, double (*func_rand_num)());
    void init(std::string meta_path, line_node *p_u, line_node *p_v, int negative);
    void init_map(std::vector<line_map *> pointer_map);
    void train_path(int *node_lst, real *_error_vec, real *_error_p, real *_error_q, real alpha, double (*func_rand_num)(), unsigned long long &rand_index, int model, int mode = 0);
    int get_path_length();
    double eval(int *node_lst, double (*func_rand_num)(), int num_path);
};