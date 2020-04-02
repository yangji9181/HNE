#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <iostream>

#define MAX_STRING 10000
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
};

struct hin_nb {
	int nb_id;
	double eg_wei;
	int eg_tp;
};

class sampler
{
	long long n;
	long long *alias;
	double *prob;

public:
	sampler();
	~sampler();

	void init(long long ndata, double *p);
	long long draw(double ran1, double ran2);
};

class line_node
{
protected:
	struct struct_node *node;
	int node_size, node_max_size, vector_size;
	char node_file[MAX_STRING];
	int *node_hash;
	real *_vec;
	Eigen::Map<BLPMatrix> vec;

	int get_hash(char *word);
	int add_node(char *word);
public:
	line_node();
	~line_node();

	friend class line_hin;
	friend class line_trainer;

	void init(char *file_name, int vector_dim);
	int search(char *word);
	void output(char *file_name, int binary);
};

class line_hin
{
protected:
	char hin_file[MAX_STRING];

	line_node *node_u, *node_v;
	std::vector<hin_nb> *hin;
	long long hin_size;

public:
	line_hin();
	~line_hin();

	friend class line_trainer;

	void init(char *file_name, line_node *p_u, line_node *p_v);
};

class line_trainer
{
protected:
	line_hin *phin;

	int *u_nb_cnt; int **u_nb_id; double **u_nb_wei;
	double *u_wei, *v_wei;
	sampler smp_u, *smp_u_nb;
	real *expTable;
	int neg_samples, *neg_table;

	int edge_tp;
public:
	line_trainer();
	~line_trainer();

	void init(int edge_type, line_hin *p_hin, int negative);
	void train_sample(real alpha, real *_error_vec, double(*func_rand_num)(), unsigned long long &rand_index);
};
