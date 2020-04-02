#include "linelib.h"

sampler::sampler()
{
	n = 0;
	alias = 0;
	prob = 0;
}

sampler::~sampler()
{
	n = 0;
	if (alias != NULL) { free(alias); alias = NULL; }
	if (prob != NULL) { free(prob); prob = NULL; }
}

void sampler::init(long long ndata, double *p)
{
	n = ndata;

	alias = (long long *)malloc(n * sizeof(long long));
	prob = (double *)malloc(n * sizeof(double));

	long long i, a, g;

	// Local workspace:
	double *P;
	long long *S, *L;
	P = (double *)malloc(n * sizeof(double));
	S = (long long *)malloc(n * sizeof(long long));
	L = (long long *)malloc(n * sizeof(long long));

	// Normalise given probabilities:
	double sum = 0;
	for (i = 0; i < n; ++i)
	{
		if (p[i] < 0)
		{
			fprintf(stderr, "ransampl: invalid probability p[%d]<0\n", (int)(i));
			exit(1);
		}
		sum += p[i];
	}
	if (!sum)
	{
		fprintf(stderr, "ransampl: no nonzero probability\n");
		exit(1);
	}
	for (i = 0; i < n; ++i) P[i] = p[i] * n / sum;

	// Set separate index lists for small and large probabilities:
	long long nS = 0, nL = 0;
	for (i = n - 1; i >= 0; --i)
	{
		// at variance from Schwarz, we revert the index order
		if (P[i] < 1)
			S[nS++] = i;
		else
			L[nL++] = i;
	}

	// Work through index lists
	while (nS && nL)
	{
		a = S[--nS]; // Schwarz's l
		g = L[--nL]; // Schwarz's g
		prob[a] = P[a];
		alias[a] = g;
		P[g] = P[g] + P[a] - 1;
		if (P[g] < 1)
			S[nS++] = g;
		else
			L[nL++] = g;
	}

	while (nL) prob[L[--nL]] = 1;

	while (nS) prob[S[--nS]] = 1;

	free(P);
	free(S);
	free(L);
}

long long sampler::draw(double ran1, double ran2)
{
	long long i = n * ran1;
	return ran2 < prob[i] ? i : alias[i];
}

line_node::line_node() : vec(NULL, 0, 0)
{
	node = NULL;
	node_size = 0;
	node_max_size = 1000;
	vector_size = 0;
	node_file[0] = 0;
	node_hash = NULL;
	_vec = NULL;
}

line_node::~line_node()
{
	if (node != NULL) { free(node); node = NULL; }
	node_size = 0;
	node_max_size = 0;
	vector_size = 0;
	node_file[0] = 0;
	if (node_hash != NULL) { free(node_hash); node_hash = NULL; }
	if (_vec != NULL) { free(_vec); _vec = NULL; }
	new (&vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
}

int line_node::get_hash(char *word)
{
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % hash_table_size;
	return hash;
}

int line_node::search(char *word)
{
	unsigned int hash = get_hash(word);
	while (1) {
		if (node_hash[hash] == -1) return -1;
		if (!strcmp(word, node[node_hash[hash]].word)) return node_hash[hash];
		hash = (hash + 1) % hash_table_size;
	}
	return -1;
}

int line_node::add_node(char *word)
{
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	node[node_size].word = (char *)calloc(length, sizeof(char));
	strcpy(node[node_size].word, word);
	node_size++;
	// Reallocate memory if needed
	if (node_size + 2 >= node_max_size) {
		node_max_size += 1000;
		node = (struct struct_node *)realloc(node, node_max_size * sizeof(struct struct_node));
	}
	hash = get_hash(word);
	while (node_hash[hash] != -1) hash = (hash + 1) % hash_table_size;
	node_hash[hash] = node_size - 1;
	return node_size - 1;
}

void line_node::init(char *file_name, int vector_dim)
{
	strcpy(node_file, file_name);
	vector_size = vector_dim;

	node = (struct struct_node *)calloc(node_max_size, sizeof(struct struct_node));
	node_hash = (int *)calloc(hash_table_size, sizeof(int));
	for (int k = 0; k != hash_table_size; k++) node_hash[k] = -1;

	FILE *fi = fopen(node_file, "rb");
	if (fi == NULL)
	{
		printf("ERROR: node file not found!\n");
		printf("%s\n", node_file);
		exit(1);
	}

	char word[MAX_STRING];
	node_size = 0;
	while (1)
	{
		if (fscanf(fi, "%s", word) != 1) break;
		add_node(word);
	}
	fclose(fi);

	long long a, b;
	//a = posix_memalign((void **)&_vec, 128, (long long)node_size * vector_size * sizeof(real));
	_vec = (real *)malloc(node_size * vector_size * sizeof(real));
	if (_vec == NULL) { printf("Memory allocation failed\n"); exit(1); }
	for (b = 0; b < vector_size; b++) for (a = 0; a < node_size; a++)
		_vec[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
	new (&vec) Eigen::Map<BLPMatrix>(_vec, node_size, vector_size);

	printf("Reading nodes from file: %s, DONE!\n", node_file);
	printf("Node size: %d\n", node_size);
	printf("Node dims: %d\n", vector_size);
}

void line_node::output(char *file_name, int binary)
{
	FILE *fo = fopen(file_name, "ab");
	for (int a = 0; a != node_size; a++)
	{
		fprintf(fo, "%s\t", node[a].word);
		if (binary) for (int b = 0; b != vector_size; b++) fwrite(&_vec[a * vector_size + b], sizeof(real), 1, fo);
		else for (int b = 0; b != vector_size; b++) fprintf(fo, "%lf ", _vec[a * vector_size + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

line_hin::line_hin()
{
	hin_file[0] = 0;
	node_u = NULL;
	node_v = NULL;
	hin = NULL;
	hin_size = 0;
}

line_hin::~line_hin()
{
	hin_file[0] = 0;
	node_u = NULL;
	node_v = NULL;
	if (hin != NULL) { delete[] hin; hin = NULL; }
	hin_size = 0;
}

void line_hin::init(char *file_name, line_node *p_u, line_node *p_v)
{
	strcpy(hin_file, file_name);

	node_u = p_u;
	node_v = p_v;

	int node_size = node_u->node_size;
	hin = new std::vector<hin_nb>[node_size];

	FILE *fi = fopen(hin_file, "rb");
	char word1[MAX_STRING], word2[MAX_STRING];
	int u, v, tp;
	double w;
	hin_nb curnb;
	while (fscanf(fi, "%s %s %d %lf", word1, word2, &tp, &w) == 4)
	{
		if (hin_size % 10000 == 0)
		{
			printf("%lldK%c", hin_size / 1000, 13);
			fflush(stdout);
		}

		u = node_u->search(word1);
		v = node_v->search(word2);

		if (u != -1 && v != -1)
		{
			curnb.nb_id = v;
			curnb.eg_tp = tp;
			curnb.eg_wei = w;
			hin[u].push_back(curnb);
			hin_size++;
		}
        
	}
	fclose(fi);

	printf("Reading edges from file: %s, DONE!\n", hin_file);
	printf("Edge size: %lld\n", hin_size);
}

line_trainer::line_trainer()
{
	edge_tp = 0;
	phin = NULL;
	expTable = NULL;
	u_nb_cnt = NULL;
	u_nb_id = NULL;
	u_nb_wei = NULL;
	u_wei = NULL;
	v_wei = NULL;
	smp_u_nb = NULL;
	expTable = NULL;
	neg_samples = 0;
	neg_table = NULL;
}

line_trainer::~line_trainer()
{
	edge_tp = 0;
	phin = NULL;
	if (expTable != NULL) { free(expTable); expTable = NULL; }
	if (u_nb_cnt != NULL) { free(u_nb_cnt); u_nb_cnt = NULL; }
	if (u_nb_id != NULL) { free(u_nb_id); u_nb_id = NULL; }
	if (u_nb_wei != NULL) { free(u_nb_wei); u_nb_wei = NULL; }
	if (u_wei != NULL) { free(u_wei); u_wei = NULL; }
	if (v_wei != NULL) { free(v_wei); v_wei = NULL; }
	if (smp_u_nb != NULL)
	{
		delete[] smp_u_nb;
		smp_u_nb = NULL;
	}
	neg_samples = 0;
	if (neg_table != NULL) { free(neg_table); neg_table = NULL; }
}

void line_trainer::init(int edge_type, line_hin *p_hin, int negative)
{
	edge_tp = edge_type;
	phin = p_hin;
	neg_samples = negative;
	line_node *node_u = phin->node_u, *node_v = phin->node_v;
	if (node_u->vector_size != node_v->vector_size)
	{
		printf("ERROR: vector dimsions are not same!\n");
		exit(1);
	}

	// compute the degree of vertices
	u_nb_cnt = (int *)calloc(node_u->node_size, sizeof(int));
	u_wei = (double *)calloc(node_u->node_size, sizeof(double));
	v_wei = (double *)calloc(node_v->node_size, sizeof(double));
	for (int u = 0; u != node_u->node_size; u++)
	{
		for (int k = 0; k != (int)(phin->hin[u].size()); k++)
		{
			int v = phin->hin[u][k].nb_id;
			int cur_edge_type = phin->hin[u][k].eg_tp;
			double wei = phin->hin[u][k].eg_wei;

			if (cur_edge_type != edge_tp) continue;

			u_nb_cnt[u]++;
			u_wei[u] += wei;
			v_wei[v] += wei;
		}
	}
    
	// allocate spaces for edges
	u_nb_id = (int **)malloc(node_u->node_size * sizeof(int *));
	u_nb_wei = (double **)malloc(node_u->node_size * sizeof(double *));
	for (int k = 0; k != node_u->node_size; k++)
	{
		u_nb_id[k] = (int *)malloc(u_nb_cnt[k] * sizeof(int));
		u_nb_wei[k] = (double *)malloc(u_nb_cnt[k] * sizeof(double));
	}

	// read neighbors
	int *pst = (int *)calloc(node_u->node_size, sizeof(int));
	for (int u = 0; u != node_u->node_size; u++)
	{
		for (int k = 0; k != (int)(phin->hin[u].size()); k++)
		{
			int v = phin->hin[u][k].nb_id;
			int cur_edge_type = phin->hin[u][k].eg_tp;
			double wei = phin->hin[u][k].eg_wei;

			if (cur_edge_type != edge_tp) continue;

			u_nb_id[u][pst[u]] = v;
			u_nb_wei[u][pst[u]] = wei;
			pst[u]++;
		}
	}
	free(pst);

	// init sampler for edges
	smp_u.init(node_u->node_size, u_wei);
	smp_u_nb = new sampler[node_u->node_size];
	for (int k = 0; k != node_u->node_size; k++)
	{
		if (u_nb_cnt[k] == 0) continue;
		smp_u_nb[k].init(u_nb_cnt[k], u_nb_wei[k]);
	}

	// Init negative sampling table
	neg_table = (int *)malloc(neg_table_size * sizeof(int));

	int a, i;
	double total_pow = 0, d1;
	double power = 0.75;
	for (a = 0; a < node_v->node_size; a++) total_pow += pow(v_wei[a], power);
	a = 0; i = 0;
	d1 = pow(v_wei[i], power) / (double)total_pow;
	while (a < neg_table_size) {
		if ((a + 1) / (double)neg_table_size > d1) {
			i++;
			if (i >= node_v->node_size) { i = node_v->node_size - 1; d1 = 2; }
			d1 += pow(v_wei[i], power) / (double)total_pow;
		}
		else
			neg_table[a++] = i;
	}

	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (int i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
}

void line_trainer::train_sample(real alpha, real *_error_vec, double(*func_rand_num)(), unsigned long long &rand_index)
{
	int target, label, u, v, index, vector_size;
	real f, g;
	line_node *node_u = phin->node_u, *node_v = phin->node_v;

	u = smp_u.draw(func_rand_num(), func_rand_num());
	if (u_nb_cnt[u] == 0) return;
	index = (int)(smp_u_nb[u].draw(func_rand_num(), func_rand_num()));
	v = u_nb_id[u][index];

	vector_size = node_u->vector_size;
	Eigen::Map<BLPVector> error_vec(_error_vec, vector_size);
	error_vec.setZero();

	for (int d = 0; d < neg_samples + 1; d++)
	{
		if (d == 0)
		{
			target = v;
			label = 1;
		}
		else
		{
			rand_index = rand_index * (unsigned long long)25214903917 + 11;
			target = neg_table[(rand_index >> 16) % neg_table_size];
			if (target == v) continue;
			label = 0;
		}
		f = node_u->vec.row(u) * node_v->vec.row(target).transpose();
		if (f > MAX_EXP) g = (label - 1) * alpha;
		else if (f < -MAX_EXP) g = (label - 0) * alpha;
		else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
		error_vec += g * ((node_v->vec.row(target)));
		node_v->vec.row(target) += g * ((node_u->vec.row(u)));
	}
	node_u->vec.row(u) += error_vec;
	new (&error_vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
}