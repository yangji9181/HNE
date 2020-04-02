#include "linelib.h"

line_node::line_node() : vec(NULL, 0, 0)
{
    node = NULL;
    node_size = 0;
    node_max_size = 1000;
    vector_size = 0;
    node_file[0] = 0;
    link_file[0] = 0;
    node_hash = NULL;
    _vec = NULL;
    hin = NULL;
    hin_size = 0;
}

line_node::~line_node()
{
    if (node != NULL) {free(node); node = NULL;}
    node_size = 0;
    node_max_size = 0;
    vector_size = 0;
    node_file[0] = 0;
    link_file[0] = 0;
    if (node_hash != NULL) {free(node_hash); node_hash = NULL;}
    if (_vec != NULL) {free(_vec); _vec = NULL;}
    if (hin != NULL) {delete [] hin; hin = NULL;}
    hin_size = 0;
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

int line_node::add_node(char *word, char type)
{
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    node[node_size].word = (char *)calloc(length, sizeof(char));
    strcpy(node[node_size].word, word);
    node[node_size].type = type;
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

    char word[MAX_STRING], type;
    node_size = 0;
    while (1)
    {
        if (fscanf(fi, "%s %c", word, &type) != 2) break;
        //int cn;
        //if (fscanf(fi, "%s %d %c", word, &cn, &type) != 3) break;
        add_node(word, type);
    }
    fclose(fi);
    
    long long a, b;
    a = posix_memalign((void **)&_vec, 128, (long long)node_size * vector_size * sizeof(real));
    if (_vec == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < vector_size; b++) for (a = 0; a < node_size; a++)
        _vec[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    new (&vec) Eigen::Map<BLPMatrix>(_vec, node_size, vector_size);
    
    printf("Reading nodes from file: %s, DONE!\n", node_file);
    printf("Node size: %d\n", node_size);
    printf("Node dims: %d\n", vector_size);
}


void line_node::init_hin(char *file_name)
{
    strcpy(link_file, file_name);
    
    hin = new std::vector<int>[node_size];
    
    FILE *fi = fopen(link_file, "rb");
    char word1[MAX_STRING], word2[MAX_STRING];
    int u, v;
    while (fscanf(fi, "%s %s", word1, word2) == 2)
    {
        if (hin_size % 10000 == 0)
        {
            printf("%lldK%c", hin_size / 1000, 13);
            fflush(stdout);
        }
        
        u = search(word1);
        v = search(word2);
        
        if (u != -1 && v != -1)
        {
         hin_size++;
         hin[u].push_back(v);
        }
    }
    fclose(fi);
    
    edge_u = (int *)malloc(hin_size * sizeof(int));
    edge_v = (int *)malloc(hin_size * sizeof(int));
    
    fi = fopen(link_file, "rb");
    int pst = 0;
    while (fscanf(fi, "%s %s", word1, word2) == 2)
    {
        u = search(word1);
        v = search(word2);
        
        if (u == -1 || v == -1) continue;

       edge_u[pst] = u;
       edge_v[pst] = v;
       pst++;

    }
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

line_link::line_link()
{
    node_u = NULL;
    node_v = NULL;
    expTable = NULL;
    neg_samples = 0;
    dp_cnt = NULL;
    dp_cnt_fd = NULL;
    path[0] = 0;
    path_size = 0;
    neg_table = NULL;
    smp_init = NULL;
    smp_init_weight = NULL;
    smp_init_index = NULL;
    smp_dp = NULL;
    smp_dp_weight = NULL;
    smp_dp_index = NULL;
    pmap.clear();
}

line_link::~line_link()
{
    node_u = NULL;
    node_v = NULL;
    if (expTable != NULL) {free(expTable); expTable = NULL;}
    neg_samples = 0;
    if (dp_cnt != NULL)
    {
        for (int k = 0; k != node_u->node_size; k++) if (dp_cnt[k] != NULL)
            free(dp_cnt[k]);
        free(dp_cnt);
        dp_cnt = NULL;
    }
    if (dp_cnt_fd != NULL)
    {
        for (int k = 0; k != node_u->node_size; k++) if (dp_cnt_fd[k] != NULL)
            free(dp_cnt_fd[k]);
        free(dp_cnt_fd);
        dp_cnt_fd = NULL;
    }
    path[0] = 0;
    path_size = 0;
    if (neg_table != NULL)
    {
        for (int k = 0; k != path_size - 1; k++) if (neg_table[k] != NULL)
            free(neg_table[k]);
        free(neg_table);
        neg_table = NULL;
    }
    if (smp_init != NULL)
    {
        ransampl_free(smp_init);
        smp_init = NULL;
    }
    if (smp_dp != NULL)
    {
        for (int i = 0; i != path_size; i++) for (int j = 0; j != node_u->node_size; j++) if (smp_dp[i][j] != NULL)
        {
            ransampl_free(smp_dp[i][j]);
            smp_dp[i][j] = NULL;
        }
        for (int i = 0; i != path_size; i++) {free(smp_dp[i]); smp_dp[i] = NULL;}
        free(smp_dp);
        smp_dp = NULL;
    }
    pmap.clear();
}

void line_link::init(std::string meta_path, line_node *p_u, line_node *p_v, int negative)
{
    path = meta_path;
    path_size = path.size();
    
    node_u = p_u;
    node_v = p_v;
    neg_samples = negative;
    if (node_u->vector_size != node_v->vector_size)
    {
        printf("ERROR: vector dimsions are not same!\n");
        exit(1);
    }
    
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    int node_size = node_u->node_size;
    std::vector<int> *hin = node_u->hin;
    
    dp_cnt = (double **)malloc(node_size * sizeof(double *));
    for (int k = 0; k != node_size; k++) dp_cnt[k] = (double *)malloc(path_size * sizeof(double));
    for (int i = 0; i != node_size; i++) for (int j = 0; j != path_size; j++) dp_cnt[i][j] = 0;
    
    dp_cnt_fd = (double **)malloc(node_size * sizeof(double *));
    for (int k = 0; k != node_size; k++) dp_cnt_fd[k] = (double *)malloc(path_size * sizeof(double));
    for (int i = 0; i != node_size; i++) for (int j = 0; j != path_size; j++) dp_cnt_fd[i][j] = 0;
    
    for (int step = path_size - 1; step >= 0; step--)
    {
        char type = path[step];
        if (step == path_size - 1)
        {
            for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == type)
                dp_cnt[u][step] = 1;
        }
        else
        {
            for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == type)
            {
                int neighbor_size = hin[u].size();
                for (int i = 0; i != neighbor_size; i++)
                {
                    int v = hin[u][i];
                    if ((node_u->node[v]).type == path[step + 1])
                        dp_cnt[u][step] += dp_cnt[v][step + 1];
                }
            }
        }
    }
    
    for (int step = 0; step < path_size - 1; step++)
    {
        char type = path[step];
        for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == type)
        {
            if (step == 0) dp_cnt_fd[u][step] = 1;
            
            int neighbor_size = hin[u].size();
            for (int i = 0; i != neighbor_size; i++)
            {
                int v = hin[u][i];
                if ((node_u->node[v]).type == path[step + 1])
                    dp_cnt_fd[v][step + 1] += dp_cnt_fd[u][step];
            }
        }
    }
    

    // Init negative sampling table
    neg_table = (int **)malloc(path_size * sizeof(int *));
    for (int k = 0; k != path_size; k++) neg_table[k] = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 0; k != path_size; k++)
    {
        int a, i;
        double total_pow = 0, d1;
        double power = 0.75;
        for (a = 0; a < node_size; a++) total_pow += pow(dp_cnt_fd[a][k], power);
        a = 0; i = 0;
        d1 = pow(dp_cnt_fd[i][k], power) / (double)total_pow;
        while (a < neg_table_size) {
            if ((a + 1) / (double)neg_table_size > d1) {
                i++;
                if (i >= node_size) {i = node_size - 1; d1 = 2;}
                d1 += pow(dp_cnt_fd[i][k], power) / (double)total_pow;
            }
            else
                neg_table[k][a++] = i;
        }
    }
    
    int node_cnt;
    char type;
    
    
    // Init the sampling table of step 0
    node_cnt = 0;
    type = path[0];
    for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == type)
        node_cnt++;
    smp_init_index = (int *)malloc(node_cnt * sizeof(int));
    smp_init_weight = (double *)malloc(node_cnt * sizeof(double));
    smp_init = ransampl_alloc(node_cnt);
    node_cnt = 0;
    for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == type)
    {
        smp_init_index[node_cnt] = u;
        smp_init_weight[node_cnt] = dp_cnt[u][0];
        node_cnt++;
    }
    ransampl_set(smp_init, smp_init_weight);
    
    
    // Init sampling tables of the following steps
    smp_dp_index = (int ***)malloc(path_size * sizeof(int **));
    for (int k = 0; k != path_size; k++) smp_dp_index[k] = (int **)malloc(node_size * sizeof(int *));
    for (int i = 0; i != path_size; i++) for (int j = 0; j != node_size; j++) smp_dp_index[i][j] = NULL;
    smp_dp_weight = (double ***)malloc(path_size * sizeof(double **));
    for (int k = 0; k != path_size; k++) smp_dp_weight[k] = (double **)malloc(node_size * sizeof(double *));
    for (int i = 0; i != path_size; i++) for (int j = 0; j != node_size; j++) smp_dp_weight[i][j] = NULL;
    smp_dp = (ransampl_ws ***)malloc(path_size * sizeof(ransampl_ws));
    for (int k = 0; k != path_size; k++) smp_dp[k] = (ransampl_ws **)malloc(node_size * sizeof(ransampl_ws));
    for (int i = 0; i != path_size; i++) for (int j = 0; j != node_size; j++) smp_dp[i][j] = NULL;
    
    for (int step = 0; step != path_size - 1; step++)
    {
        type = path[step];
        for (int u = 0; u != node_size; u++) if((node_u->node[u]).type == type)
        {
            node_cnt = 0;
            int neighbor_size = hin[u].size();
            for (int i = 0; i != neighbor_size; i++)
            {
                int v = hin[u][i];
                if ((node_u->node[v]).type == path[step + 1])
                    node_cnt++;
            }
            if (node_cnt == 0) continue;
            
            smp_dp_index[step][u] = (int *)malloc(node_cnt * sizeof(int));
            smp_dp_weight[step][u] = (double *)malloc(node_cnt * sizeof(double));
            smp_dp[step][u] = ransampl_alloc(node_cnt);
            node_cnt = 0;
            double sum = 0;
            for (int i = 0; i != neighbor_size; i++)
            {
                int v = hin[u][i];
                if ((node_u->node[v]).type == path[step + 1])
                {
                    smp_dp_index[step][u][node_cnt] = v;
                    smp_dp_weight[step][u][node_cnt] = dp_cnt[v][step + 1];
                    
                    
                    sum += smp_dp_weight[step][u][node_cnt];
                    
                    
                    node_cnt++;
                }
            }
            ransampl_set(smp_dp[step][u], smp_dp_weight[step][u]);
        }
    }
}

void line_link::init_map(std::vector<line_map *> pointer_map)
{
    pmap = pointer_map;
    tp2mid.clear();
    for (std::vector<line_map *>::size_type k = 0; k != pmap.size(); k++)
        tp2mid[pmap[k]->type] = (int)(k + 1);
}

void line_link::sample_path(int *node_lst, double (*func_rand_num)())
{
    long long cur_entry;
    // Sample the first node
    cur_entry = ransampl_draw(smp_init, func_rand_num(), func_rand_num());
    node_lst[0] = smp_init_index[cur_entry];
    // Sample following nodes
    for (int step = 0; step != path_size - 1; step++)
    {
        int u = node_lst[step];
        cur_entry = ransampl_draw(smp_dp[step][u], func_rand_num(), func_rand_num());
        node_lst[step + 1] = smp_dp_index[step][u][cur_entry];
    }
}

void line_link::train_path(int *node_lst, real *_error_vec, real *_error_p, real *_error_q, real alpha, double (*func_rand_num)(), unsigned long long &rand_index, int model, int mode)
{
    int target, label, u, v, vector_size, mid;
    real f, g;
    std::string tp;
    int step_bg = 0, step_ed = 0, pst_bg = 0, pst_ed = 0;
    
    vector_size = node_u->vector_size;
    sample_path(node_lst, func_rand_num);
    
    Eigen::Map<BLPVector> error_vec(_error_vec, vector_size);
    Eigen::Map<BLPVector> error_p(_error_p, vector_size);
    Eigen::Map<BLPVector> error_q(_error_q, vector_size);
    
    if (model == 0)
    {
        step_bg = path_size - 1; step_ed = path_size;
        pst_bg = 0; pst_ed = 1;
    }
    else if (model == 1)
    {
        step_bg = 1; step_ed = 2;
        pst_bg = 0; pst_ed = path_size - 1;
    }
    else if (model == 2)
    {
        step_bg = 1; step_ed = path_size;
        pst_bg = 0; pst_ed = path_size - 1;
    }
    
    for (int step = step_bg; step != step_ed; step++) for (int pst = pst_bg; pst != pst_ed; pst++)
    {
        if (pst + step >= path_size) continue;
        
        u = node_lst[pst];
        v = node_lst[pst + step];
        
        tp = path.substr(pst, step + 1);
        if (path[pst] > path[pst + step]) for (int k = 0; k <= step; k++)
            tp[k] = path[pst + step - k];
        mid = tp2mid[tp];
        if (mid == 0)
        {
            printf("Map not found!\n");
            continue;
        }
        mid--;
        
        error_vec.setZero();
        error_p.setZero();
        error_q.setZero();
        
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
                //target = neg_table[pst][(rand_index >> 16) % neg_table_size];
                target = neg_table[pst + step][(rand_index >> 16) % neg_table_size];
                if (target == v) continue;
                label = 0;
            }
            
            f = (node_u->vec.row(u)) * (node_v->vec.row(target).transpose());
            f += (node_u->vec.row(u)) * (pmap[mid]->P.transpose());
            f += (node_v->vec.row(target)) * (pmap[mid]->Q.transpose());
            f += (pmap[mid]->bias);
            
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

            if (mode == 1 || mode == 2)
            {
                error_p += g * (node_u->vec.row(u));
                error_q += g * (node_v->vec.row(target));
            }
            if (mode == 0 || mode == 2)
            {
                error_vec += g * ((node_v->vec.row(target)));
                error_vec += g * (pmap[mid]->P);
                    
                node_v->vec.row(target) += g * ((node_u->vec.row(u)));
                node_v->vec.row(target) += g * (pmap[mid]->Q);
            }
        }
        if (mode == 0 || mode == 2) node_u->vec.row(u) += error_vec;
        if (mode == 1 || mode == 2)
        {
            pmap[mid]->P += error_p;
            pmap[mid]->Q += error_q;
        }
    }
    new (&error_vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_p) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_q) Eigen::Map<BLPMatrix>(NULL, 0, 0);
}

double line_link::eval(int *node_lst, double (*func_rand_num)(), int num_path)
{
    double log_prob = 0;
    int cnt = 0;
    for (int n = 0; n != num_path; n++)
    {
        sample_path(node_lst, func_rand_num);
        for (int k = 0; k != path_size - 1; k++)
        {
            int u = node_lst[k], v = node_lst[k + 1];
            real f = (node_u->vec.row(u).array() * node_v->vec.row(v).array()).sum();
            f = 1 / (1 + exp(-f));
            
            log_prob += log(f);
            cnt++;
        }
    }
    return exp(log_prob / cnt);
}

int line_link::get_path_length()
{
    return path_size;
}

line_map::line_map() : S(NULL, 0, 0), P(NULL, 0), Q(NULL, 0)
{
    type = "";
    vector_size = 0;
    _S = NULL;
    _P = NULL;
    _Q = NULL;
    bias = 0;
}

line_map::~line_map()
{
    type = "";
    vector_size = 0;
    if (_S != NULL) {free(_S); _S = NULL;}
    if (_P != NULL) {free(_P); _P = NULL;}
    if (_Q != NULL) {free(_Q); _Q = NULL;}
    new (&S) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&P) Eigen::Map<BLPVector>(NULL, 0);
    new (&Q) Eigen::Map<BLPVector>(NULL, 0);
    bias = 0;
}

void line_map::init(std::string map_type, int vector_dim)
{
    type = map_type;
    vector_size = vector_dim;
    _S = (real *)malloc(vector_size * vector_size * sizeof(real));
    _P = (real *)malloc(vector_size * sizeof(real));
    _Q = (real *)malloc(vector_size * sizeof(real));
    
    for (int a = 0; a < vector_size; a++) for (int b = 0; b != vector_size; b++)
        _S[a * vector_size + b] = 0;
    for (int a = 0; a < vector_size; a++) _S[a * vector_size + a] = 1;
    new (&S) Eigen::Map<BLPMatrix>(_S, vector_size, vector_size);
    for (int a = 0; a < vector_size; a++) _P[a] = 0;
    new (&P) Eigen::Map<BLPVector>(_P, vector_size);
    for (int a = 0; a < vector_size; a++) _Q[a] = 0;
    new (&Q) Eigen::Map<BLPVector>(_Q, vector_size);
}

