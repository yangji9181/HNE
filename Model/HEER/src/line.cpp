//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "ransampl.h"


#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
    double cn;
    char *word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 0, num_threads = 1, min_reduce = 1, order = 2;
long long samples = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long edge_count_actual = 0;
real alpha = 0.025, starting_alpha;
real *syn0, *syn1, *expTable;
clock_t start;

int negative = 5;
const int table_size = 1e8;
int *table;

long long nedges;
int *edge_from, *edge_to;
double *edge_weight;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;
ransampl_ws* ws;

void InitUnigramTable() {
    int a, i;
    double train_words_pow = 0;
    real d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (real)table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
        {
            if (a > 0) break;
            else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    double wei_a = ((struct vocab_word *)a)->cn;
    double wei_b = ((struct vocab_word *)b)->cn;
    if (wei_b > wei_a) return 1;
    else if (wei_b < wei_a) return -1;
    else return 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    for (a = 0; a < size; a++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if (vocab[a].cn < min_count) {
            vocab_size--;
            free(vocab[vocab_size].word);
        }
        else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash = GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
        vocab[b].cn = vocab[a].cn;
        vocab[b].word = vocab[a].word;
        b++;
    }
    else free(vocab[a].word);
    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

void LearnVocabFromTrainFile() {
    char word1[MAX_STRING], word2[MAX_STRING], edgetype[MAX_STRING];
    double weight;
    FILE *fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *)"</s>");
    while (1) {
        ReadWord(word1, fin);
        ReadWord(word2, fin);
        if (fscanf(fin, "%lf %s", &weight, edgetype) != 2) break;
        nedges++;
        
        if ((debug_mode > 1) && (nedges % 10000 == 0))
        {
            printf("%lldK%c", nedges / 1000, 13);
            fflush(stdout);
        }
        
        i = SearchVocab(word1);
        if (i == -1) {
            a = AddWordToVocab(word1);
            vocab[a].cn = weight;
        }
        else vocab[i].cn += weight;
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
        
        i = SearchVocab(word2);
        if (i == -1) {
            a = AddWordToVocab(word2);
            vocab[a].cn = weight;
        }
        else vocab[i].cn += weight;
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Number of edges: %lld\n", nedges);
    }
    SortVocab();
    fclose(fin);
}

void InitNet() {
    long long a, b;
    
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn1[a * layer1_size + b] = 0;
}

void ReadNet()
{
    FILE *fin = fopen(train_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: biterm data file not found!\n");
        exit(1);
    }
    double edgetype;
    edge_weight = (double *)malloc(nedges*sizeof(double));
    edge_from = (int *)malloc(nedges*sizeof(int));
    edge_to = (int *)malloc(nedges*sizeof(int));
    char word1[MAX_STRING], word2[MAX_STRING];
    if (edge_weight == NULL || edge_from == NULL || edge_to == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }
    
    for (long long k = 0; k != nedges; k++)
    {
        if (k % 10000 == 0)
        {
            printf("%cRead biterms: %.3lf%%", 13, k / (double)(nedges + 1) * 100);
            fflush(stdout);
        }
        ReadWord(word1, fin);
        ReadWord(word2, fin);
        fscanf(fin, "%lf %lf", &edge_weight[k], &edgetype);
        edge_from[k] = (int)(SearchVocab(word1));
        edge_to[k] = (int)(SearchVocab(word2));
    }
    printf("\n");
    fclose(fin);
    
    ws = ransampl_alloc(nedges);
    ransampl_set(ws, edge_weight);
}

void *TrainModelThread(void *id) {
    long long c, d, node, last_node;
    long long edge_count = 0, last_edge_count = 0, curedge;
    long long l1, l2, target, label;
    unsigned long long next_random = (long long)id;
    real f, g;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    
    while (1)
    {
        //judge for exit
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
        
        // use last_node to predict node
        curedge = ransampl_draw(ws, gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
        last_node = edge_from[curedge];
        node = edge_to[curedge];
        
        // last_node -> last_node*wei
        l1 = last_node * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1[c] = syn0[c + l1];
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        
        // NEGATIVE SAMPLING
        
        for (d = 0; d < negative + 1; d++)
        {
            if (d == 0)
            {
                target = node;
                label = 1;
            }
            else
            {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                if (target == node) continue;
                label = 0;
            }
            l2 = target * layer1_size;
            if (order == 1)
            {
                f = 0;
                for (c = 0; c < layer1_size; c++) f += neu1[c] * syn0[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn0[c + l2];
                for (c = 0; c < layer1_size; c++) syn0[c + l2] += g * neu1[c];
            }
            if (order == 2)
            {
                f = 0;
                for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
            }
        }
        // hidden -> in
        for (int c = 0; c < layer1_size; c++)
            syn0[c + l1] += neu1e[c];
        
        edge_count++;
    }
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void TrainModel() {
    long a, b;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;
    LearnVocabFromTrainFile();
    if (output_file[0] == 0) return;
    InitNet();
    ReadNet();
    InitUnigramTable();
    
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    
    start = clock();
    printf("Training process:\n");
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    
    fo = fopen(output_file, "wb");
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
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

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("LINE: Large Information Network Embedding toolkit v 0.1b\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse network data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting vectors\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-order <int>\n");
        printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./line -train net.txt -output vec.txt -debug 2 -binary 1 -size 200 -order 2 -negative 5 -samples 100\n\n");
        return 0;
    }
    output_file[0] = 0;
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-order", argc, argv)) > 0) order = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = atoi(argv[i + 1])*(long long)(1000000);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
    }
    TrainModel();
    return 0;
}