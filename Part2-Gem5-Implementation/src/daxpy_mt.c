/*
 * Multi-threaded DAXPY kernel for gem5 TLP exploration
 * Computes: Y = a * X + Y (scaled vector addition)
 * 
 * Usage: ./daxpy_mt <vector_size> <num_threads>
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

// gem5 magic instruction annotations (optional - add gem5/m5ops.h if available)
// #include "gem5/m5ops.h"
// For now, use placeholder macros
#define m5_work_begin(workid, threadid) 
#define m5_work_end(workid, threadid)
#define m5_dump_reset_stats(delay, period)

typedef struct {
    double *x;
    double *y;
    double a;
    size_t start;
    size_t end;
    int thread_id;
} thread_data_t;

// Thread worker function
void* daxpy_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    // Signal start of computation region to gem5
    m5_work_begin(0, data->thread_id);
    
    // Perform DAXPY on assigned chunk
    for (size_t i = data->start; i < data->end; i++) {
        data->y[i] = data->a * data->x[i] + data->y[i];
    }
    
    // Signal end of computation region to gem5
    m5_work_end(0, data->thread_id);
    
    return NULL;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <vector_size> <num_threads>\n", argv[0]);
        return 1;
    }
    
    size_t N = atol(argv[1]);
    int num_threads = atoi(argv[2]);
    
    if (N == 0 || num_threads <= 0) {
        fprintf(stderr, "Error: Invalid parameters\n");
        return 1;
    }
    
    printf("=== Multi-threaded DAXPY Benchmark ===\n");
    printf("Vector size: %zu\n", N);
    printf("Threads: %d\n", num_threads);
    
    // Allocate and initialize vectors
    double *x = (double*)malloc(N * sizeof(double));
    double *y = (double*)malloc(N * sizeof(double));
    double a = 2.5;
    
    if (!x || !y) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }
    
    // Initialize with deterministic values
    for (size_t i = 0; i < N; i++) {
        x[i] = (double)(i % 100) / 100.0;
        y[i] = (double)(i % 50) / 50.0;
    }
    
    // Allocate thread resources
    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = (thread_data_t*)malloc(num_threads * sizeof(thread_data_t));
    
    if (!threads || !thread_data) {
        fprintf(stderr, "Error: Thread allocation failed\n");
        return 1;
    }
    
    // Calculate work distribution
    size_t chunk_size = N / num_threads;
    size_t remainder = N % num_threads;
    
    // Reset gem5 stats before computation
    m5_dump_reset_stats(0, 0);
    
    double start_time = get_time();
    
    // Launch threads
    size_t offset = 0;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].x = x;
        thread_data[i].y = y;
        thread_data[i].a = a;
        thread_data[i].start = offset;
        thread_data[i].end = offset + chunk_size + (i < remainder ? 1 : 0);
        thread_data[i].thread_id = i;
        
        if (pthread_create(&threads[i], NULL, daxpy_worker, &thread_data[i]) != 0) {
            fprintf(stderr, "Error: Failed to create thread %d\n", i);
            return 1;
        }
        
        offset = thread_data[i].end;
    }
    
    // Join all threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    double end_time = get_time();
    double elapsed = end_time - start_time;
    
    // Verification (check first and last few elements)
    printf("\n=== Verification ===\n");
    printf("First 3 results:\n");
    for (int i = 0; i < 3 && i < N; i++) {
        double expected = a * ((double)(i % 100) / 100.0) + ((double)(i % 50) / 50.0);
        printf("  y[%d] = %.6f (expected: %.6f)\n", i, y[i], expected);
    }
    
    // Performance metrics
    double gflops = (2.0 * N) / (elapsed * 1e9);  // 2 ops per element
    double bandwidth = (3.0 * N * sizeof(double)) / (elapsed * 1e9);  // GB/s (read x, read/write y)
    
    printf("\n=== Performance ===\n");
    printf("Execution time: %.6f seconds\n", elapsed);
    printf("Throughput: %.3f GFLOP/s\n", gflops);
    printf("Bandwidth: %.3f GB/s\n", bandwidth);
    printf("Elements/second: %.3e\n", N / elapsed);
    
    // Cleanup
    free(x);
    free(y);
    free(threads);
    free(thread_data);
    
    return 0;
}
