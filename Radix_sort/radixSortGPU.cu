#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(int *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = rand() % 10000;
    }
}

void cpu_radixSort(int *arr, int size)
{
    int *temp = (int *)malloc(size * sizeof(int));
    int *src = arr;
    int *dst = temp;
    for (int bit = 0; bit < 32; bit += 8)
    {
        int count[256] = {0};
        for (int i = 0; i < size; i++)
        {
            count[(src[i] >> bit) & 255]++;
        }
        for (int i = 1; i < 256; i++)
        {
            count[i] += count[i - 1];
        }
        for (int i = size - 1; i >= 0; i--)
        {
            int bucket = (src[i] >> bit) & 255;
            dst[count[bucket] - 1] = src[i];
            count[bucket]--;
        }
        int *tmp = src;
        src = dst;
        dst = tmp;
    }
    if (src != arr)
    {
        memcpy(arr, src, size * sizeof(int));
    }
    free(temp);
}

__global__ void blockScan(int *d_out, int *d_in, int *d_block_sums, int N)
{
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = 1;

    int idx1 = 2 * blockIdx.x * blockDim.x + 2 * thid;
    int idx2 = idx1 + 1;

    temp[2 * thid] = (idx1 < N) ? d_in[idx1] : 0;
    temp[2 * thid + 1] = (idx2 < N) ? d_in[idx2] : 0;

    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0)
    {
        if (d_block_sums != NULL)
        {
            d_block_sums[blockIdx.x] = temp[2 * blockDim.x - 1];
        }
        temp[2 * blockDim.x - 1] = 0;
    }

    for (int d = 1; d <= blockDim.x; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (idx1 < N) d_out[idx1] = temp[2 * thid];
    if (idx2 < N) d_out[idx2] = temp[2 * thid + 1];
}

__global__ void addBlockSums(int *d_out, int *d_block_sums, int N)
{
    int idx1 = blockIdx.x * blockDim.x * 2 + 2 * threadIdx.x;
    int idx2 = idx1 + 1;
    int block_val = d_block_sums[blockIdx.x];
    if (idx1 < N) d_out[idx1] += block_val;
    if (idx2 < N) d_out[idx2] += block_val;
}

void gpu_scan(int *d_out, int *d_in, int N)
{
    int block_size = 512;
    int num_elements_per_block = 2 * block_size;
    int num_blocks = (N + num_elements_per_block - 1) / num_elements_per_block;

    int *d_block_sums = NULL;
    CHECK(cudaMalloc((void **)&d_block_sums, num_blocks * sizeof(int)));

    blockScan<<<num_blocks, block_size, num_elements_per_block * sizeof(int)>>>(d_out, d_in, d_block_sums, N);
    CHECK(cudaDeviceSynchronize());

    if (num_blocks > 1)
    {
        int *d_block_sums_scanned = NULL;
        CHECK(cudaMalloc((void **)&d_block_sums_scanned, num_blocks * sizeof(int)));

        gpu_scan(d_block_sums_scanned, d_block_sums, num_blocks);

        addBlockSums<<<num_blocks, block_size, 0>>>(d_out, d_block_sums_scanned, N);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaFree(d_block_sums_scanned));
    }

    CHECK(cudaFree(d_block_sums));
}

__global__ void computePredicate(int *d_p, int *d_in, int bit, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_p[idx] = ((d_in[idx] >> bit) & 1) == 0 ? 1 : 0;
    }
}

__global__ void scatter(int *d_out, int *d_in, int *d_p, int *d_scanned_p, int total_zeros, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int val = d_in[idx];
    int p = d_p[idx];
    int scanned_p = d_scanned_p[idx];

    int new_idx;
    if (p == 1)
    {
        new_idx = scanned_p;
    }
    else
    {
        new_idx = total_zeros + (idx - scanned_p);
    }

    d_out[new_idx] = val;
}

void gpu_radixSort(int *d_out, int *d_in, int N)
{
    int *d_p, *d_scanned_p;
    CHECK(cudaMalloc((void **)&d_p, N * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_scanned_p, N * sizeof(int)));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    int *d_src = d_in;
    int *d_dst = d_out;

    for (int bit = 0; bit < 32; bit++)
    {
        computePredicate<<<grid_size, block_size>>>(d_p, d_src, bit, N);
        CHECK(cudaDeviceSynchronize());

        gpu_scan(d_scanned_p, d_p, N);

        int last_val = 0;
        int last_pred = 0;
        CHECK(cudaMemcpy(&last_val, &d_scanned_p[N - 1], sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&last_pred, &d_p[N - 1], sizeof(int), cudaMemcpyDeviceToHost));
        int total_zeros = last_val + last_pred;

        scatter<<<grid_size, block_size>>>(d_dst, d_src, d_p, d_scanned_p, total_zeros, N);
        CHECK(cudaDeviceSynchronize());

        int *temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }

    if (d_src != d_out)
    {
        CHECK(cudaMemcpy(d_out, d_src, N * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    CHECK(cudaFree(d_p));
    CHECK(cudaFree(d_scanned_p));
}

void checkResult(int *hostRef, int *gpuRef, const int N)
{
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            match = 0;
            printf("Arrays do not match! at %d: host %d gpu %d\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int size = 1 << 20;
    printf("Data size %d\n", size);

    size_t nBytes = size * sizeof(int);
    int *h_A = (int *)malloc(nBytes);
    int *hostRef = (int *)malloc(nBytes);
    int *gpuRef = (int *)malloc(nBytes);

    initialData(h_A, size);
    memcpy(hostRef, h_A, nBytes);
    memcpy(gpuRef, h_A, nBytes);

    double iStart, iElaps;

    iStart = seconds();
    cpu_radixSort(hostRef, size);
    iElaps = seconds() - iStart;
    printf("CPU Radix Sort elapsed %f sec\n", iElaps);

    int *d_A, *d_B;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    iStart = seconds();
    gpu_radixSort(d_B, d_A, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("GPU Radix Sort elapsed %f sec\n", iElaps);

    CHECK(cudaMemcpy(gpuRef, d_B, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, size);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
