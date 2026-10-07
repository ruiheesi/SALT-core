#include <math.h>
#include <stdio.h>
#include <string.h>

#include "salt/gpu.h"

static int run_case(int start_position) {
    const int rows = start_position + 1;
    const int stride = 512;
    const size_t row_floats = (size_t)rows * (size_t)stride;
    SaltGpuSharedBuffer queries = {0};
    SaltGpuSharedBuffer kv = {0};
    SaltGpuSharedBuffer outputs = {0};
    float *query;
    float *cache;
    float *output;
    int rc = -1;

    if (salt_gpu_shared_buffer_alloc(&queries,
            (size_t)stride * sizeof(float)) != 0 ||
        salt_gpu_shared_buffer_alloc(&kv,
            2u * row_floats * sizeof(float)) != 0 ||
        salt_gpu_shared_buffer_alloc(&outputs,
            (size_t)stride * sizeof(float)) != 0)
        goto done;
    query = (float *)queries.contents;
    cache = (float *)kv.contents;
    output = (float *)outputs.contents;
    memset(query, 0, (size_t)stride * sizeof(float));
    memset(cache, 0, row_floats * sizeof(float));
    for (int row = 0; row < rows; row++)
        for (int dimension = 0; dimension < stride; dimension++)
            cache[row_floats + (size_t)row * stride + (size_t)dimension] =
                dimension & 1 ? -3.0f : 2.0f;
    if (salt_gpu_attention_batch(
            1, 1, 1, stride, 0, &queries, 0, &kv, 0, row_floats,
            start_position, 1, stride, stride, &outputs, 0) != 0)
        goto done;
    for (int dimension = 0; dimension < stride; dimension++) {
        float expected = dimension & 1 ? -3.0f : 2.0f;
        if (!isfinite(output[dimension]) ||
                fabsf(output[dimension] - expected) > 1e-5f) {
            fprintf(stderr, "position %d dimension %d: output %.9g\n",
                    start_position, dimension, output[dimension]);
            goto done;
        }
    }
    rc = 0;

done:
    if (outputs.backend && salt_gpu_shared_buffer_free(&outputs) != 0)
        rc = -1;
    if (kv.backend && salt_gpu_shared_buffer_free(&kv) != 0)
        rc = -1;
    if (queries.backend && salt_gpu_shared_buffer_free(&queries) != 0)
        rc = -1;
    return rc;
}

int main(void) {
    if (salt_gpu_init() != 0) {
        fputs("Metal unavailable\n", stderr);
        return 1;
    }
    if (run_case(8191) != 0 || run_case(8192) != 0) {
        (void)salt_gpu_free();
        fputs("Metal long attention: FAIL\n", stderr);
        return 1;
    }
    if (salt_gpu_free() != 0) return 1;
    puts("Metal long attention: PASS (8192 boundary crossed)");
    return 0;
}
