#include <webgpu/webgpu.h>
#include <cstdio>
int main() {
    printf("sizeof(WGPUBindGroupEntry) = %zu\n", sizeof(WGPUBindGroupEntry));
    printf("sizeof(WGPUBindGroupLayoutEntry) = %zu\n", sizeof(WGPUBindGroupLayoutEntry));
    printf("sizeof(WGPUBindGroupDescriptor) = %zu\n", sizeof(WGPUBindGroupDescriptor));
    return 0;
}
