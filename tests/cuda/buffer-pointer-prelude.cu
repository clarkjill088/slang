//TEST(smoke):COMPILE: -pass-through nvrtc -target ptx -entry testBufferPointers -I prelude tests/cuda/buffer-pointer-prelude.cu

#include "slang-cuda-prelude.h"

static_assert(sizeof(StructuredBuffer<float4>) == 8 && alignof(StructuredBuffer<float4>) == 8);
static_assert(sizeof(RWStructuredBuffer<float4>) == 8 && alignof(RWStructuredBuffer<float4>) == 8);
static_assert(sizeof(ByteAddressBuffer) == 8 && alignof(ByteAddressBuffer) == 8);
static_assert(sizeof(RWByteAddressBuffer) == 8 && alignof(RWByteAddressBuffer) == 8);

// Instantiate both byte-to-structured conversions as well as direct buffer accesses.
__global__ void testBufferPointers(ByteAddressBuffer input, RWByteAddressBuffer output)
{
    output.asStructuredBuffer<uint32_t>()[0] = input.asStructuredBuffer<uint32_t>().Load(0);
    output.Store(4, input.Load(4));
}
