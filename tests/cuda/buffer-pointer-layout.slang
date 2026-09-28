//TEST:SIMPLE(filecheck=CHECK): -target cuda -entry computeMain -stage compute
//TEST:REFLECTION(filecheck=LAYOUT): -target cuda -entry computeMain -stage compute -no-codegen

// The resource reference, its descriptor handle, and the emitted prelude wrapper must agree.
// The element type's size is independent of the size of the reference.
struct Element
{
    float4 value;
}

struct Buffers
{
    StructuredBuffer<Element> input;
    uint afterInput;
    RWStructuredBuffer<Element> output;
    uint afterOutput;
    ByteAddressBuffer bytes;
    uint afterBytes;
    RWByteAddressBuffer mutableBytes;
    uint afterMutableBytes;
    StructuredBuffer<Element> array[2];
}

Buffers buffers;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    static_assert(sizeof(StructuredBuffer<Element>) == 8, "Incorrect CUDA buffer layout.");
    static_assert(sizeof(RWStructuredBuffer<Element>) == 8, "Incorrect CUDA buffer layout.");
    static_assert(sizeof(ByteAddressBuffer) == 8, "Incorrect CUDA buffer layout.");
    static_assert(sizeof(RWByteAddressBuffer) == 8, "Incorrect CUDA buffer layout.");
    static_assert(alignof(StructuredBuffer<Element>) == 8, "Incorrect CUDA buffer layout.");
    static_assert(alignof(RWStructuredBuffer<Element>) == 8, "Incorrect CUDA buffer layout.");
    static_assert(alignof(ByteAddressBuffer) == 8, "Incorrect CUDA buffer layout.");
    static_assert(alignof(RWByteAddressBuffer) == 8, "Incorrect CUDA buffer layout.");
    static_assert(
        sizeof(DescriptorHandle<StructuredBuffer<Element>>) == 8,
        "Incorrect CUDA buffer layout.");
    static_assert(
        sizeof(DescriptorHandle<RWStructuredBuffer<Element>>) == 8,
        "Incorrect CUDA buffer layout.");
    static_assert(
        sizeof(DescriptorHandle<ByteAddressBuffer>) == 8,
        "Incorrect CUDA buffer layout.");
    static_assert(
        sizeof(DescriptorHandle<RWByteAddressBuffer>) == 8,
        "Incorrect CUDA buffer layout.");
    static_assert(sizeof(AppendStructuredBuffer<Element>) == 16, "Incorrect CUDA buffer layout.");
    static_assert(sizeof(ConsumeStructuredBuffer<Element>) == 16, "Incorrect CUDA buffer layout.");
    static_assert(sizeof(Buffers) == 80, "Incorrect CUDA buffer layout.");

    buffers.output[0] = buffers.input[0];
    buffers.mutableBytes.Store(0, buffers.bytes.Load(0));
}

// CHECK: __global__ void computeMain
// LAYOUT: "name": "input"
// LAYOUT: "offset": 0, "size": 8
// LAYOUT: "name": "afterInput"
// LAYOUT: "offset": 8, "size": 4
// LAYOUT: "name": "output"
// LAYOUT: "offset": 16, "size": 8
// LAYOUT: "name": "afterOutput"
// LAYOUT: "offset": 24, "size": 4
// LAYOUT: "name": "bytes"
// LAYOUT: "offset": 32, "size": 8
// LAYOUT: "name": "afterBytes"
// LAYOUT: "offset": 40, "size": 4
// LAYOUT: "name": "mutableBytes"
// LAYOUT: "offset": 48, "size": 8
// LAYOUT: "name": "afterMutableBytes"
// LAYOUT: "offset": 56, "size": 4
// LAYOUT: "name": "array"
// LAYOUT: "offset": 64, "size": 16
