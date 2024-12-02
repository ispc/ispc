# ISPC Intrinsics Documentation

This document describes the LLVM IR intrinsics used in ISPC and their lowering to LLVM IR instructions. These intrinsics provide essential functionality for atomic operations, type manipulation, vector operations, and memory access.

## Usage Notes

1. All intrinsics support ISPC's varying and uniform types through appropriate vector widths in LLVM IR.
2. Memory ordering options follow LLVM's memory ordering semantics (seq_cst, acquire, release, etc.).
3. Vector widths are determined by the target architecture and ISPC's compilation settings.

## Atomic Operations

### @llvm.ispc.atomicrmw
Performs an atomic read-modify-write operation.
```
@llvm.ispc.atomicrmw.<operation>.<ordering>(ptr, value)
```
- Lowered to: `atomicrmw` LLVM instruction with the corresponding operation and ordering option.
- Parameters:
  - ptr: Pointer to the memory location
  - value: Value to exchange
- Ordering options: seq_cst, acquire, release, etc.
- Supported operations: xchg, add, sub, and, or, etc.

### @llvm.ispc.cmpxchg
Atomic compare-and-swap operation.
```
@llvm.ispc.cmpxchg.<success_ordering>.<failure_ordering>(ptr, cmp, val)
```
- Lowered to: `cmpxchg` LLVM instruction with corresponding ordering options
- Parameters:
  - ptr: Pointer to the memory location
  - cmp: Expected value
  - val: New value
- Returns: Original value at the memory location

## Data Manipulation

### @llvm.ispc.bitcast
Performs bitwise conversion between types of the same size.
```
@llvm.ispc.bitcast.<source_type>(value, dummy)
```
- Lowered to: `bitcast` LLVM instruction
- Parameters:
  - value: Source value
  - dummy: Destination type placeholder used for type deduction

### @llvm.ispc.concat
Concatenates two vectors into a single vector of double width.
```
@llvm.ispc.concat(vector1, vector2)
```
- Lowered to: `shufflevector` LLVM instruction
- Creates a vector containing all elements from both input vectors

## Vector Operations

### @llvm.ispc.extract
Extracts a scalar element from a vector at specified index.
```
@llvm.ispc.extract(vector, index)
```
- Lowered to: `extractelement` LLVM instruction
- Parameters:
  - vector: Source vector
  - index: Element index to extract

### @llvm.ispc.insert
Inserts a scalar value into a vector at specified index.
```
@llvm.ispc.insert(vector, index, value)
```
- Lowered to: `insertelement` LLVM instruction
- Parameters:
  - vector: Target vector
  - index: Position to insert
  - value: Scalar value to insert

### @llvm.ispc.select
Performs element-wise selection between two vectors based on a mask.
```
@llvm.ispc.select(mask, true_value, false_value)
```
- Lowered to: `select` LLVM instruction
- Parameters:
  - mask: Vector of boolean values
  - true_value: Values to select when mask is true
  - false_value: Values to select when mask is false

### @llvm.ispc.packmask
Packs a vector mask into a scalar integer value.
```
@llvm.ispc.packmask(mask)
```
- Lowered to: Bitcast followed by zero extension
- Converts vector of i1 to integer where each bit represents a mask element

## Memory Operations

### @llvm.ispc.stream_load
Performs non-temporal (streaming) load from memory.
```
@llvm.ispc.stream_load(ptr, dummy)
```
- Lowered to: `load` with `!nontemporal` metadata
- Parameters:
  - ptr: Source memory address
  - dummy: Unused parameter for type information

### @llvm.ispc.stream_store
Performs non-temporal (streaming) store to memory.
```
@llvm.ispc.stream_store(ptr, value)
```
- Lowered to: `store` with `!nontemporal` metadata
- Parameters:
  - ptr: Destination memory address
  - value: Value to store

## Synchronization

### @llvm.ispc.fence
Inserts a memory fence for synchronization.
```
@llvm.ispc.fence.<ordering>()
```
- Lowered to: `fence` LLVM instruction
- Ordering options: seq_cst, acquire, release, etc.
- Ensures memory ordering across threads
