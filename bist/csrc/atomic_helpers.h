#ifndef ATOMIC_HELPERS_H
#define ATOMIC_HELPERS_H


extern "C" {


__device__ uint64_t combine(uint32_t size, uint32_t id);
__device__ uint32_t extract_size(uint64_t combined);
__device__ uint32_t extract_id(uint64_t combined);

__device__ void atomic_max_update(uint64_t* addr, uint32_t new_size, uint32_t new_id);
///__device__ void atomic_min_update(uint64_t* addr, uint32_t new_size, uint32_t new_id);
__device__ void atomic_min_update_int(uint64_t* addr, int new_size, int new_id);
__device__ void atomic_max_update_int(uint64_t* addr, int new_size, int new_id);
__device__ void atomic_min_update_float(uint64_t* addr, float new_size, int new_id);
__device__ float atomicMaxFloat(float* address, float val);


// -- basic --
#include <stdio.h>
#include <cuda_runtime.h>

/*******************************************************


   Atomically,
   1. compare an old_size vs new_size
   2. update the old_size with the new_size AND update the old_id with the new_id?

   We do this by combining the 32-bit ints into a 64-bit uint when swapping :D

********************************************************/

// Combine size and id into a 64-bit value
__device__ inline uint64_t combine(uint32_t size, uint32_t id) {
  return (uint64_t(size) << 32) | id;
}
// Extract size from the combined 64-bit value
__device__ inline uint32_t extract_size(uint64_t combined) {
  return uint32_t(combined >> 32);
}

// Extract id from the combined 64-bit value
__device__ inline uint32_t extract_id(uint64_t combined) {
    return uint32_t(combined);
}



// Combine size and id into a 64-bit value
__device__ inline uint64_t _combine(uint32_t size, uint32_t id) {
    return (uint64_t(size) << 32) | id;
}


// Extract size from the combined 64-bit value
__device__ inline uint32_t _extract_size(uint64_t combined) {
    return uint32_t(combined >> 32);
}

// Extract id from the combined 64-bit value
__device__ inline uint32_t _extract_id(uint64_t combined) {
    return uint32_t(combined);
}

// Atomic compare-and-update
__device__ void inline atomic_max_update(uint64_t* addr, uint32_t new_size, uint32_t new_id) {
    uint64_t old = *addr;
    uint64_t new_value = _combine(new_size, new_id);

    while (true) {
        uint32_t old_size = _extract_size(old);
        if (new_size > old_size) {
          uint64_t prev = (uint64_t)atomicCAS((unsigned long long*)addr,
                                                (unsigned long long)old,
                                                (unsigned long long)new_value);
          if (prev == old) break; // Success
          old = prev;             // Retry with updated value
        } else {
            break; // No update needed
        }
    }
}



// Atomic compare-and-update
__device__ void inline atomic_min_update(uint64_t* addr, uint32_t new_size, uint32_t new_id) {
    uint64_t old = *addr;
    uint64_t new_value = _combine(new_size, new_id);

    while (true) {
        uint32_t old_size = _extract_size(old);
        if (new_size < old_size) {
          uint64_t prev = (uint64_t)atomicCAS((unsigned long long*)addr,
                                                (unsigned long long)old,
                                                (unsigned long long)new_value);
          if (prev == old) break; // Success
          old = prev;             // Retry with updated value
        } else {
            break; // No update needed
        }
    }
}

// Atomic compare-and-update
__device__ void inline atomic_min_update_int(uint64_t* addr, int new_size, int new_id) {
    uint64_t old = *addr;
    // uint64_t new_value = _combine(new_size, new_id);
    uint32_t new_size_32 = *reinterpret_cast<uint32_t*>(&new_size);
    uint32_t new_id_32 = *reinterpret_cast<uint32_t*>(&new_id);
    uint64_t new_value = _combine(new_size_32,new_id_32);


    while (true) {
        // uint32_t old_size = _extract_size(old);
        uint32_t old_size_32 = _extract_size(old);
        int old_size = *reinterpret_cast<int*>(&old_size_32);
        if (new_size < old_size) {
          uint64_t prev = (uint64_t)atomicCAS((unsigned long long*)addr,
                                                (unsigned long long)old,
                                                (unsigned long long)new_value);
          if (prev == old) break; // Success
          old = prev;             // Retry with updated value
        } else {
            break; // No update needed
        }
    }
}


// Atomic compare-and-update
__device__ void inline atomic_max_update_int(uint64_t* addr, int new_size, int new_id) {
    uint64_t old = *addr;
    // uint64_t new_value = _combine(new_size, new_id);
    uint32_t new_size_32 = *reinterpret_cast<uint32_t*>(&new_size);
    uint32_t new_id_32 = *reinterpret_cast<uint32_t*>(&new_id);
    uint64_t new_value = _combine(new_size_32,new_id_32);


    while (true) {
        // uint32_t old_size = _extract_size(old);
        uint32_t old_size_32 = _extract_size(old);
        int old_size = *reinterpret_cast<int*>(&old_size_32);
        if (new_size > old_size) {
          uint64_t prev = (uint64_t)atomicCAS((unsigned long long*)addr,
                                              (unsigned long long)old,
                                              (unsigned long long)new_value);
          if (prev == old) break; // Success
          old = prev;             // Retry with updated value
        } else {
            break; // No update needed
        }
    }
}

// Atomic compare-and-update
__device__ void inline atomic_min_update_float(uint64_t* addr, float new_size, int new_id) {
    uint64_t old = *addr;
    // uint64_t new_value = _combine((uint32_t)new_size, (uint32_t)new_id);
    uint32_t new_size_32 = *reinterpret_cast<uint32_t*>(&new_size);
    uint32_t new_id_32 = *reinterpret_cast<uint32_t*>(&new_id);
    uint64_t new_value = _combine(new_size_32,new_id_32);

    while (true) {
      // float old_size = static_cast<float>(_extract_size(old));
      uint32_t old_size_32 = _extract_size(old);
      float old_size = *reinterpret_cast<float*>(&old_size_32);
      if (new_size < old_size) {
          uint64_t prev = (uint64_t)atomicCAS((unsigned long long*)addr,
                                                (unsigned long long)old,
                                                (unsigned long long)new_value);
          if (prev == old) break; // Success
          old = prev;             // Retry with updated value
        } else {
            break; // No update needed
        }
    }
}


__device__ float inline atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;  // Treat float as int
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        // Convert int to float, check if `val` is greater, and update if needed
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);  // Retry if the value changed during update

    return __int_as_float(old);  // Return the final max value
}



}

#endif


/* __device__ uint64_t combine(uint32_t size, uint32_t id); */
/* __device__ uint32_t extract_size(uint64_t combined); */
/* __device__ uint32_t extract_id(uint64_t combined); */

/* __device__ void atomic_max_update(uint64_t* addr, uint32_t new_size, uint32_t new_id); */
/* ///__device__ void atomic_min_update(uint64_t* addr, uint32_t new_size, uint32_t new_id); */
/* __device__ void atomic_min_update_int(uint64_t* addr, int new_size, int new_id); */
/* __device__ void atomic_min_update_float(uint64_t* addr, float new_size, int new_id); */
/* __device__ float atomicMaxFloat(float* address, float val); */
