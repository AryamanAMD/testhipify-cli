/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SAMPLES_SHFL_SCAN_UTIL_H_
#define SAMPLES_SHFL_SCAN_UTIL_H_

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                                  \
  do {                                                                        \
    /* Check synchronous errors, i.e. pre-launch */                           \
    hipError_t err = hipGetLastError();                                     \
    if (hipSuccess != err) {                                                 \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, \
              __LINE__, hipGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */                 \
    err = hipDeviceSynchronize();                                            \
    if (hipSuccess != err) {                                                 \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s!\n", __FILE__, \
              __LINE__, hipGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    hipError_t err = call;                                                   \
    if (hipSuccess != err) {                                                 \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, \
              __LINE__, hipGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#endif  // SAMPLES_SHFL_SCAN_UTIL_H_