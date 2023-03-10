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


#include <hip/hip_runtime.h>
#include "common.h"
#include "helper_cuda_hipified.h"
// include kernels
#include "downscaleKernel_hipified.cuh"
#include "upscaleKernel_hipified.cuh"
#include "warpingKernel_hipified.cuh"
#include "derivativesKernel_hipified.cuh"
#include "solverKernel_hipified.cuh"
#include "addKernel_hipified.cuh"

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  nLevels      number of levels in a pyramid
/// \param[in]  nWarpIters   number of warping iterations per pyramid level
/// \param[in]  nSolverIters number of solver iterations (Jacobi iterations)
/// \param[out] u            horizontal displacement
/// \param[out] v            vertical displacement
///////////////////////////////////////////////////////////////////////////////
void ComputeFlowCUDA(const float *I0, const float *I1, int width, int height,
                     int stride, float alpha, int nLevels, int nWarpIters,
                     int nSolverIters, float *u, float *v) {
  printf("Computing optical flow on GPU...\n");

  // pI0 and pI1 will hold device pointers
  const float **pI0 = new const float *[nLevels];
  const float **pI1 = new const float *[nLevels];

  int *pW = new int[nLevels];
  int *pH = new int[nLevels];
  int *pS = new int[nLevels];

  // device memory pointers
  float *d_tmp;
  float *d_du0;
  float *d_dv0;
  float *d_du1;
  float *d_dv1;

  float *d_Ix;
  float *d_Iy;
  float *d_Iz;

  float *d_u;
  float *d_v;
  float *d_nu;
  float *d_nv;

  const int dataSize = stride * height * sizeof(float);

  HIPCHECK(hipMalloc(&d_tmp, dataSize));
  HIPCHECK(hipMalloc(&d_du0, dataSize));
  HIPCHECK(hipMalloc(&d_dv0, dataSize));
  HIPCHECK(hipMalloc(&d_du1, dataSize));
  HIPCHECK(hipMalloc(&d_dv1, dataSize));

  HIPCHECK(hipMalloc(&d_Ix, dataSize));
  HIPCHECK(hipMalloc(&d_Iy, dataSize));
  HIPCHECK(hipMalloc(&d_Iz, dataSize));

  HIPCHECK(hipMalloc(&d_u, dataSize));
  HIPCHECK(hipMalloc(&d_v, dataSize));
  HIPCHECK(hipMalloc(&d_nu, dataSize));
  HIPCHECK(hipMalloc(&d_nv, dataSize));

  // prepare pyramid

  int currentLevel = nLevels - 1;
  // allocate GPU memory for input images
  HIPCHECK(hipMalloc(pI0 + currentLevel, dataSize));
  HIPCHECK(hipMalloc(pI1 + currentLevel, dataSize));

  HIPCHECK(hipMemcpy((void *)pI0[currentLevel], I0, dataSize,
                             hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy((void *)pI1[currentLevel], I1, dataSize,
                             hipMemcpyHostToDevice));

  pW[currentLevel] = width;
  pH[currentLevel] = height;
  pS[currentLevel] = stride;

  for (; currentLevel > 0; --currentLevel) {
    int nw = pW[currentLevel] / 2;
    int nh = pH[currentLevel] / 2;
    int ns = iAlignUp(nw);

    HIPCHECK(
        hipMalloc(pI0 + currentLevel - 1, ns * nh * sizeof(float)));
    HIPCHECK(
        hipMalloc(pI1 + currentLevel - 1, ns * nh * sizeof(float)));

    Downscale(pI0[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI0[currentLevel - 1]);

    Downscale(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI1[currentLevel - 1]);

    pW[currentLevel - 1] = nw;
    pH[currentLevel - 1] = nh;
    pS[currentLevel - 1] = ns;
  }

  HIPCHECK(hipMemset(d_u, 0, stride * height * sizeof(float)));
  HIPCHECK(hipMemset(d_v, 0, stride * height * sizeof(float)));

  // compute flow
  for (; currentLevel < nLevels; ++currentLevel) {
    for (int warpIter = 0; warpIter < nWarpIters; ++warpIter) {
      HIPCHECK(hipMemset(d_du0, 0, dataSize));
      HIPCHECK(hipMemset(d_dv0, 0, dataSize));

      HIPCHECK(hipMemset(d_du1, 0, dataSize));
      HIPCHECK(hipMemset(d_dv1, 0, dataSize));

      // on current level we compute optical flow
      // between frame 0 and warped frame 1
      WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                pS[currentLevel], d_u, d_v, d_tmp);

      ComputeDerivatives(pI0[currentLevel], d_tmp, pW[currentLevel],
                         pH[currentLevel], pS[currentLevel], d_Ix, d_Iy, d_Iz);

      for (int iter = 0; iter < nSolverIters; ++iter) {
        SolveForUpdate(d_du0, d_dv0, d_Ix, d_Iy, d_Iz, pW[currentLevel],
                       pH[currentLevel], pS[currentLevel], alpha, d_du1, d_dv1);

        Swap(d_du0, d_du1);
        Swap(d_dv0, d_dv1);
      }

      // update u, v
      Add(d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u);
      Add(d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v);
    }

    if (currentLevel != nLevels - 1) {
      // prolongate solution
      float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];

      Upscale(d_u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleX, d_nu);

      float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];

      Upscale(d_v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleY, d_nv);

      Swap(d_u, d_nu);
      Swap(d_v, d_nv);
    }
  }

  HIPCHECK(hipMemcpy(u, d_u, dataSize, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(v, d_v, dataSize, hipMemcpyDeviceToHost));

  // cleanup
  for (int i = 0; i < nLevels; ++i) {
    HIPCHECK(hipFree((void *)pI0[i]));
    HIPCHECK(hipFree((void *)pI1[i]));
  }

  delete[] pI0;
  delete[] pI1;
  delete[] pW;
  delete[] pH;
  delete[] pS;

  HIPCHECK(hipFree(d_tmp));
  HIPCHECK(hipFree(d_du0));
  HIPCHECK(hipFree(d_dv0));
  HIPCHECK(hipFree(d_du1));
  HIPCHECK(hipFree(d_dv1));
  HIPCHECK(hipFree(d_Ix));
  HIPCHECK(hipFree(d_Iy));
  HIPCHECK(hipFree(d_Iz));
  HIPCHECK(hipFree(d_nu));
  HIPCHECK(hipFree(d_nv));
  HIPCHECK(hipFree(d_u));
  HIPCHECK(hipFree(d_v));
}
