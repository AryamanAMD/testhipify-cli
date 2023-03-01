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

/*
 * This sample demonstrates how 2D convolutions
 * with very large kernel sizes
 * can be efficiently implemented
 * using FFT transformations.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

// Helper functions for CUDA
#include "helper_functions.h"
#include "helper_cuda_hipified.h"
#include "HIPCHECK.h"
#include "convolutionFFT2D_common_hipified.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
int snapTransformSize(int dataSize) {
  int hiBit;
  unsigned int lowPOT, hiPOT;

  dataSize = iAlignUp(dataSize, 16);

  for (hiBit = 31; hiBit >= 0; hiBit--)
    if (dataSize & (1U << hiBit)) {
      break;
    }

  lowPOT = 1U << hiBit;

  if (lowPOT == (unsigned int)dataSize) {
    return dataSize;
  }

  hiPOT = 1U << (hiBit + 1);

  if (hiPOT <= 1024) {
    return hiPOT;
  } else {
    return iAlignUp(dataSize, 512);
  }
}

float getRand(void) { return (float)(rand() % 16); }

bool test0(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_PaddedData, *d_Kernel, *d_PaddedKernel;

  fComplex *d_DataSpectrum, *d_KernelSpectrum;

  hipfftHandle fftPlanFwd, fftPlanInv;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing built-in R2C / C2R FFT-based convolution\n");
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  HIPCHECK(hipMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  HIPCHECK(
      hipMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  HIPCHECK(
      hipMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  HIPCHECK(
      hipMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  HIPCHECK(hipMalloc((void **)&d_DataSpectrum,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));
  HIPCHECK(hipMalloc((void **)&d_KernelSpectrum,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));
  HIPCHECK(hipMemset(d_KernelSpectrum, 0,
                             fftH * (fftW / 2 + 1) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
  HIPCHECK(hipfftPlan2d(&fftPlanFwd, fftH, fftW, HIPFFT_R2C));
  HIPCHECK(hipfftPlan2d(&fftPlanInv, fftH, fftW, HIPFFT_C2R));

  printf("...uploading to GPU and padding convolution kernel and input data\n");
  HIPCHECK(hipMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             hipMemcpyHostToDevice));
  HIPCHECK(hipMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
  HIPCHECK(hipMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  // Not including kernel transformation into time measurement,
  // since convolution kernel is not changed very frequently
  printf("...transforming convolution kernel\n");
  HIPCHECK(hipfftExecR2C(fftPlanFwd, (hipfftReal *)d_PaddedKernel,
                               (hipfftComplex *)d_KernelSpectrum));

  printf("...running GPU FFT convolution: ");
  HIPCHECK(hipDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  HIPCHECK(hipfftExecR2C(fftPlanFwd, (hipfftReal *)d_PaddedData,
                               (hipfftComplex *)d_DataSpectrum));
  modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
  HIPCHECK(hipfftExecC2R(fftPlanInv, (hipfftComplex *)d_DataSpectrum,
                               (hipfftReal *)d_PaddedData));

  HIPCHECK(hipDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU convolution results\n");
  HIPCHECK(hipMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             hipMemcpyDeviceToHost));

  printf("...running reference CPU convolution\n");
  convolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++)
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);

  HIPCHECK(hipfftDestroy(fftPlanInv));
  HIPCHECK(hipfftDestroy(fftPlanFwd));

  HIPCHECK(hipFree(d_DataSpectrum));
  HIPCHECK(hipFree(d_KernelSpectrum));
  HIPCHECK(hipFree(d_PaddedData));
  HIPCHECK(hipFree(d_PaddedKernel));
  HIPCHECK(hipFree(d_Data));
  HIPCHECK(hipFree(d_Kernel));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Data);
  free(h_Kernel);

  return bRetVal;
}

bool test1(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;

  fComplex *d_DataSpectrum0, *d_KernelSpectrum0, *d_DataSpectrum,
      *d_KernelSpectrum;

  hipfftHandle fftPlan;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing custom R2C / C2R FFT-based convolution\n");
  const uint fftPadding = 16;
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  HIPCHECK(hipMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  HIPCHECK(
      hipMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  HIPCHECK(
      hipMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  HIPCHECK(
      hipMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  HIPCHECK(hipMalloc((void **)&d_DataSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  HIPCHECK(hipMalloc((void **)&d_KernelSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  HIPCHECK(
      hipMalloc((void **)&d_DataSpectrum,
                 fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));
  HIPCHECK(
      hipMalloc((void **)&d_KernelSpectrum,
                 fftH * (fftW / 2 + fftPadding) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
  HIPCHECK(hipfftPlan2d(&fftPlan, fftH, fftW / 2, HIPFFT_C2C));

  printf("...uploading to GPU and padding convolution kernel and input data\n");
  HIPCHECK(hipMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             hipMemcpyHostToDevice));
  HIPCHECK(hipMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
  HIPCHECK(hipMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  // HIPFFT_BACKWARD works just as well...
  const int FFT_DIR = HIPFFT_FORWARD;

  // Not including kernel transformation into time measurement,
  // since convolution kernel is not changed very frequently
  printf("...transforming convolution kernel\n");
  HIPCHECK(hipfftExecC2C(fftPlan, (hipfftComplex *)d_PaddedKernel,
                               (hipfftComplex *)d_KernelSpectrum0, FFT_DIR));
  spPostprocess2D(d_KernelSpectrum, d_KernelSpectrum0, fftH, fftW / 2,
                  fftPadding, FFT_DIR);

  printf("...running GPU FFT convolution: ");
  HIPCHECK(hipDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  HIPCHECK(hipfftExecC2C(fftPlan, (hipfftComplex *)d_PaddedData,
                               (hipfftComplex *)d_DataSpectrum0, FFT_DIR));

  spPostprocess2D(d_DataSpectrum, d_DataSpectrum0, fftH, fftW / 2, fftPadding,
                  FFT_DIR);
  modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW,
                       fftPadding);
  spPreprocess2D(d_DataSpectrum0, d_DataSpectrum, fftH, fftW / 2, fftPadding,
                 -FFT_DIR);

  HIPCHECK(hipfftExecC2C(fftPlan, (hipfftComplex *)d_DataSpectrum0,
                               (hipfftComplex *)d_PaddedData, -FFT_DIR));

  HIPCHECK(hipDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU FFT results\n");
  HIPCHECK(hipMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             hipMemcpyDeviceToHost));

  printf("...running reference CPU convolution\n");
  convolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++)
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);
  HIPCHECK(hipfftDestroy(fftPlan));

  HIPCHECK(hipFree(d_KernelSpectrum));
  HIPCHECK(hipFree(d_DataSpectrum));
  HIPCHECK(hipFree(d_KernelSpectrum0));
  HIPCHECK(hipFree(d_DataSpectrum0));
  HIPCHECK(hipFree(d_PaddedKernel));
  HIPCHECK(hipFree(d_PaddedData));
  HIPCHECK(hipFree(d_Kernel));
  HIPCHECK(hipFree(d_Data));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Kernel);
  free(h_Data);

  return bRetVal;
}

bool test2(void) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;

  fComplex *d_DataSpectrum0, *d_KernelSpectrum0;

  hipfftHandle fftPlan;

  bool bRetVal;
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);

  printf("Testing updated custom R2C / C2R FFT-based convolution\n");
  const int kernelH = 7;
  const int kernelW = 6;
  const int kernelY = 3;
  const int kernelX = 4;
  const int dataH = 2000;
  const int dataW = 2000;
  const int fftH = snapTransformSize(dataH + kernelH - 1);
  const int fftW = snapTransformSize(dataW + kernelW - 1);

  printf("...allocating memory\n");
  h_Data = (float *)malloc(dataH * dataW * sizeof(float));
  h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
  h_ResultCPU = (float *)malloc(dataH * dataW * sizeof(float));
  h_ResultGPU = (float *)malloc(fftH * fftW * sizeof(float));

  HIPCHECK(hipMalloc((void **)&d_Data, dataH * dataW * sizeof(float)));
  HIPCHECK(
      hipMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

  HIPCHECK(
      hipMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
  HIPCHECK(
      hipMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

  HIPCHECK(hipMalloc((void **)&d_DataSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));
  HIPCHECK(hipMalloc((void **)&d_KernelSpectrum0,
                             fftH * (fftW / 2) * sizeof(fComplex)));

  printf("...generating random input data\n");
  srand(2010);

  for (int i = 0; i < dataH * dataW; i++) {
    h_Data[i] = getRand();
  }

  for (int i = 0; i < kernelH * kernelW; i++) {
    h_Kernel[i] = getRand();
  }

  printf("...creating C2C FFT plan for %i x %i\n", fftH, fftW / 2);
  HIPCHECK(hipfftPlan2d(&fftPlan, fftH, fftW / 2, HIPFFT_C2C));

  printf("...uploading to GPU and padding convolution kernel and input data\n");
  HIPCHECK(hipMemcpy(d_Data, h_Data, dataH * dataW * sizeof(float),
                             hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(d_Kernel, h_Kernel,
                             kernelH * kernelW * sizeof(float),
                             hipMemcpyHostToDevice));
  HIPCHECK(hipMemset(d_PaddedData, 0, fftH * fftW * sizeof(float)));
  HIPCHECK(hipMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));

  padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW, dataH, dataW, kernelH,
                       kernelW, kernelY, kernelX);

  padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY,
            kernelX);

  // HIPFFT_BACKWARD works just as well...
  const int FFT_DIR = HIPFFT_FORWARD;

  // Not including kernel transformation into time measurement,
  // since convolution kernel is not changed very frequently
  printf("...transforming convolution kernel\n");
  HIPCHECK(hipfftExecC2C(fftPlan, (hipfftComplex *)d_PaddedKernel,
                               (hipfftComplex *)d_KernelSpectrum0, FFT_DIR));

  printf("...running GPU FFT convolution: ");
  HIPCHECK(hipDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  HIPCHECK(hipfftExecC2C(fftPlan, (hipfftComplex *)d_PaddedData,
                               (hipfftComplex *)d_DataSpectrum0, FFT_DIR));
  spProcess2D(d_DataSpectrum0, d_DataSpectrum0, d_KernelSpectrum0, fftH,
              fftW / 2, FFT_DIR);
  HIPCHECK(hipfftExecC2C(fftPlan, (hipfftComplex *)d_DataSpectrum0,
                               (hipfftComplex *)d_PaddedData, -FFT_DIR));

  HIPCHECK(hipDeviceSynchronize());
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer);
  printf("%f MPix/s (%f ms)\n",
         (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

  printf("...reading back GPU FFT results\n");
  HIPCHECK(hipMemcpy(h_ResultGPU, d_PaddedData,
                             fftH * fftW * sizeof(float),
                             hipMemcpyDeviceToHost));

  printf("...running reference CPU convolution\n");
  convolutionClampToBorderCPU(h_ResultCPU, h_Data, h_Kernel, dataH, dataW,
                              kernelH, kernelW, kernelY, kernelX);

  printf("...comparing the results: ");
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  double max_delta_ref = 0;

  for (int y = 0; y < dataH; y++) {
    for (int x = 0; x < dataW; x++) {
      double rCPU = (double)h_ResultCPU[y * dataW + x];
      double rGPU = (double)h_ResultGPU[y * fftW + x];
      double delta = (rCPU - rGPU) * (rCPU - rGPU);
      double ref = rCPU * rCPU + rCPU * rCPU;

      if ((delta / ref) > max_delta_ref) {
        max_delta_ref = delta / ref;
      }

      sum_delta2 += delta;
      sum_ref2 += ref;
    }
  }

  double L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
  bRetVal = (L2norm < 1e-6) ? true : false;
  printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

  printf("...shutting down\n");
  sdkDeleteTimer(&hTimer);
  HIPCHECK(hipfftDestroy(fftPlan));

  HIPCHECK(hipFree(d_KernelSpectrum0));
  HIPCHECK(hipFree(d_DataSpectrum0));
  HIPCHECK(hipFree(d_PaddedKernel));
  HIPCHECK(hipFree(d_PaddedData));
  HIPCHECK(hipFree(d_Kernel));
  HIPCHECK(hipFree(d_Data));

  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Kernel);
  free(h_Data);

  return bRetVal;
}

int main(int argc, char **argv) {
  printf("[%s] - Starting...\n", argv[0]);

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  int nFailures = 0;

  if (!test0()) {
    nFailures++;
  }

  if (!test1()) {
    nFailures++;
  }

  if (!test2()) {
    nFailures++;
  }

  printf("Test Summary: %d errors\n", nFailures);

  if (nFailures > 0) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
