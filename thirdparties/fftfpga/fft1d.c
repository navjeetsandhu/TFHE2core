// Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#define CL_VERSION_2_0
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA
#include "CL/opencl.h"

#include "fpga_state.h"
#include "fftfpga.h"
#include "svm.h"
#include "opencl_utils.h"
#include "misc.h"

/**
 * \brief  compute an out-of-place single precision complex 1D-FFT on the FPGA
 * \param  N    : unsigned integer to the number of points in FFT1d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : toggle for backward transforms
 * \param  batch : number of batched executions of 1D FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_1d(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned batch){

    fpga_t fft_time = {0.0, 0.0, 0.0, 0};
    cl_kernel kernel1 = NULL, kernel2 = NULL, kernel3 = NULL, kernel4 = NULL;
    cl_int status = 0;

    // if N is not a power of 2
    if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
        return fft_time;
    }

    //printf("-- Launching%s 1D FFT of %d batches \n", inv ? " inverse":"", batch);

    queue_setup();

    cl_mem d_inData, d_outData, d_inData_2, d_outData_2;
    //printf("Launching%s FFT transform for %d batch \n", inv ? " inverse":"", batch);

    // Create device buffers - assign the buffers in different banks for more efficient memory access
    d_inData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * N * batch, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");

    d_inData_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * N * batch, NULL, &status);
    checkError(status, "Failed to allocate input device2 buffer\n");

    d_outData = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * batch, NULL, &status);
    checkError(status, "Failed to allocate output device buffer\n");

    d_outData_2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * batch, NULL, &status);
    checkError(status, "Failed to allocate output device2 buffer\n");

    //printf("-- Copying data from host to device\n");
    // Copy data from host to device
    status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N * batch, inp, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");

    status = clEnqueueWriteBuffer(queue3, d_inData, CL_TRUE, 0, sizeof(float2) * N * batch, inp, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device2");

    status = clFinish(queue1);
    checkError(status, "failed to finish writing buffer using PCIe");

    status = clFinish(queue3);
    checkError(status, "failed to finish writing buffer using PCIe 2");

    // Can't pass bool to device, so convert it to int
    int inverse_int = (int)inv;

    // Create Kernels - names must match the kernel name in the original CL file
    kernel1 = clCreateKernel(program, "fetch", &status);
    checkError(status, "Failed to create fetch kernel");

    kernel3 = clCreateKernel(program, "fetch_1", &status);
    checkError(status, "Failed to create fetch kernel3");

    kernel2 = clCreateKernel(program, "fft1d", &status);
    checkError(status, "Failed to create fft1d kernel");

    kernel4 = clCreateKernel(program, "fft1d_1", &status);
    checkError(status, "Failed to create fft1d kernel4");

    // Set the kernel arguments
    status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
    checkError(status, "Failed to set kernel1 arg 0");
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&d_outData);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*)&batch);
    checkError(status, "Failed to set kernel arg 1");
    status = clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set kernel arg 2");


    status = clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void *)&d_inData_2);
    checkError(status, "Failed to set kernel1 arg 0");
    status = clSetKernelArg(kernel4, 0, sizeof(cl_mem), (void *)&d_outData_2);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(kernel4, 1, sizeof(cl_int), (void*)&batch);
    checkError(status, "Failed to set kernel arg 1");
    status = clSetKernelArg(kernel4, 2, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set kernel arg 2");


    size_t ls = N/8;
    size_t gs = batch * ls;

    //printf("-- Executing kernels\n");
    cl_event startExec_event, endExec_event, startExec_event_2, endExec_event_2;
    // Measure execution time
    // Launch the kernel - we launch a single work item hence enqueue a task
    // FFT1d kernel is the SWI kernel
    status = clEnqueueTask(queue1, kernel2, 0, NULL, &endExec_event);
    checkError(status, "Failed to launch fft1d kernel");

    status = clEnqueueTask(queue3, kernel4, 0, NULL, &endExec_event_2);
    checkError(status, "Failed to launch fft1d kernel4");

    status = clEnqueueNDRangeKernel(queue2, kernel1, 1, NULL, &gs, &ls, 0, NULL, &startExec_event);
    checkError(status, "Failed to launch fetch kernel");

    status = clEnqueueNDRangeKernel(queue4, kernel3, 1, NULL, &gs, &ls, 0, NULL, &startExec_event_2);
    checkError(status, "Failed to launch fetch kernel3");

    // Wait for command queue to complete pending events
    status = clFinish(queue1);
    checkError(status, "Failed to finish queue1");
    status = clFinish(queue2);
    checkError(status, "Failed to finish queue2");
    status = clFinish(queue3);
    checkError(status, "Failed to finish queue3");
    status = clFinish(queue4);
    checkError(status, "Failed to finish queue4");

    // Record execution time
    cl_ulong kernel_start = 0, kernel_end = 0;
    clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
    fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

    // Copy results from device to host
    //printf("-- Transferring results back to host\n");
    status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * batch, out, 0, NULL, NULL);
    checkError(status, "Failed to copy data from device");

    status = clEnqueueReadBuffer(queue3, d_outData_2, CL_TRUE, 0, sizeof(float2) * N * batch, out, 0, NULL, NULL);
    checkError(status, "Failed to copy data from device2");

    status = clFinish(queue1);
    checkError(status, "failed to finish reading buffer using PCIe");

    status = clFinish(queue3);
    checkError(status, "failed to finish reading buffer using PCIe queue3");

    // Cleanup
    if (d_inData)
        clReleaseMemObject(d_inData);
    if (d_outData)
        clReleaseMemObject(d_outData);
    if (d_inData_2)
        clReleaseMemObject(d_inData_2);
    if (d_outData_2)
        clReleaseMemObject(d_outData_2);
    if(kernel1)
        clReleaseKernel(kernel1);
    if(kernel2)
        clReleaseKernel(kernel2);
    if(kernel3)
        clReleaseKernel(kernel3);
    if(kernel4)
        clReleaseKernel(kernel4);
    queue_cleanup();

    fft_time.valid = 1;
    return fft_time;
}

fpga_t fftfpgaf_c2c_1d_(const unsigned N, const float2 *inp, float2 *out, const bool inv){

    const unsigned batch = 1;
    fpga_t fft_time = {0.0, 0.0, 0.0, 0};
    cl_kernel kernel1 = NULL, kernel2 = NULL;
    cl_int status = 0;

    // if N is not a power of 2
    if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
        return fft_time;
    }

    //printf("-- Launching%s 1D FFT of %d batches \n", inv ? " inverse":"", batch);

    queue_setup();

    cl_mem d_inData, d_outData;
    //printf("Launching%s FFT transform for %d batch \n", inv ? " inverse":"", batch);

    // Create device buffers - assign the buffers in different banks for more efficient memory access
    d_inData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * N, NULL, &status);
    checkError(status, "Failed to allocate input device buffer\n");

    d_outData = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N, NULL, &status);
    checkError(status, "Failed to allocate output device buffer\n");

    //printf("-- Copying data from host to device\n");
    // Copy data from host to device
    status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N, inp, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");

    status = clFinish(queue1);
    checkError(status, "failed to finish writing buffer using PCIe");

    // Can't pass bool to device, so convert it to int
    int inverse_int = (int)inv;

    // Create Kernels - names must match the kernel name in the original CL file
    kernel1 = clCreateKernel(program, "fetch", &status);
    checkError(status, "Failed to create fetch kernel");

    kernel2 = clCreateKernel(program, "fft1d", &status);
    checkError(status, "Failed to create fft1d kernel");
    // Set the kernel arguments
    status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
    checkError(status, "Failed to set kernel1 arg 0");
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&d_outData);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*)&batch);
    checkError(status, "Failed to set kernel arg 1");
    status = clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set kernel arg 2");

    size_t ls = N/8;
    size_t gs = ls;

    //printf("-- Executing kernels\n");
    cl_event startExec_event, endExec_event;
    // Measure execution time
    // Launch the kernel - we launch a single work item hence enqueue a task
    // FFT1d kernel is the SWI kernel
    status = clEnqueueTask(queue1, kernel2, 0, NULL, &endExec_event);
    checkError(status, "Failed to launch fft1d kernel");

    status = clEnqueueNDRangeKernel(queue2, kernel1, 1, NULL, &gs, &ls, 0, NULL, &startExec_event);
    checkError(status, "Failed to launch fetch kernel");

    // Wait for command queue to complete pending events
    status = clFinish(queue1);
    checkError(status, "Failed to finish queue1");
    status = clFinish(queue2);
    checkError(status, "Failed to finish queue2");

    // Record execution time
    cl_ulong kernel_start = 0, kernel_end = 0;
    clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
    fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

    // Copy results from device to host
    //printf("-- Transferring results back to host\n");
    status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N, out, 0, NULL, NULL);
    checkError(status, "Failed to copy data from device");

    status = clFinish(queue1);
    checkError(status, "failed to finish reading buffer using PCIe");

    // Cleanup
    if (d_inData)
        clReleaseMemObject(d_inData);
    if (d_outData)
        clReleaseMemObject(d_outData);
    if(kernel1)
        clReleaseKernel(kernel1);
    if(kernel2)
        clReleaseKernel(kernel2);
    queue_cleanup();

    fft_time.valid = 1;
    return fft_time;
}
