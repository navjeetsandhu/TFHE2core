#include <iostream>
#include "fpga.h"
#include <math.h>
#include <filesystem>

using namespace std;

unsigned bit_reversed(unsigned x, const unsigned bits) {
    unsigned y = 0;
    for (unsigned i = 0; i < bits; i++) {
        y <<= 1;
        y |= x & 1;
        x >>= 1;
    }
    return y;
}

void correct_data_order(float2* fpgaOut, const unsigned num, const unsigned batch)
{
    unsigned total_sz = batch * num;
    float2* tmp = new float2[total_sz]();
    unsigned log_dim = log2(num);
    for(unsigned j = 0; j < batch; j++) {
        for (unsigned i = 0; i < num; i++) {
            unsigned index = (j*num) + i;
            unsigned bit_rev = (j*num) + bit_reversed(i, log_dim);
            tmp[index].x = fpgaOut[bit_rev].x;
            tmp[index].y = fpgaOut[bit_rev].y;
        }
    }
    for(unsigned i = 0; i < total_sz; i++ ){
        fpgaOut[i].x = tmp[i].x;
        fpgaOut[i].y = tmp[i].y;
    }
    delete tmp;
}



void fpga_initialize() {
    //const char* platform = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
    const char* platform = "Intel(R) FPGA SDK for OpenCL(TM)";
    std::filesystem::path currentPath(__FILE__);
    std::string str =  currentPath.parent_path().string() + "/libs/aocx/fft1d.aocx";
    int isInit = fpga_initialize(platform, str.c_str(), false);
    if(isInit != 0){
        cerr << "FPGA initialization error\n";
    }
}


void fpga_close() {
    // destroy fpga state
    fpga_final();
}

inline bool isEven(int num) {
    return num % 2 == 0;
}


/**
 * @brief  compute an out-of-place single precision complex 1D-FFT on the FPGA
 * @param  N    : integer pointer to size of FFT3d
 * @param  inp  : float2 pointer to input data of size N
 * @param  out  : float2 pointer to output data of size N
 * @param  inv  : int toggle to activate backward FFT
 * @return int : time taken in milliseconds for data transfers and execution
 */
fpga_t fpga_fft(const unsigned num, const float2 *inp, float2 *out, const bool inv, const unsigned batch)
{
    fpga_t runtime ={0, 0, 0, 0,0, false};

    try {
        if(batch == 1){
            runtime = fftfpgaf_c2c_1d(num, inp, out, inv);
            correct_data_order(out, num, batch);
        }
        else if(isEven(batch)) {
            runtime = fftfpgaf_c2c_1d(num, inp, out, inv, batch);
            correct_data_order(out, num, batch);
        } else {
            cerr << batch << " batch has to be even" << endl;
        }
    }
    catch(const char* msg){
        cerr << msg << endl;
    }
    return runtime;
}