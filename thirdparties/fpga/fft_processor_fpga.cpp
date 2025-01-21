#include "fft_processor_fpga.h"
#include "fpga.h"
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <params.hpp>


#define CAST_DOUBLE_TO_UINT32(d) ((uint32_t)((int64_t)(d)))
#define max_batch 200


FFT_Processor_FPGA::FFT_Processor_FPGA(const int32_t N)
    : _2N(2 * N), N(N), Ns2(N / 2)
{
    inbuf = new float2[Ns2*max_batch]();
    outbuf = new float2[Ns2*max_batch]();

    for (int i = 0; i < Ns2; i++) {
        double value = (double)i * M_PI / (double)N;
        twist.push_back(std::complex<double>(std::cos(value), std::sin(value)));
    }
    fpga_initialize();
}

void FFT_Processor_FPGA::execute_reverse_int(double *res, const int32_t *a, unsigned batch)
{
    for(unsigned j = 0; j < batch; j++) {
        for (unsigned i = 0; i < Ns2; i++) {
            unsigned aIndex = (j*N) + i;
            unsigned index = (j*Ns2) + i;
            auto tmp =
                twist[i] * std::complex((double)a[aIndex], (double)a[Ns2 + aIndex]);
            inbuf[index].x = tmp.real();
            inbuf[index].y = tmp.imag();
        }
    }
    runTimeRc = fpga_fft(Ns2, inbuf, outbuf, false, batch);
    for(unsigned j = 0; j < batch; j++) {
        for (unsigned i = 0; i < Ns2; i++) {
            unsigned resIndex = (j*N) + i;
            unsigned index = (j*Ns2) + i;
            res[resIndex] = outbuf[index].x;
            res[resIndex + Ns2] = outbuf[index].y;
        }
    }
}

void FFT_Processor_FPGA::execute_reverse_torus32(double *res, const uint32_t *a, unsigned batch)
{
    execute_reverse_int(res, (int32_t *)a, batch);
}


void FFT_Processor_FPGA::execute_reverse_torus64(double *res, const uint64_t *a)
{
    for (int i = 0; i < Ns2; i++) {
        auto tmp = twist[i] * std::complex((double)((int64_t)a[i]),
                                           (double)((int64_t)a[Ns2 + i]));
        inbuf[i].x = tmp.real();
        inbuf[i].y = tmp.imag();
    }
    runTimeRc = fpga_fft(Ns2, inbuf, outbuf, false);
    for (int i = 0; i < Ns2; i++) {
        res[i] = outbuf[i].x;
        res[i + Ns2] = outbuf[i].y;
    }
}

void FFT_Processor_FPGA::execute_direct_torus32(uint32_t *res, const double *a, unsigned batch)
{
    for(unsigned j = 0; j < batch; j++) {
        for (unsigned i = 0; i < Ns2; i++) {
            unsigned aIndex = (j * N) + i;
            unsigned index = (j * Ns2) + i;
            inbuf[index].x = a[aIndex] / Ns2;
            inbuf[index].y = a[Ns2 + aIndex] / Ns2;
        }
    }
    runTimeRc = fpga_fft(Ns2, inbuf, outbuf, true, batch);
    for(unsigned j = 0; j < batch; j++) {
        for (unsigned i = 0; i < Ns2; i++) {
            unsigned resIndex = (j*N) + i;
            unsigned index = (j*Ns2) + i;
            auto res_tmp = std::complex<double>(outbuf[index].x, outbuf[index].y) *
                           std::conj(twist[i]);
            res[resIndex] = CAST_DOUBLE_TO_UINT32(res_tmp.real());
            res[resIndex + Ns2] = CAST_DOUBLE_TO_UINT32(res_tmp.imag());
        }
    }
}

void FFT_Processor_FPGA::execute_direct_torus32_rescale(uint32_t *res,
                                                        const double *a,
                                                        const double delta)
{
    for (int i = 0; i < Ns2; i++) {
        inbuf[i].x = a[i] / Ns2;
        inbuf[i].y  = a[Ns2 + i] / Ns2;
    }
    runTimeRc = fpga_fft(Ns2, inbuf, outbuf, true);
    for (int i = 0; i < Ns2; i++) {
        auto res_tmp = std::complex<double>(outbuf[i].x , outbuf[i].y) *
                       std::conj(twist[i]);
        res[i] = CAST_DOUBLE_TO_UINT32(res_tmp.real() / (delta / 4));
        res[i + Ns2] = CAST_DOUBLE_TO_UINT32(res_tmp.imag() / (delta / 4));
    }
}

void FFT_Processor_FPGA::execute_direct_torus64(uint64_t *res, const double *a)
{
    for (int i = 0; i < Ns2; i++) {
        inbuf[i].x = a[i] / Ns2;
        inbuf[i].y  = a[Ns2 + i] / Ns2;
    }
    runTimeRc = fpga_fft(Ns2, inbuf, outbuf, true);
    double tmp[N];
    for (int i = 0; i < Ns2; i++) {
        auto res_tmp = std::complex<double>(outbuf[i].x, outbuf[i].y) *
                       std::conj(twist[i]);
        tmp[i] = res_tmp.real();
        tmp[i + Ns2] = res_tmp.imag();
    }
    const uint64_t *const vals = (const uint64_t *)tmp;
    constexpr uint64_t valmask0 = 0x000FFFFFFFFFFFFFul;
    constexpr uint64_t valmask1 = 0x0010000000000000ul;
    constexpr uint16_t expmask0 = 0x07FFu;
    for (int i = 0; i < N; i++) {
        uint64_t val = (vals[i] & valmask0) | valmask1;  // mantissa on 53 bits
        uint16_t expo = (vals[i] >> 52) & expmask0;      // exponent 11 bits
        // 1023 -> 52th pos -> 0th pos
        // 1075 -> 52th pos -> 52th pos
        int16_t trans = expo - 1075;
        uint64_t val2 = trans > 0 ? (val << trans) : (val >> -trans);
        res[i] = (vals[i] >> 63) ? -val2 : val2;
    }
}

void FFT_Processor_FPGA::execute_direct_torus64_rescale(uint64_t *res,
                                                        const double *a,
                                                        const double delta)
{
    for (int i = 0; i < Ns2; i++) {
        inbuf[i].x = a[i] / Ns2;
        inbuf[i].y  = a[Ns2 + i] / Ns2;
    }
    runTimeRc = fpga_fft(Ns2, inbuf, outbuf, true);
    double tmp[N];
    for (int i = 0; i < Ns2; i++) {
        auto res_tmp = std::complex<double>(outbuf[i].x, outbuf[i].y) *
                       std::conj(twist[i]);
        tmp[i] = res_tmp.real();
        tmp[i + Ns2] = res_tmp.imag();
    }
    for (int i = 0; i < N; i++) res[i] = uint64_t(std::round(tmp[i] / (delta / 4)));
}

FFT_Processor_FPGA::~FFT_Processor_FPGA()
{

    delete[] inbuf;
    delete[] outbuf;

    fpga_close();
}


FFT_Processor_FPGA fftFpgaLvl1(TFHEpp::lvl1param::n);
