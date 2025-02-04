#pragma once
#include <array>
#include <cmath>
#include <cstdint>

namespace TFHEpp {

template <class T, size_t N>
struct alignas(64) aligned_array : public std::array<T, N> {};

enum class ErrorDistribution { ModularGaussian, CenteredBinomial };
#include "params/128bit.hpp"
struct lvl01param {
    using domainP = lvl0param;
    using targetP = lvl1param;
    static constexpr uint32_t Addends = 1;
};

struct lvlh1param {
    using domainP = lvlhalfparam;
    using targetP = lvl1param;
    static constexpr uint32_t Addends = 1;
};

struct lvl02param {
    using domainP = lvl0param;
    using targetP = lvl2param;
    static constexpr uint32_t Addends = 1;
};

struct lvlh2param {
    using domainP = lvlhalfparam;
    using targetP = lvl2param;
    static constexpr uint32_t Addends = 1;
};

template <class P>
using Key = std::array<typename P::T, P::k * P::n>;

template <class P>
using TLWE = std::array<typename P::T, P::k * P::n + 1>;

template <class P, int batch>
using TLWEn = std::array<TLWE<P>, batch>;

template <int N>
using BooleanArray = std::array<bool, N>;

template <int N, int batch>
using BooleanArrayn = std::array<BooleanArray<N>, batch>;

template <int batch>
using intArray = std::array<int, batch>;

template <class P>
using Polynomial = std::array<typename P::T, P::n>;

template <class P, int batch>
using Polynomialn = std::array<Polynomial<P>, batch>;

template <class P>
using UnsignedPolynomial = Polynomial<P>;
template <class P>
using PolynomialInFD = std::array<double, P::n>;

template <class P, int batch>
using PolynomialInFDn = std::array<PolynomialInFD<P>, batch>;

template <class P>
using DecomposedPolynomial = std::array<Polynomial<P>, P::l>;

template <class P, int batch>
using DecomposedPolynomialn = std::array<Polynomialn<P, batch>, P::l>;

template <class P>
using TRLWE = std::array<Polynomial<P>, P::k + 1>;

template <class P, int batch>
using TRLWEn = std::array<Polynomialn<P, batch>, P::k + 1>;

template <class P>
using TRLWEInFD = std::array<PolynomialInFD<P>, P::k + 1>;

template <class P, int batch>
using TRLWEInFDn = std::array<PolynomialInFDn<P, batch>, P::k + 1>;


template <class P>
using TRGSW = std::array<TRLWE<P>, (P::k + 1) * P::l>;

template <class P, int batch>
using TRGSWn = std::array<TRLWEn<P, batch>, (P::k + 1) * P::l>;

template <class P>
using TRGSWFFT = aligned_array<TRLWEInFD<P>, (P::k + 1) * P::l>;

template <class P, int batch>
using TRGSWFFTn = aligned_array<TRLWEInFDn<P, batch>, (P::k + 1) * P::l>;

template <class P>
using BootstrappingKeyElement =
    std::array<TRGSW<typename P::targetP>, P::domainP::key_value_diff>;
template <class P>
using BootstrappingKeyElementFFT =
    std::array<TRGSWFFT<typename P::targetP>, P::domainP::key_value_diff>;


template <class P>
using BootstrappingKey =
    std::array<BootstrappingKeyElement<P>, P::domainP::k * P::domainP::n>;
template <class P>
using BootstrappingKeyFFT =
    std::array<BootstrappingKeyElementFFT<P>,
               P::domainP::k * P::domainP::n / P::Addends>;


template <class P>
using KeySwitchingKey = std::array<
    std::array<std::array<TLWE<typename P::targetP>, (1 << P::basebit) - 1>,
               P::t>,
    P::domainP::k * P::domainP::n>;
template <class P>
using SubsetKeySwitchingKey = std::array<
    std::array<std::array<TLWE<typename P::targetP>, (1 << P::basebit) - 1>,
               P::t>,
    P::domainP::k * P::domainP::n - P::targetP::k * P::targetP::n>;
template <class P>
using TLWE2TRLWEIKSKey = std::array<
    std::array<std::array<TRLWE<typename P::targetP>, (1 << P::basebit) - 1>,
               P::t>,
    P::domainP::n>;
template <class P>
using AnnihilateKey = std::array<TRGSWFFT<P>, P::nbit>;
template <class P>
using PrivateKeySwitchingKey = std::array<
    std::array<std::array<TRLWE<typename P::targetP>, (1 << P::basebit) - 1>,
               P::t>,
    P::domainP::k * P::domainP::n + 1>;
template <class P>
using SubsetPrivateKeySwitchingKey = std::array<
    std::array<std::array<TRLWE<typename P::targetP>, (1 << P::basebit) - 1>,
               P::t>,
    P::targetP::k * P::targetP::n + 1>;
template <class P>
using relinKey = std::array<TRLWE<P>, P::l>;
template <class P>
using relinKeyFFT = std::array<TRLWEInFD<P>, P::l>;

#define TFHEPP_EXPLICIT_INSTANTIATION_TLWE(fun) \
    fun(lvl0param);                             \
    fun(lvlhalfparam);                          \
    fun(lvl1param);                             \
    fun(lvl2param);                             \
    fun(lvl3param);
#define TFHEPP_EXPLICIT_INSTANTIATION_TRLWE(fun) \
    fun(lvl1param);                              \
    fun(lvl2param);
#define TFHEPP_EXPLICIT_INSTANTIATION_BLIND_ROTATE(fun) \
    fun(lvl01param);                                    \
    fun(lvl02param);
#define TFHEPP_EXPLICIT_INSTANTIATION_KEY_SWITCH_TO_TLWE(fun) \
    fun(lvl10param);                                          \
    fun(lvl1hparam);                                          \
    fun(lvl20param);                                          \
    fun(lvl21param);
#define TFHEPP_EXPLICIT_INSTANTIATION_KEY_SWITCH_TO_TRLWE(fun) \
    fun(lvl11param);                                           \
    fun(lvl21param);                                           \
    fun(lvl22param);
#define TFHEPP_EXPLICIT_INSTANTIATION_SUBSET_KEY_SWITCH_TO_TLWE(fun) \
    fun(lvl21param);
#define TFHEPP_EXPLICIT_INSTANTIATION_SUBSET_KEY_SWITCH_TO_TRLWE(fun) \
    fun(lvl21param);
#define TFHEPP_EXPLICIT_INSTANTIATION_GATE_BATCH_IKSBR(fun) \
    fun(lvl10param, lvl01param, lvl1param::mu, otherparam::batch);
#define TFHEPP_EXPLICIT_INSTANTIATION_GATE_IKSBR(fun) \
    fun(lvl10param, lvl01param, lvl1param::mu);
#define TFHEPP_EXPLICIT_INSTANTIATION_GATE_BRIKS(fun) \
    fun(lvl01param, lvl1param::mu, lvl10param);
#define TFHEPP_EXPLICIT_INSTANTIATION_GATE(fun) \
    fun(lvl10param, lvl01param);                \
    fun(lvl10param, lvl02param);                \
    fun(lvl20param, lvl01param);                \
    fun(lvl20param, lvl02param);
#define TFHEPP_EXPLICIT_INSTANTIATION_CIRCUIT_KEY(fun) \
    fun(lvl02param, lvl21param);                       \
    fun(lvl02param, lvl22param);
#define TFHEPP_EXPLICIT_INSTANTIATION_CIRCUIT_BOOTSTRAPPING(fun) \
    fun(lvl10param, lvl02param, lvl21param);                     \
    fun(lvl10param, lvl02param, lvl22param);
#define TFHEPP_EXPLICIT_INSTANTIATION_CIRCUIT_BOOTSTRAPPING_SUBIKS(fun) \
    fun(lvl10param, lvl02param, lvl21param);
}  // namespace TFHEpp