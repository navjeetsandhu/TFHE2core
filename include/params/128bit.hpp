#pragma once

#include <cmath>
#include <limits>

struct otherparam {
    static constexpr int batch = 150;
};

struct lvl0param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = 0;
    static constexpr int32_t key_value_diff = key_value_max - key_value_min;
    static constexpr std::uint32_t n = 636;  // dimension
    static constexpr std::uint32_t k = 1;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = 0.000'092'511'997'467'675'6;  // fresh noise  0.0000925119974676756
    using T = uint16_t;  // Torus representation
    static constexpr std::make_signed_t<T> mu =
        1LL << (std::numeric_limits<T>::digits - 3);
    static constexpr uint32_t plain_modulus = 8;
    static constexpr double delta =
        static_cast<double>(1ULL << std::numeric_limits<T>::digits) /
        plain_modulus;
};

struct lvlhalfparam {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = 0;
    static constexpr int32_t key_value_diff = key_value_max - key_value_min;
    static constexpr std::uint32_t n = 760;  // dimension
    static constexpr std::uint32_t k = 1;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = std::pow(2.0, -17);  // fresh noise
    using T = uint32_t;                                 // Torus representation
    static constexpr uint32_t plain_modulusbit = 3; // MSB 3 bits are for message 001_ (0x2) or 111_ (0xE)
    static constexpr T mu = 1U << (std::numeric_limits<T>::digits - plain_modulusbit); // mu = 1U << 29 if T is uint32_t;.
    static constexpr uint32_t plain_modulus = 1<<plain_modulusbit;
    static constexpr double maximum = static_cast<double>(1ULL << std::numeric_limits<T>::digits);
    static constexpr double delta = maximum/plain_modulus;
};

struct lvl1param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = -1;
    static constexpr std::uint32_t nbit =
        10;  // dimension must be a power of 2 for ease of polynomial
             // multiplication.
    static constexpr std::uint32_t n = 1 << nbit;  // dimension: 1024
    static constexpr std::uint32_t k = 1;
    static constexpr std::uint32_t l = 3; // used for decomposition
    static constexpr std::uint32_t Bgbit = 6; // used for decomposition
    static constexpr std::uint32_t Bg = 1 << Bgbit;  // 64: for decomposition
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = std::pow(2.0, -25);  // Gaussian fresh noise 2^{-25}, stdev
    using T = uint32_t;                                 // Torus representation
                                                          // mu is value for message if message is = 1
    static constexpr std::make_signed_t<T> mu = 1 << 29; // 536870912 (0x2000000) for 1
    static constexpr uint32_t plain_modulusbit = 3; // MSB 3 bits are for message 001_ (0x2) or 111_ (0xE)
    static constexpr uint32_t plain_modulus = 1<<plain_modulusbit;
    static constexpr double maximum = static_cast<double>(1ULL << std::numeric_limits<T>::digits);
    static constexpr double delta = maximum/plain_modulus;
};

struct lvl2param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = -1;
    static const std::uint32_t nbit = 11;  // dimension must be a power of 2 for
                                                   // ease of polynomial multiplication.
    static constexpr std::uint32_t n = 1 << nbit;  // dimension
    static constexpr std::uint32_t k = 1;
    static constexpr std::uint32_t l = 4; // used for decomposition
    static constexpr std::uint32_t Bgbit = 9;   // used for decomposition of polynomial
    static constexpr std::uint32_t Bg = 1 << Bgbit;  //512
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = std::pow(2.0, -47);  // fresh noise
    using T = uint64_t;                                 // Torus representation
    static constexpr std::make_signed_t<T> mu = 1LL << 61;
    static constexpr uint32_t plain_modulusbit = 3;
    static constexpr uint32_t plain_modulus = 1<<plain_modulusbit;
    static constexpr double delta = static_cast<double>(mu);  //amplify the message

};

struct lvl3param {
    static constexpr int32_t key_value_max = 1;
    static constexpr int32_t key_value_min = -1;
    static const std::uint32_t nbit = 13;  // dimension must be a power of 2 for
    // ease of polynomial multiplication.
    static constexpr std::uint32_t n = 1 << nbit;  // dimension
    static constexpr std::uint32_t k = 1;
    static constexpr std::uint32_t l = 4;
    static constexpr std::uint32_t Bgbit = 9;
    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = std::pow(2.0, -47);  // fresh noise
    using T = uint64_t;                                 // Torus representation
    static constexpr T mu = 1ULL << 61;
    static constexpr uint32_t plain_modulusbit = 31;
    static constexpr uint64_t plain_modulus = 1ULL << plain_modulusbit;
    static constexpr double delta = 1ULL << (64 - plain_modulusbit - 1);
};

// Key Switching parameters
struct lvl10param {
    static constexpr std::uint32_t t = 7;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        2;  // how many bit should be encrypted in keyswitching key
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = lvl0param::alpha;  // key noise
    using domainP = lvl1param;
    using targetP = lvl0param;
};

struct lvl1hparam {
    static constexpr std::uint32_t t =
        10;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        3;  // how many bit should be encrypted in keyswitching key
    static const inline double alpha = lvlhalfparam::alpha;  // key noise
    using domainP = lvl1param;
    using targetP = lvlhalfparam;
};

struct lvl11param {
    static constexpr std::uint32_t t = 6;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        4;  // how many bit should be encrypted in keyswitching key
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = lvl1param::alpha;  // key noise
    using domainP = lvl1param;
    using targetP = lvl1param;
};

struct lvl20param {
    static constexpr std::uint32_t t = 7;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        2;  // how many bit should be encrypted in keyswitching key
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = lvl0param::alpha;  // key noise
    using domainP = lvl2param;
    using targetP = lvl0param;
};

struct lvl21param {
    static constexpr std::uint32_t t = 8;  // number of addition in
                                           // keyswitching
    static constexpr std::uint32_t basebit =
        3;  // how many bit should be encrypted in keyswitching key
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = lvl1param::alpha;  // key noise
    using domainP = lvl2param;
    using targetP = lvl1param;
};

struct lvl22param {
    static constexpr std::uint32_t t =
        38;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        1;  // how many bit should be encrypted in keyswitching key
    static constexpr ErrorDistribution errordist =
        ErrorDistribution::ModularGaussian;
    static const inline double alpha = lvl2param::alpha;  // key noise
    using domainP = lvl2param;
    using targetP = lvl2param;
};

struct lvl31param {
    static constexpr std::uint32_t t = 7;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        2;  // how many bit should be encrypted in keyswitching key
    static const inline double alpha = lvl1param::alpha;  // key noise
    using domainP = lvl3param;
    using targetP = lvl1param;
};