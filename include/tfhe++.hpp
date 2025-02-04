#pragma once

#include "cloudkey.hpp"
#include "cmuxmem.hpp"
#include "detwfa.hpp"
#include "externs/cloudkey.hpp"
#include "externs/detwfa.hpp"
#include "externs/keyswitch.hpp"
#include "externs/tlwe.hpp"
#include "externs/trgsw.hpp"
#include "externs/trlwe.hpp"
#include "gate.hpp"
#include "blindrotate.hpp"
#include "gatebootstrapping.hpp"
#include "homdecomp.hpp"
#include "io-packet.hpp"
#include "key.hpp"
#include "keyswitch.hpp"
#include "params.hpp"
#include "tlwe.hpp"
#include "trgsw.hpp"
#include "decomposition.hpp"
#include "externalproduct.hpp"
#include "trlwe.hpp"
#include "c_assert.hpp"

#ifndef __clang__
// Because of some resons (may be clang bug?) this will gives linking error
// caused by mismatching name mangling.
#include "externs/gate.hpp"
#include "externs/gatebootstrapping.hpp"
#endif