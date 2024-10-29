#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

#include <string>

#include <cstdint>

namespace mlir::clift::c_backend {

struct PlatformInfo {
  uint8_t sizeof_char;
  uint8_t sizeof_short;
  uint8_t sizeof_int;
  uint8_t sizeof_long;
  uint8_t sizeof_longlong;
  uint8_t sizeof_pointer;
};

std::string emit(const PlatformInfo &Platform,
                 const model::Binary &Model,
                 ExpressionOpInterface Expr,
                 bool GeneratePlainC = false);

std::string emit(const PlatformInfo &Platform,
                 const model::Binary &Model,
                 FunctionOp Function,
                 bool GeneratePlainC = false,
                 bool EmitDeclaration = false);

} // namespace mlir::clift::c_backend
