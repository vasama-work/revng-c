#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace revng::mlir_llvm {

using LocType = mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>;

inline mlir::Block &getModuleBlock(mlir::ModuleOp Module) {
  revng_assert(Module);
  revng_assert(Module->getNumRegions() == 1);
  return Module->getRegion(0).front();
}

inline decltype(auto) getModuleOperations(mlir::ModuleOp Module) {
  return getModuleBlock(Module).getOperations();
}

inline std::optional<llvm::StringRef> getLinkageName(mlir::Operation *const O) {
  if (const auto L = mlir::dyn_cast<LocType>(O->getLoc()))
    return L.getMetadata().getLinkageName();
  return std::nullopt;
}

inline std::optional<llvm::StringRef>
getFunctionName(mlir::FunctionOpInterface F) {
  if (const auto Name = getLinkageName(F)) {
    static constexpr llvm::StringRef Path = "/function/";
    revng_assert(Name->starts_with(Path));
    return Name->substr(Path.size());
  }
  return std::nullopt;
}

template<typename R, typename C, typename P>
P visitHelper(R (C::*)(P));
template<typename R, typename C, typename P>
P visitHelper(R (C::*)(P) const);

// Helper for visiting module operations non-recursively.
// Allows erasing the visited operation during visitation.
template<typename Visitor>
void visit(mlir::ModuleOp Module, Visitor visitor) {
  auto &OpList = getModuleOperations(Module);

  auto Begin = OpList.begin();
  const auto End = OpList.end();

  while (Begin != End) {
    using Type = decltype(visitHelper(&Visitor::operator()));
    mlir::Operation *const Operation = &*Begin++;
    if constexpr (std::is_same_v<Type, mlir::Operation *>) {
      visitor(Operation);
    } else if (auto O = mlir::dyn_cast<Type>(Operation)) {
      visitor(O);
    }
  }
}

} // namespace revng::mlir_llvm
