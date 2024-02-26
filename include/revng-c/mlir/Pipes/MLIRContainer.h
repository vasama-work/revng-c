#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "revng/Pipeline/Container.h"

namespace revng::pipes {

class MLIRContainer : public pipeline::Container<MLIRContainer> {
public:
  static const char ID;
  static constexpr auto Name = "mlir-module-2";
  static constexpr auto MIMEType = "text/mlir";

private:
  // NOTE: Mutability of the context is required for cloneFiltered.
  //       The thread safety of this is not clear at the time of writing,
  //       but it was decided that thread safety of cloneFiltered was not
  //       necessary at this time.
  mutable mlir::MLIRContext Context;
  mutable mlir::OwningOpRef<mlir::ModuleOp> Module;

public:
  MLIRContainer(llvm::StringRef Name);

  mlir::MLIRContext &getContext() {
    return Context;
  }

  void setModule(mlir::OwningOpRef<mlir::ModuleOp> &&NewModule) {
    Module = std::move(NewModule);
  }

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Targets) const override;

  void mergeBackImpl(MLIRContainer &&Container) override;

  pipeline::TargetsList enumerate() const override;

  bool remove(const pipeline::TargetsList &Targets) override;

  void clear() override;

  llvm::Error serialize(llvm::raw_ostream &OS) const override;
  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override;

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override;

  static std::vector<pipeline::Kind *> possibleKinds();
};

} // namespace revng::pipes
