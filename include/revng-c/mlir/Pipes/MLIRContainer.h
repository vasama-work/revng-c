#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

#include "revng/Pipeline/Container.h"

namespace revng::pipes {

class MLIRContainer : public pipeline::Container<MLIRContainer>
{
public:
  static const char ID;
  static constexpr auto MIMEType = "text/mlir";

private:
  mlir::MLIRContext Context;
  mlir::ModuleOp Module;

public:
  MLIRContainer();

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Targets) const override;

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override;

  pipeline::TargetsList enumerate() const override;

  bool remove(const pipeline::TargetsList &Targets) override;

  llvm::Error serialize(llvm::raw_ostream &OS) const override;
  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override;

  static std::vector<pipeline::Kind *> possibleKinds();
};

} // namespace revng::pipes
