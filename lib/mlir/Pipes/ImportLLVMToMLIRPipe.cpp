#include "revng-c/mlir/Pipes/MLIRContainer.h"

namespace {

class ImportLLVMToMLIRPipe {
public:
  static constexpr auto Name = "ImportLLVMToMLIRPipe";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(StackAccessesSegregated,
                                      0,
                                      MLIRLLVMModule,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(const pipeline::ExecutionContext &Ctx,
           pipeline::LLVMContainer &IRContainer,
           revng::pipes::MLIRContainer &MLIRContainer) {

    
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    OS << "mlir-translate -import-llvm -mlir-print-debuginfo module.ll -o "
          "module.mlir\n";
  }
};

static pipeline::RegisterPipe<ImportLLVMToMLIRPipe> X;
} // namespace
