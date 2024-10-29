
#include "revng/Pipeline/RegisterPipe.h"

#include "revng-c/mlir/Dialect/Clift/Utils/CBackend.h"

using namespace mlir::clift;

namespace {

class EmitCPipe {
public:
  static constexpr auto Name = "emit-clift-to-c";

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::MLIRContainer &MLIRContainer,
           DecompiledFileContainer &OutCFile) {
    mlir::ModuleOp Module = MLIRContainer.getModule();

    PlatformInfo Platform = {
      .sizeof_char = 1,
      .sizeof_short = 2,
      .sizeof_int = 4,
      .sizeof_long = 8,
      .sizeof_longlong = 8,
    };

    Module->walk([&](clift::FunctionOp F) {
      c_backend::emit(Platform, F, true);
    });
  }
};

static pipeline::RegisterPipe<EmitCPipe> X;

} // namespace
