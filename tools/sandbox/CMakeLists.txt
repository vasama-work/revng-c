#
# Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_executable(revng-sandbox Main.cpp)

target_link_libraries(revng-sandbox revngcSupport revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES} MLIRCliftDialect
					  MLIRDialect MLIRIR MLIRTransforms
					  MLIRBytecodeReader MLIRBytecodeWriter MLIRParser
         MLIRLLVMDialect
         MLIRDLTIDialect)
