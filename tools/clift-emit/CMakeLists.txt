#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(clift-emit Main.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(MLIR_LIBRARIES ${dialect_libs} ${conversion_libs})

target_link_libraries(clift-emit MLIRCliftDialect MLIRCliftUtils
                      ${MLIR_LIBRARIES} ${LLVM_LIBRARIES})
