#
# Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_executable(revng-clift-import-model Main.cpp)

target_link_libraries(revng-clift-import-model revngcSupport revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES} MLIRCliftDialect)
