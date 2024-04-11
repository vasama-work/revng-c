#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_mlir_dialect_library(MLIRCliftTransforms ImportModelTypes.cpp DEPENDS
                         LINK_LIBS PUBLIC MLIRCliftDialect)
