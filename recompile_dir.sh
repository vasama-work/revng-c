#!/bin/sh

set -e

DIRECTORY_NAME=$1

####
#### Warnings that could be enabled in the future if we become better behaved
####
#
# -pedantic-errors

####
#### Warnings that are currently disabled, explained
####
#
##
## Warnings that need more work to be removed
##
#
# -Wno-self-assign
#  We still have global variables representing CSVs with the same name of
#  function arguments that were promoted from them.
#  This causes situations where we emit self-assignments, but they are actually
#  serializations of aguments to CSVs, or vice versa.
#  Once we manage to detect and avoid this situation with CSVs this warning can
#  be re-enabled.
#
# -Wno-unused-parameter
#  The ABI analysis may detect a register as argument, but that argumentn may be
#  unused, or even become unused after some optimizations.
#  This warning could be used to detect chances for improvements in ABI
#  analysis.
#  Also, we can try to detect when this happens, meaning that we don't emit the
#  argument name if we detect that it't unused.
#  If we manage to do so, we can re-enable this warning.
#
# -Wno-return-type
# -Wno-unused-variable
# -Wno-unused-value
#  These two tend to be emitted when some optimizations remove some branches
#  that are considered dead.
#  For now we have disabled them because they happen quite often, but we should
#  really look into this and figure out if something is wrong in the lifting
#  process or in the optimizations we do, because this should not really happen.
#
# -Wno-uninitialized
#  We happen to have uninitialized local variable that are used before being
#  written. They are leftover and should go away, but for now they are still
#  common, so this warning should be disabled.
#
##
## Warnings that are a symptom of broken emitted code and should be investigated
##
#
# -Wno-parentheses \
# -Wno-shift-count-negative \
# -Wno-tautological-constant-out-of-range-compare \
# -Wno-implicitly-unsigned-literal \
# -Wno-int-to-pointer-cast \
# -Wno-infinite-recursion \
# -Wno-constant-conversion \
# -Wno-shift-count-overflow \


for C_FILE in $(ls $DIRECTORY_NAME/*.c) ; do
  clang -c -fsyntax-only -Wall -Wextra -Werror -ferror-limit=0 \
    -Wno-self-assign \
    -Wno-unused-parameter \
    -Wno-return-type \
    -Wno-unused-variable \
    -Wno-unused-value \
    -Wno-uninitialized \
    -Wno-parentheses \
    -Wno-constant-conversion \
    -Wno-shift-count-negative \
    -Wno-shift-count-overflow \
    -Wno-implicitly-unsigned-literal \
    -Wno-tautological-constant-out-of-range-compare \
    -Wno-int-to-pointer-cast \
    -Wno-infinite-recursion \
    -Wno-incompatible-library-redeclaration \
    $C_FILE \
    -o /dev/null
done
