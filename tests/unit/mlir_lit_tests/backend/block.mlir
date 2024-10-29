//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftemit --tagless --model %S/model.yml %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: {
  clift.func @f<!f>() attributes { unique_handle = "0x40001001:Code_x86_64" } {
    // CHECK: {
    // CHECK: }
    clift.block {}

    // CHECK: {
    clift.block {
      // CHECK: 0;
      clift.expr {
        %0 = clift.imm 0 : !int32_t
        clift.yield %0 : !int32_t
      }
    }
    // CHECK: }

    // CHECK: if (1)
    // CHECK: {
    clift.if {
      %1 = clift.imm 1 : !int32_t
      clift.yield %1 : !int32_t
    } {
      // CHECK: {
      clift.block {
        // CHECK: 2;
        clift.expr {
          %2 = clift.imm 2 : !int32_t
          clift.yield %2 : !int32_t
        }
      }
      // CHECK: }
    }
    // CHECK: }
  }
  // CHECK: }
}
