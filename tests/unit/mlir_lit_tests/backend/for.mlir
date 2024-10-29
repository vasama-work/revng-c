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
    // CHECK: for (
    // CHECK: ;
    // CHECK: ;
    // CHECK: )
    clift.for {} {} {} {
      // CHECK: 1;
      clift.expr {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      }
    }

    // CHECK: for (
    // CHECK: ;
    // CHECK: 2;
    // CHECK: 3)
    // CHECK: {
    clift.for {} {
      %2 = clift.imm 2 : !int32_t
      clift.yield %2 : !int32_t
    }{
      %3 = clift.imm 3 : !int32_t
      clift.yield %3 : !int32_t
    } {
      // CHECK: 4;
      clift.expr {
        %4 = clift.imm 4 : !int32_t
        clift.yield %4 : !int32_t
      }
      // CHECK: 5;
      clift.expr {
        %5 = clift.imm 5 : !int32_t
        clift.yield %5 : !int32_t
      }
    }
    // CHECK: }
  }
  // CHECK: }
}
