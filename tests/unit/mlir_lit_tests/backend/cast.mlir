//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftemit --tagless --model %S/model.yml %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!int32_t = !clift.primitive<SignedKind 4>
!uint32_t = !clift.primitive<UnsignedKind 4>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: {
  clift.func @f<!f>() attributes { unique_handle = "0x40001001:Code_x86_64" } {
    // CHECK: (uint32_t)0;
    clift.expr {
      %0 = clift.imm 0 : !int32_t
      %1 = clift.cast<reinterpret> %0 : !int32_t -> !uint32_t
      clift.yield %1 : !uint32_t
    }
  }
  // CHECK: }
}
