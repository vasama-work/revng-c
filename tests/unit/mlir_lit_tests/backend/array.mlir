//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftemit --tagless --model %S/model.yml %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!int32_t = !clift.primitive<SignedKind 4>
!int32_t$p = !clift.pointer<pointer_size = 8, pointee_type = !int32_t>

!int32_t$1 = !clift.array<element_type = !int32_t, elements_count = 1>
!int32_t$1$p = !clift.pointer<pointer_size = 8, pointee_type = !int32_t$1>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: {
  clift.func @f<!f>() attributes { unique_handle = "0x40001001:Code_x86_64" } {
    // CHECK: int32_t array[1];
    %array = clift.local !int32_t$1 "array"

    // CHECK: array[0];
    clift.expr {
      %p = clift.cast<decay> %array : !int32_t$1 -> !int32_t$p
      %i = clift.imm 0 : !int32_t
      %r = clift.subscript %p, %i : (!int32_t$p, !int32_t)
      clift.yield %r : !int32_t
    }

    // CHECK: int32_t
    // CHECK: (
    // CHECK: *
    // CHECK: p_array
    // CHECK: )
    // CHECK: [1]
    %p_array = clift.local !int32_t$1$p "p_array" = {
      %r = clift.addressof %array : !int32_t$1$p
      clift.yield %r : !int32_t$1$p
    }

    // CHECK: (*p_array)[(0, 0)]
    clift.expr {
      %q = clift.indirection %p_array : !int32_t$1$p
      %p = clift.cast<decay> %q : !int32_t$1 -> !int32_t$p
      %i = clift.imm 0 : !int32_t
      %comma = clift.comma %i, %i : !int32_t, !int32_t
      %r = clift.subscript %p, %comma : (!int32_t$p, !int32_t)
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
