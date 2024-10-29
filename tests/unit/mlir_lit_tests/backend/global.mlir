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
  clift.global !int32_t @g {
    unique_handle = "0x40002001:Generic64-4"
  }

  // CHECK: {
  clift.func @f<!f>() attributes {
    unique_handle = "0x40001001:Code_x86_64"
  } {
    // CHECK: seg_0x40002001;
    clift.expr {
      %y = clift.use @g : !int32_t
      clift.yield %y : !int32_t
    }
  }
  // CHECK: }
}
