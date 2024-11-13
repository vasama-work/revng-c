//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt --global-to-local %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: clift.global !int32_t @g1
  clift.global !int32_t @g1

  // CHECK: clift.global !int32_t @g2
  clift.global !int32_t @g2

  // CHECK: @f1
  // CHECK: {
  clift.func @f1<!f>() {
    // CHECK: [[F1_G1:%[0-9]+]] = clift.local !int32_t "g1"
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: [[F1_G2:%[0-9]+]] = clift.use @g2 : !int32_t
      %g1 = clift.use @g1 : !int32_t
      %g2 = clift.use @g2 : !int32_t
      // CHECK: [[F1_R:%[0-9]+]] = clift.add [[F1_G1]], [[F1_G2]] : !int32_t
      %r = clift.add %g1, %g2 : !int32_t
      // CHECK: clift.yield [[F1_R]] : !int32_t
      clift.yield %r : !int32_t
    }
    // CHECK: }
  }
  // CHECK: }

  // CHECK: @f2
  // CHECK: {
  clift.func @f2<!f>() {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: [[F2_G2:%[0-9]+]] = clift.use @g2 : !int32_t
      %g2 = clift.use @g2 : !int32_t
      // CHECK: clift.yield [[F2_G2]] : !int32_t
      clift.yield %g2 : !int32_t
    }
    // CHECK: }
  }
  // CHECK: }
}
