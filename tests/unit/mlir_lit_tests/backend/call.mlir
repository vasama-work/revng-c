//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftemit --tagless --model %S/model.yml %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>

!f = !clift.defined<#clift.function<
  id = 1002,
  name = "",
  return_type = !int32_t,
  argument_types = [!int32_t, !int32_t]>>

clift.module {
  // CHECK: {
  clift.func @f<!f>(%arg0 : !int32_t, %arg1 : !int32_t) attributes {
    unique_handle = "0x40001002:Code_x86_64"
  } {
    // CHECK: return
    clift.return {
      %f = clift.use @f : !f

      %comma = clift.comma %arg1, %arg1 : !int32_t, !int32_t

      // CHECK: fun_0x40001002(x, (y, y))
      %result = clift.call %f(%arg0, %comma) : !f

      clift.yield %result : !int32_t
    }
    // CHECK: ;
  }
  // CHECK: }
}
