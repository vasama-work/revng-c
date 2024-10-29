//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftemit --tagless --model %S/model.yml %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>

!f_args = !clift.defined<#clift.struct<
  id = 2004,
  name = "",
  size = 4,
  fields = [
    <
      offset = 0,
      name = "",
      type = !int32_t
    >
  ]>>

!f_args$ptr = !clift.pointer<pointer_size = 8, pointee_type = !f_args>

!f = !clift.defined<#clift.function<
  id = 1003,
  name = "",
  return_type = !int32_t,
  argument_types = [!f_args$ptr, !int32_t]>>

clift.module {
  // CHECK: {
  clift.func @f<!f>(%arg0 : !f_args$ptr, %arg1 : !int32_t) attributes {
    unique_handle = "0x40001003:Code_x86_64"
  } {
    // CHECK: return args->a + rcx;
    clift.return {
      %a = clift.access<indirect 0> %arg0 : !f_args$ptr -> !int32_t
      %r = clift.add %a, %arg1 : !int32_t
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
