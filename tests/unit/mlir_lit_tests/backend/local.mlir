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

!my_enum = !clift.defined<#clift.enum<
  id = 2001,
  name = "",
  underlying_type = !int32_t,
  fields = [
    <
      raw_value = 0,
      name = ""
    >
  ]>>

!my_struct = !clift.defined<#clift.struct<
  id = 2002,
  name = "",
  size = 4,
  fields = [
    <
      offset = 0,
      name = "",
      type = !int32_t
    >
  ]>>

!my_union = !clift.defined<#clift.union<
  id = 2003,
  name = "",
  fields = [
    <
      offset = 0,
      name = "",
      type = !int32_t
    >
  ]>>

clift.module {
  // CHECK: {
  clift.func @f<!f>() attributes { unique_handle = "0x40001001:Code_x86_64" } {
    // CHECK: enum my_enum e;
    %e = clift.local !my_enum "e"

    // CHECK: struct my_struct s;
    %s = clift.local !my_struct "s"

    // CHECK: union my_union u;
    %u = clift.local !my_union "u"

    // CHECK: int32_t i = 42;
    %i = clift.local !int32_t "i" = {
      %42 = clift.imm 42 : !int32_t
      clift.yield %42 : !int32_t
    }

    // e;
    clift.expr {
        clift.yield %e : !my_enum
    }

    // s;
    clift.expr {
        clift.yield %s : !my_struct
    }

    // u;
    clift.expr {
        clift.yield %u : !my_union
    }

    // i;
    clift.expr {
        clift.yield %i : !int32_t
    }
  }
  // CHECK: }
}
