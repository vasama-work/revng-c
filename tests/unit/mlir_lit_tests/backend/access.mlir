//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftemit --tagless --model %S/model.yml %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>

!int32_t = !clift.primitive<SignedKind 4>

!s = !clift.defined<#clift.struct<
  id = 2002,
  name = "",
  size = 4,
  fields = [
    <
      name = "",
      offset = 0,
      type = !int32_t
    >,
    <
      name = "",
      offset = 4,
      type = !int32_t
    >
  ]>>
!s$p = !clift.pointer<pointer_size = 8, pointee_type = !s>

!u = !clift.defined<#clift.union<
  id = 2003,
  name = "",
  fields = [
    <
      name = "",
      offset = 0,
      type = !int32_t
    >,
    <
      name = "",
      offset = 0,
      type = !int32_t
    >
  ]>>
!u$p = !clift.pointer<pointer_size = 8, pointee_type = !u>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: {
  clift.func @f<!f>() attributes { unique_handle = "0x40001001:Code_x86_64" } {
    %s = clift.local !s$p "s"
    %u = clift.local !u$p "u"

    // CHECK: s->x;
    clift.expr {
      %a = clift.access<indirect 0> %s : !s$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: u->x;
    clift.expr {
      %a = clift.access<indirect 0> %u : !u$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: s->y;
    clift.expr {
      %a = clift.access<indirect 1> %s : !s$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: u->y;
    clift.expr {
      %a = clift.access<indirect 1> %u : !u$p -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: (*s).x;
    clift.expr {
      %v = clift.indirection %s : !s$p
      %a = clift.access<0> %v : !s -> !int32_t
      clift.yield %a : !int32_t
    }

    // CHECK: (*u).x;
    clift.expr {
      %v = clift.indirection %u : !u$p
      %a = clift.access<0> %v : !u -> !int32_t
      clift.yield %a : !int32_t
    }
  }
  // CHECK: }
}
