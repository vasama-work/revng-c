//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftemit --tagless --model %S/model.yml %s 2>&1 | FileCheck %s

!void = !clift.primitive<VoidKind 0>
!int32_t = !clift.primitive<SignedKind 4>
!int32_t$p = !clift.pointer<pointer_size = 8, pointee_type = !int32_t>

!f = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !void,
  argument_types = []>>

clift.module {
  // CHECK: {
  clift.func @f<!f>() attributes { unique_handle = "0x40001001:Code_x86_64" } {
    // CHECK: int32_t x;
    %x = clift.local !int32_t "x"

    // CHECK: int32_t* p;
    %p = clift.local !int32_t$p "p"

    // CHECK: -x;
    clift.expr {
      %r = clift.neg %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: ~x;
    clift.expr {
      %r = clift.bitnot %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: !x;
    clift.expr {
      %r = clift.not %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: ++x;
    clift.expr {
      %r = clift.inc %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: --x;
    clift.expr {
      %r = clift.dec %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x++;
    clift.expr {
      %r = clift.post_inc %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x--;
    clift.expr {
      %r = clift.post_dec %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: &x;
    clift.expr {
      %r = clift.addressof %x : !int32_t$p
      clift.yield %r : !int32_t$p
    }

    // CHECK: *p;
    clift.expr {
      %r = clift.indirection %p : !int32_t$p
      clift.yield %r : !int32_t
    }

    // CHECK: x + x;
    clift.expr {
      %r = clift.add %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x - x;
    clift.expr {
      %r = clift.sub %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x * x;
    clift.expr {
      %r = clift.mul %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x / x;
    clift.expr {
      %r = clift.div %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x % x;
    clift.expr {
      %r = clift.rem %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x && x;
    clift.expr {
      %r = clift.and %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x || x;
    clift.expr {
      %r = clift.or %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x & x;
    clift.expr {
      %r = clift.bitand %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x | x;
    clift.expr {
      %r = clift.bitor %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x ^ x;
    clift.expr {
      %r = clift.bitxor %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x << x;
    clift.expr {
      %r = clift.shl %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x >> x;
    clift.expr {
      %r = clift.shr %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x == x;
    clift.expr {
      %r = clift.eq %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x != x;
    clift.expr {
      %r = clift.ne %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x < x;
    clift.expr {
      %r = clift.lt %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x > x;
    clift.expr {
      %r = clift.gt %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x <= x;
    clift.expr {
      %r = clift.le %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x >= x;
    clift.expr {
      %r = clift.ge %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: x, x;
    clift.expr {
      %r = clift.comma %x, %x : !int32_t, !int32_t
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
