!int32_t = !clift.primitive<SignedKind 4>

!factorial = !clift.defined<#clift.function<
  id = 1001,
  name = "",
  return_type = !int32_t,
  argument_types = [!int32_t]>>

clift.module {
  clift.global !int32_t @i
  clift.global !int32_t @r

  clift.func @factorial<!factorial>(%n : !int32_t) attributes {
    unique_handle = "0x40001001:Code_x86_64"
  } {
    clift.expr {
      %r = clift.use @r : !int32_t
      %1 = clift.imm 1 : !int32_t
      %y = clift.assign %r, %1 : !int32_t
      clift.yield %y : !int32_t
    }

    clift.expr {
      %i = clift.use @i : !int32_t
      %2 = clift.imm 2 : !int32_t
      %y = clift.assign %i, %2 : !int32_t
      clift.yield %y : !int32_t
    }

    clift.while {
      %i = clift.use @i : !int32_t
      %c = clift.le %i, %n : !int32_t -> !int32_t
      clift.yield %c : !int32_t
    } {
      clift.expr {
        %i = clift.use @i : !int32_t
        %r = clift.use @r : !int32_t
        %m = clift.mul %i, %r : !int32_t
        %y = clift.assign %r, %m : !int32_t
        clift.yield %y : !int32_t
      }
    }

    clift.return {
      %r = clift.use @r : !int32_t
      clift.yield %r : !int32_t
    }
  }
}
