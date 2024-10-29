//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"

#include "revng/ADT/RecursiveCoroutine.h"

#include "revng-c/Support/PTMLC.h"

#include "revng-c/mlir/Dialect/Clift/Utils/CBackend.h"

namespace clift = mlir::clift;
using namespace mlir::clift;
using namespace mlir::clift::c_backend;

namespace {

enum class CInteger {
  Char,
  Short,
  Int,
  Long,
  LongLong,
};

static CInteger getCInteger(const PlatformInfo &Platform,
                            const uint64_t IntegerSize) {
  if (IntegerSize == Platform.sizeof_char)
    return CInteger::Char;
  if (IntegerSize == Platform.sizeof_short)
    return CInteger::Short;
  if (IntegerSize == Platform.sizeof_int)
    return CInteger::Int;
  if (IntegerSize == Platform.sizeof_long)
    return CInteger::Long;
  if (IntegerSize == Platform.sizeof_longlong)
    return CInteger::LongLong;
  revng_abort("Invalid integer size");
}

static std::pair<CInteger, bool> getCInteger(const PlatformInfo &Platform,
                                             PrimitiveType IntegerType) {
  const bool Signed = IntegerType.getKind() == PrimitiveKind::SignedKind;
  return { getCInteger(Platform, IntegerType.getSize()), Signed };
}

static llvm::StringRef getCIntegerLiteralSuffix(const CInteger Integer,
                                                const bool Signed) {
  switch (Integer) {
  case CInteger::Char:
  case CInteger::Short:
    break;

  case CInteger::Int:
    return Signed ? "" : "u";
  case CInteger::Long:
    return Signed ? "l" : "ul";
  case CInteger::LongLong:
    return Signed ? "ll" : "ull";
  }
  revng_abort("The requested integer literal suffix does not exist");
}


using Keyword = ptml::CBuilder::Keyword;
using Operator = ptml::CBuilder::Operator;

enum class OperatorPrecedence {
  Parentheses,
  Comma,
  Assignment,
  Or,
  And,
  Bitor,
  Bitxor,
  Bitand,
  Equality,
  Relational,
  Shift,
  Additive,
  Multiplicative,
  UnaryPrefix,
  UnaryPostfix,
  Primary,
};

class CEmitter {
public:
  explicit CEmitter(const PlatformInfo &Platform,
                    const model::Binary &Model,
                    llvm::raw_ostream &Out,
                    const bool GeneratePlainC) :
    Platform(Platform),
    Model(Model),
    NameBuilder(Model),
    C(GeneratePlainC),
    Out(Out, C) {}


  ptml::IndentedOstream::Scope enter() {
    return ptml::IndentedOstream::Scope(Out);
  }


  const model::Segment &getModelSegment(GlobalVariableOp Op) {
    auto SK = fromString<model::Segment::Key>(Op.getUniqueHandle());
    if (!SK)
      revng_abort("GlobalVariableOp missing segment key.");
    auto It = Model.Segments().find(*SK);
    if (It == Model.Segments().end())
      revng_abort("No matching model segment.");
    return *It;
  }

  const model::Function &getModelFunction(FunctionOp Op) {
    auto MA = MetaAddress::fromString(Op.getUniqueHandle());
    if (MA.isInvalid())
      revng_abort("FunctionOp missing meta address.");
    auto It = Model.Functions().find(MA);
    if (It == Model.Functions().end())
      revng_abort("No matching model function.");
    return *It;
  }

  const model::TypeDefinition *getModelTypeDefinition(uint64_t ID,
                                                      model::TypeDefinitionKind::Values Kind) {
    auto It = Model.TypeDefinitions().find({ ID, Kind });
    if (It == Model.TypeDefinitions().end())
      return nullptr;
    return It->get();
  }

  const model::TypeDefinition &getModelTypeDefinition(TypeDefinitionAttr Type) {
    if (mlir::isa<FunctionTypeAttr>(Type)) {
      auto CF = getModelTypeDefinition(Type.id(),
                                       model::TypeDefinitionKind::CABIFunctionDefinition);
      auto RF = getModelTypeDefinition(Type.id(),
                                       model::TypeDefinitionKind::RawFunctionDefinition);
      revng_assert(CF == nullptr or RF == nullptr);

      if (CF != nullptr)
        return *CF;

      if (RF != nullptr)
        return *RF;

      revng_abort("No matching model function type definition.");
    }

    model::TypeDefinitionKind::Values Kind;
    if (mlir::isa<TypedefTypeAttr>(Type))
      Kind = model::TypeDefinitionKind::TypedefDefinition;
    else if (mlir::isa<EnumTypeAttr>(Type))
      Kind = model::TypeDefinitionKind::EnumDefinition;
    else if (mlir::isa<StructTypeAttr>(Type))
      Kind = model::TypeDefinitionKind::StructDefinition;
    else if (mlir::isa<UnionTypeAttr>(Type))
      Kind = model::TypeDefinitionKind::UnionDefinition;
    else
      revng_abort();

    auto ModelType = getModelTypeDefinition(Type.id(), Kind);
    if (ModelType == nullptr)
      revng_abort("No matching model type definition.");
    return *ModelType;
  }

  void emitPrimitiveType(PrimitiveType Type) {
    llvm::SmallString<32> String;
    llvm::raw_svector_ostream Stream(String);

    const auto PrintStdTypedef = [&](llvm::StringRef Category) {
      Stream << Category;
      Stream << Type.getSize() * 8;
      Stream << "_t";
    };

    using enum PrimitiveKind;
    switch (Type.getKind()) {
    case VoidKind: {
      Stream << C.getKeyword(Keyword::Void);
    } break;
    case GenericKind: {
      PrintStdTypedef("generic");
    } break;
    case PointerOrNumberKind: {
      PrintStdTypedef("pointer_or_number");
    } break;
    case NumberKind: {
      PrintStdTypedef("number");
    } break;
    case UnsignedKind: {
      PrintStdTypedef("uint");
    } break;
    case SignedKind: {
      PrintStdTypedef("int");
    } break;
    case FloatKind: {
      PrintStdTypedef("float");
    } break;
    }

    Out << String;
  }

  struct DeclaratorNames {
    model::Identifier DeclaratorName;
    std::optional<llvm::ArrayRef<model::Identifier>> ParameterNames;
  };

  RecursiveCoroutine<void> emitDeclaration(ValueType Type,
                                           const DeclaratorNames *Names) {
    llvm::SmallVector<ValueType> Stack;

    while (true) {
      Stack.push_back(Type);
      if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
        emitPrimitiveType(T);
        break;
      } else if (auto T = mlir::dyn_cast<ArrayType>(Type)) {
        Type = T.getElementType();
      } else if (auto T = mlir::dyn_cast<PointerType>(Type)) {
        Type = T.getPointeeType();
      } else if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
        auto D = T.getElementType();

        #if 1
        if (auto F = mlir::dyn_cast<FunctionTypeAttr>(D)) {
          Type = F.getReturnType();
          continue;
        }
        #endif

        if (mlir::isa<EnumTypeAttr>(D))
          Out << C.getKeyword(Keyword::Enum);
        else if (mlir::isa<StructTypeAttr>(D))
          Out << C.getKeyword(Keyword::Struct);
        else if (mlir::isa<UnionTypeAttr>(D))
          Out << C.getKeyword(Keyword::Union);

        Out << ' ' << getModelTypeDefinition(D).CustomName();
        break;
      }
    }

    for (auto [RI, ST] : llvm::enumerate(std::views::reverse(Stack))) {
      const size_t I = Stack.size() - RI - 1;

      if (auto T = mlir::dyn_cast<PointerType>(ST)) {
        if (T.getPointerSize() != Platform.sizeof_pointer) {
          //WIP: How to deal with non-native-sized pointers?
          //Out << " __revng_ptr_size(" << T.getPointerSize() << ")";
        }
        Out << '*';
      } else if (mlir::isa<ArrayType>(ST)) {
        if (I != 0 and not mlir::isa<ArrayType>(Stack[I - 1]))
          Out << '(';
      } else if (auto DT = mlir::dyn_cast<DefinedType>(ST)) {
        if (I != 0 and mlir::isa<FunctionTypeAttr>(DT.getElementType()))
          Out << '(';
      }

      if (ST.isConst())
        Out << ' ' << C.getKeyword(Keyword::Const);
    }

    if (Names != nullptr)
      Out << ' ' << Names->DeclaratorName;

    for (auto [I, ST] : llvm::enumerate(Stack)) {
      if (auto T = mlir::dyn_cast<ArrayType>(ST)) {
        if (I != 0 and not mlir::isa<ArrayType>(Stack[I - 1]))
          Out << ')';

        Out << '[';
        Out << T.getElementsCount();
        Out << ']';
      } else if (auto T = mlir::dyn_cast<DefinedType>(ST)) {
        if (auto F = mlir::dyn_cast<FunctionTypeAttr>(T.getElementType())) {
          #if 1
          if (I != 0)
            Out << ')';

          Out << '(';
          for (auto [J, PT] : llvm::enumerate(F.getArgumentTypes())) {
            if (J != 0)
              Out << ',' << ' ';

            DeclaratorNames ParameterNames;
            DeclaratorNames *ParameterNamesPtr = nullptr;

            if (Names != nullptr && Names->ParameterNames) {
              ParameterNames.DeclaratorName = (*Names->ParameterNames)[J];
              ParameterNamesPtr = &ParameterNames;
            }

            rc_recur emitDeclaration(PT, ParameterNamesPtr);
          }
          Out << ')';
          #endif
        }
      }
    }
  }

  RecursiveCoroutine<void> emitType(ValueType Type) {
    return emitDeclaration(Type, nullptr);
  }


  static OperatorPrecedence decrementPrecedence(OperatorPrecedence Precedence) {
    revng_assert(Precedence != static_cast<OperatorPrecedence>(0));
    using T = std::underlying_type_t<OperatorPrecedence>;
    return static_cast<OperatorPrecedence>(static_cast<T>(Precedence) - 1);
  }

  ptml::Tag getIntegerConstant(uint64_t Value, CInteger Integer, bool Signed) {
    llvm::SmallString<64> String;
    {
      llvm::raw_svector_ostream Stream(String);

      if (Signed and static_cast<int64_t>(Value) < 0) {
        Stream << static_cast<int64_t>(Value);
      } else {
        Stream << Value;
      }

      Stream << getCIntegerLiteralSuffix(Integer, Signed);
    }
    return C.getConstantTag(String);
  }

  ptml::Tag getIntegerConstant(uint64_t Value, PrimitiveType Type) {
    auto [Integer, Signed] = getCInteger(Platform, Type);
    return getIntegerConstant(Value, Integer, Signed);
  }

  void emitIntegerImmediate(const uint64_t Value, ValueType Type) {
    Type = dealias(Type, /*IgnoreQualifiers=*/true);

    if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
      Out << getIntegerConstant(Value, T);
    } else {
      auto TypeAttr = mlir::cast<DefinedType>(Type).getElementType();
      const auto &ModelType = getModelTypeDefinition(TypeAttr);
      const auto &ModelEnum = llvm::cast<model::EnumDefinition>(ModelType);
      Out << NameBuilder.name(ModelEnum, Value);
    }
  }

  RecursiveCoroutine<void> emitImmediateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<ImmediateOp>();
    emitIntegerImmediate(E.getValue(), E.getResult().getType());
    rc_return;
  }

  RecursiveCoroutine<void> emitParameterExpression(mlir::Value V) {
    auto Arg = mlir::cast<mlir::BlockArgument>(V);
    Out << ParameterNames[Arg.getArgNumber()];
    rc_return;
  }

  RecursiveCoroutine<void> emitLocalVariableExpression(mlir::Value V) {
    auto Local = V.getDefiningOp<LocalVariableOp>();
    //WIP: Emit variable name from the model?
    Out << Local.getSymName();
    rc_return;
  }

  RecursiveCoroutine<void> emitUseExpression(mlir::Value V) {
    auto E = V.getDefiningOp<UseOp>();

    auto Module = E->getParentOfType<clift::ModuleOp>();
    revng_assert(Module);

    mlir::Operation *SymbolOp =
      mlir::SymbolTable::lookupSymbolIn(Module, E.getSymbolNameAttr());
    revng_assert(SymbolOp);

    if (auto G = mlir::dyn_cast<GlobalVariableOp>(SymbolOp)) {
      Out << NameBuilder.name(getModelSegment(G));
    } else if (auto F = mlir::dyn_cast<FunctionOp>(SymbolOp)) {
      Out << NameBuilder.name(getModelFunction(F));
    } else {
      revng_abort();
    }

    rc_return;
  }

  RecursiveCoroutine<void> emitAccessExpression(mlir::Value V) {
    auto E = V.getDefiningOp<AccessOp>();

    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);
    rc_recur emitExpression(E.getValue());

    Out << C.getOperator(E.isIndirect() ? Operator::Arrow : Operator::Dot);

    const model::TypeDefinition &ModelType =
      getModelTypeDefinition(E.getClassTypeAttr());

    if (auto *T = llvm::dyn_cast<model::StructDefinition>(&ModelType)) {
      Out << NameBuilder.name(*T, E.getFieldAttr().getOffset());
    } else if (auto *T = llvm::dyn_cast<model::UnionDefinition>(&ModelType)) {
      Out << NameBuilder.name(*T, E.getMemberIndex());
    }
  }

  RecursiveCoroutine<void> emitSubscriptExpression(mlir::Value V) {
    auto E = V.getDefiningOp<SubscriptOp>();

    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);
    rc_recur emitExpression(E.getPointer());

    // The precedence here could be parentheses and still preserve semantics,
    // but given that a comma expression within a subscript ( array[i, j] ) is
    // not only very confusing, but has a different meaning in C++23, we force
    // comma expressions to be parenthesized, the same way they are in argument
    // lists. The output in this case is as: array[(i, j)]
    CurrentPrecedence = OperatorPrecedence::Comma;

    Out << '[';
    rc_recur emitExpression(E.getIndex());
    Out << ']';
  }

  RecursiveCoroutine<void> emitCallExpression(mlir::Value V) {
    auto E = V.getDefiningOp<CallOp>();

    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);
    rc_recur emitExpression(E.getFunction());

    // The precedence here must be comma, because an argument list cannot
    // contain an unparenthesized comma expression. It would be parsed as two
    // arguments instead.
    CurrentPrecedence = OperatorPrecedence::Comma;

    Out << '(';
    for (auto [I, A] : llvm::enumerate(E.getArguments())) {
      if (I != 0)
        Out << ',' << ' ';

      rc_recur emitExpression(A);
    }
    Out << ')';
  }

  RecursiveCoroutine<void> emitCastExpression(mlir::Value V) {
    auto E = V.getDefiningOp<CastOp>();

    if (E.getKind() != CastKind::Decay) {
      Out << '(';
      rc_recur emitType(E.getResult().getType());
      Out << ')';
    }

    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);
    rc_recur emitExpression(E.getValue());
  }

  static ptml::CBuilder::Operator getOperator(mlir::Operation *Op) {
    if (mlir::isa<NegOp>(Op))
      return Operator::UnaryMinus;
    if (mlir::isa<AddOp>(Op))
      return Operator::Add;
    if (mlir::isa<SubOp>(Op))
      return Operator::Sub;
    if (mlir::isa<MulOp>(Op))
      return Operator::Mul;
    if (mlir::isa<DivOp>(Op))
      return Operator::Div;
    if (mlir::isa<RemOp>(Op))
      return Operator::Modulo;
    if (mlir::isa<LogicalNotOp>(Op))
      return Operator::BoolNot;
    if (mlir::isa<LogicalAndOp>(Op))
      return Operator::BoolAnd;
    if (mlir::isa<LogicalOrOp>(Op))
      return Operator::BoolOr;
    if (mlir::isa<BitwiseNotOp>(Op))
      return Operator::BinaryNot;
    if (mlir::isa<BitwiseAndOp>(Op))
      return Operator::And;
    if (mlir::isa<BitwiseOrOp>(Op))
      return Operator::Or;
    if (mlir::isa<BitwiseXorOp>(Op))
      return Operator::Xor;
    if (mlir::isa<ShiftLeftOp>(Op))
      return Operator::LShift;
    if (mlir::isa<ShiftRightOp>(Op))
      return Operator::RShift;
    if (mlir::isa<EqualOp>(Op))
      return Operator::CmpEq;
    if (mlir::isa<NotEqualOp>(Op))
      return Operator::CmpNeq;
    if (mlir::isa<LessThanOp>(Op))
      return Operator::CmpLt;
    if (mlir::isa<GreaterThanOp>(Op))
      return Operator::CmpGt;
    if (mlir::isa<LessThanOrEqualOp>(Op))
      return Operator::CmpLte;
    if (mlir::isa<GreaterThanOrEqualOp>(Op))
      return Operator::CmpGte;
    if (mlir::isa<IncrementOp, PostIncrementOp>(Op))
      return Operator::Increment;
    if (mlir::isa<DecrementOp, PostDecrementOp>(Op))
      return Operator::Decrement;
    if (mlir::isa<AddressofOp>(Op))
      return Operator::AddressOf;
    if (mlir::isa<IndirectionOp>(Op))
      return Operator::PointerDereference;
    if (mlir::isa<AssignOp>(Op))
      return Operator::Assign;
    if (mlir::isa<CommaOp>(Op))
      return Operator::Comma;
    revng_abort("This operation does not represent a C operator.");
  }

  RecursiveCoroutine<void> emitPrefixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();
    Out << C.getOperator(getOperator(Op));

    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPrefix);
    return emitExpression(Op->getOperand(0));
  }

  RecursiveCoroutine<void> emitPostfixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();
    rc_recur emitExpression(Op->getOperand(0));

    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);
    Out << C.getOperator(getOperator(Op));
  }

  RecursiveCoroutine<void> emitInfixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();

    auto LhsPrecedence = decrementPrecedence(CurrentPrecedence);
    auto RhsPrecedence = CurrentPrecedence;

    // Assignment operators are right-associative.
    if (CurrentPrecedence == OperatorPrecedence::Assignment)
      std::swap(LhsPrecedence, RhsPrecedence);

    CurrentPrecedence = LhsPrecedence;
    rc_recur emitExpression(Op->getOperand(0));

    if (not mlir::isa<CommaOp>(Op))
      Out << ' ';

    Out << C.getOperator(getOperator(Op)) << ' ';

    CurrentPrecedence = RhsPrecedence;
    rc_recur emitExpression(Op->getOperand(1));
  }

  struct ExpressionEmitInfo {
    OperatorPrecedence Precedence;
    RecursiveCoroutine<void>(CEmitter::* Emit)(mlir::Value V);
  };

  ExpressionEmitInfo getExpressionEmitInfo(mlir::Value V) {
    auto E = V.getDefiningOp<ExpressionOpInterface>();

    if (not E) {
      if (auto Variable = V.getDefiningOp<LocalVariableOp>()) {
        return {
          .Precedence = OperatorPrecedence::Primary,
          .Emit = &CEmitter::emitLocalVariableExpression,
        };
      }

      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CEmitter::emitParameterExpression,
      };
    }

    if (mlir::isa<ImmediateOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CEmitter::emitImmediateExpression,
      };
    }
    if (mlir::isa<UseOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CEmitter::emitUseExpression,
      };
    }
    if (mlir::isa<AccessOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CEmitter::emitAccessExpression,
      };
    }
    if (mlir::isa<SubscriptOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CEmitter::emitSubscriptExpression,
      };
    }
    if (mlir::isa<CallOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CEmitter::emitCallExpression,
      };
    }
    if (mlir::isa<PostIncrementOp, PostDecrementOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CEmitter::emitPostfixExpression,
      };
    }
    if (auto Cast = mlir::dyn_cast<CastOp>(E.getOperation())) {
      if (Cast.getKind() == CastKind::Decay) {
        return {
          .Precedence = OperatorPrecedence::Primary,
          .Emit = &CEmitter::emitCastExpression,
        };
      }
      return {
        .Precedence = OperatorPrecedence::UnaryPrefix,
        .Emit = &CEmitter::emitCastExpression,
      };
    }
    if (mlir::isa<NegOp, BitwiseNotOp, LogicalNotOp, IncrementOp, DecrementOp, AddressofOp, IndirectionOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPrefix,
        .Emit = &CEmitter::emitPrefixExpression,
      };
    }
    if (mlir::isa<MulOp, DivOp, RemOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Multiplicative,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<AddOp, SubOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Additive,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<ShiftLeftOp, ShiftRightOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Shift,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<LessThanOp, GreaterThanOp, LessThanOrEqualOp, GreaterThanOrEqualOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Relational,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<EqualOp, NotEqualOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Equality,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<BitwiseAndOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitand,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<BitwiseXorOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitxor,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<BitwiseOrOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitor,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<LogicalAndOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::And,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<LogicalOrOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Or,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<AssignOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Assignment,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    if (mlir::isa<CommaOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Comma,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }
    revng_abort("This operation is not supported.");
  }

  RecursiveCoroutine<void> emitExpression(mlir::Value V) {
    const ExpressionEmitInfo Info = getExpressionEmitInfo(V);

    bool PrintParentheses = Info.Precedence <= CurrentPrecedence and
                            Info.Precedence != OperatorPrecedence::Primary;

    if (PrintParentheses)
      Out << '(';

    // CurrentPrecedence is changed within this scope:
    {
      const auto PreviousPrecedence = CurrentPrecedence;
      const auto PrecedenceGuard = llvm::make_scope_exit([&]() {
        CurrentPrecedence = PreviousPrecedence;
      });
      CurrentPrecedence = Info.Precedence;

      rc_recur (this->*Info.Emit)(V);
    }

    if (PrintParentheses)
      Out << ')';
  }

  RecursiveCoroutine<void> emitExpressionRegion(mlir::Region &R) {
    revng_assert(R.hasOneBlock());
    mlir::Block &B = R.front();

    auto End = B.end();
    revng_assert(End != B.begin());

    auto Yield = mlir::cast<YieldOp>(*--End);
    return emitExpression(Yield.getValue());
  }

  RecursiveCoroutine<void> emitStatement(StatementOpInterface Op) {
    if (auto S = mlir::dyn_cast<MakeLabelOp>(Op.getOperation())) {
      // Do nothing.
    } else if (auto S = mlir::dyn_cast<AssignLabelOp>(Op.getOperation())) {
      Out << S.getLabelOp().getName() << ':';

      // Until C23, labels cannot be placed at the end of a block.
      if (Op.getOperation() == &S->getBlock()->back())
        Out << ' ' << ';';

      Out << '\n';
    } else if (auto S = mlir::dyn_cast<BlockStatementOp>(Op.getOperation())) {
      Out << '{' << '\n';
      enter(), rc_recur emitStatementRegion(S.getBody());
      Out << '}' << '\n';
    } else if (auto S = mlir::dyn_cast<DoWhileOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::Do) << '\n';
      rc_recur emitImplicitBlockStatement(S.getBody());

      Out << '\n' << C.getKeyword(Keyword::While) << ' ' << '(';
      rc_recur emitExpressionRegion(S.getCondition());
      Out << ')' << ';' << '\n';
    } else if (auto S = mlir::dyn_cast<ExpressionStatementOp>(Op.getOperation())) {
      rc_recur emitExpressionRegion(S.getExpression());
      Out << ';' << '\n';
    } else if (auto S = mlir::dyn_cast<ForOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::For) << ' ' << '(' << ';';

      if (not S.getCondition().empty()) {
        Out << ' ';
        rc_recur emitExpressionRegion(S.getCondition());
      }

      Out << ';';
      if (not S.getExpression().empty()) {
        Out << ' ';
        rc_recur emitExpressionRegion(S.getExpression());
      }

      Out << ')' << '\n';

      rc_recur emitImplicitBlockStatement(S.getBody());
    } else if (auto S = mlir::dyn_cast<GoToOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::Goto) << ' ' << S.getLabelOp().getName() << ';' << '\n';
    } else if (auto S = mlir::dyn_cast<IfOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::If) << ' ' << '(';
      rc_recur emitExpressionRegion(S.getCondition());
      Out << ')' << '\n';

      rc_recur emitImplicitBlockStatement(S.getThen());

      if (not S.getElse().empty()) {
        Out << C.getKeyword(Keyword::Else) << '\n';
        rc_recur emitImplicitBlockStatement(S.getElse());
      }
    } else if (auto S = mlir::dyn_cast<LocalVariableOp>(Op.getOperation())) {
      //WIP: Emit variable name from the model?
      DeclaratorNames Names = { S.getSymName(), {} };
      rc_recur emitDeclaration(S.getResult().getType(), &Names);

      if (not S.getInitializer().empty()) {
        Out << ' ' << '=' << ' ';

        // Comma expressions in a variable initialiser must be parenthesized.
        CurrentPrecedence = OperatorPrecedence::Comma;

        rc_recur emitExpressionRegion(S.getInitializer());
      }

      Out << ';' << '\n';
    } else if (auto S = mlir::dyn_cast<LoopBreakOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::Break) << ';' << '\n';
    } else if (auto S = mlir::dyn_cast<LoopContinueOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::Continue) << ';' << '\n';
    } else if (auto S = mlir::dyn_cast<ReturnOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::Return) << ' ';

      if (not S.getResult().empty())
        rc_recur emitExpressionRegion(S.getResult());

      Out << ';' << '\n';
    } else if (auto S = mlir::dyn_cast<SwitchBreakOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::Break) << ';' << '\n';
    } else if (auto S = mlir::dyn_cast<SwitchOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::Switch) << ' ' << '(';
      rc_recur emitExpressionRegion(S.getCondition());
      Out << ')' << '\n' << '{';

      ValueType Type = S.getConditionType();
      for (unsigned I = 0, Count = S.getNumCases(); I < Count; ++I) {
        Out << C.getKeyword(Keyword::Case) << ' ';
        emitIntegerImmediate(S.getCaseValue(I), Type);
        Out << ':' << '\n' << '{' << '\n';
        enter(), emitStatementRegion(S.getCaseRegion(I));
        Out << '}' << '\n';
      }

      if (S.hasDefaultCase()) {
        Out << C.getKeyword(Keyword::Default) << ':' << '\n' << '{' << '\n';
        enter(), rc_recur emitStatementRegion(S.getDefaultCaseRegion());
        Out << '}' << '\n';
      }

      Out << '}' << '\n';
    } else if (auto S = mlir::dyn_cast<WhileOp>(Op.getOperation())) {
      Out << C.getKeyword(Keyword::While) << ' ' << '(';
      rc_recur emitExpressionRegion(S.getCondition());
      Out << ')' << '\n';

      rc_recur emitImplicitBlockStatement(S.getBody());
    } else {
      revng_abort();
    }
  }

  RecursiveCoroutine<void> emitStatementRegion(mlir::Region &R) {
    for (mlir::Operation &Stmt : R.getOps())
      rc_recur emitStatement(mlir::cast<StatementOpInterface>(&Stmt));
  }

  static mlir::Operation *getOnlyOperation(mlir::Region &R) {
    revng_assert(R.hasOneBlock());
    mlir::Block &B = R.front();
    auto Beg = B.begin();
    auto End = B.end();

    if (Beg == End)
      return nullptr;

    mlir::Operation *Op = &*Beg;

    if (++Beg != End)
      return nullptr;

    return Op;
  }

  RecursiveCoroutine<void> emitImplicitBlockStatement(mlir::Region &R) {
    mlir::Operation *OnlyOp = getOnlyOperation(R);
    bool PrintBlock = OnlyOp == nullptr or mlir::isa<BlockStatementOp>(OnlyOp);

    if (PrintBlock)
      Out << '{' << '\n';

    enter(), rc_recur emitStatementRegion(R);

    if (PrintBlock)
      Out << '}' << '\n';
  }

  RecursiveCoroutine<void> emitFunction(FunctionOp Op,
                                        bool EmitDeclaration) {
    const model::Function &ModelFunction = getModelFunction(Op);

    auto *MFT = llvm::cast<model::DefinedType>(ModelFunction.Prototype().get());
    const model::TypeDefinition *MFD = MFT->Definition().get();

    auto ParameterNamesGuard = llvm::make_scope_exit([&]() {
      ParameterNames.clear();
    });

    if (auto F = llvm::dyn_cast<model::CABIFunctionDefinition>(MFD)) {
      for (const model::Argument &Parameter : F->Arguments())
        ParameterNames.push_back(NameBuilder.argumentName(*F, Parameter));
    } else if (auto F = llvm::dyn_cast<model::RawFunctionDefinition>(MFD)) {
      if (not F->StackArgumentsType().isEmpty())
        ParameterNames.push_back(model::Identifier::fromString("args"));

      for (const model::NamedTypedRegister &Register : F->Arguments())
        ParameterNames.push_back(NameBuilder.argumentName(*F, Register));
    } else {
      revng_abort();
    }

    if (EmitDeclaration) {
      const DeclaratorNames Names = {
        .DeclaratorName = NameBuilder.name(ModelFunction),
        .ParameterNames = ParameterNames,
      };

      rc_recur emitDeclaration(Op.getFunctionType(), &Names);
      Out << '\n';
    }

    Out << '{' << '\n';
    enter(), rc_recur emitStatementRegion(Op.getBody());
    Out << '}';
  }

private:
  const PlatformInfo &Platform;
  const model::Binary &Model;
  model::NameBuilder NameBuilder;

  ptml::CBuilder C;
  ptml::IndentedOstream Out;

  // Parameter names of the current function.
  llvm::SmallVector<model::Identifier> ParameterNames;

  // Ambient precedence of the current expression.
  OperatorPrecedence CurrentPrecedence = {};
};

} // namespace

std::string c_backend::emit(const PlatformInfo &Platform,
                            const model::Binary &Model,
                            FunctionOp Function,
                            const bool GeneratePlainC,
                            const bool EmitDeclaration) {
  std::string Result;
  llvm::raw_string_ostream Out(Result);

  CEmitter(Platform, Model, Out, GeneratePlainC).emitFunction(Function,
                                                              EmitDeclaration);

  return Result;
}
