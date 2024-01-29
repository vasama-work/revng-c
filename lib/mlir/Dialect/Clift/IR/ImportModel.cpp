//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/ADT/RecursiveCoroutine.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"
#include "revng-c/mlir/Dialect/Clift/IR/ImportModel.h"

namespace {

namespace clift = mlir::clift;

template<typename Attribute>
using AttributeVector = llvm::SmallVector<Attribute, 16>;

struct CliftConverter {
  mlir::MLIRContext *Context;
  llvm::function_ref<mlir::InFlightDiagnostic()> EmitError;
  llvm::SmallSet<uint64_t, 16> IncompleteTypes;

  class DefinitionGuard {
    CliftConverter *Self = nullptr;
    uint64_t ID;

  public:
    explicit DefinitionGuard(CliftConverter &Self, const uint64_t ID) {
      if (Self.IncompleteTypes.insert(ID).second) {
        this->Self = &Self;
        this->ID = ID;
      }
    }

    DefinitionGuard(const DefinitionGuard &) = delete;
    DefinitionGuard &operator=(const DefinitionGuard &) = delete;

    ~DefinitionGuard() {
      if (Self != nullptr) {
        Self->IncompleteTypes.erase(ID);
      }
    }

    explicit operator bool() const { return Self != nullptr; }
  };

  explicit CliftConverter(mlir::MLIRContext &Context,
                          llvm::function_ref<mlir::InFlightDiagnostic()>
                            EmitError = nullptr) :
    Context(&Context), EmitError(EmitError) {}

  CliftConverter(const CliftConverter &) = delete;
  CliftConverter &operator=(const CliftConverter &) = delete;

  ~CliftConverter() { revng_assert(IncompleteTypes.empty()); }

  mlir::BoolAttr getBool(bool const Value) {
    return mlir::BoolAttr::get(Context, Value);
  }

  mlir::BoolAttr getFalse() { return getBool(false); }

  template<typename T, typename... ArgTypes>
  T make(const ArgTypes &...Args) {
    if (EmitError and failed(T::verify(EmitError, Args...)))
      return {};
    return T::get(Context, Args...);
  }

  static clift::PrimitiveKind
  getPrimitiveKind(const model::PrimitiveTypeKind::Values K) {
    switch (K) {
    case model::PrimitiveTypeKind::Void:
      return clift::PrimitiveKind::VoidKind;
    case model::PrimitiveTypeKind::Generic:
      return clift::PrimitiveKind::GenericKind;
    case model::PrimitiveTypeKind::PointerOrNumber:
      return clift::PrimitiveKind::PointerOrNumberKind;
    case model::PrimitiveTypeKind::Number:
      return clift::PrimitiveKind::NumberKind;
    case model::PrimitiveTypeKind::Unsigned:
      return clift::PrimitiveKind::UnsignedKind;
    case model::PrimitiveTypeKind::Signed:
      return clift::PrimitiveKind::SignedKind;
    case model::PrimitiveTypeKind::Float:
      return clift::PrimitiveKind::FloatKind;
    default:
      revng_abort();
    }
  }

  clift::ValueType getPrimitiveType(const model::PrimitiveType &ModelType,
                                    const bool Const) {
    return make<clift::PrimitiveType>(getPrimitiveKind(ModelType
                                                         .PrimitiveKind()),
                                      ModelType.Size(),
                                      getBool(Const));
  }

  RecursiveCoroutine<clift::TypeDefinition>
  getTypeAttribute(const model::CABIFunctionType &ModelType) {
    DefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard)
      rc_return nullptr;

    AttributeVector<clift::FunctionArgumentAttr> Args;
    Args.reserve(ModelType.Arguments().size());

    for (const model::Argument &Argument : ModelType.Arguments()) {
      const auto Type = rc_recur getQualifiedType(Argument.Type());
      if (not Type)
        rc_return nullptr;
      const llvm::StringRef Name = Argument.OriginalName();
      const auto Attribute = make<clift::FunctionArgumentAttr>(Type, Name);
      if (not Attribute)
        rc_return nullptr;
      Args.push_back(Attribute);
    }

    const auto ReturnType = rc_recur getQualifiedType(ModelType.ReturnType());
    if (not ReturnType)
      rc_return nullptr;

    rc_return make<clift::FunctionAttr>(ModelType.ID(),
                                        ModelType.OriginalName(),
                                        ReturnType,
                                        Args);
  }

  RecursiveCoroutine<clift::TypeDefinition>
  getTypeAttribute(const model::EnumType &ModelType) {
    DefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard)
      rc_return nullptr;

    auto const UnderlyingType = rc_recur getQualifiedType(ModelType
                                                            .UnderlyingType());
    if (not UnderlyingType)
      rc_return nullptr;

    AttributeVector<clift::EnumFieldAttr> Fields;
    Fields.reserve(ModelType.Entries().size());

    for (const model::EnumEntry &Entry : ModelType.Entries()) {
      const auto Attribute = make<clift::EnumFieldAttr>(Entry.Value(),
                                                        Entry.CustomName());
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::EnumAttr>(ModelType.ID(),
                                    ModelType.OriginalName(),
                                    UnderlyingType,
                                    Fields);
  }

  RecursiveCoroutine<clift::TypeDefinition>
  getTypeAttribute(const model::RawFunctionType &ModelType) {
    DefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard)
      rc_return nullptr;

    clift::FunctionArgumentAttr StackArgument;
    size_t ArgumentsCount = 0;

    if (ModelType.StackArgumentsType().isValid()) {
      const auto Type = rc_recur
        getUnwrappedType(*ModelType.StackArgumentsType().get());
      if (not Type)
        rc_return nullptr;

      // WIP: Choose an appropriate pointer size.
      const uint64_t PointerSize = 8;
      const auto PointerType = make<clift::PointerType>(Type,
                                                        PointerSize,
                                                        getFalse());
      if (not PointerType)
        rc_return nullptr;

      StackArgument = make<clift::FunctionArgumentAttr>(PointerType, "");
      if (not StackArgument)
        rc_return nullptr;

      ++ArgumentsCount;
    }

    ArgumentsCount += ModelType.Arguments().size();
    AttributeVector<clift::FunctionArgumentAttr> Args;
    Args.reserve(ArgumentsCount);

    for (const model::NamedTypedRegister &Register : ModelType.Arguments()) {
      const auto Type = rc_recur getQualifiedType(Register.Type());
      if (not Type)
        rc_return nullptr;
      const llvm::StringRef Name = Register.OriginalName();
      const auto Argument = make<clift::FunctionArgumentAttr>(Type, Name);
      if (not Argument)
        rc_return nullptr;
      Args.push_back(Argument);
    }

    if (StackArgument)
      Args.push_back(StackArgument);

    clift::ValueType ReturnType;
    switch (ModelType.ReturnValues().size()) {
    case 0:
      ReturnType = make<clift::PrimitiveType>(clift::PrimitiveKind::VoidKind,
                                              0,
                                              getFalse());
      break;

    case 1:
      ReturnType = rc_recur
        getQualifiedType(ModelType.ReturnValues().begin()->Type());
      break;

    default:
      // WIP: Revisit multi-register return type.
      AttributeVector<clift::FieldAttr> Fields;
      Fields.reserve(ModelType.ReturnValues().size());

      uint64_t Offset = 0;
      for (const model::NamedTypedRegister &Register :
           ModelType.ReturnValues()) {
        auto const RegisterType = rc_recur getQualifiedType(Register.Type());
        if (not RegisterType)
          rc_return nullptr;
        auto const Attribute = make<clift::FieldAttr>(Offset,
                                                      RegisterType,
                                                      Register.CustomName());
        if (not Attribute)
          rc_return nullptr;
        Fields.push_back(Attribute);

        const auto Size = Register.Type().size();
        // WIP: Is it possible that the register type size cannot be computed?
        if (not Size)
          rc_return nullptr;

        Offset += Register.Type().size().value();
      }

      // WIP: Revisit ID selection if the struct is kept.
      const uint64_t ID = ModelType.ID() + 1'000'000'000u;
      const std::string Name = std::string(llvm::formatv("RawFunctionType-{0}-"
                                                         "ReturnType",
                                                         ModelType.ID()));
      const auto Attribute = make<clift::StructType>(ID, Name, Offset, Fields);
      if (not Attribute)
        rc_return nullptr;
      ReturnType = make<clift::DefinedType>(clift::TypeDefinition(Attribute),
                                            getFalse());
      break;
    }
    if (not ReturnType)
      rc_return nullptr;

    rc_return make<clift::FunctionAttr>(ModelType.ID(),
                                        ModelType.OriginalName(),
                                        ReturnType,
                                        Args);
  }

  RecursiveCoroutine<clift::TypeDefinition>
  getTypeAttribute(const model::StructType &ModelType) {
    DefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard)
      rc_return make<clift::StructType>(ModelType.ID());

    AttributeVector<clift::FieldAttr> Fields;
    Fields.reserve(ModelType.Fields().size());

    for (const model::StructField &Field : ModelType.Fields()) {
      const auto FieldType = rc_recur getQualifiedType(Field.Type());
      if (not FieldType)
        rc_return nullptr;
      const auto Attribute = make<clift::FieldAttr>(Field.Offset(),
                                                    FieldType,
                                                    Field.CustomName());
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::StructType>(ModelType.ID(),
                                      ModelType.OriginalName(),
                                      ModelType.Size(),
                                      Fields);
  }

  RecursiveCoroutine<clift::TypeDefinition>
  getTypeAttribute(const model::TypedefType &ModelType) {
    DefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard)
      rc_return nullptr;

    auto const UnderlyingType = rc_recur getQualifiedType(ModelType
                                                            .UnderlyingType());
    if (not UnderlyingType)
      rc_return nullptr;
    rc_return make<clift::TypedefAttr>(ModelType.ID(),
                                       ModelType.OriginalName(),
                                       UnderlyingType);
  }

  RecursiveCoroutine<clift::TypeDefinition>
  getTypeAttribute(const model::UnionType &ModelType) {
    DefinitionGuard Guard(*this, ModelType.ID());
    if (not Guard)
      rc_return make<clift::StructType>(ModelType.ID());

    AttributeVector<clift::FieldAttr> Fields;
    Fields.reserve(ModelType.Fields().size());

    for (const model::UnionField &Field : ModelType.Fields()) {
      auto const FieldType = rc_recur getQualifiedType(Field.Type());
      if (not FieldType)
        rc_return nullptr;
      auto const Attribute = make<clift::FieldAttr>(0,
                                                    FieldType,
                                                    Field.CustomName());
      if (not Attribute)
        rc_return nullptr;
      Fields.push_back(Attribute);
    }

    rc_return make<clift::UnionType>(ModelType.ID(),
                                     ModelType.OriginalName(),
                                     Fields);
  }

  RecursiveCoroutine<clift::ValueType>
  getUnwrappedType(const model::Type &ModelType, const bool Const = false) {
    auto const getDefinedType = [&](auto const &Attribute) -> clift::ValueType {
      if (not Attribute)
        return nullptr;
      return make<clift::DefinedType>(Attribute, getBool(Const));
    };
    switch (ModelType.Kind()) {
    case model::TypeKind::CABIFunctionType: {
      auto const &T = llvm::cast<model::CABIFunctionType>(ModelType);
      rc_return getDefinedType(rc_recur getTypeAttribute(T));
    }
    case model::TypeKind::EnumType: {
      auto const &T = llvm::cast<model::EnumType>(ModelType);
      rc_return getDefinedType(rc_recur getTypeAttribute(T));
    }
    case model::TypeKind::PrimitiveType: {
      auto const &T = llvm::cast<model::PrimitiveType>(ModelType);
      rc_return getPrimitiveType(T, Const);
    }
    case model::TypeKind::RawFunctionType: {
      auto const &T = llvm::cast<model::RawFunctionType>(ModelType);
      rc_return getDefinedType(rc_recur getTypeAttribute(T));
    }
    case model::TypeKind::StructType: {
      auto const &T = llvm::cast<model::StructType>(ModelType);
      rc_return getDefinedType(rc_recur getTypeAttribute(T));
    }
    case model::TypeKind::TypedefType: {
      auto const &T = llvm::cast<model::TypedefType>(ModelType);
      rc_return getDefinedType(rc_recur getTypeAttribute(T));
    }
    case model::TypeKind::UnionType: {
      auto const &T = llvm::cast<model::UnionType>(ModelType);
      rc_return getDefinedType(rc_recur getTypeAttribute(T));
    }
    default:
      revng_abort();
    }
  }

  RecursiveCoroutine<clift::ValueType>
  getQualifiedType(const model::QualifiedType &ModelType) {
    if (not ModelType.UnqualifiedType().isValid())
      rc_return nullptr;

    auto Qualifiers = llvm::ArrayRef(ModelType.Qualifiers());

    const auto takeQualifier = [&]() -> const model::Qualifier & {
      const model::Qualifier &Qualifier = Qualifiers.back();
      Qualifiers = Qualifiers.slice(0, Qualifiers.size() - 1);
      return Qualifier;
    };

    const auto takeConst = [&]() -> bool {
      if (not Qualifiers.empty()
          and Qualifiers.back().Kind() == model::QualifierKind::Const) {
        Qualifiers = Qualifiers.slice(0, Qualifiers.size() - 1);
        return true;
      }
      return false;
    };

    clift::ValueType QualifiedType = rc_recur
      getUnwrappedType(*ModelType.UnqualifiedType().get(), takeConst());

    if (not QualifiedType)
      rc_return nullptr;

    // Loop over (qualifier, const (optional)) pairs wrapping the type at each
    // iteration, until the list of qualifiers is exhausted.
    while (not Qualifiers.empty()) {
      switch (const model::Qualifier &Qualifier = takeQualifier();
              Qualifier.Kind()) {
      case model::QualifierKind::Pointer:
        QualifiedType = make<clift::PointerType>(QualifiedType,
                                                 Qualifier.Size(),
                                                 getBool(takeConst()));
        break;

      case model::QualifierKind::Array:
        QualifiedType = make<clift::ArrayType>(QualifiedType,
                                               Qualifier.Size(),
                                               getBool(takeConst()));
        break;

      default:
        if (EmitError)
          EmitError() << "invalid type qualifiers";
        rc_return nullptr;
      }

      if (not QualifiedType)
        rc_return nullptr;
    }

    rc_return QualifiedType;
  }
};

} // namespace

clift::ValueType revng::getUnqualifiedType(mlir::MLIRContext &Context,
                                           const model::Type &ModelType) {
  return CliftConverter(Context).getUnwrappedType(ModelType);
}

clift::ValueType
revng::getUnqualifiedTypeChecked(llvm::function_ref<mlir::InFlightDiagnostic()>
                                   EmitError,
                                 mlir::MLIRContext &Context,
                                 const model::Type &ModelType) {
  return CliftConverter(Context, EmitError).getUnwrappedType(ModelType);
}

clift::ValueType
revng::getQualifiedType(mlir::MLIRContext &Context,
                        const model::QualifiedType &ModelType) {
  return CliftConverter(Context).getQualifiedType(ModelType);
}

clift::ValueType
revng::getQualifiedTypeChecked(llvm::function_ref<mlir::InFlightDiagnostic()>
                                 EmitError,
                               mlir::MLIRContext &Context,
                               const model::QualifiedType &ModelType) {
  return CliftConverter(Context, EmitError).getQualifiedType(ModelType);
}
