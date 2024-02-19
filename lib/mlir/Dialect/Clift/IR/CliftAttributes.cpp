//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftInterfaces.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"

// This include should stay here for correct build procedure
//
#define GET_ATTRDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"

static thread_local std::map<uint64_t, mlir::Attribute> CurrentlyPrintedTypes;

void mlir::clift::CliftDialect::registerAttributes() {
  addAttributes<StructType, UnionType, /* Include the auto-generated clift types
                                        */
#define GET_ATTRDEF_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.cpp.inc"
                /* End of types list */>();
}

mlir::LogicalResult
mlir::clift::FieldAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()>
                                 EmitError,
                               uint64_t Offset,
                               mlir::Type ElementType,
                               llvm::StringRef Name) {
  if (auto Definition = mlir::dyn_cast<mlir::clift::DefinedType>(ElementType))
    if (mlir::isa<mlir::clift::FunctionAttr>(Definition.getElementType()))
      return EmitError() << "Underlying type of field attr cannot be a "
                            "function type";
  mlir::clift::ValueType
    Casted = mlir::dyn_cast<mlir::clift::ValueType>(ElementType);
  if (Casted == nullptr) {
    return EmitError() << "Underlying type of a field attr must be a value "
                          "type";
  }
  if (Casted.getByteSize() == 0) {
    return EmitError() << "Field cannot be of zero size";
  }
  return mlir::success();
}

using ArgAttr = mlir::clift::FunctionArgumentAttr;
mlir::LogicalResult
ArgAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                mlir::clift::ValueType underlying,
                llvm::StringRef Name) {
  if (underlying.getByteSize() == 0) {
    return EmitError() << "type of argument of function cannot be zero size";
  }
  return mlir::success();
}

using mlir::clift::FunctionAttr;
mlir::LogicalResult
FunctionAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                     uint64_t Id,
                     llvm::StringRef,
                     mlir::clift::ValueType ReturnType,
                     llvm::ArrayRef<mlir::clift::FunctionArgumentAttr> Args) {
  if (const auto Type = mlir::dyn_cast<mlir::clift::DefinedType>(ReturnType)) {
    if (mlir::isa<FunctionAttr>(Type.getElementType()))
      return EmitError() << "function type cannot return another function type";
  }

  return mlir::success();
}

/// Parse a type registered to this dialect
::mlir::Attribute
mlir::clift::CliftDialect::parseAttribute(::mlir::DialectAsmParser &Parser,
                                          mlir::Type Type) const {
  ::llvm::SMLoc typeLoc = Parser.getCurrentLocation();
  ::llvm::StringRef Mnemonic;
  ::mlir::Attribute GenAttr;

  auto ParseResult = generatedAttributeParser(Parser, &Mnemonic, Type, GenAttr);
  if (ParseResult.has_value())
    return GenAttr;
  if (Mnemonic == StructType::getMnemonic()) {
    return StructType::parse(Parser);
  }
  if (Mnemonic == UnionType::getMnemonic()) {
    return UnionType::parse(Parser);
  }

  Parser.emitError(typeLoc) << "unknown  attr `" << Mnemonic << "` in dialect `"
                            << getNamespace() << "`";
  return {};
}

/// Print a type registered to this dialect
void mlir::clift::CliftDialect::printAttribute(::mlir::Attribute Attr,
                                               ::mlir::DialectAsmPrinter
                                                 &Printer) const {

  if (::mlir::succeeded(generatedAttributePrinter(Attr, Printer)))
    return;
  if (auto Casted = Attr.dyn_cast<StructType>()) {
    Casted.print(Printer);
    return;
  }
  if (auto Casted = Attr.dyn_cast<UnionType>()) {
    Casted.print(Printer);
    return;
  }
  revng_abort("cannot print attribute");
}

template<typename AttrType>
static mlir::Attribute printImpl(mlir::AsmPrinter &P, AttrType Attr) {
  P << Attr.getMnemonic();
  P << "<id = ";
  size_t ID = Attr.getImpl()->getID();
  P << ID;
  if (not Attr.getImpl()->isInitialized()) {
    P << ">";
    return Attr;
  }
  if (auto Iter = CurrentlyPrintedTypes.find(ID);
      Iter != CurrentlyPrintedTypes.end()) {
    P << ">";
    return Attr;
  }

  CurrentlyPrintedTypes[ID] = Attr;
  auto guard = llvm::make_scope_exit([&]() {
    CurrentlyPrintedTypes.erase(ID);
  });

  P << ", name = ";
  P << "\"" << Attr.getName() << "\"";
  P << ", ";
  if constexpr (std::is_same_v<AttrType, mlir::clift::StructType>) {
    P.printKeywordOrString("size");
    P << " = ";
    P << Attr.getByteSize();
    P << ", ";
  }
  P << "fields = [";
  P.printStrippedAttrOrType(Attr.getImpl()->getFields());
  P << "]>";
  return Attr;
}

mlir::Attribute mlir::clift::UnionType::print(AsmPrinter &p) const {
  return printImpl(p, *this);
}

mlir::Attribute mlir::clift::StructType::print(AsmPrinter &p) const {
  return printImpl(p, *this);
}

template<typename AttrType>
AttrType parseImpl(mlir::AsmParser &parser, llvm::StringRef TypeName) {
  const auto OnUnepxectedToken = [&parser,
                                  TypeName](llvm::StringRef name) -> AttrType {
    parser.emitError(parser.getCurrentLocation(),
                     "Expected " + name + " while parsing mlir " + TypeName
                       + "type");
    return AttrType();
  };

  if (parser.parseLess()) {
    return OnUnepxectedToken("<");
  }

  if (parser.parseKeyword("id").failed()) {
    return OnUnepxectedToken("keyword 'id'");
  }

  if (parser.parseEqual().failed()) {
    return OnUnepxectedToken("=");
  }

  uint64_t ID;
  if (parser.parseInteger(ID).failed()) {
    return OnUnepxectedToken("<integer>");
  }

  if (auto Iterator = CurrentlyPrintedTypes.find(ID);
      Iterator != CurrentlyPrintedTypes.end()) {
    if (parser.parseGreater().failed()) {
      return OnUnepxectedToken(">");
    }

    return Iterator->second.cast<AttrType>();
  }

  AttrType ToReturn = AttrType::get(parser.getContext(), ID);

  CurrentlyPrintedTypes[ID] = ToReturn;
  auto guard = llvm::make_scope_exit([&]() {
    CurrentlyPrintedTypes.erase(ID);
  });

  if (parser.parseComma().failed()) {
    return OnUnepxectedToken(",");
  }

  if (parser.parseKeyword("name").failed()) {
    return OnUnepxectedToken("keyword 'name'");
  }

  if (parser.parseEqual().failed()) {
    return OnUnepxectedToken("=");
  }

  std::string OptionalName = "";
  if (parser.parseOptionalString(&OptionalName).failed()) {
    return OnUnepxectedToken("<string>");
  }

  if (parser.parseComma().failed()) {
    return OnUnepxectedToken(",");
  }

  uint64_t Size;
  if constexpr (std::is_same_v<mlir::clift::StructType, AttrType>) {
    if (parser.parseKeyword("size").failed()) {
      return OnUnepxectedToken("keyword 'size'");
    }

    if (parser.parseEqual().failed()) {
      return OnUnepxectedToken("=");
    }

    if (parser.parseInteger(Size).failed()) {
      return OnUnepxectedToken("<size_t>");
    }

    if (parser.parseComma().failed()) {
      return OnUnepxectedToken(",");
    }
  }

  if (parser.parseKeyword("fields").failed()) {
    return OnUnepxectedToken("keyword 'fields'");
  }

  if (parser.parseEqual().failed()) {
    return OnUnepxectedToken("=");
  }

  if (parser.parseLSquare().failed()) {
    return OnUnepxectedToken("[");
  }

  using SmallVectorType = ::llvm::SmallVector<mlir::clift::FieldAttr>;
  using ParserType = ::mlir::FieldParser<SmallVectorType>;
  ::mlir::FailureOr<::llvm::SmallVector<mlir::clift::FieldAttr>>
    Attrs = ParserType::parse(parser);
  if (::mlir::failed(Attrs)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse Clift_EnumAttr parameter 'fields' "
                     "which "
                     "is to be a "
                     "`::llvm::ArrayRef<mlir::clift::FieldAttr>`");
  }

  if (parser.parseRSquare().failed()) {
    return OnUnepxectedToken("]");
  }

  if (parser.parseGreater().failed()) {
    return OnUnepxectedToken(">");
  }
  if constexpr (std::is_same_v<mlir::clift::StructType, AttrType>) {
    ToReturn.setBody(OptionalName, Size, *Attrs);
  } else {
    ToReturn.setBody(OptionalName, *Attrs);
  }
  return ToReturn;
}

mlir::Attribute mlir::clift::UnionType::parse(AsmParser &parser) {
  return parseImpl<UnionType>(parser, "union");
}

mlir::Attribute mlir::clift::StructType::parse(AsmParser &parser) {
  return parseImpl<StructType>(parser, "union");
}

static bool isIncompleteType(const mlir::Type T) {
  if (T.isa<mlir::clift::DefinedType>()) {
    const mlir::clift::TypeDefinition D = T.cast<mlir::clift::DefinedType>()
                                            .getElementType();
    if (D.isa<mlir::clift::StructType>())
      return not D.cast<mlir::clift::StructType>().isDefinition();
    if (D.isa<mlir::clift::UnionType>())
      return not D.cast<mlir::clift::UnionType>().isDefinition();
  }
  return false;
}

mlir::LogicalResult
mlir::clift::StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                uint64_t id) {
  return mlir::success();
}

mlir::LogicalResult
mlir::clift::StructType::verify(const function_ref<InFlightDiagnostic()>
                                  EmitError,
                                const uint64_t ID,
                                llvm::StringRef,
                                const uint64_t Size,
                                const llvm::ArrayRef<FieldAttr> Fields) {
  if (Size == 0)
    return EmitError() << "struct type cannot have a size of zero";

  if (not Fields.empty()) {
    uint64_t LastEndOffset = 0;

    for (const auto &Field : Fields) {
      if (isIncompleteType(Field.getType()))
        return EmitError() << "Fields of structs must be complete types";

      if (Field.getOffset() < LastEndOffset)
        return EmitError() << "Fields of structs must be ordered by offset, "
                              "and "
                              "they cannot overlap";

      LastEndOffset = Field.getOffset()
                      + Field.getType()
                          .cast<mlir::clift::ValueType>()
                          .getByteSize();
    }

    if (LastEndOffset > Size)
      return EmitError() << "offset + size of field of struct type is greater "
                            "than the struct type size.";
  }

  return mlir::success();
}

mlir::LogicalResult
mlir::clift::UnionType::verify(function_ref<InFlightDiagnostic()> EmitError,
                               uint64_t ID,
                               llvm::StringRef Name,
                               llvm::ArrayRef<FieldAttr> Fields) {
  if (Fields.size() != 0) {
    return EmitError() << "union types must have at least a field";
  }
  for (auto Field : Fields) {
    if (Field.getOffset() != 0) {
      return EmitError() << "union types offsets must be zero";
    }
  }
  std::set<llvm::StringRef> Names;
  for (auto Field : Fields) {
    if (Field.getName().empty())
      continue;
    if (Names.contains(Field.getName())) {
      return EmitError() << "multiple definitions of union field named "
                         << Field.getName();
    }
    Names.insert(Field.getName());
  }
  return mlir::success();
}

mlir::clift::StructType mlir::clift::StructType::get(MLIRContext *ctx,
                                                     uint64_t ID) {
  return Base::get(ctx, ID);
}

mlir::clift::StructType
mlir::clift::StructType::getChecked(llvm::function_ref<InFlightDiagnostic()>
                                      EmitError,
                                    MLIRContext *ctx,
                                    uint64_t ID) {
  if (failed(verify(EmitError, ID)))
    return {};
  return get(ctx, ID);
}

mlir::clift::StructType
mlir::clift::StructType::get(MLIRContext *ctx,
                             uint64_t ID,
                             llvm::StringRef Name,
                             uint64_t Size,
                             llvm::ArrayRef<FieldAttr> Fields) {
  auto Result = Base::get(ctx, ID);
  Result.setBody(Name, Size, Fields);
  return Result;
}

mlir::clift::StructType
mlir::clift::StructType::getChecked(llvm::function_ref<InFlightDiagnostic()>
                                      EmitError,
                                    MLIRContext *ctx,
                                    uint64_t ID,
                                    llvm::StringRef Name,
                                    uint64_t Size,
                                    llvm::ArrayRef<FieldAttr> Fields) {
  if (failed(verify(EmitError, ID, Name, Size, Fields)))
    return {};
  return get(ctx, ID, Name, Size, Fields);
}

mlir::clift::UnionType mlir::clift::UnionType::get(MLIRContext *ctx,
                                                   uint64_t ID) {
  // Call into the base to get a uniqued instance of this type. The parameter
  // (name) is passed after the context.
  return Base::get(ctx, ID);
}

mlir::clift::UnionType
mlir::clift::UnionType::getChecked(llvm::function_ref<InFlightDiagnostic()>
                                     EmitError,
                                   MLIRContext *ctx,
                                   uint64_t ID) {
  if (failed(verify(EmitError, ID)))
    return {};
  return get(ctx, ID);
}

mlir::clift::UnionType
mlir::clift::UnionType::get(MLIRContext *ctx,
                            uint64_t ID,
                            llvm::StringRef Name,
                            llvm::ArrayRef<FieldAttr> Fields) {
  // Call into the base to get a uniqued instance of this type. The parameter
  // (name) is passed after the context.
  auto Result = Base::get(ctx, ID);
  Result.setBody(Name, Fields);
  return Result;
}

mlir::clift::UnionType
mlir::clift::UnionType::getChecked(llvm::function_ref<InFlightDiagnostic()>
                                     EmitError,
                                   MLIRContext *ctx,
                                   uint64_t ID,
                                   llvm::StringRef Name,
                                   llvm::ArrayRef<FieldAttr> Fields) {
  if (failed(verify(EmitError, ID, Name, Fields)))
    return {};
  return get(ctx, ID, Name, Fields);
}
