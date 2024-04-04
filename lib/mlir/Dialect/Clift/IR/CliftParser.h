#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "mlir/IR/OpImplementation.h"

#include "revng/Support/Assert.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"

namespace mlir::clift {

class AsmRecursionGuardBase {
  using Map = std::map<uint64_t, void *>;

  std::pair<Map::iterator, bool> Insert;

public:
  explicit AsmRecursionGuardBase(const uint64_t ID);

  AsmRecursionGuardBase(const AsmRecursionGuardBase &) = delete;
  AsmRecursionGuardBase &operator=(const AsmRecursionGuardBase &) = delete;

  ~AsmRecursionGuardBase();

protected:
  void *get() const { return Insert.first->second; }

  void set(void *const Value) {
    revng_assert(Insert.second);
    revng_assert(Insert.first->second == nullptr);
    Insert.first->second = Value;
  }
};

template<typename ObjectT>
class AsmRecursionGuard : AsmRecursionGuardBase {
public:
  using AsmRecursionGuardBase::AsmRecursionGuardBase;

  [[nodiscard]] ObjectT get() const {
    return ObjectT(static_cast<
                   typename ObjectT::ImplType *>(AsmRecursionGuardBase::get()));
  }

  void set(ObjectT Value) { AsmRecursionGuardBase::set(Value.getImpl()); }
};

template<typename ObjectT>
constexpr bool hasExplicitSize() {
  return requires(MLIRContext *C, uint64_t U, llvm::StringRef S) {
    ObjectT::get(C, U, S, U);
  };
}

template<typename ObjectT>
void printCompositeType(AsmPrinter &Printer, ObjectT Object) {
  const uint64_t ID = Object.getImpl()->getID();

  Printer << Object.getMnemonic();
  Printer << "<id = ";
  Printer << ID;

  AsmRecursionGuard<ObjectT> Guard(ID);
  if (Guard.get()) {
    Printer << ">";
    return;
  }
  Guard.set(Object);

  Printer << ", name = ";
  Printer << "\"" << Object.getName() << "\"";

  if constexpr (hasExplicitSize<ObjectT>()) {
    Printer << ", ";
    Printer.printKeywordOrString("size");
    Printer << " = ";
    Printer << Object.getByteSize();
  }

  Printer << ", fields = [";
  Printer.printStrippedAttrOrType(Object.getImpl()->getSubobjects());
  Printer << "]>";
}

template<typename ObjectT>
ObjectT parseCompositeType(AsmParser &Parser, const size_t MinSubobjects) {
  const auto OnUnexpectedToken = [&](const llvm::StringRef Name) -> ObjectT {
    Parser.emitError(Parser.getCurrentLocation(),
                     "Expected " + Name + " while parsing mlir "
                       + ObjectT::getMnemonic() + "type");
    return {};
  };

  if (Parser.parseLess())
    return OnUnexpectedToken("<");

  if (Parser.parseKeyword("id").failed())
    return OnUnexpectedToken("keyword 'id'");

  if (Parser.parseEqual().failed())
    return OnUnexpectedToken("=");

  uint64_t ID;
  if (Parser.parseInteger(ID).failed())
    return OnUnexpectedToken("<integer>");

  AsmRecursionGuard<ObjectT> Guard(ID);
  if (const auto Attr = Guard.get()) {
    if (Parser.parseGreater().failed())
      return OnUnexpectedToken(">");

    return Attr;
  }

  if (Parser.parseComma().failed())
    return OnUnexpectedToken(",");

  if (Parser.parseKeyword("name").failed())
    return OnUnexpectedToken("keyword 'name'");

  if (Parser.parseEqual().failed())
    return OnUnexpectedToken("=");

  std::string OptionalName = "";
  if (Parser.parseOptionalString(&OptionalName).failed())
    return OnUnexpectedToken("<string>");

  ObjectT ToReturn;
  if constexpr (hasExplicitSize<ObjectT>()) {
    if (Parser.parseComma().failed())
      return OnUnexpectedToken(",");

    if (Parser.parseKeyword("size").failed())
      return OnUnexpectedToken("keyword 'size'");

    if (Parser.parseEqual().failed())
      return OnUnexpectedToken("=");

    uint64_t Size;
    if (Parser.parseInteger(Size).failed())
      return OnUnexpectedToken("<uint64_t>");

    ToReturn = ObjectT::get(Parser.getContext(), ID, OptionalName, Size);
  } else {
    ToReturn = ObjectT::get(Parser.getContext(), ID, OptionalName);
  }
  Guard.set(ToReturn);

  if (Parser.parseComma().failed())
    return OnUnexpectedToken(",");

  if (Parser.parseKeyword("fields").failed())
    return OnUnexpectedToken("keyword 'fields'");

  if (Parser.parseEqual().failed())
    return OnUnexpectedToken("=");

  if (Parser.parseLSquare().failed())
    return OnUnexpectedToken("[");

  using SubobjectType = typename ObjectT::ImplType::SubobjectTy;
  using SubobjectVectorType = ::llvm::SmallVector<SubobjectType>;
  using SubobjectParserType = ::mlir::FieldParser<SubobjectVectorType>;
  ::mlir::FailureOr<SubobjectVectorType> Fields(SubobjectVectorType{});

  if (MinSubobjects > 0 or Parser.parseOptionalRSquare().failed()) {
    Fields = SubobjectParserType::parse(Parser);

    if (::mlir::failed(Fields)) {
      Parser.emitError(Parser.getCurrentLocation(),
                       "failed to parse class type parameter 'fields' "
                       "which is to be a "
                       "`::llvm::ArrayRef<mlir::clift::FieldAttr>`");
    }

    if (Fields->size() < MinSubobjects) {
      Parser.emitError(Parser.getCurrentLocation(),
                       llvm::Twine(ObjectT::getMnemonic())
                         + " requires at least " + llvm::Twine(MinSubobjects)
                         + " fields");
      return {};
    }

    if (Parser.parseRSquare().failed())
      return OnUnexpectedToken("]");
  }

  if (Parser.parseGreater().failed())
    return OnUnexpectedToken(">");

  ToReturn.setBody(*Fields);
  return ToReturn;
}

} // namespace mlir::clift
