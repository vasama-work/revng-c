#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <utility>

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir::clift {

template<typename StorageT,
         typename BaseT,
         typename SubobjectT,
         typename ValueT = std::monostate>
class ClassTypeStorage : public BaseT {
  struct KeyBase {
    uint64_t ID;
    llvm::StringRef Name;
    std::optional<llvm::SmallVector<SubobjectT, 2>> Subobjects;
  };

  // Some internal mechanism of mlir have access to the key of the storage
  // instead of the entire storage or the key using that storage. For that
  // reason it is best to throw everything related to the type inside the key,
  // and then override operators to pretend they don't know about the non key
  // fields. That way when it is needed one can access everything.
  struct Key : KeyBase, ValueT {
    template<typename... ArgsT>
    Key(const uint64_t ID, const llvm::StringRef Name, ArgsT &&...Args) :
      KeyBase{ ID, Name, std::nullopt },
      ValueT{ std::forward<ArgsT>(Args)... } {}

    // struct storages are never exposed to the user, they are only used
    // internally to figure out how to create unique objects. only operator== is
    // every used, everything else is handled by hasValue
    friend bool operator==(const Key &LHS, const Key &RHS) {
      return LHS.KeyBase::ID == RHS.KeyBase::ID;
    }

    [[nodiscard]] llvm::hash_code hashValue() const {
      return llvm::hash_value(this->KeyBase::ID);
    }
  };

  Key TheKey;

public:
  using KeyTy = Key;

  using SubobjectTy = SubobjectT;

  const Key &getAsKey() const { return TheKey; }

  static llvm::hash_code hashKey(const KeyTy &Key) { return Key.hashValue(); }

  bool operator==(const Key &Other) const { return TheKey == Other; }

  template<typename... ArgsT>
  ClassTypeStorage(const uint64_t ID,
                   const llvm::StringRef Name,
                   ArgsT &&...Args) :
    TheKey(ID, Name, std::forward<ArgsT>(Args)...) {}

  static StorageT *construct(mlir::StorageUniquer::StorageAllocator &Allocator,
                             const KeyTy &Key) {
    return new (Allocator.allocate<StorageT>())
      StorageT(Key.ID,
               Allocator.copyInto(Key.Name),
               static_cast<const ValueT &>(Key));
  }

  /// Define a mutation method for changing the type after it is created. In
  /// many cases, we only want to set the mutable component once and reject
  /// any further modification, which can be achieved by returning failure
  /// from this function.
  template<typename... ArgsT>
  [[nodiscard]] mlir::LogicalResult
  mutate(mlir::StorageUniquer::StorageAllocator &Allocator,
         const llvm::ArrayRef<SubobjectT> Subobjects) {
    if (TheKey.Subobjects.has_value()) {
      if (not std::equal(TheKey.Subobjects->begin(),
                         TheKey.Subobjects->end(),
                         Subobjects.begin(),
                         Subobjects.end()))
        return mlir::failure();
    }

    TheKey.Subobjects.emplace(Subobjects.begin(), Subobjects.end());
    return mlir::success();
  }

  [[nodiscard]] uint64_t getID() const { return TheKey.ID; }

  [[nodiscard]] llvm::StringRef getName() const { return TheKey.Name; }

  [[nodiscard]] bool isInitialized() const {
    return TheKey.Subobjects.has_value();
  }

  [[nodiscard]] llvm::ArrayRef<SubobjectT> getSubobjects() const {
    return TheKey.Subobjects.has_value() ? *TheKey.Subobjects :
                                           llvm::ArrayRef<SubobjectT>{};
  }

protected:
  [[nodiscard]] const ValueT &getValue() const { return TheKey; }
};

struct StructTypeStorageValue {
  uint64_t Size;
  StructTypeStorageValue(const uint64_t Size) : Size(Size) {}
};

struct StructTypeStorage : ClassTypeStorage<StructTypeStorage,
                                            mlir::AttributeStorage,
                                            FieldAttr,
                                            StructTypeStorageValue> {
  using ClassTypeStorage::ClassTypeStorage;

  [[nodiscard]] uint64_t getSize() const { return getValue().Size; }
};

struct UnionTypeStorage
  : ClassTypeStorage<UnionTypeStorage, mlir::AttributeStorage, FieldAttr> {
  using ClassTypeStorage::ClassTypeStorage;
};

struct ScalarTupleTypeStorage : ClassTypeStorage<ScalarTupleTypeStorage,
                                                 mlir::TypeStorage,
                                                 ScalarTupleElementAttr> {
  using ClassTypeStorage::ClassTypeStorage;
};

} // namespace mlir::clift
