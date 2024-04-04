//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "CliftParser.h"

using namespace mlir::clift;

static thread_local std::map<uint64_t, void *> CurrentlyVisited;

AsmRecursionGuardBase::AsmRecursionGuardBase(const uint64_t ID) :
  Insert(CurrentlyVisited.try_emplace(ID)) {
}

AsmRecursionGuardBase::~AsmRecursionGuardBase() {
  if (Insert.second)
    CurrentlyVisited.erase(Insert.first);
}
