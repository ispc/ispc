/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "GatherCoalescePass.h"
#include "builtins-decl.h"

namespace ispc {

/** Representation of a memory load that the gather coalescing code has
    decided to generate.
 */
struct CoalescedLoadOp {
    CoalescedLoadOp(int64_t s, int c) {
        start = s;
        count = c;
        load = element0 = element1 = nullptr;
    }

    /** Starting offset of the load from the common base pointer (in terms
        of numbers of items of the underlying element type--*not* in terms
        of bytes). */
    int64_t start;

    /** Number of elements to load at this location */
    int count;

    /** Value loaded from memory for this load op */
    llvm::Value *load;

    /** For 2-wide loads (i.e. 64-bit loads), these store the lower and
        upper 32 bits of the result, respectively. */
    llvm::Value *element0, *element1;
};

/** This function determines whether it makes sense (and is safe) to
    generate a vector load of width vectorWidth, starting at *iter.  It
    returns true if so, setting *newIter to point to the next element in
    the set that isn't taken care of by the generated load.  If a vector
    load of the given width doesn't make sense, then false is returned.
 */
static bool lVectorLoadIsEfficient(std::set<int64_t>::iterator iter, std::set<int64_t>::iterator end,
                                   std::set<int64_t>::iterator *newIter, int vectorWidth) {
    // We're considering a vector load of width vectorWidth, starting at
    // the offset "start".
    int64_t start = *iter;

    // The basic idea is that we'll look at the subsequent elements in the
    // load set after the initial one at start.  As long as subsequent
    // elements:
    //
    // 1. Aren't so far separated that they no longer fit into the range
    //    [start, start+vectorWidth)
    //
    // 2. And don't have too large a gap in between them (e.g., it's not
    //    worth generating an 8-wide load for two elements with offsets 0
    //    and 7, but no loads requested in between).
    //
    // Then we continue moving forward through the elements until we either
    // fill up the vector or run out of elements.

    // lastAccepted holds the last offset we've processed and accepted as
    // valid for the vector load underconsideration
    int64_t lastAccepted = start;

    while (iter != end) {
        // What is the separation in offset values from the last element we
        // added to the set for this load?
        int64_t delta = *iter - lastAccepted;
        if (delta > 3)
            // If there's too big a gap, then we won't issue the load
            return false;

        int64_t span = *iter - start + 1;

        if (span == vectorWidth) {
            // We've extended far enough that we have exactly filled up the
            // entire vector width; we can't go any further, so return with
            // success.  (Update *newIter to point at the next element
            // after the last one accepted here.)
            *newIter = ++iter;
            return true;
        } else if (span > vectorWidth) {
            // The current offset won't fit into a vectorWidth-wide load
            // starting from start.  It's still generally worthwhile
            // issuing the load we've been considering, though, since it
            // will provide values for a number of previous offsets.  This
            // load will have one or more elements at the end of its range
            // that is not needed by any of the offsets under
            // consideration.  As such, there are three cases where issuing
            // this load is a bad idea:
            //
            // 1. 2-wide loads: we know that we haven't completely filled
            //    the 2-wide vector, since otherwise the if() test above
            //    would have succeeded previously.  Therefore, we must have
            //    a situation with offsets like (4,6,...); it would be a
            //    silly idea to issue a 2-wide load to get the value for
            //    the 4 offset, versus failing here and issuing a scalar
            //    load instead.
            //
            // 2. If there are too many unnecessary values at the end of
            //    the load extent (defined as more than half of them)--in
            //    this case, it'd be better to issue a vector load of
            //    smaller width anyway.
            //
            // 3. If the gap between the last accepted offset and the
            //    current one under consideration is more than the page
            //    size.  In this case we can't be sure whether or not some
            //    of the unused elements at the end of the load will
            //    straddle a page boundary and thus lead to an undesirable
            //    fault.  (It's hard to imagine this happening in practice,
            //    except under contrived circumstances, but better safe
            //    than sorry.)
            const int pageSize = 4096;
            if (vectorWidth != 2 && (lastAccepted - start) > (vectorWidth / 2) && (*iter - lastAccepted) < pageSize) {
                *newIter = iter;
                return true;
            } else
                return false;
        }

        // Continue moving forward
        lastAccepted = *iter;
        ++iter;
    }

    return false;
}

/** Given a set of offsets from a common base pointer that we need to get
    loaded into memory, determine a reasonable set of load operations that
    gets all of the corresponding values in memory (ideally, including as
    many as possible wider vector loads rather than scalar loads).  Return
    a CoalescedLoadOp for each one in the *loads array.
 */
static void lSelectLoads(const std::vector<int64_t> &loadOffsets, std::vector<CoalescedLoadOp> *loads) {
    // First, get a sorted set of unique offsets to load from.
    std::set<int64_t> allOffsets;
    for (unsigned int i = 0; i < loadOffsets.size(); ++i)
        allOffsets.insert(loadOffsets[i]);

    std::set<int64_t>::iterator iter = allOffsets.begin();
    while (iter != allOffsets.end()) {
        Debug(SourcePos(), "Load needed at %" PRId64 ".", *iter);
        ++iter;
    }

    // Now, iterate over the offsets from low to high.  Starting at the
    // current offset, we see if a vector load starting from that offset
    // will cover loads at subsequent offsets as well.
    iter = allOffsets.begin();
    while (iter != allOffsets.end()) {
        // Consider vector loads of width of each of the elements of
        // spanSizes[], in order.
        int vectorWidths[] = {8, 4, 2};
        int nVectorWidths = sizeof(vectorWidths) / sizeof(vectorWidths[0]);
        bool gotOne = false;
        for (int i = 0; i < nVectorWidths; ++i) {
            // See if a load of vector with width vectorWidths[i] would be
            // effective (i.e. would cover a reasonable number of the
            // offsets that need to be loaded from).
            std::set<int64_t>::iterator newIter;
            if (lVectorLoadIsEfficient(iter, allOffsets.end(), &newIter, vectorWidths[i])) {
                // Yes: create the corresponding coalesced load and update
                // the iterator to the returned iterator; doing so skips
                // over the additional offsets that are taken care of by
                // this load.
                loads->push_back(CoalescedLoadOp(*iter, vectorWidths[i]));
                iter = newIter;
                gotOne = true;
                break;
            }
        }

        if (gotOne == false) {
            // We couldn't find a vector load starting from this offset
            // that made sense, so emit a scalar load and continue onward.
            loads->push_back(CoalescedLoadOp(*iter, 1));
            ++iter;
        }
    }
}

/** Print a performance message with the details of the result of
    coalescing over a group of gathers. */
static void lCoalescePerfInfo(const std::vector<llvm::CallInst *> &coalesceGroup,
                              const std::vector<CoalescedLoadOp> &loadOps) {
    SourcePos pos;
    LLVMGetSourcePosFromMetadata(coalesceGroup[0], &pos);

    // Create a string that indicates the line numbers of the subsequent
    // gathers from the first one that were coalesced here.
    char otherPositions[512];
    otherPositions[0] = '\0';
    if (coalesceGroup.size() > 1) {
        const char *plural = (coalesceGroup.size() > 2) ? "s" : "";
        char otherBuf[32];
        snprintf(otherBuf, sizeof(otherBuf), "(other%s at line%s ", plural, plural);
        strncat(otherPositions, otherBuf, sizeof(otherPositions) - strlen(otherPositions) - 1);

        for (int i = 1; i < (int)coalesceGroup.size(); ++i) {
            SourcePos p;
            bool ok = LLVMGetSourcePosFromMetadata(coalesceGroup[i], &p);
            if (ok) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%d", p.first_line);
                strncat(otherPositions, buf, sizeof(otherPositions) - strlen(otherPositions) - 1);
                if (i < (int)coalesceGroup.size() - 1)
                    strncat(otherPositions, ", ", sizeof(otherPositions) - strlen(otherPositions) - 1);
            }
        }
        strncat(otherPositions, ") ", sizeof(otherPositions) - strlen(otherPositions) - 1);
    }

    // Count how many loads of each size there were.
    std::map<int, int> loadOpsCount;
    for (int i = 0; i < (int)loadOps.size(); ++i)
        ++loadOpsCount[loadOps[i].count];

    // Generate a string the describes the mix of load ops
    char loadOpsInfo[512];
    loadOpsInfo[0] = '\0';
    std::map<int, int>::const_iterator iter = loadOpsCount.begin();
    while (iter != loadOpsCount.end()) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d x %d-wide", iter->second, iter->first);
        if ((strlen(loadOpsInfo) + strlen(buf)) >= 512) {
            break;
        }
        strncat(loadOpsInfo, buf, sizeof(loadOpsInfo) - strlen(loadOpsInfo) - 1);
        ++iter;
        if (iter != loadOpsCount.end())
            strncat(loadOpsInfo, ", ", sizeof(loadOpsInfo) - strlen(loadOpsInfo) - 1);
    }

    if (g->opt.level > 0) {
        if (coalesceGroup.size() == 1)
            PerformanceWarning(pos, "Coalesced gather into %d load%s (%s).", (int)loadOps.size(),
                               (loadOps.size() > 1) ? "s" : "", loadOpsInfo);
        else
            PerformanceWarning(pos,
                               "Coalesced %d gathers starting here %sinto %d "
                               "load%s (%s).",
                               (int)coalesceGroup.size(), otherPositions, (int)loadOps.size(),
                               (loadOps.size() > 1) ? "s" : "", loadOpsInfo);
    }
}

/** Utility routine that computes an offset from a base pointer and then
    returns the result of a load of the given type from the resulting
    location:

    return *((type *)(basePtr + offset))
 */
static llvm::Value *lGEPAndLoad(llvm::Value *basePtr, llvm::Type *baseType, int64_t offset, int align,
                                llvm::Instruction *insertBefore, llvm::Type *type) {
    llvm::Value *ptr = LLVMGEPInst(basePtr, baseType, LLVMInt64(offset), "new_base", insertBefore);
    ptr = new llvm::BitCastInst(ptr, llvm::PointerType::get(type, 0), "ptr_cast", insertBefore);
    Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
    return new llvm::LoadInst(type, ptr, "gather_load", false /* not volatile */, llvm::MaybeAlign(align).valueOrOne(),
                              insertBefore);
}

/* Having decided that we're doing to emit a series of loads, as encoded in
   the loadOps array, this function emits the corresponding load
   instructions.
 */
static void lEmitLoads(llvm::Value *basePtr, llvm::Type *baseType, std::vector<CoalescedLoadOp> &loadOps,
                       int elementSize, llvm::Instruction *insertBefore) {
    Debug(SourcePos(), "Coalesce doing %d loads.", (int)loadOps.size());
    for (int i = 0; i < (int)loadOps.size(); ++i) {
        Debug(SourcePos(), "Load #%d @ %" PRId64 ", %d items", i, loadOps[i].start, loadOps[i].count);

        // basePtr is an i8 *, so the offset from it should be in terms of
        // bytes, not underlying i32 elements.
        int64_t start = loadOps[i].start * elementSize;

        int align = 4;
        switch (loadOps[i].count) {
        case 1:
            // Single 32-bit scalar load
            loadOps[i].load = lGEPAndLoad(basePtr, baseType, start, align, insertBefore, LLVMTypes::Int32Type);
            break;
        case 2: {
            // Emit 2 x i32 loads as i64 loads and then break the result
            // into two 32-bit parts.
            loadOps[i].load = lGEPAndLoad(basePtr, baseType, start, align, insertBefore, LLVMTypes::Int64Type);
            // element0 = (int32)value;
            loadOps[i].element0 =
                new llvm::TruncInst(loadOps[i].load, LLVMTypes::Int32Type, "load64_elt0", insertBefore);
            // element1 = (int32)(value >> 32)
            llvm::Value *shift = llvm::BinaryOperator::Create(llvm::Instruction::LShr, loadOps[i].load, LLVMInt64(32),
                                                              "load64_shift", insertBefore);
            loadOps[i].element1 = new llvm::TruncInst(shift, LLVMTypes::Int32Type, "load64_elt1", insertBefore);
            break;
        }
        case 4: {
            // 4-wide vector load
            if (g->opt.forceAlignedMemory) {
                align = g->target->getNativeVectorAlignment();
            }
            llvm::VectorType *vt = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
            loadOps[i].load = lGEPAndLoad(basePtr, baseType, start, align, insertBefore, vt);
            break;
        }
        case 8: {
            // 8-wide vector load
            if (g->opt.forceAlignedMemory) {
                align = g->target->getNativeVectorAlignment();
            }
            llvm::VectorType *vt = LLVMVECTOR::get(LLVMTypes::Int32Type, 8);
            loadOps[i].load = lGEPAndLoad(basePtr, baseType, start, align, insertBefore, vt);
            break;
        }
        default:
            FATAL("Unexpected load count in lEmitLoads()");
        }
    }
}

/** Convert any loads of 8-wide vectors into two 4-wide vectors
    (logically).  This allows the assembly code below to always operate on
    4-wide vectors, which leads to better code.  Returns a new vector of
    load operations.
 */
static std::vector<CoalescedLoadOp> lSplit8WideLoads(const std::vector<CoalescedLoadOp> &loadOps,
                                                     llvm::Instruction *insertBefore) {
    std::vector<CoalescedLoadOp> ret;
    for (unsigned int i = 0; i < loadOps.size(); ++i) {
        if (loadOps[i].count == 8) {
            // Create fake CoalescedLOadOps, where the load llvm::Value is
            // actually a shuffle that pulls either the first 4 or the last
            // 4 values out of the original 8-wide loaded value.
            int32_t shuf[2][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}};

            ret.push_back(CoalescedLoadOp(loadOps[i].start, 4));
            ret.back().load = LLVMShuffleVectors(loadOps[i].load, loadOps[i].load, shuf[0], 4, insertBefore);

            ret.push_back(CoalescedLoadOp(loadOps[i].start + 4, 4));
            ret.back().load = LLVMShuffleVectors(loadOps[i].load, loadOps[i].load, shuf[1], 4, insertBefore);
        } else
            ret.push_back(loadOps[i]);
    }

    return ret;
}

/** Given a 1-wide load of a 32-bit value, merge its value into the result
    vector for any and all elements for which it applies.
 */
static llvm::Value *lApplyLoad1(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    for (int elt = 0; elt < 4; ++elt) {
        if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 1 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);
            // If this load gives one of the values that we need, then we
            // can just insert it in directly
            Assert(set[elt] == false);
            result = llvm::InsertElementInst::Create(result, load.load, LLVMInt32(elt), "insert_load", insertBefore);
            set[elt] = true;
        }
    }

    return result;
}

/** Similarly, incorporate the values from a 2-wide load into any vector
    elements that they apply to. */
static llvm::Value *lApplyLoad2(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    int elt = 0;
    while (elt < 4) {
        // First, try to do a 64-bit-wide insert into the result vector.
        // We can do this when we're currently at an even element, when the
        // current and next element have consecutive values, and where the
        // original 64-bit load is at the offset needed by the current
        // element.
        if ((elt & 1) == 0 && offsets[elt] + 1 == offsets[elt + 1] && offsets[elt] == load.start) {
            Debug(SourcePos(),
                  "Load 2 @ %" PRId64 " matches for elements #%d,%d "
                  "(values %" PRId64 ",%" PRId64 ")",
                  load.start, elt, elt + 1, offsets[elt], offsets[elt + 1]);
            Assert(set[elt] == false && ((elt < 3) && set[elt + 1] == false));

            // In this case, we bitcast from a 4xi32 to a 2xi64 vector
            llvm::Type *vec2x64Type = LLVMVECTOR::get(LLVMTypes::Int64Type, 2);
            result = new llvm::BitCastInst(result, vec2x64Type, "to2x64", insertBefore);

            // And now we can insert the 64-bit wide value into the
            // appropriate elment
            result = llvm::InsertElementInst::Create(result, load.load, LLVMInt32(elt / 2), "insert64", insertBefore);

            // And back to 4xi32.
            llvm::Type *vec4x32Type = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
            result = new llvm::BitCastInst(result, vec4x32Type, "to4x32", insertBefore);

            set[elt] = true;
            if (elt < 3) {
                set[elt + 1] = true;
            }
            // Advance elt one extra time, since we just took care of two
            // elements
            ++elt;
        } else if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 2 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);
            // Otherwise, insert one of the 32-bit pieces into an element
            // of the final vector
            Assert(set[elt] == false);
            llvm::Value *toInsert = (offsets[elt] == load.start) ? load.element0 : load.element1;
            result = llvm::InsertElementInst::Create(result, toInsert, LLVMInt32(elt), "insert_load", insertBefore);
            set[elt] = true;
        }
        ++elt;
    }

    return result;
}

#if 1
/* This approach works better with AVX, while the #else path generates
   slightly better code with SSE.  Need to continue to dig into performance
   details with this stuff in general... */

/** And handle a 4-wide load */
static llvm::Value *lApplyLoad4(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    // Conceptually, we're doing to consider doing a shuffle vector with
    // the 4-wide load and the 4-wide result we have so far to generate a
    // new 4-wide vector.  We'll start with shuffle indices that just
    // select each element of the result so far for the result.
    int32_t shuf[4] = {4, 5, 6, 7};

    for (int elt = 0; elt < 4; ++elt) {
        if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 4 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);

            // If the current element falls within the range of locations
            // that the 4-wide load covers, then compute the appropriate
            // shuffle index that extracts the appropriate element from the
            // load.
            Assert(set[elt] == false);
            shuf[elt] = int32_t(offsets[elt] - load.start);
            set[elt] = true;
        }
    }

    // Now, issue a shufflevector instruction if any of the values from the
    // load we just considered were applicable.
    if (shuf[0] != 4 || shuf[1] != 5 || shuf[2] != 6 || shuf[3] != 7)
        result = LLVMShuffleVectors(load.load, result, shuf, 4, insertBefore);

    return result;
}

/** We're need to fill in the values for a 4-wide result vector.  This
    function looks at all of the generated loads and extracts the
    appropriate elements from the appropriate loads to assemble the result.
    Here the offsets[] parameter gives the 4 offsets from the base pointer
    for the four elements of the result.
*/
static llvm::Value *lAssemble4Vector(const std::vector<CoalescedLoadOp> &loadOps, const int64_t offsets[4],
                                     llvm::Instruction *insertBefore) {
    llvm::Type *returnType = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
    llvm::Value *result = llvm::UndefValue::get(returnType);

    Debug(SourcePos(), "Starting search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    // Track whether we have found a valid value for each of the four
    // elements of the result
    bool set[4] = {false, false, false, false};

    // Loop over all of the loads and check each one to see if it provides
    // a value that's applicable to the result
    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &li = loadOps[load];

        switch (li.count) {
        case 1:
            result = lApplyLoad1(result, li, offsets, set, insertBefore);
            break;
        case 2:
            result = lApplyLoad2(result, li, offsets, set, insertBefore);
            break;
        case 4:
            result = lApplyLoad4(result, li, offsets, set, insertBefore);
            break;
        default:
            FATAL("Unexpected load count in lAssemble4Vector()");
        }
    }

    Debug(SourcePos(), "Done with search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    for (int i = 0; i < 4; ++i)
        Assert(set[i] == true);

    return result;
}

#else

static llvm::Value *lApplyLoad4s(llvm::Value *result, const std::vector<CoalescedLoadOp> &loadOps,
                                 const int64_t offsets[4], bool set[4], llvm::Instruction *insertBefore) {
    int32_t firstMatchElements[4] = {-1, -1, -1, -1};
    const CoalescedLoadOp *firstMatch = nullptr;

    Assert(llvm::isa<llvm::UndefValue>(result));

    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &loadop = loadOps[load];
        if (loadop.count != 4)
            continue;

        int32_t matchElements[4] = {-1, -1, -1, -1};
        bool anyMatched = false;
        for (int elt = 0; elt < 4; ++elt) {
            if (offsets[elt] >= loadop.start && offsets[elt] < loadop.start + loadop.count) {
                Debug(SourcePos(),
                      "Load 4 @ %" PRId64 " matches for element #%d "
                      "(value %" PRId64 ")",
                      loadop.start, elt, offsets[elt]);
                anyMatched = true;
                Assert(set[elt] == false);
                matchElements[elt] = offsets[elt] - loadop.start;
                set[elt] = true;
            }
        }

        if (anyMatched) {
            if (llvm::isa<llvm::UndefValue>(result)) {
                if (firstMatch == nullptr) {
                    firstMatch = &loadop;
                    for (int i = 0; i < 4; ++i)
                        firstMatchElements[i] = matchElements[i];
                } else {
                    int32_t shuffle[4] = {-1, -1, -1, -1};
                    for (int i = 0; i < 4; ++i) {
                        if (firstMatchElements[i] != -1)
                            shuffle[i] = firstMatchElements[i];
                        else
                            shuffle[i] = 4 + matchElements[i];
                    }
                    result = LLVMShuffleVectors(firstMatch->load, loadop.load, shuffle, 4, insertBefore);
                    firstMatch = nullptr;
                }
            } else {
                int32_t shuffle[4] = {-1, -1, -1, -1};
                for (int i = 0; i < 4; ++i) {
                    if (matchElements[i] != -1)
                        shuffle[i] = 4 + matchElements[i];
                    else
                        shuffle[i] = i;
                }
                result = LLVMShuffleVectors(result, loadop.load, shuffle, 4, insertBefore);
            }
        }
    }

    if (firstMatch != nullptr && llvm::isa<llvm::UndefValue>(result))
        return LLVMShuffleVectors(firstMatch->load, result, firstMatchElements, 4, insertBefore);
    else
        return result;
}

static llvm::Value *lApplyLoad12s(llvm::Value *result, const std::vector<CoalescedLoadOp> &loadOps,
                                  const int64_t offsets[4], bool set[4], llvm::Instruction *insertBefore) {
    // Loop over all of the loads and check each one to see if it provides
    // a value that's applicable to the result
    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &loadop = loadOps[load];
        Assert(loadop.count == 1 || loadop.count == 2 || loadop.count == 4);

        if (loadop.count == 1)
            result = lApplyLoad1(result, loadop, offsets, set, insertBefore);
        else if (loadop.count == 2)
            result = lApplyLoad2(result, loadop, offsets, set, insertBefore);
    }
    return result;
}

/** We're need to fill in the values for a 4-wide result vector.  This
    function looks at all of the generated loads and extracts the
    appropriate elements from the appropriate loads to assemble the result.
    Here the offsets[] parameter gives the 4 offsets from the base pointer
    for the four elements of the result.
*/
static llvm::Value *lAssemble4Vector(const std::vector<CoalescedLoadOp> &loadOps, const int64_t offsets[4],
                                     llvm::Instruction *insertBefore) {
    llvm::Type *returnType = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
    llvm::Value *result = llvm::UndefValue::get(returnType);

    Debug(SourcePos(), "Starting search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    // Track whether we have found a valid value for each of the four
    // elements of the result
    bool set[4] = {false, false, false, false};

    result = lApplyLoad4s(result, loadOps, offsets, set, insertBefore);
    result = lApplyLoad12s(result, loadOps, offsets, set, insertBefore);

    Debug(SourcePos(), "Done with search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    for (int i = 0; i < 4; ++i)
        Assert(set[i] == true);

    return result;
}
#endif

/** Given the set of loads that we've done and the set of result values to
    be computed, this function computes the final llvm::Value *s for each
    result vector.
 */
static void lAssembleResultVectors(const std::vector<CoalescedLoadOp> &loadOps,
                                   const std::vector<int64_t> &constOffsets, std::vector<llvm::Value *> &results,
                                   llvm::Instruction *insertBefore) {
    // We work on 4-wide chunks of the final values, even when we're
    // computing 8-wide or 16-wide vectors.  This gives better code from
    // LLVM's SSE/AVX code generators.
    Assert((constOffsets.size() % 4) == 0);
    std::vector<llvm::Value *> vec4s;
    for (int i = 0; i < (int)constOffsets.size(); i += 4)
        vec4s.push_back(lAssemble4Vector(loadOps, &constOffsets[i], insertBefore));

    // And now concatenate 1, 2, or 4 of the 4-wide vectors computed above
    // into 4, 8, or 16-wide final result vectors.
    int numGathers = constOffsets.size() / g->target->getVectorWidth();
    for (int i = 0; i < numGathers; ++i) {
        llvm::Value *result = nullptr;
        switch (g->target->getVectorWidth()) {
        case 4:
            result = vec4s[i];
            break;
        case 8:
            result = LLVMConcatVectors(vec4s[2 * i], vec4s[2 * i + 1], insertBefore);
            break;
        case 16: {
            llvm::Value *v1 = LLVMConcatVectors(vec4s[4 * i], vec4s[4 * i + 1], insertBefore);
            llvm::Value *v2 = LLVMConcatVectors(vec4s[4 * i + 2], vec4s[4 * i + 3], insertBefore);
            result = LLVMConcatVectors(v1, v2, insertBefore);
            break;
        }
        // The code for 32 and 64 width may be triggered when --opt=disable-gathers option is used.
        case 32: {
            llvm::Value *v1 = LLVMConcatVectors(vec4s[4 * i], vec4s[4 * i + 1], insertBefore);
            llvm::Value *v2 = LLVMConcatVectors(vec4s[4 * i + 2], vec4s[4 * i + 3], insertBefore);
            llvm::Value *v3 = LLVMConcatVectors(vec4s[4 * i + 4], vec4s[4 * i + 5], insertBefore);
            llvm::Value *v4 = LLVMConcatVectors(vec4s[4 * i + 6], vec4s[4 * i + 7], insertBefore);
            llvm::Value *res1 = LLVMConcatVectors(v1, v2, insertBefore);
            llvm::Value *res2 = LLVMConcatVectors(v3, v4, insertBefore);
            result = LLVMConcatVectors(res1, res2, insertBefore);
            break;
        }
        case 64: {
            llvm::Value *v1 = LLVMConcatVectors(vec4s[4 * i], vec4s[4 * i + 1], insertBefore);
            llvm::Value *v2 = LLVMConcatVectors(vec4s[4 * i + 2], vec4s[4 * i + 3], insertBefore);
            llvm::Value *v3 = LLVMConcatVectors(vec4s[4 * i + 4], vec4s[4 * i + 5], insertBefore);
            llvm::Value *v4 = LLVMConcatVectors(vec4s[4 * i + 6], vec4s[4 * i + 7], insertBefore);
            llvm::Value *v5 = LLVMConcatVectors(vec4s[4 * i + 8], vec4s[4 * i + 9], insertBefore);
            llvm::Value *v6 = LLVMConcatVectors(vec4s[4 * i + 10], vec4s[4 * i + 11], insertBefore);
            llvm::Value *v7 = LLVMConcatVectors(vec4s[4 * i + 12], vec4s[4 * i + 13], insertBefore);
            llvm::Value *v8 = LLVMConcatVectors(vec4s[4 * i + 14], vec4s[4 * i + 15], insertBefore);
            llvm::Value *res1 = LLVMConcatVectors(v1, v2, insertBefore);
            llvm::Value *res2 = LLVMConcatVectors(v3, v4, insertBefore);
            llvm::Value *res3 = LLVMConcatVectors(v5, v6, insertBefore);
            llvm::Value *res4 = LLVMConcatVectors(v7, v8, insertBefore);
            llvm::Value *res12 = LLVMConcatVectors(res1, res2, insertBefore);
            llvm::Value *res34 = LLVMConcatVectors(res3, res4, insertBefore);
            result = LLVMConcatVectors(res12, res34, insertBefore);
            break;
        }
        default:
            FATAL("Unhandled vector width in lAssembleResultVectors()");
        }

        results.push_back(result);
    }
}

/** Given a call to a gather function, extract the base pointer, the 2/4/8
    scale, and the first varying offsets value to use them to compute that
    scalar base pointer that is shared by all of the gathers in the group.
    (Thus, this base pointer plus the constant offsets term for each gather
    gives the set of addresses to use for each gather.
 */
static llvm::Value *lComputeBasePtr(llvm::CallInst *gatherInst, llvm::Type *baseType, llvm::Instruction *insertBefore) {
    llvm::Value *basePtr = gatherInst->getArgOperand(0);
    llvm::Value *variableOffsets = gatherInst->getArgOperand(1);
    llvm::Value *offsetScale = gatherInst->getArgOperand(2);
    // All of the variable offsets values should be the same, due to
    // checking for this in GatherCoalescePass::runOnBasicBlock().  Thus,
    // extract the first value and use that as a scalar.
    llvm::Value *variable = LLVMExtractFirstVectorElement(variableOffsets);
    Assert(variable != nullptr);
    if (variable->getType() == LLVMTypes::Int64Type)
        offsetScale = new llvm::ZExtInst(offsetScale, LLVMTypes::Int64Type, "scale_to64", insertBefore);
    llvm::Value *offset =
        llvm::BinaryOperator::Create(llvm::Instruction::Mul, variable, offsetScale, "offset", insertBefore);

    return LLVMGEPInst(basePtr, baseType, offset, "new_base", insertBefore);
}

/** Extract the constant offsets (from the common base pointer) from each
    of the gathers in a set to be coalesced.  These come in as byte
    offsets, but we'll transform them into offsets in terms of the size of
    the base scalar type being gathered.  (e.g. for an i32 gather, we might
    have offsets like <0,4,16,20>, which would be transformed to <0,1,4,5>
    here.)
 */
static void lExtractConstOffsets(const std::vector<llvm::CallInst *> &coalesceGroup, int elementSize,
                                 std::vector<int64_t> *constOffsets) {
    int width = g->target->getVectorWidth();
    *constOffsets = std::vector<int64_t>(coalesceGroup.size() * width, 0);

    int64_t *endPtr = &((*constOffsets)[0]);
    for (int i = 0; i < (int)coalesceGroup.size(); ++i, endPtr += width) {
        llvm::Value *offsets = coalesceGroup[i]->getArgOperand(3);
        int nElts;
        bool ok = LLVMExtractVectorInts(offsets, endPtr, &nElts);
        Assert(ok && nElts == width);
    }

    for (int i = 0; i < (int)constOffsets->size(); ++i)
        (*constOffsets)[i] /= elementSize;
}

/** Actually do the coalescing.  We have a set of gathers all accessing
    addresses of the form:

    (ptr + {1,2,4,8} * varyingOffset) + constOffset, a.k.a.
    basePtr + constOffset

    where varyingOffset actually has the same value across all of the SIMD
    lanes and where the part in parenthesis has the same value for all of
    the gathers in the group.
 */
static bool lCoalesceGathers(const std::vector<llvm::CallInst *> &coalesceGroup, llvm::Type *baseType) {
    llvm::Instruction *insertBefore = coalesceGroup[0];

    // First, compute the shared base pointer for all of the gathers
    llvm::Value *basePtr = lComputeBasePtr(coalesceGroup[0], baseType, insertBefore);

    int elementSize = 0;
    if (coalesceGroup[0]->getType() == LLVMTypes::Int32VectorType ||
        coalesceGroup[0]->getType() == LLVMTypes::FloatVectorType)
        elementSize = 4;
    else if (coalesceGroup[0]->getType() == LLVMTypes::Int64VectorType ||
             coalesceGroup[0]->getType() == LLVMTypes::DoubleVectorType)
        elementSize = 8;
    else
        FATAL("Unexpected gather type in lCoalesceGathers");

    // Extract the constant offsets from the gathers into the constOffsets
    // vector: the first vectorWidth elements will be those for the first
    // gather, the next vectorWidth those for the next gather, and so
    // forth.
    std::vector<int64_t> constOffsets;
    lExtractConstOffsets(coalesceGroup, elementSize, &constOffsets);

    // Determine a set of loads to perform to get all of the values we need
    // loaded.
    std::vector<CoalescedLoadOp> loadOps;
    lSelectLoads(constOffsets, &loadOps);

    lCoalescePerfInfo(coalesceGroup, loadOps);

    // Actually emit load instructions for them
    lEmitLoads(basePtr, baseType, loadOps, elementSize, insertBefore);

    // Now, for any loads that give us <8 x i32> vectors, split their
    // values into two <4 x i32> vectors; it turns out that LLVM gives us
    // better code on AVX when we assemble the pieces from 4-wide vectors.
    loadOps = lSplit8WideLoads(loadOps, insertBefore);

    // Given all of these chunks of values, shuffle together a vector that
    // gives us each result value; the i'th element of results[] gives the
    // result for the i'th gather in coalesceGroup.
    std::vector<llvm::Value *> results;
    lAssembleResultVectors(loadOps, constOffsets, results, insertBefore);

    // Finally, replace each of the original gathers with the instruction
    // that gives the value from the coalescing process.
    Assert(results.size() == coalesceGroup.size());
    for (int i = 0; i < (int)results.size(); ++i) {
        llvm::Instruction *ir = llvm::dyn_cast<llvm::Instruction>(results[i]);
        Assert(ir != nullptr);

        llvm::Type *origType = coalesceGroup[i]->getType();
        if (origType != ir->getType())
            ir = new llvm::BitCastInst(ir, origType, ir->getName(), coalesceGroup[i]);

        // Previously, all of the instructions to compute the final result
        // were into the basic block here; here we remove the very last one
        // of them (that holds the final result) from the basic block.
        // This way, the following ReplaceInstWithInst() call will operate
        // successfully. (It expects that the second argument not be in any
        // basic block.)
        ir->removeFromParent();

        llvm::ReplaceInstWithInst(coalesceGroup[i], ir);
    }

    return true;
}

/** Given an instruction, returns true if the instructon may write to
    memory.  This is a conservative test in that it may return true for
    some instructions that don't actually end up writing to memory, but
    should never return false for an instruction that does write to
    memory. */
static bool lInstructionMayWriteToMemory(llvm::Instruction *inst) {
    if (llvm::isa<llvm::StoreInst>(inst) || llvm::isa<llvm::AtomicRMWInst>(inst) ||
        llvm::isa<llvm::AtomicCmpXchgInst>(inst))
        // FIXME: we could be less conservative and try to allow stores if
        // we are sure that the pointers don't overlap..
        return true;

    // Otherwise, any call instruction that doesn't have an attribute
    // indicating it won't write to memory has to be treated as a potential
    // store.
    llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst);
    if (ci != nullptr) {
        llvm::Function *calledFunc = ci->getCalledFunction();
        if (calledFunc == nullptr)
            return true;

        if (calledFunc->onlyReadsMemory() || calledFunc->doesNotAccessMemory())
            return false;
        return true;
    }

    return false;
}

bool GatherCoalescePass::coalesceGathersFactored(llvm::BasicBlock &bb) {
    DEBUG_START_BB("GatherCoalescePass");

    llvm::Module *M = bb.getModule();
    llvm::Function *gatherFuncs[] = {
        M->getFunction(builtin::__pseudo_gather_factored_base_offsets32_i32),
        M->getFunction(builtin::__pseudo_gather_factored_base_offsets32_float),
        M->getFunction(builtin::__pseudo_gather_factored_base_offsets64_i32),
        M->getFunction(builtin::__pseudo_gather_factored_base_offsets64_float),
    };
    int nGatherFuncs = sizeof(gatherFuncs) / sizeof(gatherFuncs[0]);

    bool modifiedAny = false;

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::BasicBlock::iterator curIter = iter++;
        // Iterate over all of the instructions and look for calls to
        // __pseudo_gather_factored_base_offsets{32,64}_{i32,float} calls.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*curIter);
        if (callInst == nullptr)
            continue;

        llvm::Function *calledFunc = callInst->getCalledFunction();
        if (calledFunc == nullptr)
            continue;

        int i;
        for (i = 0; i < nGatherFuncs; ++i)
            if (gatherFuncs[i] != nullptr && calledFunc == gatherFuncs[i])
                break;
        if (i == nGatherFuncs)
            // Doesn't match any of the types of gathers we care about
            continue;

        SourcePos pos;
        LLVMGetSourcePosFromMetadata(callInst, &pos);
        Debug(pos, "Checking for coalescable gathers starting here...");

        llvm::Value *base = callInst->getArgOperand(0);
        llvm::Value *variableOffsets = callInst->getArgOperand(1);
        llvm::Value *offsetScale = callInst->getArgOperand(2);
        llvm::Value *mask = callInst->getArgOperand(4);

        // To apply this optimization, we need a set of one or more gathers
        // that fulfill the following conditions:
        //
        // - Mask all on
        // - The variable offsets to all have the same value (i.e., to be
        //   uniform).
        // - Same base pointer, variable offsets, and offset scale (for
        //   more than one gather)
        //
        // Then and only then do we have a common base pointer with all
        // offsets from that constants (in which case we can potentially
        // coalesce).
        if (GetMaskStatusFromValue(mask) != MaskStatus::all_on)
            continue;

        if (!LLVMVectorValuesAllEqual(variableOffsets))
            continue;

        // coalesceGroup stores the set of gathers that we're going to try to
        // coalesce over
        std::vector<llvm::CallInst *> coalesceGroup;
        coalesceGroup.push_back(callInst);

        // Start iterating at the instruction after the initial gather;
        // look at the remainder of instructions in the basic block (up
        // until we reach a write to memory) to try to find any other
        // gathers that can coalesce with this one.
        llvm::BasicBlock::iterator fwdIter = curIter;
        ++fwdIter;
        for (; fwdIter != bb.end(); ++fwdIter) {
            // Must stop once we come to an instruction that may write to
            // memory; otherwise we could end up moving a read before this
            // write.
            if (lInstructionMayWriteToMemory(&*fwdIter))
                break;

            llvm::CallInst *fwdCall = llvm::dyn_cast<llvm::CallInst>(&*fwdIter);
            if (fwdCall == nullptr || fwdCall->getCalledFunction() != calledFunc)
                continue;

            SourcePos fwdPos;
            // TODO: need to redesign metadata attached to pseudo calls,
            // LLVM drops metadata frequently and it results in bad disgnostics.
            LLVMGetSourcePosFromMetadata(fwdCall, &fwdPos);

            if (g->debugPrint) {
                if (base != fwdCall->getArgOperand(0)) {
                    Debug(fwdPos, "base pointers mismatch");
                    LLVMDumpValue(base);
                    LLVMDumpValue(fwdCall->getArgOperand(0));
                }
                if (variableOffsets != fwdCall->getArgOperand(1)) {
                    Debug(fwdPos, "varying offsets mismatch");
                    LLVMDumpValue(variableOffsets);
                    LLVMDumpValue(fwdCall->getArgOperand(1));
                }
                if (offsetScale != fwdCall->getArgOperand(2)) {
                    Debug(fwdPos, "offset scales mismatch");
                    LLVMDumpValue(offsetScale);
                    LLVMDumpValue(fwdCall->getArgOperand(2));
                }
                if (mask != fwdCall->getArgOperand(4)) {
                    Debug(fwdPos, "masks mismatch");
                    LLVMDumpValue(mask);
                    LLVMDumpValue(fwdCall->getArgOperand(4));
                }
            }

            if (base == fwdCall->getArgOperand(0) && variableOffsets == fwdCall->getArgOperand(1) &&
                offsetScale == fwdCall->getArgOperand(2) && mask == fwdCall->getArgOperand(4)) {
                Debug(fwdPos, "This gather can be coalesced.");
                coalesceGroup.push_back(fwdCall);
                // We deal with a group of instructions handled in a single pass of the optimization.
                // "iter" points to the insturction which needs to be handled on the next iteration.
                // By default it's the next instruction after the first one in the gorup.
                // If this happens to be another group instruction, move further.
                if (fwdCall == &*iter) {
                    iter++;
                }

                if (coalesceGroup.size() == 4)
                    // FIXME: untested heuristic: don't try to coalesce
                    // over a window of more than 4 gathers, so that we
                    // don't cause too much register pressure and end up
                    // spilling to memory anyway.
                    break;
            } else
                Debug(fwdPos, "This gather doesn't match the initial one.");
        }

        Debug(pos, "Done with checking for matching gathers");

        // Now that we have a group of gathers, see if we can coalesce them
        // into something more efficient than the original set of gathers.
        if (lCoalesceGathers(coalesceGroup, baseType)) {
            modifiedAny = true;
        }
    }
    DEBUG_END_BB("GatherCoalescePass");

    return modifiedAny;
}

llvm::PreservedAnalyses GatherCoalescePass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("GatherCoalescePass::run", F.getName());

    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= coalesceGathersFactored(BB);
    }

    if (!modifiedAny) {
        // No changes, all analyses are preserved.
        return llvm::PreservedAnalyses::all();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
