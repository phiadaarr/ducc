/*! \file ducc0/infra/mav.h
 *  Classes for dealing with multidimensional arrays
 *
 *  \copyright Copyright (C) 2019-2026 Max-Planck-Society
 *  \copyright Copyright (C) 2026 Philipp Arras
 *  \author Martin Reinecke
 *  \author Philipp Arras
 *  */

/* SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-or-later */

/*
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef DUCC0_MAV_H
#define DUCC0_MAV_H

#include <algorithm>
#include <array>
#include <concepts>
#include <span>
#include <vector>
#include <memory>
#include <numeric>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <tuple>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/threading.h"

namespace DUCC0_NAMESPACE {

namespace detail_mav {

using namespace std;

struct uninitialized_dummy {};
constexpr uninitialized_dummy UNINITIALIZED;

static inline void page_in_memory2(char *ptr, size_t sz, size_t nthreads=1)
  {
  if (sz==0) return;
// FIXME: can we determine the real page size for the underlying chunk of memory?
  size_t pagesize = 4096;
  size_t npages = (sz+pagesize-1)/pagesize;
// FIXME: perhaps reduce nthreads if number of pages is low?
  nthreads = min(nthreads, max(size_t(1), npages/1024));
  execParallel(npages, nthreads, [&](size_t lo, size_t hi) {
  for (size_t i=lo; i<hi; ++i)
    ptr[i*pagesize]=char(1);  // touch the memory page with a write access
    });
  }
template<typename T> void page_in_memory(T *ptr, size_t sz, size_t nthreads=1)
  { page_in_memory2(reinterpret_cast<char *>(ptr), sizeof(T)*sz, nthreads); }

struct PAGE_IN
  {
  size_t nthreads;
  explicit PAGE_IN(size_t nthreads_) : nthreads(nthreads_) {}
  };

/// Reference-counted memory buffer.
/** \a owner keeps the memory alive (it is empty for externally owned data),
 *  \a d points to the first element of the view, which may lie anywhere
 *  inside the owned memory. Writability is decided by the array classes
 *  holding the buffer, so \a d is stored as a non-const pointer. */
template<typename T> class membuf
  {
  private:
    // Uninitialized memory. Deliberately not make_shared_for_overwrite<T[]>,
    // which runs default constructors (e.g. zeroing for std::complex) and
    // thereby touches all pages from a single thread.
    void alloc_uninitialized(size_t sz)
      {
      auto tmp = make_shared<quick_array<T>>(sz);
      d = tmp->data();
      owner = std::move(tmp);
      }

    shared_ptr<void> owner;
    T *d;

  public:
    // share ownership with another buffer, but view a different location
    membuf(T *d_, const membuf &other)
      : owner(other.owner), d(d_) {}

    // externally owned data pointer. This is the only place where constness
    // is cast away; read-only classes never write through \a d.
    explicit membuf(const T *d_)
      : d(const_cast<T *>(d_)) {}
    explicit membuf(size_t sz)
      {
      // value-initialized. Not make_shared<T[]>(sz): libstdc++ copies the
      // elements from a prototype, which fails for e.g. std::mutex.
      d = new T[sz]();
      owner = shared_ptr<T[]>(d);
      }
#if 1
    membuf(size_t sz, uninitialized_dummy)
      { alloc_uninitialized(sz); }
    membuf(size_t sz, PAGE_IN page_in)
      {
      alloc_uninitialized(sz);
      page_in_memory(d, sz, page_in.nthreads);
      }
# else // "poison" the array with a fixed value; use for debugging
    membuf(size_t sz, uninitialized_dummy)
      {
      alloc_uninitialized(sz);
      for (size_t i=0; i<sz; ++i) d[i]=T(42000000);
      }
    membuf(size_t sz, PAGE_IN)
      : membuf(sz, UNINITIALIZED) {}
#endif
    membuf(): d(nullptr) {}

    T *data() const
      { return d; }
  };

/// Rank of arrays whose number of dimensions is only known at runtime
inline constexpr size_t dynamic_rank = numeric_limits<size_t>::max();


constexpr size_t MAXIDX=numeric_limits<size_t>::max();

struct slice
  {
  size_t beg, end;
  ptrdiff_t step;
  slice() : beg(0), end(MAXIDX), step(1) {}
  slice(size_t idx) : beg(idx), end(idx), step(1) {}
  slice(size_t beg_, size_t end_, ptrdiff_t step_=1)
    : beg(beg_), end(end_), step(step_)
    {
    MR_assert(step!=0, "slice step must not be 0");
    // FIXME: support (MAXIDX,MAXIDX,-1) and similar? (reversal)
    if ((beg!=end) && (end!=MAXIDX))
      MR_assert((beg<end)!=(step<0), "step does not match begin, end ordering");
    }

  size_t size(size_t shp) const
    {
    if (beg==end) return 0;
    if (step>0) return (min(shp,end)-beg+step-1)/step;
    // negative step
    if (end==MAXIDX)
      return (beg-step)/(-step);
    return (beg-end-step-1)/(-step);
    }
  };

// C-contiguous strides for shape \a shp, written to \a str
inline void shape2stride(span<const size_t> shp, span<ptrdiff_t> str)
  {
  MR_assert(shp.size()==str.size(), "dimensions mismatch");
  ptrdiff_t stride=1;
  for (size_t i=shp.size(); i>0; --i)
    {
    str[i-1]=stride;
    stride*=ptrdiff_t(shp[i-1]);
    }
  }
template<size_t N> inline void shape2stride(const array<size_t, N> &shp,
  array<ptrdiff_t, N> &str)
  {
  [&]<size_t... Is>(index_sequence<Is...>)
    {
    [[maybe_unused]] ptrdiff_t stride=1;
    ((str[N-1-Is]=stride, stride*=ptrdiff_t(shp[N-1-Is])), ...);
    }(make_index_sequence<N>{});
  }

// Number of entries of an array with shape \a shp
inline size_t shape_size(span<const size_t> shp)
  { return ranges::fold_left(shp, size_t(1), multiplies<>()); }
template<size_t N> inline size_t shape_size(const array<size_t, N> &shp)
  {
  return [&]<size_t... Is>(index_sequence<Is...>)
    { return (size_t(1) * ... * shp[Is]); }(make_index_sequence<N>{});
  }

inline bool is_contiguous(span<const size_t> shp, span<const ptrdiff_t> str)
  {
  ptrdiff_t stride=1;
  for (size_t i=shp.size(); i>0; --i)
    {
    if ((shp[i-1]!=1) && (str[i-1]!=stride)) return false;
    stride *= ptrdiff_t(shp[i-1]);
    }
  return true;
  }

// Offset of the entry at multi-index \a ns for strides \a str
template<integral... Ns> inline ptrdiff_t strided_idx(
  [[maybe_unused]] span<const ptrdiff_t> str, Ns... ns)
  {
  ptrdiff_t res=0;
  [[maybe_unused]] size_t dim=0;
  ((res += str[dim++]*ptrdiff_t(ns)), ...);
  return res;
  }

// Strides after inserting the axes of (shp, str) at positions \a axpos of a
// higher-dimensional array; new axes get stride 0. \a new_str must be zeroed.
inline void extend_and_broadcast_info(span<const size_t> shp,
  span<const ptrdiff_t> str, span<const size_t> new_shape,
  span<const size_t> axpos, span<ptrdiff_t> new_str)
  {
  MR_assert(new_shape.size()>=shp.size(), "new shape smaller than original one");
  MR_assert(axpos.size()==shp.size(), "bad axpos size");
  MR_assert(new_str.size()==new_shape.size(), "bad stride size");
  for (size_t i=0; i<shp.size(); ++i)
    {
    MR_assert(axpos[i]<new_shape.size(), "bad axis number");
    MR_assert(shp[i]==new_shape[axpos[i]], "axis length mismatch");
    for (size_t j=0; j<i; ++j)
      MR_assert(axpos[j]!=axpos[i], "repeated axis position");
    new_str[axpos[i]] = str[i];
    }
  }

inline void transpose_info(span<const size_t> shp, span<const ptrdiff_t> str,
  span<const size_t> axes, span<size_t> shp2, span<ptrdiff_t> str2)
  {
  MR_assert(axes.size()==shp.size(), "bad axes length");
  for (size_t i=0; i<axes.size(); ++i)
    {
    MR_assert(axes[i]<shp.size(), "invalid axis number");
    for (size_t j=0; j<i; ++j)
      MR_assert(axes[j]!=axes[i], "repeated axis");
    shp2[i] = shp[axes[i]];
    str2[i] = str[axes[i]];
    }
  }

/// Number of dimensions that remain after applying \a slices
inline size_t subdata_ndim(const vector<slice> &slices)
  { return size_t(ranges::count_if(slices, [](const slice &x){ return x.beg!=x.end; })); }

// Returns the offset; writes the output shape/stride (sized by the caller)
inline ptrdiff_t subdata_info(span<const size_t> shp, span<const ptrdiff_t> str,
  const vector<slice> &slices, span<size_t> nshp, span<ptrdiff_t> nstr)
  {
  MR_assert(slices.size()==shp.size(), "incorrect number of slices");
  MR_assert(nshp.size()==subdata_ndim(slices), "bad extent");
  MR_assert(nstr.size()==nshp.size(), "bad extent");
  ptrdiff_t nofs=0;
  for (size_t i=0, i2=0; i<shp.size(); ++i)
    {
    if (shp[i]==0)
      {
      MR_assert((slices[i].beg==0) && (slices[i].end==MAXIDX), "bad subset");
      nshp[i2]=0;
      nstr[i2]=0;
      ++i2;
      }
    else
      {
      MR_assert(slices[i].beg<shp[i], "bad subset");
      nofs+=slices[i].beg*str[i];
      if (slices[i].beg!=slices[i].end)
        {
        auto ext = slices[i].size(shp[i]);
        MR_assert(slices[i].beg+(ext-1)*slices[i].step<shp[i], "bad subset");
        nshp[i2]=ext; nstr[i2]=slices[i].step*str[i];
        ++i2;
        }
      }
    }
  return nofs;
  }

/// Helper class containing shape and stride information of a `mav` object.
/** \a N is the number of dimensions, or \a dynamic_rank if it is only known
 *  at runtime. For fixed rank, shape and strides are stored in
 *  \a Tcontainer (normally std::array; kept generic so that e.g. a device-side
 *  array type can be used), for dynamic rank in std::vector. */
template<template<typename, size_t> typename Tcontainer, size_t N> class mav_info_proto
  {
  public:
    static constexpr bool is_dynamic = (N==dynamic_rank);
    /// Rank of the result of prepend_1()
    static constexpr size_t rank_plus_1 = is_dynamic ? dynamic_rank : N+1;
    /// nonnegative integers for storing the array shape
    using shape_t = conditional_t<is_dynamic, vector<size_t>, Tcontainer<size_t, N>>;
    /// integers for storing the array strides
    using stride_t = conditional_t<is_dynamic, vector<ptrdiff_t>, Tcontainer<ptrdiff_t, N>>;

    /// Zero-filled shape with \a nd entries (\a nd must equal N for fixed rank)
    static shape_t zero_shape(size_t nd)
      {
      if constexpr (is_dynamic)
        return shape_t(nd, 0);
      else
        {
        MR_assert(nd==N, "dimensionality mismatch");
        return shape_t{};
        }
      }
    /// Zero-filled strides with \a nd entries (\a nd must equal N for fixed rank)
    static stride_t zero_stride(size_t nd)
      {
      if constexpr (is_dynamic)
        return stride_t(nd, 0);
      else
        {
        MR_assert(nd==N, "dimensionality mismatch");
        return stride_t{};
        }
      }

  protected:
    shape_t shp;
    stride_t str;
    size_t sz{0};

    static stride_t shape2stride(const shape_t &shp)
      {
      auto res = zero_stride(shp.size());
      detail_mav::shape2stride(shp, res);
      return res;
      }

  public:
    /// Constructs an object with all extents and strides set to zero.
    /// With dynamic rank, the object is one-dimensional.
    mav_info_proto()
      : shp(zero_shape(is_dynamic ? 1 : N)), str(zero_stride(is_dynamic ? 1 : N)) {}
    /// Constructs an object with the given shape and stride.
    mav_info_proto(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_), sz(shape_size(shp))
      {
      if constexpr (is_dynamic)
        MR_assert(shp.size()==str.size(), "dimensions mismatch");
      }
    /// Constructs an object with the given shape and computes the strides
    /// automatically, assuming a C-contiguous memory layout.
    explicit mav_info_proto(const shape_t &shape_)
      : mav_info_proto(shape_, shape2stride(shape_)) {}
    /// Converts between fixed and dynamic rank: implicitly towards dynamic
    /// rank, explicitly (with a runtime check) towards fixed rank.
    template<size_t N2> requires ((N2!=N) && (is_dynamic || (N2==dynamic_rank)))
    explicit(!is_dynamic) mav_info_proto(const mav_info_proto<Tcontainer, N2> &other)
      : shp(zero_shape(other.ndim())), str(zero_stride(other.ndim())),
        sz(other.size())
      {
      for (size_t i=0; i<ndim(); ++i)
        {
        shp[i] = other.shape(i);
        str[i] = other.stride(i);
        }
      }
    void assign(const mav_info_proto &other)
      {
      shp = other.shp;
      str = other.str;
      sz = other.sz;
      }
    /// Returns the dimensionality of the object.
    constexpr size_t ndim() const
      {
      if constexpr (is_dynamic)
        return shp.size();
      else
        return N;
      }
    /// Returns the total number of entries in the object.
    size_t size() const { return sz; }
    /// Returns the shape of the object.
    const shape_t &shape() const { return shp; }
    /// Returns the length along dimension \a i.
    size_t shape(size_t i) const { return shp[i]; }
    /// Returns the strides of the object.
    const stride_t &stride() const { return str; }
    /// Returns the stride along dimension \a i.
    ptrdiff_t stride(size_t i) const { return str[i]; }
    /// Returns true iff the last dimension has stride 1.
    /**  Typically used for optimization purposes. */
    bool last_contiguous() const
      { return ((ndim()==0) || (str.back()==1)); }
    /** Returns true iff the object is C-contiguous, i.e. if the stride of the
     *  last dimension is 1, the stride for the next-to-last dimension is the
     *  shape of the last dimension etc. */
    bool contiguous() const
      { return is_contiguous(shp, str); }
    /// Returns true iff this->shape and \a other.shape match.
    bool conformable(const mav_info_proto &other) const
      { return shp==other.shp; }
    /// Returns true iff this->shape and \a other match.
    bool conformable(const shape_t &other) const
      { return shp==other; }
    /// Returns the one-dimensional index of an entry from the given
    /// multi-dimensional index tuple, taking strides into account.
    template<integral... Ns> ptrdiff_t idx(Ns... ns) const
      {
      if constexpr (is_dynamic)
        MR_assert(ndim()==sizeof...(ns), "incorrect number of indices");
      else
        static_assert(N==sizeof...(ns), "incorrect number of indices");
      return strided_idx(str, ns...);
      }
    ptrdiff_t idx(const shape_t &ns) const
      {
      MR_assert(ndim()==ns.size(), "incorrect number of indices");
      ptrdiff_t res = 0;
      for (size_t i=0; i<ndim(); ++i) res += str[i]*ptrdiff_t(ns[i]);
      return res;
      }
    template<typename RAiter> ptrdiff_t idxval(RAiter beg, RAiter end) const
      {
      MR_assert(ndim()==size_t(end-beg), "incorrect number of indices");
      ptrdiff_t res = 0;
      for (size_t i=0; i<ndim(); ++i, ++beg) res += str[i]* ptrdiff_t(*beg);
      return res;
      }
    /// Returns the common broadcast shape of *this and \a shp2
    shape_t bcast_shape(const shape_t &shp2) const requires is_dynamic
      {
      shape_t res(max(shp.size(), shp2.size()), 1);
      for (size_t i=0; i<shp.size(); ++i)
        res[i+res.size()-shp.size()] = shp[i];
      for (size_t i=0; i<shp2.size(); ++i)
        {
        size_t i2 = i+res.size()-shp2.size();
        if (res[i2]==1)
          res[i2] = shp2[i];
        else
          MR_assert((res[i2]==shp2[i])||(shp2[i]==1),
            "arrays cannot be broadcast together");
        }
      return res;
      }
    void bcast_to_shape(const shape_t &shp2) requires is_dynamic
      {
      MR_assert(shp2.size()>=shp.size(), "cannot reduce dimensionality");
      stride_t newstr(shp2.size(), 0);
      for (size_t i=0; i<shp.size(); ++i)
        {
        size_t i2 = i+shp2.size()-shp.size();
        if (shp[i]!=1)
          {
          MR_assert(shp[i]==shp2[i2], "arrays cannot be broadcast together");
          newstr[i2] = str[i];
          }
        }
      shp = shp2;
      str = newstr;
      sz = shape_size(shp);
      }

    void swap_axes(size_t ax0, size_t ax1)
      {
      MR_assert(ax0<ndim() && ax1<ndim(), "bad axes");
      if (ax0==ax1) return;
      swap(shp[ax0], shp[ax1]);
      swap(str[ax0], str[ax1]);
      }

    // dynamic rank: the result has dynamic rank as well
    [[nodiscard]] mav_info_proto extend_and_broadcast(const shape_t &new_shape,
      const vector<size_t> &axpos) const requires is_dynamic
      {
      stride_t new_stride(new_shape.size(), 0);
      extend_and_broadcast_info(shp, str, new_shape, axpos, new_stride);
      return mav_info_proto(new_shape, new_stride);
      }
    [[nodiscard]] mav_info_proto extend_and_broadcast(const shape_t &new_shape,
      size_t firstaxis) const requires is_dynamic
      {
      vector<size_t> axpos(ndim());
      std::iota(axpos.begin(), axpos.end(), firstaxis);
      return extend_and_broadcast(new_shape, axpos);
      }
    // fixed rank: the result rank is given by the new shape
    template<size_t nd2> requires (!is_dynamic)
    [[nodiscard]] mav_info_proto<Tcontainer, nd2> extend_and_broadcast(const Tcontainer<size_t, nd2> &new_shape,
      const vector<size_t> &axpos) const
      {
      static_assert(nd2>=N, "new shape smaller than original one");
      Tcontainer<ptrdiff_t, nd2> new_stride{};
      extend_and_broadcast_info(shp, str, new_shape, axpos, new_stride);
      return mav_info_proto<Tcontainer, nd2>(new_shape, new_stride);
      }
    template<size_t nd2> requires (!is_dynamic)
    [[nodiscard]] mav_info_proto<Tcontainer, nd2> extend_and_broadcast(const Tcontainer<size_t, nd2> &new_shape,
      size_t firstaxis) const
      {
      vector<size_t> axpos(N);
      std::iota(axpos.begin(), axpos.end(), firstaxis);
      return extend_and_broadcast(new_shape, axpos);
      }
    [[nodiscard]] mav_info_proto transpose(const shape_t &axes) const
      {
      auto shp2 = zero_shape(ndim());
      auto str2 = zero_stride(ndim());
      transpose_info(shp, str, axes, shp2, str2);
      return mav_info_proto(shp2, str2);
      }
    [[nodiscard]] mav_info_proto<Tcontainer, rank_plus_1> prepend_1() const
      {
      using res_t = mav_info_proto<Tcontainer, rank_plus_1>;
      auto newshp = res_t::zero_shape(ndim()+1);
      auto newstr = res_t::zero_stride(ndim()+1);
      newshp[0] = 1;
      newstr[0] = 0;
      for (size_t i=0; i<ndim(); ++i)
        {
        newshp[i+1] = shp[i];
        newstr[i+1] = str[i];
        }
      return res_t(newshp, newstr);
      }

  protected:
    // Info and offset of the sub-array described by \a slices; the result
    // has rank \a nd2 (which may be dynamic_rank).
    template<size_t nd2> auto subdata(const vector<slice> &slices) const
      {
      using res_t = mav_info_proto<Tcontainer, nd2>;
      auto nd = subdata_ndim(slices);
      auto nshp = res_t::zero_shape(nd);
      auto nstr = res_t::zero_stride(nd);
      auto nofs = subdata_info(shp, str, slices, nshp, nstr);
      return make_tuple(res_t(nshp, nstr), nofs);
      }
  };

template<size_t N> using mav_info = mav_info_proto<std::array, N>;
using fmav_info = mav_info<dynamic_rank>;

// Tag selecting the internal constructors of mav_storage / basic_mav
struct mav_construct {};

/// Shape, stride and memory of an array; base of basic_mav.
/** All real construction happens here; basic_mav only forwards, so that its
 *  constructors can be written once for the read-only and the writable
 *  variant. */
template<typename T, size_t N> class mav_storage: public mav_info<N>
  {
  protected:
    template<typename T2, size_t N2> friend class mav_storage;

    using tinfo = mav_info<N>;
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tinfo::shp, tinfo::str, tinfo::ndim, tinfo::size;

    membuf<T> buf;

    // NOTE: assert_compactness is false for a fully reversed contiguous array.
    void assert_compactness() const
      {
      ptrdiff_t ofs=0;
      for (size_t i=0; i<ndim(); ++i)
        ofs += (ptrdiff_t(shp[i])-1)*str[i];
      MR_assert(ofs+1==ptrdiff_t(size()), "array is not compact");
      }

    explicit mav_storage(mav_construct) {}
    // newly allocated memory
    mav_storage(mav_construct, const shape_t &shp_)
      : tinfo(shp_), buf(size()) {}
    mav_storage(mav_construct, const shape_t &shp_, uninitialized_dummy)
      : tinfo(shp_), buf(size(), UNINITIALIZED) {}
    mav_storage(mav_construct, const shape_t &shp_, PAGE_IN page_in)
      : tinfo(shp_), buf(size(), page_in) {}
    mav_storage(mav_construct, const shape_t &shp_, const stride_t &str_, uninitialized_dummy)
      : tinfo(shp_, str_), buf(size(), UNINITIALIZED)
      { assert_compactness(); }
    mav_storage(mav_construct, const shape_t &shp_, const stride_t &str_, PAGE_IN page_in)
      : tinfo(shp_, str_), buf(size(), page_in)
      { assert_compactness(); }
    // externally owned memory
    mav_storage(mav_construct, const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), buf(d_) {}
    mav_storage(mav_construct, const T *d_, const shape_t &shp_)
      : tinfo(shp_), buf(d_) {}
    mav_storage(mav_construct, const T *d_, const tinfo &info)
      : tinfo(info), buf(d_) {}
    // memory shared with an existing buffer
    mav_storage(mav_construct, const membuf<T> &buf_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), buf(buf_) {}
    mav_storage(mav_construct, const tinfo &info, const membuf<T> &buf_)
      : tinfo(info), buf(buf_) {}
    mav_storage(mav_construct, const tinfo &info, T *d_, const membuf<T> &buf_)
      : tinfo(info), buf(d_, buf_) {}
    // rank conversion
    template<size_t N2>
    mav_storage(mav_construct, const mav_storage<T, N2> &other)
      : tinfo(static_cast<const mav_info<N2> &>(other)), buf(other.buf) {}
  };

/// View of an array with \a N dimensions (or a runtime number of dimensions,
/// if N==dynamic_rank). Elements are writable iff \a W is true; the writable
/// variant derives from the read-only one, so it can be passed wherever a
/// read-only array is expected. Memory is reference-counted, unless it is
/// externally owned.
/** Use the aliases cmav<T,N> (read-only) and vmav<T,N> (writable).
 *  The writable variant re-declares every accessor with a non-const element
 *  type, which hides the read-only version of its base. */
template<typename T, size_t N, bool W> class basic_mav
  : public conditional_t<W, basic_mav<T, N, false>, mav_storage<T, N>>
  {
  protected:
    template<typename T2, size_t N2, bool W2> friend class basic_mav;

    using base_t = conditional_t<W, basic_mav<T, N, false>, mav_storage<T, N>>;
    using tinfo = mav_info<N>;
    using tbuf = membuf<T>;
    using mav_storage<T, N>::buf;
    static constexpr bool is_dynamic = (N==dynamic_rank);

    // forwards to the matching constructor of mav_storage
    template<typename... Args> explicit basic_mav(mav_construct, Args &&... args)
      : base_t(mav_construct(), std::forward<Args>(args)...) {}

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using element_type = conditional_t<W, T, const T>;
    using value_type = T;
    using tinfo::ndim, tinfo::contiguous, tinfo::size, tinfo::idx,
          tinfo::conformable;

    basic_mav() : basic_mav(mav_construct()) {}
    basic_mav(element_type *d_, const shape_t &shp_, const stride_t &str_)
      : basic_mav(mav_construct(), d_, shp_, str_) {}
    basic_mav(element_type *d_, const shape_t &shp_)
      : basic_mav(mav_construct(), d_, shp_) {}
    basic_mav(element_type *d_, const tinfo &info)
      : basic_mav(mav_construct(), d_, info) {}
    // Allocating constructors; a newly allocated read-only array would be
    // useless, so they only exist for the writable variant.
    explicit basic_mav(const shape_t &shp_) requires W
      : basic_mav(mav_construct(), shp_) {}
    basic_mav(const shape_t &shp_, uninitialized_dummy) requires W
      : basic_mav(mav_construct(), shp_, UNINITIALIZED) {}
    basic_mav(const shape_t &shp_, PAGE_IN page_in) requires W
      : basic_mav(mav_construct(), shp_, page_in) {}
    basic_mav(const shape_t &shp_, const stride_t &str_, uninitialized_dummy) requires W
      : basic_mav(mav_construct(), shp_, str_, UNINITIALIZED) {}
    basic_mav(const shape_t &shp_, const stride_t &str_, PAGE_IN page_in) requires W
      : basic_mav(mav_construct(), shp_, str_, page_in) {}
    // same memory as \a other, but with the given shape and strides
    basic_mav(const basic_mav &other, const shape_t &shp_, const stride_t &str_)
      : basic_mav(mav_construct(), other.buf, shp_, str_) {}
    /// Converts between fixed and dynamic rank: implicitly towards dynamic
    /// rank, explicitly (with a runtime check) towards fixed rank.
    template<size_t N2> requires ((N2!=N) && (is_dynamic || (N2==dynamic_rank)))
    explicit(!is_dynamic) basic_mav(const basic_mav<T, N2, W> &other)
      : basic_mav(mav_construct(), static_cast<const mav_storage<T, N2> &>(other)) {}

    basic_mav(const basic_mav &) = default;
    basic_mav(basic_mav &&) = default;
    // Deleted, because a cmav reference may refer to a vmav, which must
    // not end up pointing to read-only data. Use assign() instead.
    basic_mav &operator=(const basic_mav &) = delete;

    void assign(const basic_mav &other)
      {
      tinfo::assign(other);
      buf = other.buf;
      }
    void unassign()
      { assign(basic_mav()); }
    // Needed for template tricks.
    basic_mav<T, dynamic_rank, W> to_fmav() const
      { return basic_mav<T, dynamic_rank, W>(*this); }

    /// Returns a pointer to the first element of the array.
    element_type *data() const
      { return buf.data(); }
    /// Returns the element at memory offset \a i from data().
    template<typename I> element_type &raw(I i) const
      { return data()[i]; }

    /// Returns the data entry at the given set of indices.
    template<integral... Ns> element_type &operator()(Ns... ns) const
      { return raw(idx(ns...)); }
    element_type &operator()(const shape_t &ns) const
      { return raw(idx(ns)); }
    template<typename RAiter> element_type &val(RAiter beg, RAiter end) const
      { return raw(tinfo::idxval(beg, end)); }

    template<integral... Ns> void prefetch_r(Ns... ns) const
      { DUCC0_PREFETCH_R(&raw(idx(ns...))); }
    void prefetch_r(const shape_t &ns) const
      { DUCC0_PREFETCH_R(&raw(idx(ns))); }
    template<integral... Ns> void prefetch_w(Ns... ns) const requires W
      { DUCC0_PREFETCH_W(&raw(idx(ns...))); }
    void prefetch_w(const shape_t &ns) const requires W
      { DUCC0_PREFETCH_W(&raw(idx(ns))); }

    /// Returns a view of the sub-array described by \a slices, with rank
    /// \a nd2 (dynamic by default).
    template<size_t nd2=dynamic_rank> basic_mav<T, nd2, W> subarray(const vector<slice> &slices) const
      {
      auto [ninfo, nofs] = tinfo::template subdata<nd2>(slices);
      return basic_mav<T, nd2, W>(mav_construct(), ninfo, buf.data()+nofs, buf);
      }
    [[nodiscard]] basic_mav extend_and_broadcast(const shape_t &new_shape,
      const vector<size_t> &axpos) const requires is_dynamic
      { return basic_mav(mav_construct(), tinfo::extend_and_broadcast(new_shape, axpos), buf); }
    [[nodiscard]] basic_mav extend_and_broadcast(const shape_t &new_shape,
      size_t firstaxis) const requires is_dynamic
      { return basic_mav(mav_construct(), tinfo::extend_and_broadcast(new_shape, firstaxis), buf); }
    // For fixed rank, the result is read-only (as it always has been).
    template<size_t nd2> requires (!is_dynamic)
    [[nodiscard]] basic_mav<T, nd2, false> extend_and_broadcast(const array<size_t, nd2> &new_shape,
      const vector<size_t> &axpos) const
      {
      return basic_mav<T, nd2, false>(mav_construct(),
        tinfo::template extend_and_broadcast<nd2>(new_shape, axpos), buf);
      }
    template<size_t nd2> requires (!is_dynamic)
    [[nodiscard]] basic_mav<T, nd2, false> extend_and_broadcast(const array<size_t, nd2> &new_shape,
      size_t firstaxis) const
      {
      return basic_mav<T, nd2, false>(mav_construct(),
        tinfo::template extend_and_broadcast<nd2>(new_shape, firstaxis), buf);
      }

    /// Read-only array of the given shape with all entries equal to \a value.
    static basic_mav<T, N, false> build_uniform(const shape_t &shape, const T &value)
      {
      // one-element buffer, broadcast to the full shape via zero strides
      auto tshp = tinfo::zero_shape(shape.size());
      ranges::fill(tshp, 1);
      basic_mav<T, N, true> tmp(tshp);
      tmp(tinfo::zero_shape(shape.size())) = value;
      return basic_mav<T, N, false>(mav_construct(), tmp.buf, shape,
        tinfo::zero_stride(shape.size()));
      }
    static basic_mav build_empty() requires (!is_dynamic)
      { return basic_mav(static_cast<element_type *>(nullptr), shape_t{}); }

    /** Returns a writable array with the specified shape.
     *  The strides are chosen in such a way that critical strides (multiples
     *  of 4096 bytes) along any dimension are avoided, by enlarging the
     *  allocated memory slightly if necessary.
     *  \a init is forwarded to the constructor (e.g., UNINITIALIZED or
     *  PAGE_IN). Without it, the array data is default-initialized. */
    template<typename... Tinit>
    static basic_mav build_noncritical(const shape_t &shape, Tinit... init) requires W
      {
      auto shape2 = noncritical_shape(shape, sizeof(T));
      basic_mav tmp(shape2, init...);
      if (shape2==shape) return tmp;
      vector<slice> slc(shape.size());
      for (size_t i=0; i<shape.size(); ++i) slc[i] = slice(0, shape[i]);
      return tmp.template subarray<N>(slc);
      }

    [[nodiscard]] basic_mav transpose(const shape_t &axes) const
      { return basic_mav(mav_construct(), tinfo::transpose(axes), buf); }
    [[nodiscard]] basic_mav<T, tinfo::rank_plus_1, W> prepend_1() const
      { return basic_mav<T, tinfo::rank_plus_1, W>(mav_construct(), tinfo::prepend_1(), buf); }
    template<size_t N2> basic_mav<T, N2, W> reinterpret
      (const typename mav_info<N2>::shape_t &newshp,
       const typename mav_info<N2>::stride_t &newstr) const
      { return basic_mav<T, N2, W>(mav_construct(), buf, newshp, newstr); }
  };

template<typename T, size_t N> using cmav = basic_mav<T, N, false>;
template<typename T, size_t N> using vmav = basic_mav<T, N, true>;
template<typename T> using cfmav = cmav<T, dynamic_rank>;
template<typename T> using vfmav = vmav<T, dynamic_rank>;

template<size_t nd2=dynamic_rank, typename T, size_t N, bool W> basic_mav<T, nd2, W> subarray
  (const basic_mav<T, N, W> &arr, const vector<slice> &slices)
  { return arr.template subarray<nd2>(slices); }

// various operations involving fmav objects of the same shape -- experimental

DUCC0_NOINLINE tuple<fmav_info::shape_t, vector<fmav_info::stride_t>, size_t, size_t>
  multiprep(const vector<fmav_info> &info, const vector<size_t> &tsizes);
DUCC0_NOINLINE tuple<fmav_info::shape_t, vector<fmav_info::stride_t>>
  multiprep(const vector<fmav_info> &info);

template<typename... Ts, typename Func>
inline auto tuple_transform(tuple<Ts...> const& inputs, Func &&func)
  {
  return apply([&](auto &&... a)
    { return tuple<invoke_result_t<Func, Ts>...>{func(a)...}; }, inputs);
  }

template<typename Ttuple, typename Func>
inline void tuple_for_each(Ttuple &tpl, Func &&func)
  {
  apply([&](auto &... a){ (func(a), ...); }, tpl);
  }

template<typename Ttuple, typename Func>
inline void tuple_for_each_idx(Ttuple &tpl, Func &&func)
  {
  apply([&](auto &... a){ size_t i=0; (func(a, i++), ...); }, tpl);
  }

template<typename Ttuple> inline auto to_ref (const Ttuple &tuple)
  {
  return tuple_transform(tuple, [](auto &&ptr) -> auto & { return *ptr; });
  }


template<typename Ttuple> inline void advance_contiguous (Ttuple &ptrs)
  { tuple_for_each(ptrs, [](auto &&ptr) { ++ptr; }); }
template<typename Ttuple, size_t N> inline void advance (Ttuple &ptrs,
  const array<ptrdiff_t, N> &str)
  {
  tuple_for_each_idx(ptrs, [&str](auto &&ptr, size_t idx)
                     { ptr += str[idx]; });
  }
template<typename Ttuple> inline void advance_by_n (Ttuple &ptrs,
  const vector<vector<ptrdiff_t>> &str, size_t idim, size_t n)
  {
  tuple_for_each_idx(ptrs, [idim,n,&str](auto &&ptr, size_t idx)
                     { ptr += n*str[idx][idim]; });
  }
template<typename Ttuple> inline Ttuple update_pointers (Ttuple ptrs,
  const vector<vector<ptrdiff_t>> &str, size_t idim, size_t i)
  {
  advance_by_n(ptrs, str, idim, i);
  return ptrs;
  }

template<typename Ttuple> inline auto get_strides
  (const vector<vector<ptrdiff_t>> &str, size_t idim)
  {
  array<ptrdiff_t, tuple_size_v<Ttuple>> res;
  for (size_t i=0; i<res.size(); ++i)
    res[i] = str[i][idim];
  return res;
  }

// Multithreaded mav_apply (and its variants) distribute the work along axis 0
// by default. If axis 0 is too short for a balanced static distribution among
// the threads, the combined index range of as many leading axes as necessary is
// distributed instead (definitely excluding blocks). The busiest thread is at
// most a factor of (BALANCE_FACTOR+1)/BALANCE_FACTOR slower than a perfectly
// balanced distribution (12.5% for the value 8). Larger values balance better,
// but split the work into shorter pieces increasing per-piece overhead.
constexpr size_t BALANCE_FACTOR=8;
inline size_t parallel_naxes(const vector<size_t> &shp, size_t nthreads,
  size_t block0)
  {
  size_t nmax = shp.size() - ((block0!=0) ? 1 : 0);
  size_t naxes=1, nwork=shp[0];
  while ((naxes<nmax) && (nwork<BALANCE_FACTOR*nthreads))
    nwork *= shp[naxes++];
  return naxes;
  }
inline size_t leading_size(const vector<size_t> &shp, size_t naxes)
  {
  size_t res=1;
  for (size_t i=0; i<naxes; ++i)
    res *= shp[i];
  return res;
  }
// Counterpart to parallel_naxes(). Calls func() for all parts of the iteration
// space that belong to the work items [lo; hi) of the index range that has been
// combined by parallel_naxes.
template<typename Ttuple, typename Func>
  inline void for_each_work_chunk(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, const Ttuple &ptrs, size_t naxes,
    size_t lo, size_t hi, Func &&func)
  {
  size_t idim = naxes-1;
  auto locshp(shp);
  vector<size_t> pos(naxes);
  while (lo<hi)
    {
    auto locptrs(ptrs);
    for (size_t i=naxes, rest=lo; i>0; --i)
      {
      pos[i-1] = rest%shp[i-1];
      rest /= shp[i-1];
      advance_by_n(locptrs, str, i-1, pos[i-1]);
      }
    size_t n=min(hi-lo, shp[idim]-pos[idim]);
    locshp[idim] = n;
    func(idim, locshp, locptrs);
    lo += n;
    }
  }

template<typename ReduceType, typename Ttuple, typename Func>
  DUCC0_NOINLINE ReduceType applyReduceHelper_block(size_t idim,
    const vector<size_t> &shp, const vector<vector<ptrdiff_t>> &str,
    size_t bsi, size_t bsj, const Ttuple &ptrs, Func &&func)
  {
  ReduceType rt;
  auto leni=shp[idim], lenj=shp[idim+1];
  size_t nbi = (leni+bsi-1)/bsi;
  size_t nbj = (lenj+bsj-1)/bsj;
  const auto stri = get_strides<Ttuple>(str, idim),
             strj = get_strides<Ttuple>(str, idim+1);
  for (size_t bi=0; bi<nbi; ++bi)
    for (size_t bj=0; bj<nbj; ++bj)
      {
      auto locptrs(ptrs);
      advance_by_n(locptrs, str, idim, bi*bsi);
      advance_by_n(locptrs, str, idim+1, bj*bsj);
      for (size_t i=bi*bsi; i<min(leni, (bi+1)*bsi); ++i, advance(locptrs, stri))
        {
        auto locptrs2(locptrs);
        for (size_t j=bj*bsj; j<min(lenj, (bj+1)*bsj); ++j, advance(locptrs2, strj))
          rt.reduceWith(apply(func, to_ref(locptrs2)));
        }
      }
  return rt;
  }
template<typename ReduceType, typename Ttuple, typename Func>
  inline ReduceType applyReduceHelper_leaf(size_t idim, size_t len,
    const vector<vector<ptrdiff_t>> &str, const Ttuple &ptrs, Func &&func,
    bool last_contiguous)
  {
  ReduceType rt;
  auto locptrs(ptrs);
  if (last_contiguous)
    for (size_t i=0; i<len; ++i, advance_contiguous(locptrs))
      rt.reduceWith(apply(func, to_ref(locptrs)));
  else
    {
    const auto locstr = get_strides<Ttuple>(str, idim);
    for (size_t i=0; i<len; ++i, advance(locptrs, locstr))
      rt.reduceWith(apply(func, to_ref(locptrs)));
    }
  return rt;
  }
template<typename ReduceType, typename Ttuple, typename Func>
  DUCC0_NOINLINE ReduceType applyReduceHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, size_t block0, size_t block1,
    const Ttuple &ptrs, Func &&func, bool last_contiguous)
  {
  auto len = shp[idim];
  ReduceType rt;
  if ((idim+2==shp.size()) && (block0!=0))  // we should do blocking
    rt.reduceWith(applyReduceHelper_block<ReduceType>(idim, shp, str,
      block0, block1, ptrs, func));
  else if (idim+2==shp.size())  // avoid a function call for every short row
    {
    auto locptrs(ptrs);
    const auto locstr = get_strides<Ttuple>(str, idim);
    for (size_t i=0; i<len; ++i, advance(locptrs, locstr))
      rt.reduceWith(applyReduceHelper_leaf<ReduceType>(idim+1, shp[idim+1],
        str, locptrs, func, last_contiguous));
    }
  else if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      rt.reduceWith(applyReduceHelper<ReduceType>(idim+1, shp, str, block0,
        block1, update_pointers(ptrs, str, idim, i), func, last_contiguous));
  else
    return applyReduceHelper_leaf<ReduceType>(idim, len, str, ptrs, func,
      last_contiguous);
  return rt;
  }
template<typename ReduceType, typename Func, typename Ttuple>
  inline ReduceType applyReduceHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, size_t block0, size_t block1,
    const Ttuple &ptrs, Func &&func, size_t nthreads, bool last_contiguous)
  {
  ReduceType rt;
  if (shp.size()==0)
    rt.reduceWith(apply(std::forward<Func>(func), to_ref(ptrs)));
  else if (nthreads==1)
    rt.reduceWith(applyReduceHelper<ReduceType>(0, shp, str, block0, block1,
      ptrs, std::forward<Func>(func), last_contiguous));
  else
    {
    Mutex mut;
    size_t naxes = parallel_naxes(shp, adjust_nthreads(nthreads), block0);
    execParallel(leading_size(shp, naxes), nthreads, [&](size_t lo, size_t hi)
      {
      ReduceType local_rt;
      for_each_work_chunk(shp, str, ptrs, naxes, lo, hi,
        [&](size_t idim, const vector<size_t> &locshp, const Ttuple &locptrs)
        {
        local_rt.reduceWith(applyReduceHelper<ReduceType>(idim, locshp, str,
          block0, block1, locptrs, func, last_contiguous));
        });
      {
      LockGuard lock(mut);
      rt.reduceWith(local_rt);
      }
      });
    }
  return rt;
  }

template<typename ReduceType, typename Func, typename... Targs>
  ReduceType mav_apply_reduce(Func &&func, size_t nthreads, Targs... args)
  {
  auto [shp, str, block0, block1] = multiprep({args...},
    {sizeof(typename Targs::value_type)...});
  bool last_contiguous = true;
  if (shp.size()>0)
    for (const auto &s:str)
      last_contiguous &= (s.back()==1);

  return applyReduceHelper<ReduceType>(shp, str, block0, block1,
    make_tuple(args.data()...), std::forward<Func>(func), nthreads,
    last_contiguous);
  }

struct NoReduction { void reduceWith(NoReduction) {} };
template<typename Func, typename... Targs>
  void mav_apply(Func &&func, size_t nthreads, Targs... args)
  {
  mav_apply_reduce<NoReduction>([&func](auto &... refs)
    { func(refs...); return NoReduction(); }, nthreads, std::move(args)...);
  }


template<typename T, size_t ndim> class mavref
  {
  private:
    const mav_info<ndim> &info;
    T *d;

  public:
    using shape_t = typename mav_info<ndim>::shape_t;
    using stride_t = typename mav_info<ndim>::stride_t;
    using element_type = T;
    using value_type = remove_cv_t<T>;
    mavref(const mav_info<ndim> &info_, T *d_) : info(info_), d(d_) {}
    template<integral... Ns> T &operator()(Ns... ns) const
      { return d[info.idx(ns...)]; }
    /// Returns the total number of entries in the object.
    size_t size() const { return info.size(); }
    /// Returns the shape of the object.
    const shape_t &shape() const { return info.shape(); }
    /// Returns the length along dimension \a i.
    size_t shape(size_t i) const { return info.shape(i); }
    /// Returns the strides of the object.
    const stride_t &stride() const { return info.stride(); }
    /// Returns the stride along dimension \a i.
    ptrdiff_t stride(size_t i) const { return info.stride(i); }
    /// Returns true iff the last dimension has stride 1.
    /**  Typically used for optimization purposes. */
    bool last_contiguous() const
      { return info.last_contiguous(); }
    /** Returns true iff the object is C-contiguous, i.e. if the stride of the
     *  last dimension is 1, the stride for the next-to-last dimension is the
     *  shape of the last dimension etc. */
    bool contiguous() const
      { return info.contiguous(); }
    /// Returns true iff this->shape and \a other.shape match.
    bool conformable(const mavref &other) const
      { return shape()==other.shape(); }
  };

template<typename T, size_t ndim>
  mavref<T, ndim> make_mavref(const mav_info<ndim> &info_, T *d_)
  { return mavref<T, ndim>(info_, d_); }

template<typename Tptrs, typename Tinfos>
  auto make_mavrefs(const Tptrs &ptrs, const Tinfos &infos)
  {
  return [&]<size_t... Is>(index_sequence<Is...>)
    { return make_tuple(make_mavref(get<Is>(infos), get<Is>(ptrs))...); }
    (make_index_sequence<tuple_size_v<Tptrs>>{});
  }

template<size_t ndim> auto make_infos(const fmav_info &info)
  {
  if constexpr(ndim>0)
    MR_assert(ndim<=info.ndim(), "bad dimensionality");
  auto iterdim = info.ndim()-ndim;
  fmav_info fout({info.shape().begin(),info.shape().begin()+iterdim},
                 {info.stride().begin(),info.stride().begin()+iterdim});

  typename mav_info<ndim>::shape_t shp;
  typename mav_info<ndim>::stride_t str;
  if constexpr (ndim>0)  // just to silence compiler warnings
    for (size_t i=0; i<ndim; ++i)
      {
      shp[i] = info.shape(iterdim+i);
      str[i] = info.stride(iterdim+i);
      }
  mav_info<ndim> iout(shp, str);
  return make_tuple(fout, iout);
  }

template<typename Tptrs, typename Tinfos, typename Func>
  DUCC0_NOINLINE void flexible_mav_applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, const Tptrs &ptrs,
    const Tinfos &infos, Func &&func)
  {
  auto len = shp[idim];
  auto locptrs(ptrs);
  const auto locstr = get_strides<Tptrs>(str, idim);
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i, advance(locptrs, locstr))
      flexible_mav_applyHelper(idim+1, shp, str, locptrs, infos, func);
  else
    for (size_t i=0; i<len; ++i, advance(locptrs, locstr))
      apply(func, make_mavrefs(locptrs, infos));
  }
template<typename Tptrs, typename Tinfos, typename Func>
  DUCC0_NOINLINE void flexible_mav_applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, const Tptrs &ptrs,
    const Tinfos &infos, Func &&func, size_t nthreads)
  {
  if (shp.size()==0)
    apply(func, make_mavrefs(ptrs, infos));
  else if (nthreads==1)
    flexible_mav_applyHelper(0, shp, str, ptrs, infos, std::forward<Func>(func));
  else
    {
    size_t naxes = parallel_naxes(shp, adjust_nthreads(nthreads), 0);
    execParallel(leading_size(shp, naxes), nthreads, [&](size_t lo, size_t hi)
      {
      for_each_work_chunk(shp, str, ptrs, naxes, lo, hi,
        [&](size_t idim, const vector<size_t> &locshp, const Tptrs &locptrs)
        { flexible_mav_applyHelper(idim, locshp, str, locptrs, infos, func); });
      });
    }
  }

template<size_t... nds, typename Func, typename... Targs>
  void flexible_mav_apply(Func &&func, size_t nthreads, const Targs &... args)
  {
  static_assert(sizeof...(nds)==sizeof...(Targs),
    "need exactly one dimensionality per array argument");
  auto fullinfos = make_tuple(make_infos<nds>(fmav_info(args))...);
  auto [shp, str] = apply([](const auto &... fi)
    { return multiprep({get<0>(fi)...}); }, fullinfos);
  auto infos = tuple_transform(fullinfos, [](const auto &fi)
    { return get<1>(fi); });
  flexible_mav_applyHelper(shp, str, make_tuple(args.data()...), infos,
    std::forward<Func>(func), nthreads);
  }

}

using detail_mav::UNINITIALIZED;
using detail_mav::PAGE_IN;
using detail_mav::page_in_memory;
using detail_mav::fmav_info;
using detail_mav::mav_info;
using detail_mav::slice;
using detail_mav::MAXIDX;
using detail_mav::dynamic_rank;
using detail_mav::cfmav;
using detail_mav::vfmav;
using detail_mav::basic_mav;
using detail_mav::cmav;
using detail_mav::vmav;
using detail_mav::subarray;
using detail_mav::mav_apply;
using detail_mav::mav_apply_reduce;
using detail_mav::flexible_mav_apply;
}

#endif
