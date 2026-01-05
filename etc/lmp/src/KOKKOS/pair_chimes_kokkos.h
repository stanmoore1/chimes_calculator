/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(chimesFF/kk,PairCHIMESKokkos<LMPDeviceType>);
PairStyle(chimesFF/kk/device,PairCHIMESKokkos<LMPDeviceType>);
PairStyle(chimesFF/kk/host,PairCHIMESKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_CHIMES_KOKKOS_H
#define LMP_PAIR_CHIMES_KOKKOS_H

#include "chimesFF.h"
#include "pair_chimes.h"
#include "kokkos_base.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairCHIMESKokkos : public PairCHIMES
{
 public:
  chimesFF chimes_calculator;   // chimesFF instance

  struct TagPairCHIMESComputeNeigh{};

  template<int NEIGHFLAG, int EVFLAG>
  struct TagPairCHIMESComputeForce{};

  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairCHIMESKokkos(class LAMMPS *);
  ~PairCHIMESKokkos() override;
  void init_style() override;
  void allocate() override;
  void compute(int eflag, int vflag) override;
  void build_mb_neighlists() override;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESComputeNeigh,const typename Kokkos::TeamPolicy<DeviceType, TagPairCHIMESComputeNeigh>::member_type& team) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairCHIMESComputeForce<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT get_dist(int i, int j, KK_FLOAT* dr) const;

  KOKKOS_INLINE_FUNCTION
  KK_FLOAT get_dist(int i, int j) const;

 private:
  int neighflag;
  int inum, maxneigh, chunk_size, chunk_offset;
  int host_flag;

  int eflag, vflag;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d_randomread type;

  typedef Kokkos::DualView<KK_FLOAT**, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq, k_scale;
  typedef Kokkos::View<KK_FLOAT**, DeviceType> t_fparams;
  t_fparams d_cutsq, d_scale;

  typename AT::t_int_1d d_chimes_type;

  int need_dup;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout> dup_f;
  DupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> dup_vatom;

  NonDupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout> ndup_f;
  NonDupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairCHIMESKokkos>(PairCHIMESKokkos*);

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const KK_FLOAT &fx, const KK_FLOAT &fy, const KK_FLOAT &fz,
      const KK_FLOAT &delx, const KK_FLOAT &dely, const KK_FLOAT &delz) const;

};
}    // namespace LAMMPS_NS

#endif
#endif
