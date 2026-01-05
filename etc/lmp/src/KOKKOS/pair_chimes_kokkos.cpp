// clang-format off
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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mpi.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "force.h"
#include "kokkos.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "my_page.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include "pair_chimes_kokkos.h"
#include "group.h"
#include "update.h" // Needed for mb neighlist updates and info printing for fitting
#include "output.h" // Needed for infor printing for fitting -- dump 1 must be the "main" dump file used for fitting
#include "utils.h"  // Needed for infor printing for fitting
#include <vector>
#include <iostream>
#include <sstream>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairCHIMESKokkos<DeviceType>::PairCHIMESKokkos(LAMMPS *lmp) : PairCHIMES(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

#ifdef TABULATION
 error->all(FLERR,"Cannot (yet) use tabulation with pair_style chimes/kk");
#endif

#ifdef FINGERPRINT
 error->all(FLERR,"Cannot (yet) use fingerprint with pair_style chimes/kk");
#endif
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairCHIMESKokkos<DeviceType>::~PairCHIMESKokkos()
{
  if (copymode) return;

  /*if (allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }*/
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::allocate()
{
  PairCHIMES::allocate();

  int n = atom->ntypes + 1;
  MemKK::realloc_kokkos(d_map, "chimes:map", n);

  MemKK::realloc_kokkos(k_cutsq, "chimes:cutsq", n, n);
  d_cutsq = k_cutsq.template view<DeviceType>();

  MemKK::realloc_kokkos(k_scale, "chimes:scale", n, n);
  d_scale = k_scale.template view<DeviceType>();

  allocated = 1;

  memory->create(setflag,atom->ntypes+1,atom->ntypes+1,"pair:setflag");

  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,atom->ntypes+1,atom->ntypes+1,"pair:cutsq");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::init_style()
{
  PairCHIMES::init_style();

  // adjust neighbor list request for KOKKOS

  auto request = neighbor->find_request(this);

  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);

  neighflag = lmp->kokkos->neighflag;

  if (neighflag == FULL)
    error->all(FLERR,"Must use half neighbor list style with pair chimes/kk");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairCHIMESKokkos<DeviceType>::get_dist(int i, int j, KK_FLOAT *dr) const
{
  dr[0] = x(j,0) - x(i,0);
  dr[1] = x(j,1) - x(i,1);
  dr[2] = x(j,2) - x(i,2);

  return sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairCHIMESKokkos<DeviceType>::get_dist(int i, int j) const
{
  KK_FLOAT dummy_dr[3];

  return get_dist(i,j, dummy_dr);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::build_mb_neighlists()
{
  if ((chimes_calculatorKK.poly_orders[1] == 0) &&  (chimes_calculatorKK.poly_orders[2] == 0))
    return;

  // List gets built based on atoms owned by calling proc.

  neighborlist_3mers.clear();
  neighborlist_4mers.clear();

  int i,j,k,l,inum,jnum,knum,lnum, ii, jj, kk, ll;             // Local iterator vars
  int *ilist,*jlist,*klist,*llist, *numneigh,**firstneigh; // Local neighborlist vars
  tagint *tag = atom->tag;                                       // Access to global atom indices
  int itag, jtag, ktag, ltag;                                       // holds tags
  double **x = atom->x;                                           // Access to system coordinates

  KK_FLOAT maxcut_3b_padded = maxcut_3b + neighbor->skin;
  KK_FLOAT maxcut_4b_padded = maxcut_4b + neighbor->skin;

  KK_FLOAT dist_ij, dist_ik, dist_il, dist_jk, dist_jl, dist_kl;

  inum = list->inum;              // length of the list
  ilist = list->ilist;            // list of i atoms for which neighbor list exists
  numneigh = list->numneigh;      // length of each of the ilist neighbor lists
  firstneigh = list->firstneigh;  // point to the list of neighbors of i

  Kokkos::parallel_for("ComputeNeigh",policy_neigh,*this);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESComputeNeigh,const t
{
  const int i = d_ilist[ii];
  const tagint itag = tag[i];
  const int jnum = d_numneigh[i];

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i,jj);
    j &= NEIGHMASK;
    jtag = tag[j];

    if (j == i) continue;

    if (jtag < itag) continue;

    // Check ij distance

    const KK_FLOAT dist_ij = get_dist(i,j);

    if ((dist_ij >= maxcut_3b_padded) && (dist_ij >= maxcut_4b_padded))
      continue;

    // ChIMES assumes all atoms must be within cutoff of each other for a valid interaction
    const int knum = d_numneigh[i];

    for (kk = 0; kk < knum; kk++) {
      k = d_neighbors(i,kk);
      k &= NEIGHMASK;
      ktag = tag[k];

      if ((k == i) || (k == j)) continue;

      if ((ktag < itag) || (ktag < jtag)) continue;

      // Check ik distance

      const KK_FLOAT dist_ik = get_dist(i,k);

      if ((dist_ik >= maxcut_3b_padded) && (dist_ik >= maxcut_4b_padded))
        continue;

      // Check jk distance

      const KK_FLOAT dist_jk = get_dist(j,k);

      if ((dist_ij < maxcut_3b_padded) && (dist_ik < maxcut_3b_padded) && (dist_jk < maxcut_3b_padded))
      {
        // If we're here and valid_3mer == true, then add the triplet to the chimes neigh list

        neighborlist_3mers(ii3,0) = i;
        neighborlist_3mers(ii3,1) = j;
        neighborlist_3mers(ii3,2) = k;
        ii3++; //// need atomic_fetch_add
      }

      if ((dist_ij >= maxcut_4b_padded) || (dist_ik >= maxcut_4b_padded) || (dist_jk >= maxcut_4b_padded))
        continue;

      // Now decide if we should continue on to 4-body neighbor list construction

      if (chimes_calculatorKK.poly_orders[2] == 0)
        continue;

      llist = firstneigh[i];
      lnum = numneigh[i];

      for (ll = 0; ll < lnum; ll++)
      {
        l = llist[ll];
        ltag = tag[l];
        l &= NEIGHMASK;

        if ((l == i) || (l == j) || (l == k)) continue;

        if ((ltag < itag) || (ltag < jtag) || (ltag < ktag))
          continue;

        // Check il distance

        const KK_FLOAT dist_il = get_dist(i,l);

        if (dist_il >= maxcut_4b_padded)
          continue;

        // Check jl distance

        const KK_FLOAT dist_jl = get_dist(j,l);

        if (dist_jl >= maxcut_4b_padded)
          continue;

        // Check kl distance

        const KK_FLOAT dist_kl = get_dist(k,l);

        if (dist_kl >= maxcut_4b_padded)
          continue;

        // If we're here and valid_4mer == true, then add the quadruplet to the chimes neigh list

        d_neighborlist_4mers(ii4,0) = i;
        d_neighborlist_4mers(ii4,1) = j;
        d_neighborlist_4mers(ii4,2) = k;
        d_neighborlist_4mers(ii4,3) = l;
        ii4++; //// need atomic_fetch_add
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType>* nl): k_list(*nl) {}
  ~FindMaxNumNeighs() {k_list.copymode = 1;}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii, int& maxneigh) const {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (maxneigh < num_neighs) maxneigh = num_neighs;
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCHIMESKokkos<DeviceType>::compute(int eflag, int vflag)
{
  // Vars for access to chimesFF compute_XB functions

  KK_FLOAT stensor[6];      // pointers to system stress tensor

  // Temp vars to hold chimes output for passing to ev_tally function

  KK_FLOAT fscalar[6];
  KK_FLOAT tmp_dist[3];
  KK_FLOAT tmp_dr[6];
  int atmidxlst[6][2];

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;

  // Set up vars controlling if energy/pressure (virial) contributions are computed

  if (eflag || vflag)
  {
    ev_setup(eflag,vflag);
  } else {
    evflag = 0;
    vflag_fdotr = 0;
    vflag_atom = 0;
  }

  ////////////////////////////////////////
  // Access to (2-body) neighbor list vars
  ////////////////////////////////////////

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }

  chimes2BTmpKK chimes_2btmp(chimes_calculatorKK.poly_orders[0]);
  chimes3BTmpKK chimes_3btmp(chimes_calculatorKK.poly_orders[1]);
  chimes4BTmpKK chimes_4btmp(chimes_calculatorKK.poly_orders[2]);

  // Build the ChIMES many-body neighbor lists.. only do so when LAMMPS neighborlist has been updated

  if (neighbor->ago == 0)
  {
    if (chimes_calculatorKK.rank == 0)
      std::cout << "Updating chimesFF neighbor lists..." << std::endl;

    build_mb_neighlists();
    if (chimes_calculatorKK.rank == 0)
    {
      std::cout << "      Rank " << me << " 3-body list size: " << neighborlist_3mers.size() << std::endl;
      std::cout << "      Rank " << me << " 4-body list size: " << neighborlist_4mers.size() << std::endl;
      std::cout << "      ...update complete" << std::endl;
    }
  }

  // Prepare the badness variable

  chimes_calculatorKK.reset_badness();

  maxneigh = 0;
  Kokkos::parallel_reduce("chimes::find_maxneigh", inum, FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(maxneigh));

  int vector_length_default = 1;
  int team_size_default = 1;
  if (!host_flag)
    team_size_default = 32;

  chunk_size = MIN(chunksize,inum); // "chunksize" variable is set by user
  chunk_offset = 0;

  grow(chunk_size, maxneigh);

  EV_FLOAT ev;

  while (chunk_offset < inum) { // chunk up loop to prevent running out of memory

    //Compute2Body
    {
      if (evflag) {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALF,1> > policy_force(0,chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALFTHREAD,1> > policy_force(0,chunk_size);
          Kokkos::parallel_reduce("Compute2Body",policy_force, *this, ev_tmp);
        }
      } else {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALF,0> > policy_force(0,chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute2Body<HALFTHREAD,0> > policy_force(0,chunk_size);
          Kokkos::parallel_for("Compute2Body",policy_force, *this);
        }
      }
    }
    ev += ev_tmp;

    // Document badness for configuration: current timestep, current rank, worst badness seen by rank

    if (for_fitting)
    if (update->ntimestep % output->every_dump[0] == 0)
      badness_stream << update->ntimestep << " " <<  chimes_calculatorKK.get_badness() << endl;

    //Compute3Body
    // if (chimes_calculatorKK.poly_orders[1] > 0 || tmp_FP)
    if (chimes_calculatorKK.poly_orders[1] > 0)
    {
      for (ii = 0; ii < neighborlist_3mers.size(); ii++)
      if (evflag) {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute3Body<HALF,1> > policy_force(0,chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute3Body<HALFTHREAD,1> > policy_force(0,chunk_size);
          Kokkos::parallel_reduce("Compute3Body",policy_force, *this, ev_tmp);
        }
      } else {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute3Body<HALF,0> > policy_force(0,chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute3Body<HALFTHREAD,0> > policy_force(0,chunk_size);
          Kokkos::parallel_for("Compute3Body",policy_force, *this);
        }
      }
    }
    ev += ev_tmp;

    //Compute4Body
    // if (chimes_calculatorKK.poly_orders[2] > 0 || tmp_FP)
    if (chimes_calculatorKK.poly_orders[2] > 0)
    {
      for (ii = 0; ii < neighborlist_4mers.size(); ii++)
      if (evflag) {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute4Body<HALF,1> > pol
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute4Body<HALFTHREAD,1>
          Kokkos::parallel_reduce("Compute4Body",policy_force, *this, ev_tmp);
        }
      } else {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute4Body<HALF,0> > pol
          Kokkos::parallel_for(policy_force, *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairCHIMESCompute4Body<HALFTHREAD,0>
          Kokkos::parallel_for("Compute4Body",policy_force, *this);
        }
      }
    }
    ev += ev_tmp;

    chunk_offset += chunk_size;
  } // end while

  if (need_dup)
    Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  atomKK->modified(execution_space,F_MASK);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f     = {};
    dup_vatom = {};
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 1- and 2-body interactions
  ////////////////////////////////////////

  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const int i = d_ilist[ii + chunk_offset];
  const int itype = type(i);
  const tagint itag = tag(i);
  const KK_FLOAT scale = d_scale(itype,itype);

  const int ncount = d_ncount(ii);

  KK_ACC_FLOAT fitmp[3] = {0.0,0.0,0.0};
  for (int jj = 0; jj < ncount; jj++) {
    int j = d_nearest(ii,jj);

    // First, get the single-atom energy contribution

    energy = 0.0;

    chimes_calculatorKK.compute_1B(type[i]-1, energy);

    atmidxlst[0][0] = i;

    if (evflag)
      ev_tally_mb(1, 0, atmidxlst, energy, stensor);

    // Now move on to two-body force, stress, and energy

    KK_ACC_FLOAT fitmp[3] = {0.0,0.0,0.0};

    for (int jj = 0; jj < ncount; jj++) {
      int j = d_nearest(ii,jj);

      jtag = tag[j]; // Get j's global atom index (sort of like its "parent")
      j &= NEIGHMASK; // Strip possible extra bits of j

      if (jtag <= itag) // only allow calculation for j<i, since we've requested a full neighbor list
        continue;

      // Get distance using ghost atoms... don't need MIC since we're using ghost atoms

      const KK_FLOAT dist = get_dist(i,j,&dr[0]);

      typ_idxs_2b[0] = d_chimes_type[type[i]-1]; // Type (index) of the current atom... subtract 1 to account for chimesFF vs LAMMPS numbering convention
      typ_idxs_2b[1] = d_chimes_type[type[j]-1];

      // Using std::fill for maximum efficiency.
      //std::fill(force_2b.begin(), force_2b.end(), 0.0);

      // Do the same for stress tensors
      //std::fill(stensor.begin(), stensor.end(), 0.0);

      energy = 0.0;

      chimes_calculatorKK.compute_2B(dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp);      // Auto-updates badness
      
      for (idx = 0; idx < 3; idx++) {
        a_f(i,idx) += force_2b[0*CHDIM+idx];
        a_f(j,idx) += force_2b[1*CHDIM+idx];
      }

      // "Save"/tally up the energy and stresses to the global virial/energy data objects (see pair.cpp ~ line 1000)
      // Compute pressure, (in contrast to chimes_md) AFTER penalty has been added

      if (vflag_atom)
      {
        atmidxlst[0][0] = i;
        atmidxlst[0][1] = j;
      }
      tmp_dist[0] = dist;

      if (evflag)
        ev_tally_mb(2, 1, atmidxlst, energy, stensor);
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute2Body<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 3-body interactions
  ////////////////////////////////////////

  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  i = d_neighborlist_3mers(ii,0);
  j = d_neighborlist_3mers(ii,1);
  k = d_neighborlist_3mers(ii,2);

  KK_FLOAT dist_3b[3];
  dist_3b[0] = get_dist(i,j,&dr_3b[0*CHDIM]);
  dist_3b[1] = get_dist(i,k,&dr_3b[1*CHDIM]);
  dist_3b[2] = get_dist(j,k,&dr_3b[2*CHDIM]);

  int typ_idxs_3b[3];
  typ_idxs_3b[0] = d_chimes_type[type[i]-1];
  typ_idxs_3b[1] = d_chimes_type[type[j]-1];
  typ_idxs_3b[2] = d_chimes_type[type[k]-1];

  //std::fill(force_3b.begin(), force_3b.end(), 0.0);
  //std::fill(stensor.begin(), stensor.end(), 0.0);

  //energy = 0.0;

  chimes_calculatorKK.compute_3B(dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);

  for (idx = 0; idx < 3; idx++)
  {
    a_f(i,idx) += force_3b[0*CHDIM+idx];
    a_f(j,idx) += force_3b[1*CHDIM+idx];
    a_f(k,idx) += force_3b[2*CHDIM+idx];
  }

  if (vflag_atom)
  {
    atmidxlst[0][0] = i;
    atmidxlst[0][1] = j;
    atmidxlst[1][0] = i;
    atmidxlst[1][1] = k;
    atmidxlst[2][0] = j;
    atmidxlst[2][1] = k;
  }

  if (evflag)
    ev_tally_mb(3, 3, atmidxlst, energy, stensor);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute3Body<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  ////////////////////////////////////////
  // Compute 4-body interactions
  ////////////////////////////////////////

  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  i = d_neighborlist_4mers(ii,0);
  j = d_neighborlist_4mers(ii,1);
  k = d_neighborlist_4mers(ii,2);
  l = d_neighborlist_4mers(ii,3);

  KK_FLOAT dist_4b[6];
  dist_4b[0] = get_dist(i,j,&dr_4b[0*CHDIM]);
  dist_4b[1] = get_dist(i,k,&dr_4b[1*CHDIM]);
  dist_4b[2] = get_dist(i,l,&dr_4b[2*CHDIM]);
  dist_4b[3] = get_dist(j,k,&dr_4b[3*CHDIM]);
  dist_4b[4] = get_dist(j,l,&dr_4b[4*CHDIM]);
  dist_4b[5] = get_dist(k,l,&dr_4b[5*CHDIM]);

  int typ_idxs_4b[4];
  typ_idxs_4b[0] = d_chimes_type[type[i]-1];
  typ_idxs_4b[1] = d_chimes_type[type[j]-1];
  typ_idxs_4b[2] = d_chimes_type[type[k]-1];
  typ_idxs_4b[3] = d_chimes_type[type[l]-1];

  //std::fill(force_4b.begin(), force_4b.end(), 0.0);
  //std::fill(stensor.begin(), stensor.end(), 0.0);

  //energy = 0.0;

  chimes_calculatorKK.compute_4B(dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp);

  for (idx = 0; idx < 3; idx++) {
    a_f(i,idx) += force_4b[0*CHDIM+idx];
    a_f(j,idx) += force_4b[1*CHDIM+idx];
    a_f(k,idx) += force_4b[2*CHDIM+idx];
    a_f(l,idx) += force_4b[3*CHDIM+idx];
  }

  if (vflag_atom) {
    atmidxlst[0][0] = i;
    atmidxlst[0][1] = j;
    atmidxlst[1][0] = i;
    atmidxlst[1][1] = k;
    atmidxlst[2][0] = i;
    atmidxlst[2][1] = l;
    atmidxlst[3][0] = j;
    atmidxlst[3][1] = k;
    atmidxlst[4][0] = j;
    atmidxlst[4][1] = l;
    atmidxlst[5][0] = k;
    atmidxlst[5][1] = l;
  }

  if (evflag)
    ev_tally_mb(4, 6, atmidxlst, energy, stensor);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::operator() (TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairCHIMESCompute4Body<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairCHIMESKokkos<DeviceType>::v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const KK_FLOAT &fx, const KK_FLOAT &fy, const KK_FLOAT &fz,
      const KK_FLOAT &delx, const KK_FLOAT &dely, const KK_FLOAT &delz) const
{
  // The vatom array is duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const KK_FLOAT v0 = delx*fx;
  const KK_FLOAT v1 = dely*fy;
  const KK_FLOAT v2 = delz*fz;
  const KK_FLOAT v3 = delx*fy;
  const KK_FLOAT v4 = delx*fz;
  const KK_FLOAT v5 = dely*fz;

  if (vflag_global) {
    ev.v[0] += v0;
    ev.v[1] += v1;
    ev.v[2] += v2;
    ev.v[3] += v3;
    ev.v[4] += v4;
    ev.v[5] += v5;
  }

  if (vflag_atom) {
    a_vatom(i,0) += 0.5*v0;
    a_vatom(i,1) += 0.5*v1;
    a_vatom(i,2) += 0.5*v2;
    a_vatom(i,3) += 0.5*v3;
    a_vatom(i,4) += 0.5*v4;
    a_vatom(i,5) += 0.5*v5;
    a_vatom(j,0) += 0.5*v0;
    a_vatom(j,1) += 0.5*v1;
    a_vatom(j,2) += 0.5*v2;
    a_vatom(j,3) += 0.5*v3;
    a_vatom(j,4) += 0.5*v4;
    a_vatom(j,5) += 0.5*v5;
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairCHIMESKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairCHIMESKokkos<LMPHostType>;
#endif
}
