/*
    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
    Contributing Author: Stan Moore (2025)
*/

#ifndef _chimesFF_KOKKOS_h
#define _chimesFF_KOKKOS_h

#include "chimesFF.h"
#include "kokkos_type.h"
#include "memory_kokkos.h"

#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<algorithm>
#include<cmath>
#include<map>

#define pi 3.14159265359

using namespace std;

// Notes:
//
// 1. A Morse-style coordinate transformation is hard-coded (see set_cheby_polys)
// 2. Polynomials are hard-coded over the domain [-1,1]
// 3. A cubic style cutoff is assumed, and Tersoff is the only other style considered (see get_fcut)


#define CHDIM 3 // The number of spatial dimensions.
#define USE_DISTANCE_TENSOR 1 // Use tensor of distances in computing stresses.

enum class fcutType
{
  CUBIC,
  TERSOFF,
};

template<class DeviceType>
class chimesFFKokkos : public chimesFF
{
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  // Temporary storage for ChIMES interaction

  // Two-body

  struct chimes2BTmp {
    typename AT::t_kkfloat_1d d_Tn, d_Tnd;

    chimes2BTmp(int poly_order) {
      memKK::realloc_kokkos(d_Tn,"chimes:Tn",poly_order+1);
      memKK::realloc_kokkos(d_Tnd,"chimes:Tnd",poly_order+1);
    }

    void resize(int poly_order)
    {
      if (d_Tn.extent(0) < poly_order + 1) {
        memKK::realloc_kokkos(d_Tn,"chimes:Tn",poly_order+1);
        memKK::realloc_kokkos(d_Tnd,"chimes:Tn",poly_order+1);
      }
    }
  };


  // Three-body

  struct chimes3BTmp
  {
    typename AT::t_kkfloat_1d d_Tn_ij, d_Tn_ik, d_Tn_jk;   // The Chebyshev polymonials
    typename AT::t_kkfloat_1d d_Tnd_ij, d_Tnd_ik, d_Tnd_jk;  // The Chebyshev polymonial derivatives

    chimes3BTmp(int poly_order) {
      memKK::realloc_kokkos(d_Tn_ij,"chimes:Tn_ij",poly_order+1);
      memKK::realloc_kokkos(d_Tn_ik,"chimes:Tn_ik",poly_order+1);
      memKK::realloc_kokkos(d_Tn_jk,"chimes:Tn_jk",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_ij,"chimes:Tnd_ij",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_ik,"chimes:Tnd_ik",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_jk,"chimes:Tnd_jk",poly_order+1);
    }
    
    void resize(int poly_order)
    { 
      if (d_Tn_ij.extent(0) < poly_order + 1) {
        memKK::realloc_kokkos(d_Tn_ij,"chimes:Tn_ij",poly_order+1);
        memKK::realloc_kokkos(d_Tn_ik,"chimes:Tn_ik",poly_order+1);
        memKK::realloc_kokkos(d_Tn_jk,"chimes:Tn_jk",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_ij,"chimes:Tnd_ij",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_ik,"chimes:Tnd_ik",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_jk,"chimes:Tnd_jk",poly_order+1);
      }
    }
  };

  // Four-body

  struct chimes4BTmp {
    typename AT::t_kkfloat_1d d_Tn_ij, d_Tn_ik, d_Tn_il, d_Tn_jk, d_Tn_jl, d_Tn_kl;   // The Chebyshev polymonials
    typename AT::t_kkfloat_1d d_Tnd_ij, d_Tnd_ik, d_Tnd_il, d_Tnd_jk, d_Tnd_jl, d_Tnd_kl;  // The Chebyshev polymonial derivatives

    chimes4BTmp(int poly_order) {
      memKK::realloc_kokkos(d_Tn_ij,"chimes:Tn_ij",poly_order+1);
      memKK::realloc_kokkos(d_Tn_ik,"chimes:Tn_ik",poly_order+1);
      memKK::realloc_kokkos(d_Tn_il,"chimes:Tn_il",poly_order+1);
      memKK::realloc_kokkos(d_Tn_jk,"chimes:Tn_jk",poly_order+1);
      memKK::realloc_kokkos(d_Tn_jl,"chimes:Tn_jl",poly_order+1);
      memKK::realloc_kokkos(d_Tn_kl,"chimes:Tn_kl",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_ij,"chimes:Tnd_ij",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_ik,"chimes:Tnd_ik",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_il,"chimes:Tnd_il",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_jk,"chimes:Tnd_jk",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_jl,"chimes:Tnd_jl",poly_order+1);
      memKK::realloc_kokkos(d_Tnd_kl,"chimes:Tnd_kl",poly_order+1);
    }

    void resize(int poly_order)
    {
      if (d_Tn_ij.extent(0) < poly_order + 1) {
        memKK::realloc_kokkos(d_Tn_ij,"chimes:Tn_ij",poly_order+1);
        memKK::realloc_kokkos(d_Tn_ik,"chimes:Tn_ik",poly_order+1);
        memKK::realloc_kokkos(d_Tn_il,"chimes:Tn_il",poly_order+1);
        memKK::realloc_kokkos(d_Tn_jk,"chimes:Tn_jk",poly_order+1);
        memKK::realloc_kokkos(d_Tn_jl,"chimes:Tn_jl",poly_order+1);
        memKK::realloc_kokkos(d_Tn_kl,"chimes:Tn_kl",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_ij,"chimes:Tnd_ij",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_ik,"chimes:Tnd_ik",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_il,"chimes:Tnd_il",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_jk,"chimes:Tnd_jk",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_jl,"chimes:Tnd_jl",poly_order+1);
        memKK::realloc_kokkos(d_Tnd_kl,"chimes:Tnd_kl",poly_order+1);
      }
    }
  };

  ////////////////////////
  // General parameters
  ////////////////////////

  vector<int>      poly_orders;    // [bodiedness-1]; i.e. 12 = 2-body only, 12th order; 12 5 = 2+3-body, 0 5 = 3-body only, 5th order
  vector<string>   atmtyps;        // Atom types
  vector<double>   masses;         // Atom masses

  ////////////////////////
  // Functions
  ////////////////////////

  chimesFFKokkos();
  ~chimesFFKokkos();

  void read_parameters(string paramfile) override;

  void compute_1B(const int typ_idx, double & energy);

  // 2+B compute functions overloaded with force_scalar_in var for compatibility with LAMMPS

  void compute_2B(const double dx, const vector<double> & dr, const vector<int> typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes2BTmpKokkos &tmp);
  void compute_2B(const double dx, const vector<double> & dr, const vector<int> typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes2BTmpKokkos &tmp, double & force_scalar_in);

  void compute_3B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force,vector<double> & stress, double & energy, chimes3BTmpKokkos &tmp);
  void compute_3B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force,vector<double> & stress, double & energy, chimes3BTmpKokkos &tmp, vector<double> & force_scalar_in);

  void compute_4B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes4BTmpKokkos &tmp);
  void compute_4B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes4BTmpKokkos &tmp, vector<double> & force_scalar_in);

  // Functions to aid using ChIMES Calculator for fitting

  inline int  get_badness();
  inline void reset_badness();

private:

  vector<double>    morse_var;      // [npairs]; morse_lambda
  vector<double>    penalty_params; // [2];  Second dimension: [0] = A_pen, [1] = d_pen
  vector<double>    energy_offsets; // [natmtyps]; Single atom ChIMES energies

  ////////////////////////
  // Definitions for pair, triplet, and quadruplet types
  ////////////////////////

  // 2-body maps

  vector    <string> atom_typ_pair_map; // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiO")
  vector       <int> atom_idx_pair_map; // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 5)
  vector       <int> atom_int_pair_map; // [nmaps] "fast" maps, based on atom type index
  vector    <string> atom_int_prpr_map; // [nmaps] "fast" maps, based on atom type index ... returns the "proper" pair type instead of an index

  // 3-body maps

  vector    <string>    atom_typ_trip_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiOSiOOO")
  vector       <int>    atom_idx_trip_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 3)
  vector       <int>    atom_int_trip_map;    // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of three atoms of arbitrary order
  vector<vector<int> >   pair_int_trip_map;  // Gives the atom pair indices for an arbitrary triplet of atom types.

  // 4-body maps

  vector    <string>    atom_typ_quad_map;    // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives chemical symbol list (i.e. "SiOSiOOO")
  vector      <int>        atom_idx_quad_map; // [nmaps] "slow" maps, based on atom chemical symbol    // Used to build int map -- gives correspoding parameter index (i.e. 3)
  vector      <int>        atom_int_quad_map; // [nmaps] "fast" maps, based on atom type index         // gives the correspoding parameter index (i.e. 3) for a unique integer built from type index of four atoms of arbitrary order
  vector<vector<int> >    pair_int_quad_map;  // Gives the atom pair indices for an arbitrary quad of atom types.

  ////////////////////////
  // Polynomial parameters
  ////////////////////////

  // number of coefficients for the pair/triplet/quadruplet type

  vector          <int>   ncoeffs_2b;        // [npairs]

  vector<vector<int>    > chimes_2b_pows;    // [npairs][npowers] power for the coresponding parameter

  vector<vector<double> > chimes_2b_params;    // [npairs][npowers] 2-body polynomial coefficients
  vector<vector<double> > chimes_2b_cutoff;    // [npairs][2] inner and outer cutoff for pair

  vector<int>                      ncoeffs_3b;          // [ntrips]
  vector<vector<vector<int> > >    chimes_3b_powers;    // [ntrips][nparams][constit. pair]
  vector<vector<double> >          chimes_3b_params;    // [ntrips][nparams]
  vector<vector<vector<double> > > chimes_3b_cutoff;    // [ntrips][2][constit. pair] inner and outer cutoff for pair 1


  vector<int>                      ncoeffs_4b;          // [nquads]
  vector<vector<vector<int> > >    chimes_4b_powers;    // [nquads][nparams][constit. pair]
  vector<vector<double> >          chimes_4b_params;    // [nquads][nparams]
  vector<vector<vector<double> > > chimes_4b_cutoff;    // [nquads][2][constit. pair] inner and outer cutoff for pair 1

  // Tools for compute functions

  KOKKOS_INLINE_FUNCTION
  void set_cheby_polys(typename AT::t_kkfloat_1d &d_Tn, typename AT::t_kkfloat_1d &d_Tnd, double dx, const double morse,
                       const double inner_cutoff, const double outer_cutoff, const int order);

  void set_polys_out_of_range(typename AT::t_kkfloat_1d &d_Tn, typename AT::t_kkfloat_1d &d_Tnd, double dx, double x,
                              int poly_order, double inner_cutoff, double exprlen, double dx_dr);

  inline void get_fcut(const double dx, const double outer_cutoff, double & fcut, double & fcutderiv);

  inline void get_penalty(const double dx, const int & pair_idx, double & E_penalty, double & force_scalar);

  inline void build_atom_and_pair_mappers(const int natoms, const int npairs, const vector<int> & typ_idxs,
                                          const vector<string> & clu_params_atm_chems, vector<int >  & mapped_pair_idx);

  inline void build_atom_and_pair_mappers(const int natoms, const int npairs, const vector<int> & typ_idxs, const vector<string> & clu_params_atm_chems,
                                          int *mapped_pair_idx);

  int get_proper_pair(string ty1, string ty2);

  double max_cutoff(int ntypes, vector<vector<vector<double> > > & cutoff_list);

  // Tools for reading the input file

  int split_line(string line, vector<string> & items);

  string get_next_line(istream& str);

  // Fun stuff

  void print_pretty_stuff();

  inline double dr2_3B(const double *dr2, int i, int j, int k, int l);
  inline double dr2_4B(const double *dr2, int i, int j, int k, int l);
  inline void init_distance_tensor(double *dr2, const vector<double> & dr, int natoms)   ;
};

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void chimesFFKokkos::get_fcut(const double dx, const double outer_cutoff, double & fcut, double & fcutderiv)
{
  double fcut0;
  double fcut0_deriv;

  if (fcut_type == fcutType::CUBIC) {
    fcut0 = (1.0 - dx/outer_cutoff);
    fcut = pow(fcut0,3.0);
    fcutderiv = pow(fcut0,2.0);
    fcutderiv *= -1.0 * 3.0 /outer_cutoff;
  } else if (fcut_type == fcutType::TERSOFF) {

    double THRESH = outer_cutoff-fcut_var*outer_cutoff;

    if (dx < THRESH)        // Case 1: Our pair distance is less than the fcut kick-in distance
    {
      fcut = 1.0;
      fcutderiv = 0.0;
    }
    else if (dx > outer_cutoff)        // Case 2: Our pair distance is greater than the cutoff
    {
      fcut = 0.0;
      fcutderiv = 0.0;
    }
    else                // Case 3: We'll use our modified sin function
    {
      fcut0 = (dx-THRESH) / (outer_cutoff-THRESH) * pi + pi/2.0;
      fcut0_deriv = pi / (outer_cutoff - THRESH);

      fcut = 0.5 + 0.5 * sin(fcut0);
      fcutderiv  = 0.5 * cos(fcut0) * fcut0_deriv;
    }
  }
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void chimesFFKokkos::get_penalty(const double dx, const int& pair_idx, double& E_penalty, double& force_scalar)
{
  double r_penalty = 0.0;

  E_penalty    = 0.0;
  force_scalar = 1.0;

  if (dx - penalty_params[0] < chimes_2b_cutoff[pair_idx][0]) { // Then we're within the penalty-enforced region of distance space
    r_penalty = chimes_2b_cutoff[pair_idx][0] + penalty_params[0] - dx;

    if (dx < chimes_2b_cutoff[pair_idx][0])
      badness = 2;
    else if (1 > badness) // Only update badness if candiate badness is worse than its current value
      badness = 1;
  }

  if (r_penalty > 0.0) {
    E_penalty    = r_penalty * r_penalty * r_penalty * penalty_params[1];

    force_scalar = -3.0 * r_penalty * r_penalty * penalty_params[1];

    //if (rank == 0) // Commenting out - we need all ranks to report if the penalty function has been sampled
    //{
        cout << "chimesFFKokkos: " << "Adding penalty in 2B Cheby calc, r < rmin+penalty_dist " << fixed
             << dx << " "
             << chimes_2b_cutoff[pair_idx][0] + penalty_params[0]
             << " pair type: " << pair_idx << endl;
        cout << "chimesFFKokkos: " << "\t...Penalty potential = "<< E_penalty << endl;
    //}
  }
}

inline int chimesFFKokkos::get_badness()
{
  return badness;
}

inline void chimesFFKokkos::reset_badness()
{
  badness = 0;
}

inline void chimesFFKokkos::build_atom_and_pair_mappers(const int natoms, const int npairs, const vector<int> & typ_idxs,
                                                  const vector<string> & clu_params_pair_typs, vector<int>  & mapped_pair_idx)
// Interface to array-based version

{
  build_atom_and_pair_mappers(natoms, npairs, typ_idxs, clu_params_pair_typs, mapped_pair_idx.data());
}

inline void chimesFFKokkos::build_atom_and_pair_mappers(const int natoms, const int npairs, const vector<int> & typ_idxs,
                                                  const vector<string> & clu_params_pair_typs,
                                                  int *mapped_pair_idx)
{
    // Generate permutations for atoms... all we are doing is permuting the possible indices for typ_idxs


    // build a copy of the atom type vector for permuting

    vector<int> tmp_typ_idxs;
    int         nelements;

    nelements = typ_idxs.size();
    tmp_typ_idxs.resize(nelements);

    for(int i=0; i<nelements; i++)
        tmp_typ_idxs[i] = i;

    // Build a copy of the original pairs for comparison against permuted pairs


    vector<vector<int> > tmp_pairs;
    tmp_pairs.resize(npairs,vector<int>(2));
    vector<vector<int> > runtime_pairs;
    runtime_pairs.resize(npairs,vector<int>(2));

    int idx = 0;

    for(int i=0; i<natoms; i++)
    {

        for (int j=i+1; j<natoms; j++)
        {
            tmp_pairs[idx][0] = i;
            tmp_pairs[idx][1] = j;

            idx++;
        }
    }

    vector<string> runtime_pair_typs(npairs);

    do
    {
        // Check if the permutation leads to pair types that match the order specified by the force field type

        idx = 0;

        for(int i=0; i<natoms; i++) // Associate the current atom pairs with a "proper" 2-body force field name
        {
            for (int j=i+1; j<natoms; j++)
            {
                runtime_pair_typs[idx] = atom_int_prpr_map[ typ_idxs[tmp_typ_idxs[i]]*natmtyps + typ_idxs[tmp_typ_idxs[j]] ];

                idx++;
            }
        }

        bool match = true;

        for(int i=0; i<npairs; i++)
        {
            if (clu_params_pair_typs[i] != runtime_pair_typs[i])
            {
                match = false;
                break;
            }
        }

        if (match) // Then we've found an appropriate atom ordering... now what?
        {
            idx = 0;

            for(int i=0; i<natoms; i++) // Associate the current atom pairs with a "proper" 2-body force field name
            {
                for (int j=i+1; j<natoms; j++)
                {
                    runtime_pairs[idx][0] = tmp_typ_idxs[i];
                    runtime_pairs[idx][1] = tmp_typ_idxs[j];

                    idx++;
                }
            }

            break;
        }


    } while (next_permutation(tmp_typ_idxs.begin(),tmp_typ_idxs.begin()+nelements));

    // Once we've found a re-ordering of atoms that properly maps to the force field pair types, need to figure out how to convert that to a map between *pairs*

    idx = 0;

    for(int i=0; i<npairs; i++)
        for(int j=0; j<npairs; j++)
            if (   ((runtime_pairs[i][0] == tmp_pairs[j][0]) && (runtime_pairs[i][1] == tmp_pairs[j][1]))
                 || ((runtime_pairs[i][0] == tmp_pairs[j][1]) && (runtime_pairs[i][1] == tmp_pairs[j][0]))
               )
                mapped_pair_idx[j] = i;

}

KOKKOS_INLINE_FUNCTION
void chimesFFKokkos::set_cheby_polys(vector<double> &Tn, vector<double> &Tnd, double dx, const double morse,
                                     const double inner_cutoff, const double outer_cutoff, const int order) const
{
  // Currently assumes a Morse-style transformation has been requested

  // Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd).  Tnd is the derivative
  // with respect to the interatomic distance, not the transformed distance (x).

  // Do the Morse transformation

  double x_min = exp(-1*inner_cutoff/morse);
  double x_max = exp(-1*outer_cutoff/morse);

  double x_avg   = 0.5 * (x_max + x_min);
  double x_diff  = 0.5 * (x_max - x_min);

  x_diff *= -1.0; // Special for Morse style

  bool out_of_range;
  double dx_orig = dx;

  // The case dx > outer_cutoff is not treated, because it is assumed that the outer smoothing
  //  function will be zero for dx > outer_cutoff

  if (dx < inner_cutoff) {
    out_of_range = true;
    dx = inner_cutoff;
  } else
    out_of_range = false;

  double exprlen = exp(-1*dx/morse);
  double x = (exprlen - x_avg)/x_diff;
  double dx_dr = (-exprlen/morse)/x_diff;

  if (!out_of_range) {

    // Generate Chebyshev polynomials by recursion.
    //
    // What we're doing here. Want to fit using Cheby polynomials of the 1st kinD[i]. "T_n(x)."
    // We need to calculate the derivative of these polynomials.
    // Derivatives are defined through use of Cheby polynomials of the 2nd kind "U_n(x)", as:
    //
    // d/dx[ T_n(x) = n * U_n-1(x)]
    //
    // So we need to first set up the 1st-kind polynomials ("Tn[]")
    // Then, to compute the derivatives ("Tnd[]"), first set equal to the 2nd-kind, then multiply by n to get the der's

    // First two 1st-kind Chebys:

    Tn[0] = 1.0;
    Tn[1] = x;

    // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

    Tnd[0] = 1.0;
    Tnd[1] = 2.0 * x;

    // Use recursion to set up the higher n-value Tn and Tnd's

    for (int i = 2; i <= order; i++) {
      Tn[i]  = 2.0 * x *  Tn[i-1] -  Tn[i-2];
      Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
    }

    // Now multiply by n to convert Tnd's to actual derivatives of Tn

    // The following dx_dr compuation assumes a Morse transformation
    // DERIV_CONST is no longer used. (old way: dx_dr = DERIV_CONST*cheby_var_deriv(x_diff, rlen, ff_2body.LAMBDA, ff_2body.CHEBY_TYPE, exprlen);)

    for (int i = order; i >= 1; i--)
      Tnd[i] = i * dx_dr * Tnd[i-1];

    Tnd[0] = 0.0;
  } else { // out_of_range == true
    cout << "Warning: An intermolecular distance less than the inner cutoff = " << inner_cutoff << " was found\n ";
    cout << "         Distance = " << dx_orig << endl;

    set_polys_out_of_range(Tn, Tnd, dx_orig, x, order, inner_cutoff, exprlen, dx_dr);
  }
}

#endif
