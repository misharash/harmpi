//Modified by Alexander Tchekhovskoy: MPI+3D
/***********************************************************************************
    Copyright 2006 Charles F. Gammie, Jonathan C. McKinney, Scott C. Noble, 
                   Gabor Toth, and Luca Del Zanna

                        HARM  version 1.0   (released May 1, 2006)

    This file is part of HARM.  HARM is a program that solves hyperbolic 
    partial differential equations in conservative form using high-resolution
    shock-capturing techniques.  This version of HARM has been configured to 
    solve the relativistic magnetohydrodynamic equations of motion on a 
    stationary black hole spacetime in Kerr-Schild coordinates to evolve
    an accretion disk model. 

    You are morally obligated to cite the following two papers in his/her 
    scientific literature that results from use of any part of HARM:

    [1] Gammie, C. F., McKinney, J. C., \& Toth, G.\ 2003, 
        Astrophysical Journal, 589, 444.

    [2] Noble, S. C., Gammie, C. F., McKinney, J. C., \& Del Zanna, L. \ 2006, 
        Astrophysical Journal, 641, 626.

   
    Further, we strongly encourage you to obtain the latest version of 
    HARM directly from our distribution website:
    http://rainman.astro.uiuc.edu/codelib/


    HARM is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    HARM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HARM; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/


#include "decs.h"

void bound_mpi_dim(int dim, int ispack, double prim[][N2M][N3M][NPR], int pflag[][N2M][N3M]);
int pack_prim(int ispack, int dim, int isup, double prim[][N2M][N3M][NPR], double *mpi_buf );
int pack_pflag(int ispack, int dim, int isup, int pflag[][N2M][N3M], int *mpi_buf );
void bound_x1dn(double prim[][N2M][N3M][NPR] );
void bound_x2dn(double prim[][N2M][N3M][NPR] );
void bound_x2dn_polefix(double prim[][N2M][N3M][NPR] );
void bound_x3dn(double prim[][N2M][N3M][NPR] );
void bound_x1up(double prim[][N2M][N3M][NPR] );
void bound_x2up(double prim[][N2M][N3M][NPR] );
void bound_x2up_polefix(double prim[][N2M][N3M][NPR] );
void bound_x3up(double prim[][N2M][N3M][NPR] );


/* bound array containing entire set of primitive variables */

void bound_x1dn(double prim[][N2M][N3M][NPR] )
{
  int i,j,k,m,jref ;
  void inflow_check(double *pr, int ii, int jj, int kk, int type );
  struct of_geom geom ;
  
  int iNg, jNg, kNg;
  
#if(N1!=1)
  if (!is_physical_bc(1, 0)) return;
#if (WHICHPROBLEM == NSSURFACE)
  bound_x1dn_nssurface(prim);
#else

  /* inner r boundary condition */
  for(j=0;j<N2;j++)
  {
    for(k=0; k<N3;k++)  //!!!ATCH: make sure don't need to expand to transverse ghost cells
    {
#if( RESCALE )
      get_geometry(0,j,k,CENT,&geom) ;
      rescale(prim[0][j][k],FORWARD, 1, 0,j,k,CENT,&geom) ;
#endif
      
      for (iNg=-N1G; iNg<0; iNg++)
      {
#if (PERIODIC==1)
        PLOOP prim[iNg][j][k][m] = prim[N1+iNg][j][k][m];
        pflag[iNg][j][k] = pflag[N1+iNg][j][k];
#elif(DIRICHLET==1)
        PLOOP prim[iNg][j][k][m] = pbound[iNg][j][k][m];
        pflag[iNg][j][k] = pflag[0][j][k] ;
#else //outflow
        PLOOP prim[iNg][j][k][m] = prim[0][j][k][m];
        pflag[iNg][j][k] = pflag[0][j][k] ;
#endif
      }
      
#if( RESCALE )
      for (iNg = -N1G; iNg<=0; iNg++)
      {
        get_geometry(iNg,j,k,CENT,&geom) ;
        rescale(prim[iNg][j][k],REVERSE, 1, iNg,j,k,CENT,&geom) ;
      }
#endif
    }
  }

  /* make sure there is no inflow at the inner boundary */
  if(1!=INFLOW) {
    for(i=-N1G;i<=-1;i++)  for(j=-N2G;j<N2+N2G;j++) for(k=-N3G;k<N3+N3G;k++)
    {
      //!!!ATCH: eliminated one loop that seemed excessive. verify.
      inflow_check(prim[i][j][k],i,j,k,0) ; //0 stands for -x1 boundary
    }
  }
#endif
#endif
  
}

void bound_x1up(double prim[][N2M][N3M][NPR] )
{
  int i,j,k,m,jref ;
  void inflow_check(double *pr, int ii, int jj, int kk, int type );
  struct of_geom geom ;
  
  int iNg, jNg, kNg;
#if(N1!=1)
  if (!is_physical_bc(1, 1)) return;

  /* Outer r boundary condition */
  for(j=0;j<N2;j++)
  {
    for(k=0; k<N3;k++)  //!!!ATCH: make sure don't need to expand to transverse ghost cells
    {
#if( RESCALE )
      get_geometry(N1-1,j,k,CENT,&geom) ;
      rescale(prim[N1-1][j][k],FORWARD, 1, N1-1,j,k,CENT,&geom) ;
#endif
      
      for (iNg=0; iNg<N1G; iNg++)
      {
#if (PERIODIC==1)
        PLOOP prim[N1+iNg][j][k][m] = prim[iNg][j][k][m];
        pflag[N1+iNg][j][k] = pflag[iNg][j][k] ;
#elif(DIRICHLET==1)
        PLOOP prim[N1+iNg][j][k][m] = pbound[N1+iNg][j][k][m];
        pflag[N1+iNg][j][k] = pflag[N1-1][j][k] ;
#else //outflow
        PLOOP prim[N1+iNg][j][k][m] = prim[N1-1][j][k][m];
        pflag[N1+iNg][j][k] = pflag[N1-1][j][k] ;
#endif
      }
#if( RESCALE )
      for (iNg= 0; iNg<N1G+1; iNg++) //!!!ATCH: added +1 to N1G to ensure that all ghost cells are looped over
      {
        get_geometry(N1-1+iNg,j,k,CENT,&geom) ;
        rescale(prim[N1-1+iNg][j][k],REVERSE, 1, N1-1+iNG,j,k,CENT,&geom) ;
      }
#endif
    }
  }
  /* make sure there is no inflow at the outer boundary */
  if(1!=INFLOW) {
    for(i=N1;i<=N1+N1G-1;i++)  for(j=-N2G;j<N2+N2G;j++) for(k=-N3G;k<N3+N3G;k++)
    {
      //!!!ATCH: eliminated one loop that seemed excessive. verify.
      inflow_check(prim[i][j][k],i,j,k,1) ; //1 stands for +x1 boundary
    }
  }

#endif
}

void bound_x2dn_polefix( double prim[][N2M][N3M][NPR] )
{
  int i,j,k,m,jref ;
  struct of_geom geom ;
  int iNg, jNg, kNg;
  
#if(POLEFIX && POLEFIX < N2/2 && BL)
  //only do anything if physically in an MPI process that touches the inner pole
  if (!is_physical_bc(2, 0)) return;

  //copy all densities and B^phi in; interpolate linearly transverse velocity
  jref = POLEFIX;
  for(i=-N1G;i<N1+N1G;i++) {
    for(k=-N3G;k<N3+N3G;k++) {
      for(j=0;j<jref;j++) {
        PLOOP {
          if(m==B1 || m==B2 || (N3>1 && m==B3))
            //don't touch magnetic fields
            continue;
          else if(m==U2) {
            //linear interpolation of transverse velocity (both poles)
            prim[i][j][k][m] = (j+0.5)/(jref+0.5) * prim[i][jref][k][m];
          }
          else {
            //everything else copy (both poles)
            prim[i][j][k][m] = prim[i][jref][k][m];
          }
        }
      }
    }
  }
#endif
}

void bound_x2up_polefix( double prim[][N2M][N3M][NPR] )
{
  int i,j,k,m,jref ;
  struct of_geom geom ;
  int iNg, jNg, kNg;

#if(POLEFIX && POLEFIX < N2/2 && BL)
  //only do anything if physically in an MPI process that touches the outer pole
  if (!is_physical_bc(2, 1)) return;

  //copy all densities and B^phi in; interpolate linearly transverse velocity
  jref = POLEFIX;
  for(i=-N1G;i<N1+N1G;i++) {
    for(k=-N3G;k<N3+N3G;k++) {
      for(j=0;j<jref;j++) {
        PLOOP {
          if(m==B1 || m==B2 || (N3>1 && m==B3))
            //don't touch magnetic fields
            continue;
          else if(m==U2) {
            //linear interpolation of transverse velocity (both poles)
            prim[i][N2-1-j][k][m] = (j+0.5)/(jref+0.5) * prim[i][N2-1-jref][k][m];
          }
          else {
            //everything else copy (both poles)
            prim[i][N2-1-j][k][m] = prim[i][N2-1-jref][k][m];
          }
        }
      }
    }
  }
#endif
  
}

/* polar BCs */
//inner theta boundary
void bound_x2dn( double prim[][N2M][N3M][NPR] )
{
  int i,j,k,m,jref ;
  struct of_geom geom ;
  int iNg, jNg, kNg;

#if(N2!=1)
  //only do anything if physically in an MPI process that touches the inner pole
  if (!is_physical_bc(2, 0)) return;

  bound_x2dn_polefix(prim);
  
  for (i=-N1G; i<N1+N1G; i++)
  {
    for (k=-N3G; k<N3+N3G; k++)
    {
      for (jNg=-N2G; jNg<0; jNg++)
      {
#if (PERIODIC==1)
        PLOOP prim[i][jNg][k][m] = prim[i][N2+jNg][k][m];
        pflag[i][jNg][k] = pflag[i][N2+jNg][k];
#elif(DIRICHLET==1)
        PLOOP prim[i][jNg][k][m] = pbound[i][jNg][k][m];
        pflag[i][jNg][k] = pflag[i][-jNg-1][k];
#elif (OUTFLOW==1)
        PLOOP prim[i][jNg][k][m] = prim[i][0][k][m];
        pflag[i][jNg][k] = pflag[i][0][k];
#else //symmetric/asymmetric
        PLOOP prim[i][jNg][k][m] = prim[i][-jNg-1][k][m];
        pflag[i][jNg][k] = pflag[i][-jNg-1][k];
#endif
        
      }
    }
  }
  
  /* polar BCs */
  /* make sure b and u are antisymmetric at the poles */
  if(BL){
    for(i=-N1G;i<N1+N1G;i++) {
      for(k=-N3G;k<N3+N3G;k++) {
        for(j=-N2G;j<0;j++) {
          prim[i][j][k][U2] *= -1. ;
          prim[i][j][k][B2] *= -1. ;
        }
      }
    }
  }

#endif

}

/* polar BCs */
//outer theta boundary
void bound_x2up( double prim[][N2M][N3M][NPR] )
{
  int i,j,k,m,jref ;
  struct of_geom geom ;
  int iNg, jNg, kNg;
  
#if(N2!=1)
  //only do anything if physically in an MPI process that touches the inner pole
  if (!is_physical_bc(2, 1)) return;

  bound_x2up_polefix(prim);

  for (i=-N1G; i<N1+N1G; i++)
  {
    for (k=-N3G; k<N3+N3G; k++)
    {
      //outer theta boundary
      for (jNg=0; jNg<N2G; jNg++)
      {
#if (PERIODIC==1)
        PLOOP prim[i][N2+jNg][k][m] = prim[i][jNg][k][m];
        pflag[i][N2+jNg][k] = pflag[i][jNg][k];
#elif(DIRICHLET==1)
        PLOOP prim[i][N2+jNg][k][m] = pbound[i][N2+jNg][k][m];
        pflag[i][N2+jNg][k] = pflag[i][N2-jNg-1][k];
#elif (OUTFLOW==1)
        PLOOP prim[i][N2+jNg][k][m] = prim[i][N2-1][k][m];
        pflag[i][N2+jNg][k] = pflag[i][N2-1][k];
#else //symmetric/asymmetric
        PLOOP prim[i][N2+jNg][k][m] = prim[i][N2-jNg-1][k][m];
        pflag[i][N2+jNg][k] = pflag[i][N2-jNg-1][k];
#endif
      }
    }
  }
  
  /* polar BCs */
  /* make sure b and u are antisymmetric at the poles */
  if(BL){
    for(i=-N1G;i<N1+N1G;i++) {
      for(k=-N3G;k<N3+N3G;k++) {
        for(j=N2;j<N2+N2G;j++) {
          prim[i][j][k][U2] *= -1. ;
          prim[i][j][k][B2] *= -1. ;
        }
      }
    }
  }

#endif
}

void bound_x3dn( double prim[][N2M][N3M][NPR] )
{
  int i,j,k,m,jref ;
  struct of_geom geom ;
  int iNg, jNg, kNg;

#if(N3!=1)
  //only do anything if physically in an MPI process that touches the inner pole
  if (!is_physical_bc(3, 0)) return;

  /* phi BCs */
  //inner phi-boundary
  for (i=-N1G; i<N1+N1G; i++)
  {
    for (j=-N2G; j<N2+N2G; j++)
    {
      for (kNg=-N3G; kNg<0; kNg++)
      {
#if (PERIODIC==1)
        PLOOP prim[i][j][kNg][m] = prim[i][j][N3+kNg][m];
        pflag[i][j][kNg] = pflag[i][j][N3+kNg];
#elif(DIRICHLET==1)
        PLOOP prim[i][j][kNg][m] = pbound[i][j][kNg][m];
        pflag[i][j][kNg] = pflag[i][j][0];
#elif (OUTFLOW==1)
        PLOOP prim[i][j][kNg][m] = prim[i][j][0][m];
        pflag[i][j][kNg] = pflag[i][j][0];
#else
        //periodic by default
        PLOOP prim[i][j][kNg][m] = prim[i][j][N3+kNg][m];
        pflag[i][j][kNg] = pflag[i][j][N3+kNg];
#endif
      }
    }
  }
#endif

}

void bound_x3up( double prim[][N2M][N3M][NPR] )
{
  int i,j,k,m,jref ;
  struct of_geom geom ;
  int iNg, jNg, kNg;

#if(N3!=1)
  //only do anything if physically in an MPI process that touches the inner pole
  if (!is_physical_bc(3, 1)) return;

  /* phi BCs */
  //outer phi-boundary
  for (i=-N1G; i<N1+N1G; i++)
  {
    for (j=-N2G; j<N2+N2G; j++)
    {
      for (kNg=-N3G; kNg<0; kNg++)
      {
        for (kNg=0; kNg<N3G; kNg++)
        {
#if (PERIODIC==1)
          PLOOP prim[i][j][N3+kNg][m] = prim[i][j][kNg][m];
          pflag[i][j][N3+kNg] = pflag[i][j][kNg];
#elif(DIRICHLET==1)
          PLOOP prim[i][j][N3+kNg][m] = pbound[i][j][N3+kNg][m];
          pflag[i][j][N3+kNg] = pflag[i][j][N3-1];
#elif (OUTFLOW==1)
          PLOOP prim[i][j][N3+kNg][m] = prim[i][j][N3-1][m];
          pflag[i][j][N3+kNg] = pflag[i][j][N3-1];
#else
          //periodic by default
          PLOOP prim[i][j][N3+kNg][m] = prim[i][j][kNg][m];
          pflag[i][j][N3+kNg] = pflag[i][j][kNg];
#endif
        }
      }
    }
  }
#endif
}

void bound_prim( double prim[][N2M][N3M][NPR] )
{
  int ispack, dim;
  
  //MPIMARK: could be optimized by individually passing corner zones:
  //         then, speed up by not doing comm dimension by dimension

  //x1-dim
  //packing, putting send and receive requests
  dim = 1;
  ispack = 1;
  bound_mpi_dim(dim, ispack, prim, pflag);
  //while waiting for MPI comm to complete, do physical boundaries
  bound_x1dn(prim);
  bound_x1up(prim);
  //waiting for comm to complete and unpacking
  ispack = 0;
  bound_mpi_dim(dim, ispack, prim, pflag);

  //x2-dim
  //packing, putting send and receive requests
  dim = 2;
  ispack = 1;
  bound_mpi_dim(dim, ispack, prim, pflag);
  //while waiting for MPI comm to complete, do physical boundaries
  bound_x2dn(prim);
  bound_x2up(prim);
  //waiting for comm to complete and unpacking
  ispack = 0;
  bound_mpi_dim(dim, ispack, prim, pflag);
  

  //x3-dim
  //waiting for comm to complete and unpacking
  dim = 3;
  ispack = 1;
  bound_mpi_dim(dim, ispack, prim, pflag);
  //while waiting for MPI comm to complete, do physical boundaries
  bound_x3dn(prim);
  bound_x3up(prim);
  //waiting for comm to complete and unpacking
  ispack = 0;
  bound_mpi_dim(dim, ispack, prim, pflag);
}

//packs (ispack=1) or unpacks (ispack=0) the cells to be communicated along
//dimension (dim), either upper (isup=1) or lower (isup=0) boundary
//returns the number of items packed (count)
int pack_prim(int ispack, int dim, int isup, double prim[][N2M][N3M][NPR], double *mpi_buf )
{
  int istart,istop,jstart,jstop,kstart,kstop,i,j,k,m,count;

  //if true, ensure it has value of unity for the below to work
  if(ispack) ispack=1;
  
  //do not do empty dimensions
  if(mpi_ntot[dim] == 1) return(0);
  
  //figure out the range of indices to transfer
  //x1: in x1-dim transfer only ghost cells immediately adjacent to active cells
  if(1==dim){
    jstart=0; jstop=N2-1;
    kstart=0; kstop=N3-1;
    if(isup) istart=N1-ispack*N1G; else istart=-N1G+ispack*N1G;
    istop=istart+N1G-1;
  }
  //x2: in x2-dim, additionally trasfer the ghost cells that have been communicated in x1-dim
  else if(2==dim){
    istart=-N1G; istop=N1+N1G-1;
    kstart=0; kstop=N3-1;
    if(isup) jstart=N2-ispack*N2G; else jstart=-N2G+ispack*N2G;
    jstop=jstart+N2G-1;
  }
  //x3, in x3-dim, additionally trasfer the ghost cells that have been communicated in x1-dim and x2-dim
  else if(3==dim){
    istart=-N1G; istop=N1+N1G-1;
    jstart=-N2G; jstop=N2+N2G-1;
    if(isup) kstart=N3-ispack*N3G; else kstart=-N3G+ispack*N3G;
    kstop=kstart+N3G-1;
  }
  
  //initialize the counter of the number of doubles (un)packed
  count = 0;
  
  //packing
  if(ispack){
    ZSLOOP(istart,istop,jstart,jstop,kstart,kstop) PLOOP {
      mpi_buf[count++] = prim[i][j][k][m];
    }
  }
  ///unpacking
  else {
    ZSLOOP(istart,istop,jstart,jstop,kstart,kstop) PLOOP {
      prim[i][j][k][m] = mpi_buf[count++];
    }
  }
  return(count);
}

//packs (ispack=1) or unpacks (ispack=0) the cells to be communicated along
//dimension (dim), either upper (isup=1) or lower (isup=0) boundary
//returns the number of items packed (count)
int pack_pflag(int ispack, int dim, int isup, int pflag[][N2M][N3M], int *mpi_buf )
{
  int istart,istop,jstart,jstop,kstart,kstop,i,j,k,m,count;
  //number of ghost cells to copy: need only layer of thickness one for pflag
  int n1g = (N1G>0), n2g = (N2G>0), n3g = (N3G>0);
  
  //if true, ensure it has value of unity for the below to work
  if(ispack) ispack=1;
  
  //do not do empty dimensions
  if(mpi_ntot[dim] == 1) return(0);
  
  //figure out the range of indices to transfer
  //x1: in x1-dim transfer only ghost cells immediately adjacent to active cells
  if(1==dim){
    jstart=0; jstop=N2-1;
    kstart=0; kstop=N3-1;
    if(isup) istart=N1-ispack*n1g; else istart=-n1g+ispack*n1g;
    istop=istart+n1g-1;
  }
  //x2: in x2-dim, additionally trasfer the ghost cells that have been communicated in x1-dim
  else if(2==dim){
    istart=-n1g; istop=N1+n1g-1;
    kstart=0; kstop=N3-1;
    if(isup) jstart=N2-ispack*n2g; else jstart=-n2g+ispack*n2g;
    jstop=jstart+n2g-1;
  }
  //x3, in x3-dim, additionally trasfer the ghost cells that have been communicated in x1-dim and x2-dim
  else if(3==dim){
    istart=-n1g; istop=N1+n1g-1;
    jstart=-n2g; jstop=N2+n2g-1;
    if(isup) kstart=N3-ispack*n3g; else kstart=-n3g+ispack*n3g;
    kstop=kstart+n3g-1;
  }
  
  //initialize the counter of the number of doubles (un)packed
  count = 0;
  
  //packing
  if(ispack){
    ZSLOOP(istart,istop,jstart,jstop,kstart,kstop) {
      mpi_buf[count++] = pflag[i][j][k];
    }
  }
  ///unpacking
  else {
    ZSLOOP(istart,istop,jstart,jstop,kstart,kstop) {
      pflag[i][j][k] = mpi_buf[count++];
    }
  }
  return(count);
}


int is_physical_bc( int dim, int isup )
{
  
  //dimension is trivial => boundary is not physical
  if( 1 == mpi_ntot[dim] )
    return(0);
  
  //lower boundary is physical
  if( 0 == isup && 0 == mpi_coords[dim] && (1 == mpi_dims[dim] || 1 != mpi_periods[dim]) )
    return(1);
  
  //upper boundary is physical
  if( 1 == isup && mpi_dims[dim]-1 == mpi_coords[dim] && (1 == mpi_dims[dim] || 1 != mpi_periods[dim]) )
    return(1);
  
  //MPI boundary
  return(0);
}

//initiates send (isrecv = 0) or receive (isrecv = 1) operation in dimension dim (=0,1,2)
void bound_mpi_dim(int dim, int ispack, double prim[][N2M][N3M][NPR], int pflag[][N2M][N3M])
{
#ifdef MPI
#define DOMPIPFLAG (0)
    
  int count, count_pflag, tagsend, tagrecv, isup;
  
  //packing and sending
  if(1 == ispack) {
    for(isup=0;isup<=1;isup++) {
      //skip if on the physical boundary and the BC is not periodic
      if(is_physical_bc(dim,isup)) continue;

      //pack the data from prim[] into mpi send buffers
      count = pack_prim(ispack, dim, isup, prim, mpi_buf_send[dim][isup]);

#if(DOMPIPFLAG)
      //pack the data from pflag[] into mpi send buffers
      count_pflag = pack_pflag(ispack, dim, isup, pflag, mpi_buf_send_pflag[dim][isup]);
#endif

      //prims
      tagsend = 0+2*isup;
      tagrecv = 0+2*!isup;
      MPI_Isend(mpi_buf_send[dim][isup], //buffer that's being sent
                count,              //number of items sent
                MPI_DOUBLE,         //data type
                mpi_nbrs[dim][isup],//the rank of destination process
                tagsend,                //tag
                MPI_COMM_WORLD,     //communicator
                &mpi_reqs_send[dim][isup] //error
                );

      MPI_Irecv(mpi_buf_recv[dim][isup], //buffer that's being received
                count,              //number of items received (same as those sent)
                MPI_DOUBLE,         //data type
                mpi_nbrs[dim][isup],//the rank of source process
                tagrecv,                //tag (should be same as in the send process)
                MPI_COMM_WORLD,     //communicator
                &mpi_reqs_recv[dim][isup] //error
                );
#if(DOMPIPFLAG)
      //pflags
      tagsend = 1+2*isup;
      tagrecv = 1+2*!isup;
      MPI_Isend(mpi_buf_send_pflag[dim][isup], //buffer that's being sent
                count_pflag,              //number of items sent
                MPI_INT,            //data type
                mpi_nbrs[dim][isup],//the rank of destination process
                tagsend,                //tag
                MPI_COMM_WORLD,     //communicator
                &mpi_reqs_send_pflag[dim][isup] //error
                );
      
      MPI_Irecv(mpi_buf_recv_pflag[dim][isup], //buffer that's being received
                count_pflag,              //number of items received (same as those sent)
                MPI_INT,            //data type
                mpi_nbrs[dim][isup],//the rank of source process
                tagrecv,                //tag (should be same as in the send process)
                MPI_COMM_WORLD,     //communicator
                &mpi_reqs_recv_pflag[dim][isup] //error
                );
#endif
    }
  }
  //waiting, unpacking, and putting results back
  else {
    for(isup=0;isup<=1;isup++) {
      //skip if on the physical boundary and the BC is not periodic
      if(is_physical_bc(dim,isup)) continue;
      
      //wait for comminication to complete
      //prims
      MPI_Wait(&mpi_reqs_recv[dim][isup],&mpi_stat_recv[dim][isup]);
      MPI_Wait(&mpi_reqs_send[dim][isup],&mpi_stat_send[dim][isup]);
#if(DOMPIPFLAG)
      //pflags
      MPI_Wait(&mpi_reqs_recv_pflag[dim][isup],&mpi_stat_recv_pflag[dim][isup]);
      MPI_Wait(&mpi_reqs_send_pflag[dim][isup],&mpi_stat_send_pflag[dim][isup]);
#endif
      //unpack the data from mpi recv buffers into prim[]
      pack_prim(ispack, dim, isup, prim, mpi_buf_recv[dim][isup]);
#if(DOMPIPFLAG)
      //pack the data from mpi send buffers into  pflag[]
      pack_pflag(ispack, dim, isup, pflag, mpi_buf_recv_pflag[dim][isup]);
#endif
    }
  }
#endif
}

//do not allow "sucking" on the boundary:
//type = 0: do not allow ucon[1] > 0
//type = 1: do not allow ucon[1] < 0
//if a disallowed value detected, reset vcon[1] to zero
void inflow_check(double *pr, int ii, int jj, int kk, int type )
{
        struct of_geom geom ;
        double ucon[NDIM] ;
        int j,k ;
        double alpha,beta1,gamma,vsq ;

        get_geometry(ii,jj,kk,CENT,&geom) ;
        ucon_calc(pr, &geom, ucon) ;

        if( ((ucon[1] > 0.) && (type==0)) || ((ucon[1] < 0.) && (type==1)) ) { 
                /* find gamma and remove it from primitives */
	  if( gamma_calc(pr,&geom,&gamma) ) { 
	    fflush(stderr);
	    fprintf(stderr,"\ninflow_check(): gamma failure, (%d,%d,%d) \n",
                    ii+mpi_startn[1], jj+mpi_startn[2], kk+mpi_startn[3]);
	    fflush(stderr);
	    fail(FAIL_GAMMA);
	  }
	    pr[U1] /= gamma ;
	    pr[U2] /= gamma ;
	    pr[U3] /= gamma ;
	    alpha = 1./sqrt(-geom.gcon[0][0]) ;
	    beta1 = geom.gcon[0][1]*alpha*alpha ;

	    /* reset radial velocity so radial 4-velocity
	     * is zero */
	    pr[U1] = beta1/alpha ;

	    /* now find new gamma and put it back in */
	    vsq = 0. ;
	    SLOOP vsq += geom.gcov[j][k]*pr[U1+j-1]*pr[U1+k-1] ;
	    if( fabs(vsq) < 1.e-13 )  vsq = 1.e-13;
	    if( vsq >= 1. ) { 
	      vsq = 1. - 1./(GAMMAMAX*GAMMAMAX) ;
	    }
	    gamma = 1./sqrt(1. - vsq) ;
	    pr[U1] *= gamma ;
	    pr[U2] *= gamma ;
	    pr[U3] *= gamma ;

	    /* done */
	  }
	  else
	    return ;

}

//functions that are specific for NS boundary
#if(WHICHPROBLEM == NSSURFACE)

//puts analytical EMFs on inner r boundary
//boundary check performed before call
void adjust_emfs_nssurface(double emf[][N1+D1][N2+D2][N3+D3])
{
#define DOE2 (N1>1 && N3>1)
#define DOE3 (N1>1 && N2>1)
#if (DOE2)
  double X[NDIM];
  double r, th1, th2, phi;
#endif
  int i, j, k;
  //loop on boundary
  ZSLOOP(0,0,0,N2-1+D2,0,N3-1+D3) {
#if (DOE2)
    //calculate the only nonzero component E^2
    coord(i, j, k, CORN, X);
    bl_coord(X, &r, &th1, &phi);
    coord(i, j+1, k, CORN, X);
    bl_coord(X, &r, &th2, &phi);
    emf[2][i][j][k] = OMEGA * (cos(th2)-cos(th1)) / dx[2];
#endif
    //E^3 is zero since we rotate around z-axis
    emf[3][i][j][k] = 0;
  }
}

//calculate velocity parallel to magnetic field
void compute_vpar(double pr[], struct of_geom *geom, double *vpar) {
  double Bccov[NDIM], Bccon[NDIM], vcon[NDIM], ucon[NDIM];
  double Bsq, absB, Bdotv;
  int j;
  Bccon[0] = 0;
  Bccon[1] = pr[B1];
  Bccon[2] = pr[B2];
  Bccon[3] = pr[B3];
  lower(Bccon, geom, Bccov);
  //get relative velocity
  ucon_calc(pr, geom, ucon);
  //get 3-velocity
  DLOOPA vcon[j] = ucon[j]/ucon[TT];
  //B^2 and |B|
  Bsq = SMALL;
  SLOOPA Bsq += Bccon[j] * Bccov[j];
  absB = sqrt(Bsq);
  //B_mu v^mu
  Bdotv = 0;
  SLOOPA Bdotv += Bccov[j] * ucon[j];
  //vpar
  *vpar = Bdotv/absB;
  if (Bccon[1]<0) *vpar *= -1; //positive direction is away from the star
}

//change velocity parallel to magnetic field saving perpendicular component
void set_vpar(double vpar, double gamma_max, struct of_geom *geom, double pr[]) {
  double Bccov[NDIM], Bccon[NPR];
  double vcon[NDIM], ucon[NDIM];
  double Bdotv;
  double vpar_old_vec[NDIM], vperp_old_vec[NDIM];
  double vpar_old;
  double Bsq;
  double absB;
  int j;
  double gamma_perp;
  double vperp_sq, vmax_sq;
  //magnetic field stuff
  Bccon[0] = 0;
  Bccon[1] = pr[B1];
  Bccon[2] = pr[B2];
  Bccon[3] = pr[B3];
  lower(Bccon, geom, Bccov);
  Bsq = SMALL;
  SLOOPA Bsq += Bccon[j] * Bccov[j];
  absB = sqrt(Bsq);
  //get 4-velocity
  ucon_calc(pr, geom, ucon);
  //get 3-velocity
  DLOOPA vcon[j] = ucon[j]/ucon[TT];
  //B_mu v^mu
  Bdotv = 0;
  SLOOPA Bdotv += Bccov[j] * ucon[j];
  //old velocity components
  vpar_old = Bdotv/absB;
  SLOOPA vpar_old_vec[j]  = vpar_old * Bccon[j] / absB;
  SLOOPA vperp_old_vec[j] = vcon[j] - vpar_old_vec[j];
  //gamma factor
  ut_calc_3vel(vperp_old_vec, geom, &gamma_perp);
  //squares of velocity components
  vperp_sq = 1.-1./(gamma_perp*gamma_perp);
  vmax_sq = 1.-1./(gamma_max*gamma_max);
  //check if gamma won't become too large
  if (vmax_sq <= vperp_sq) {
    vpar = 0;
  }
  else if (vperp_sq + vpar*vpar > vmax_sq) {
    vpar = sqrt(vmax_sq - vperp_sq) * copysign(1., vpar);
  }
  //set new 3-velocity
  SLOOPA vcon[j] = vperp_old_vec[j] + vpar * Bccon[j] / absB;
  //convert it to 4-velocity and put into prims
  ut_calc_3vel(vcon, geom, &ucon[TT]);
  SLOOPA ucon[j] = vcon[j]*ucon[TT];
  pr[U1] = ucon[1];
  pr[U2] = ucon[2];
  pr[U3] = ucon[3];
}

//set the velocity in primitives to be purely rotational
void set_omega_stataxi(struct of_geom *geom, double omegaf, double *X, double *pr) {
  double vcon[NDIM], vconp[NDIM], ucon[NDIM];
  double dxdxp[NDIM][NDIM], dxpdx[NDIM][NDIM];
  int j, k;
  //set the 3-velocity in spherical coordinates
  vcon[1] = vcon[2] = 0;
  vcon[3] = omegaf;
  //get jacobian from BL to code coordinates
  dxdxp_func(X, dxdxp);
  invert_matrix(dxdxp, dxpdx);
  //convert 3-velocity to code coordinates
  SLOOPA vconp[j] = 0;
  SLOOP vconp[j] += dxpdx[j][k] * vcon[k];
  //convert it to 4-velocity and put into prims
  ut_calc_3vel(vconp, geom, &ucon[TT]);
  SLOOPA ucon[j] = vconp[j]*ucon[TT];
  pr[U1] = ucon[1];
  pr[U2] = ucon[2];
  pr[U3] = ucon[3];
}

//set density and velocity in one cell
void set_den_vel(double pr[], double rprim[], int dirprim, int i, int j, int k, int ri, int rj, int rk, struct of_geom *ptrgeom, struct of_geom *ptrrgeom)
{
  double rgamma, gammamax;
  double vpar, vpar_have;
  double X[NDIM], rX[NDIM], V[NDIM], rV[NDIM];
  coord(i, j, k, dirprim, X);
  bl_coord_vec(X, V);
  coord(ri, rj, rk, CENT, rX);
  bl_coord_vec(rX, rV);
  gamma_calc(rprim, ptrrgeom, &rgamma);
  compute_vpar(rprim, ptrrgeom, &vpar_have);
  int set_bc = rprim[U1]>0; //force velocity to what we want only if flowing out
  if (set_bc) {
    vpar = VPARWANT;
    gammamax = GAMMAMAX;
  }
  else {
    vpar = vpar_have;
    gammamax = rgamma;
  }
  set_omega_stataxi(ptrgeom, OMEGA, X, pr); //set the velocity to be purely rotational
  set_vpar(vpar, gammamax, ptrgeom, pr); //adjust component parallel to magnetic field
  if (set_bc) {
    double bsq = bsq_calc(pr, ptrgeom);
    pr[RHO] = bsq/BSQORHOBND*pow(rV[1]/V[1],4.);
    pr[UU]  = bsq/BSQOUBND*pow(rV[1]/V[1],4.*gam);
  }
  else {
    pr[RHO] = rprim[RHO]*pow(rV[1]/V[1],4.);
    pr[UU]  = rprim[UU]*pow(rV[1]/V[1],4.*gam);
  }
}

//set hydrodynamic primitive variables on inner r boundary
//for one cell face, we are on physical boundary and i=0
void set_hydro_nssurface(double pr[][N2M][N3M][NPR], double p_l[], double p_r[], int i, int j, int k, struct of_geom *geom)
{
  int m;
  double prface[NPR];
  struct of_geom rgeom;
  get_geometry(i, j, k, CENT, &rgeom); //reference geometry
  PLOOP prface[m] = (p_l[m] + p_r[m])/2;
  set_den_vel(prface, pr[i][j][k], FACE1, i, j, k, i, j, k, geom, &rgeom);
  PLOOP p_l[m] = p_r[m] = prface[m];
}

//set neutron star boundary condition in ghost cells
void bound_x1dn_nssurface(double prim[][N2M][N3M][NPR]) {
  double X[NPR], V[NPR], rV[NPR];
  double dxdp[NDIM][NDIM], rdxdp[NDIM][NDIM];
  double Bur, Bth, Bph, rBur, rBth, rBph;
  struct of_geom geom, rgeom;
  int iNg, j, k;
  for(j=0;j<N2;j++) {
    for(k=0; k<N3;k++) {
#if( RESCALE )
      get_geometry(0,j,k,CENT,&geom) ;
      rescale(prim[0][j][k],FORWARD, 1, 0,j,k,CENT,&geom) ;
#endif
      //for reference physical cell
      coord(0, j, k, CENT, X);
      bl_coord_vec(X, rV);
      dxdxp_func(X, rdxdp);
      get_geometry(0, j, k, CENT, &rgeom);
      //get magnetic field
      rBur = prim[0][j][k][B1] * rdxdp[1][1];
      rBth = prim[0][j][k][B2] * rdxdp[2][2];
      rBph = prim[0][j][k][B3] * rdxdp[3][3];
      
      for (iNg=-N1G; iNg<0; iNg++) {
        //for our cell
        coord(iNg, j, k, CENT, X);
        bl_coord_vec(X, V);
        dxdxp_func(X, dxdp);
        //set radial magnetic field
        Bur = rBur * rV[1]*rV[1]/V[1]/V[1];
        prim[iNg][j][k][B1] = Bur / dxdp[1][1];
        //set other magnetic field components
        Bth = rBth * rV[1]/V[1];
        prim[iNg][j][k][B2] = Bth / dxdp[2][2];
        Bph = rBph * rV[1]/V[1];
        prim[iNg][j][k][B3] = Bph / dxdp[3][3];
        //copy other primitives
        prim[iNg][j][k][RHO] = prim[0][j][k][RHO];
        prim[iNg][j][k][UU] = prim[0][j][k][UU];
        prim[iNg][j][k][U1] = prim[0][j][k][U1];
        prim[iNg][j][k][U2] = prim[0][j][k][U2];
        prim[iNg][j][k][U3] = prim[0][j][k][U3];
        prim[iNg][j][k][KTOT] = prim[0][j][k][KTOT]; //not sure if needed
        //fix density and velocity
        get_geometry(iNg, j, k, CENT, &geom);
        set_den_vel(prim[iNg][j][k], prim[0][j][k], CENT, iNg, j, k, 0, j, k, &geom, &rgeom);
      }
#if( RESCALE )
      for (iNg = -N1G; iNg<=0; iNg++)
      {
        get_geometry(iNg,j,k,CENT,&geom) ;
        rescale(prim[iNg][j][k],REVERSE, 1, iNg,j,k,CENT,&geom) ;
      }
#endif
    }
  }
}

#endif

