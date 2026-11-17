C     SIZE.h -- minimal stub for the thsice standalone oracle.
C     Real MITgcm SIZE.h carries many more tile/grid parameters; the two
C     thsice routines used here (THSICE_SOLVE4TEMP, THSICE_CALC_THICKN)
C     only reference sNx,sNy,OLx,OLy (array bounds) and nSx,nSy (only
C     under ALLOW_AUTODIFF_TAMC, which we do not compile). One-point
C     column, no overlap, single tile/process.
C
C     Also defines the _RL/_RS real-kind macros normally pulled in
C     (indirectly) via CPP_EEMACROS.h -- MITgcm's default build has
C     REAL4_IS_SLOW defined, so both _RL and _RS are Real*8.

#ifndef _RL
#define _RL Real*8
#endif
#ifndef _RS
#define _RS Real*8
#endif

      INTEGER sNx, sNy
      INTEGER OLx, OLy
      INTEGER nSx, nSy
      INTEGER nPx, nPy
      INTEGER Nr
      PARAMETER ( sNx = 1, sNy = 1 )
      PARAMETER ( OLx = 1, OLy = 1 )
      PARAMETER ( nSx = 1, nSy = 1 )
      PARAMETER ( nPx = 1, nPy = 1 )
      PARAMETER ( Nr  = 1 )
