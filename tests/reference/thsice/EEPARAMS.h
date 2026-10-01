C     EEPARAMS.h -- minimal stub for the thsice standalone oracle.
C     Only the handful of names actually referenced by
C     thsice_solve4temp.F / thsice_calc_thickn.F / THSICE_PARAMS.h
C     are provided: MAX_LEN_MBUF, MAX_LEN_FNAM (character-buffer
C     sizes) and standardMessageUnit/errorMessageUnit (Fortran IO
C     unit numbers, normally set at runtime by the execution
C     environment; here just wired to stdout/stderr).

      INTEGER MAX_LEN_MBUF
      PARAMETER ( MAX_LEN_MBUF = 512 )
      INTEGER MAX_LEN_FNAM
      PARAMETER ( MAX_LEN_FNAM = 512 )
      INTEGER standardMessageUnit
      PARAMETER ( standardMessageUnit = 6 )
      INTEGER errorMessageUnit
      PARAMETER ( errorMessageUnit = 0 )
