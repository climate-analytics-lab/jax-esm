C     thsice_oracle.F -- standalone single-column driver ("oracle") for
C     the unmodified MITgcm pkg/thsice routines THSICE_SOLVE4TEMP and
C     THSICE_CALC_THICKN.  Reads a "key = value" text input file named
C     on the command line, sets up THSICE_PARAMS.h common-block
C     parameters (all overridable from the input file) and the routine
C     arguments (one grid column: sNx=sNy=1), calls the requested
C     routine exactly once, and prints every output argument as
C     "name = <value>" in full double-precision (ES25.17).
C
C     Usage:  thsice_oracle  <input-file>
C
C     Input file format: one "key = value" per line; blank lines and
C     lines starting with '#' or '!' are ignored.  Key matching is
C     case-insensitive.  One key selects the routine to call:
C       routine = solve4temp   (default)
C       routine = calc_thickn
C     All other recognised keys are either THSICE_PARAMS.h parameters
C     (see THSICE_PARAMS.h / thsice_readparms.F) or routine arguments,
C     named after the Fortran dummy-argument names (case-insensitive).
C     Unrecognised keys are silently ignored.

      PROGRAM THSICE_ORACLE
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "THSICE_PARAMS.h"

C--   key/value store read from the input file
      INTEGER MAXKV
      PARAMETER ( MAXKV = 400 )
      CHARACTER*64  kvKey(MAXKV)
      CHARACTER*256 kvVal(MAXKV)
      INTEGER nkv

      CHARACTER*512 infile
      CHARACTER*32  routine
      INTEGER ii, jj, nargs

C--   routine arguments (single column, halo 0:2 to match
C     (1-OLx:sNx+OLx,1-OLy:sNy+OLy) = (0:2,0:2) with sNx=sNy=OLx=OLy=1)
      _RL icMask(0:2,0:2), hIce(0:2,0:2), hSnow(0:2,0:2)
      _RL tFrz(0:2,0:2), flxSW(0:2,0:2), tSrf(0:2,0:2)
      _RL qIc1(0:2,0:2), qIc2(0:2,0:2)
      _RL tIc1(0:2,0:2), tIc2(0:2,0:2)
      _RL sHeat(0:2,0:2), flxCnB(0:2,0:2)
      _RL flxAtm(0:2,0:2), evpAtm(0:2,0:2)
      _RL flxExSW(1:1,1:1,0:2)
      _RL dTsrf(1:1,1:1)
      LOGICAL useBulkForce, useEXF

      _RL iceMaskA(0:2,0:2), tOce(0:2,0:2), v2oc(0:2,0:2)
      _RL snowP(0:2,0:2), prcAtm(0:2,0:2)
      _RL icFrac(0:2,0:2), hSnow1(0:2,0:2)
      _RL frwAtm(0:2,0:2), fzMlOc(0:2,0:2), flx2oc(0:2,0:2)
      _RL frw2oc(0:2,0:2), fsalt(0:2,0:2), frzSeaWat(0:2,0:2)

      INTEGER bi, bj, iMin, iMax, jMin, jMax
      LOGICAL dBugFlag
      _RL myTime, dtDefault
      INTEGER myIter, myThid

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
C--   read the input file into the key/value store

      nargs = COMMAND_ARGUMENT_COUNT()
      IF ( nargs.LT.1 ) THEN
        WRITE(0,'(A)') 'usage: thsice_oracle <input-file>'
        STOP 1
      ENDIF
      CALL GET_COMMAND_ARGUMENT(1, infile)
      CALL READ_KV(infile, kvKey, kvVal, nkv, MAXKV)

      routine = 'solve4temp'
      DO ii = 1, nkv
        IF ( SAMEKEY(kvKey(ii),'routine') ) routine = kvVal(ii)
      ENDDO
      CALL LCASE(routine)

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
C--   THSICE_PARAMS.h : physical constants (COMMON/THSICE_PHYSPAR_R/),
C     all overridable; defaults as specified in the task/README.

      rhos      = GETR(kvKey,kvVal,nkv,'rhos',330.0D0)
      rhoi      = GETR(kvKey,kvVal,nkv,'rhoi',905.0D0)
      rhosw     = GETR(kvKey,kvVal,nkv,'rhosw',1026.0D0)
      rhofw     = GETR(kvKey,kvVal,nkv,'rhofw',1000.0D0)
      floodFac  = GETR(kvKey,kvVal,nkv,'floodfac',(rhosw-rhoi)/rhos)
      cpIce     = GETR(kvKey,kvVal,nkv,'cpice',2100.0D0)
      cpWater   = GETR(kvKey,kvVal,nkv,'cpwater',3990.0D0)
      kIce      = GETR(kvKey,kvVal,nkv,'kice',2.03D0)
      kSnow     = GETR(kvKey,kvVal,nkv,'ksnow',0.31D0)
      bMeltCoef = GETR(kvKey,kvVal,nkv,'bmeltcoef',0.006D0)
      Lfresh    = GETR(kvKey,kvVal,nkv,'lfresh',3.34D5)
      qsnow     = GETR(kvKey,kvVal,nkv,'qsnow',Lfresh)
      albColdSnow = GETR(kvKey,kvVal,nkv,'albcoldsnow',0.85D0)
      albWarmSnow = GETR(kvKey,kvVal,nkv,'albwarmsnow',0.70D0)
      tempSnowAlb = GETR(kvKey,kvVal,nkv,'tempsnowalb',-10.0D0)
      albOldSnow  = GETR(kvKey,kvVal,nkv,'alboldsnow',0.55D0)
      hNewSnowAge = GETR(kvKey,kvVal,nkv,'hnewsnowage',0.02D0)
      snowAgTime  = GETR(kvKey,kvVal,nkv,'snowagtime',50.0D0*86400.0D0)
      albIceMax   = GETR(kvKey,kvVal,nkv,'albicemax',0.65D0)
      albIceMin   = GETR(kvKey,kvVal,nkv,'albicemin',0.20D0)
      hAlbIce     = GETR(kvKey,kvVal,nkv,'halbice',0.5D0)
      hAlbSnow    = GETR(kvKey,kvVal,nkv,'halbsnow',0.3D0)
      i0swFrac    = GETR(kvKey,kvVal,nkv,'i0swfrac',0.3D0)
      ksolar      = GETR(kvKey,kvVal,nkv,'ksolar',1.5D0)
      dhSnowLin   = GETR(kvKey,kvVal,nkv,'dhsnowlin',0.0D0)
      saltIce     = GETR(kvKey,kvVal,nkv,'saltice',4.0D0)
      S_winton    = GETR(kvKey,kvVal,nkv,'s_winton',1.0D0)
      mu_Tf       = GETR(kvKey,kvVal,nkv,'mu_tf',0.054D0)
      Tf0kel      = GETR(kvKey,kvVal,nkv,'tf0kel',273.15D0)
      Tmlt1       = GETR(kvKey,kvVal,nkv,'tmlt1',-mu_Tf*S_winton)
      Terrmax     = GETR(kvKey,kvVal,nkv,'terrmax',5.0D-1)
      hIceMin     = GETR(kvKey,kvVal,nkv,'hicemin',0.01D0)
      hiMax       = GETR(kvKey,kvVal,nkv,'himax',10.0D0)
      hsMax       = GETR(kvKey,kvVal,nkv,'hsmax',10.0D0)
      iceMaskMax  = GETR(kvKey,kvVal,nkv,'icemaskmax',1.0D0)
      iceMaskMin  = GETR(kvKey,kvVal,nkv,'icemaskmin',0.1D0)
      fracEnMelt  = GETR(kvKey,kvVal,nkv,'fracenmelt',0.0D0)
      fracEnFreez = GETR(kvKey,kvVal,nkv,'fracenfreez',0.0D0)
      hThinIce    = GETR(kvKey,kvVal,nkv,'hthinice',0.2D0)
      hThickIce   = GETR(kvKey,kvVal,nkv,'hthickice',2.5D0)
      hNewIceMax  = GETR(kvKey,kvVal,nkv,'hnewicemax',hiMax)

C--   THSICE_PAR_L : logicals (none of these affect the two routines
C     exposed here; set to safe/inert defaults, overridable anyway)
      stepFwd_oceMxL   = GETL(kvKey,kvVal,nkv,'stepfwd_ocemxl',.FALSE.)
      thSIce_skipThermo  = GETL(kvKey,kvVal,nkv,'thsice_skipthermo',
     &                           .FALSE.)
      thSIce_calc_albNIR = GETL(kvKey,kvVal,nkv,'thsice_calc_albnir',
     &                           .FALSE.)
      thSIce_snapshot_mdsio = .FALSE.
      thSIce_mon_stdio       = .FALSE.
      thSIce_snapshot_mnc    = .FALSE.
      thSIce_mon_mnc         = .FALSE.
      thSIce_pickup_read_mnc = .FALSE.
      thSIce_pickup_write_mdsio = .FALSE.
      thSIce_pickup_write_mnc   = .FALSE.

C--   THSICE_PAR_I : integers
      startIceModel   = GETI(kvKey,kvVal,nkv,'starticemodel',0)
      nitMaxTsf        = GETI(kvKey,kvVal,nkv,'nitmaxtsf',20)
      thSIceAdvScheme  = GETI(kvKey,kvVal,nkv,'thsiceadvscheme',0)
      thSIceBalanceAtmFW=GETI(kvKey,kvVal,nkv,'thsicebalanceatmfw',0)

C--   THSICE_PAR_R : timesteps and mixed-layer/misc reals
C     "dt" is a convenience key that sets all three model timesteps
C     at once; each can still be overridden individually.
      dtDefault = GETR(kvKey,kvVal,nkv,'dt',21600.0D0)
      thSIce_deltaT = GETR(kvKey,kvVal,nkv,'thsice_deltat',dtDefault)
      thSIce_dtTemp = GETR(kvKey,kvVal,nkv,'thsice_dttemp',dtDefault)
      ocean_deltaT  = GETR(kvKey,kvVal,nkv,'ocean_deltat',dtDefault)
      tauRelax_MxL      = GETR(kvKey,kvVal,nkv,'taurelax_mxl',0.0D0)
      tauRelax_MxL_salt = GETR(kvKey,kvVal,nkv,'taurelax_mxl_salt',
     &                          0.0D0)
      hMxL_default  = GETR(kvKey,kvVal,nkv,'hmxl_default',50.0D0)
      sMxL_default  = GETR(kvKey,kvVal,nkv,'smxl_default',35.0D0)
      vMxL_default  = GETR(kvKey,kvVal,nkv,'vmxl_default',0.05D0)
      thSIce_diffK  = GETR(kvKey,kvVal,nkv,'thsice_diffk',0.0D0)
      stressReduction = GETR(kvKey,kvVal,nkv,'stressreduction',1.0D0)
      thSIce_diagFreq = GETR(kvKey,kvVal,nkv,'thsice_diagfreq',0.0D0)
      thSIce_monFreq  = GETR(kvKey,kvVal,nkv,'thsice_monfreq',0.0D0)

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
C--   common routine arguments

      bi = GETI(kvKey,kvVal,nkv,'bi',1)
      bj = GETI(kvKey,kvVal,nkv,'bj',1)
      iMin = 1
      iMax = 1
      jMin = 1
      jMax = 1
      dBugFlag = .FALSE.
      myTime = GETR(kvKey,kvVal,nkv,'mytime',0.0D0)
      myIter = GETI(kvKey,kvVal,nkv,'myiter',0)
      myThid = GETI(kvKey,kvVal,nkv,'mythid',1)

      IF ( routine.EQ.'solve4temp' ) THEN

C--    THSICE_SOLVE4TEMP inputs (see thsice_solve4temp.F header):
C      icMask  : sea-ice fractional mask [0-1]
C      hIce    : ice height [m]
C      hSnow   : snow height [m]
C      tFrz    : sea-water freezing temperature [oC]
C      flxExSW(0) = net (minus SW) surf. heat flux (+=down) [W/m2]
C                   evaluated at the melting surface temperature (0 oC)
C      flxExSW(1) = same, evaluated at the current surface temp tSrf
C      flxExSW(2) = d(flxExSW(1))/d(tSrf)                    [W/m2/K]
C      flxSW   : net Short-Wave flux (+=down) at the surface [W/m2]
C                (input; overwritten on output with the SW flux passed
C                 through the ice to the ocean)
C      tSrf    : surface (ice/snow) temperature [oC]           (in/out)
C      qIc1,qIc2 : ice-layer enthalpy [J/kg], thsice convention (in/out)
        icMask(1,1) = GETR(kvKey,kvVal,nkv,'icmask',1.0D0)
        hIce(1,1)   = GETR(kvKey,kvVal,nkv,'hice',1.0D0)
        hSnow(1,1)  = GETR(kvKey,kvVal,nkv,'hsnow',0.0D0)
        tFrz(1,1)   = GETR(kvKey,kvVal,nkv,'tfrz',-1.8D0)
        flxExSW(1,1,0) = GETR(kvKey,kvVal,nkv,'flxexsw0',0.0D0)
        flxExSW(1,1,1) = GETR(kvKey,kvVal,nkv,'flxexsw1',0.0D0)
        flxExSW(1,1,2) = GETR(kvKey,kvVal,nkv,'flxexsw2',0.0D0)
        flxSW(1,1)  = GETR(kvKey,kvVal,nkv,'flxsw',0.0D0)
        tSrf(1,1)   = GETR(kvKey,kvVal,nkv,'tsrf',-10.0D0)
        qIc1(1,1)   = GETR(kvKey,kvVal,nkv,'qic1',-3.0D5)
        qIc2(1,1)   = GETR(kvKey,kvVal,nkv,'qic2',-3.2D5)
        useBulkForce = .FALSE.
        useEXF       = .FALSE.

        CALL THSICE_SOLVE4TEMP(
     I       bi, bj,
     I       iMin, iMax, jMin, jMax, dBugFlag,
     I       useBulkForce, useEXF,
     I       icMask, hIce, hSnow, tFrz, flxExSW,
     U       flxSW, tSrf, qIc1, qIc2,
     O       tIc1, tIc2, dTsrf,
     O       sHeat, flxCnB, flxAtm, evpAtm,
     I       myTime, myIter, myThid )

        WRITE(6,'(A,ES25.17)') 'flxSW = ',  flxSW(1,1)
        WRITE(6,'(A,ES25.17)') 'tSrf = ',   tSrf(1,1)
        WRITE(6,'(A,ES25.17)') 'qIc1 = ',   qIc1(1,1)
        WRITE(6,'(A,ES25.17)') 'qIc2 = ',   qIc2(1,1)
        WRITE(6,'(A,ES25.17)') 'tIc1 = ',   tIc1(1,1)
        WRITE(6,'(A,ES25.17)') 'tIc2 = ',   tIc2(1,1)
        WRITE(6,'(A,ES25.17)') 'dTsrf = ',  dTsrf(1,1)
        WRITE(6,'(A,ES25.17)') 'sHeat = ',  sHeat(1,1)
        WRITE(6,'(A,ES25.17)') 'flxCnB = ', flxCnB(1,1)
        WRITE(6,'(A,ES25.17)') 'flxAtm = ', flxAtm(1,1)
        WRITE(6,'(A,ES25.17)') 'evpAtm = ', evpAtm(1,1)

      ELSEIF ( routine.EQ.'calc_thickn' ) THEN

C--    THSICE_CALC_THICKN inputs (see thsice_calc_thickn.F header):
C      iceMask : sea-ice fractional mask [0-1]
C      tFrz    : sea-water freezing temperature [oC]
C      tOce    : surface level oceanic temperature [oC]
C      v2oc    : square of ocean surface-level velocity [m2/s2]
C      snowP   : snow precipitation [kg/m2/s]
C      prcAtm  : total precip from the atmosphere [kg/m2/s]
C      sHeat   : surf heating flux left to melt snow or ice [W/m2]
C                (= Atmos-conduction; output of THSICE_SOLVE4TEMP)
C      flxCnB  : heat flux conducted through the ice to bottom [W/m2]
C                (output of THSICE_SOLVE4TEMP)
C      icFrac,hIce,hSnow1,tSrf,qIc1,qIc2 : sea-ice state          (in/out)
C      frwAtm  : evaporation to the atmosphere [kg/m2/s], >0 evap (in/out)
C      fzMlOc  : ocean mixed-layer freezing/melting potential [W/m2]
C                = (tFrzOce-tOceMxL)*cpWater*rhosw*hOceMxL/dt      (in/out)
C      flx2oc  : net heat flux to ocean [W/m2] (>0 downward)      (in/out)
        iceMaskA(1,1)= GETR(kvKey,kvVal,nkv,'icemask',1.0D0)
        tFrz(1,1)    = GETR(kvKey,kvVal,nkv,'tfrz',-1.8D0)
        tOce(1,1)    = GETR(kvKey,kvVal,nkv,'toce',-1.8D0)
        v2oc(1,1)    = GETR(kvKey,kvVal,nkv,'v2oc',0.0D0)
        snowP(1,1)   = GETR(kvKey,kvVal,nkv,'snowp',0.0D0)
        prcAtm(1,1)  = GETR(kvKey,kvVal,nkv,'prcatm',0.0D0)
        sHeat(1,1)   = GETR(kvKey,kvVal,nkv,'sheat',0.0D0)
        flxCnB(1,1)  = GETR(kvKey,kvVal,nkv,'flxcnb',0.0D0)
        icFrac(1,1)  = GETR(kvKey,kvVal,nkv,'icfrac',1.0D0)
        hIce(1,1)    = GETR(kvKey,kvVal,nkv,'hice',1.0D0)
        hSnow1(1,1)  = GETR(kvKey,kvVal,nkv,'hsnow1',0.0D0)
        tSrf(1,1)    = GETR(kvKey,kvVal,nkv,'tsrf',-10.0D0)
        qIc1(1,1)    = GETR(kvKey,kvVal,nkv,'qic1',-3.0D5)
        qIc2(1,1)    = GETR(kvKey,kvVal,nkv,'qic2',-3.2D5)
        frwAtm(1,1)  = GETR(kvKey,kvVal,nkv,'frwatm',0.0D0)
        fzMlOc(1,1)  = GETR(kvKey,kvVal,nkv,'fzmloc',0.0D0)
        flx2oc(1,1)  = GETR(kvKey,kvVal,nkv,'flx2oc',0.0D0)

      DO jj = 0, 2
       DO ii = 0, 2
        frw2oc(ii,jj) = 0.D0
        fsalt(ii,jj) = 0.D0
        frzSeaWat(ii,jj) = 0.D0
       ENDDO
      ENDDO
        CALL THSICE_CALC_THICKN(
     I       bi, bj,
     I       iMin, iMax, jMin, jMax, dBugFlag,
     I       iceMaskA, tFrz, tOce, v2oc,
     I       snowP, prcAtm, sHeat, flxCnB,
     U       icFrac, hIce, hSnow1, tSrf, qIc1, qIc2,
     U       frwAtm, fzMlOc, flx2oc,
     O       frw2oc, fsalt, frzSeaWat,
     I       myTime, myIter, myThid )

        WRITE(6,'(A,ES25.17)') 'icFrac = ',    icFrac(1,1)
        WRITE(6,'(A,ES25.17)') 'hIce = ',      hIce(1,1)
        WRITE(6,'(A,ES25.17)') 'hSnow1 = ',    hSnow1(1,1)
        WRITE(6,'(A,ES25.17)') 'tSrf = ',      tSrf(1,1)
        WRITE(6,'(A,ES25.17)') 'qIc1 = ',      qIc1(1,1)
        WRITE(6,'(A,ES25.17)') 'qIc2 = ',      qIc2(1,1)
        WRITE(6,'(A,ES25.17)') 'frwAtm = ',    frwAtm(1,1)
        WRITE(6,'(A,ES25.17)') 'fzMlOc = ',    fzMlOc(1,1)
        WRITE(6,'(A,ES25.17)') 'flx2oc = ',    flx2oc(1,1)
        WRITE(6,'(A,ES25.17)') 'frw2oc = ',    frw2oc(1,1)
        WRITE(6,'(A,ES25.17)') 'fsalt = ',     fsalt(1,1)
        WRITE(6,'(A,ES25.17)') 'frzSeaWat = ', frzSeaWat(1,1)

      ELSE
        WRITE(0,'(A,A)') 'thsice_oracle: unknown routine = ', routine
        STOP 1
      ENDIF

      STOP

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|

      CONTAINS

      SUBROUTINE READ_KV(fname, oKey, oVal, oN, mkv)
      IMPLICIT NONE
      CHARACTER*(*) fname
      INTEGER mkv
      CHARACTER*64  oKey(mkv)
      CHARACTER*256 oVal(mkv)
      INTEGER oN
      CHARACTER*512 line
      INTEGER ios, ieq
      OPEN(UNIT=11,FILE=fname,STATUS='OLD',ACTION='READ',IOSTAT=ios)
      IF ( ios.NE.0 ) THEN
        WRITE(0,'(A,A)') 'thsice_oracle: cannot open input file: ',
     &                    TRIM(fname)
        STOP 1
      ENDIF
      oN = 0
      DO
        READ(11,'(A)',IOSTAT=ios) line
        IF ( ios.NE.0 ) EXIT
        line = ADJUSTL(line)
        IF ( LEN_TRIM(line).EQ.0 ) CYCLE
        IF ( line(1:1).EQ.'#' .OR. line(1:1).EQ.'!' ) CYCLE
        ieq = INDEX(line,'=')
        IF ( ieq.LE.1 ) CYCLE
        oN = oN + 1
        IF ( oN.GT.mkv ) STOP 'thsice_oracle: too many input keys'
        oKey(oN) = ADJUSTL(line(1:ieq-1))
        oVal(oN) = ADJUSTL(line(ieq+1:))
      ENDDO
      CLOSE(11)
      END SUBROUTINE READ_KV

      SUBROUTINE LCASE(s)
      IMPLICIT NONE
      CHARACTER*(*) s
      INTEGER kk, ic
      DO kk = 1, LEN(s)
        ic = ICHAR(s(kk:kk))
        IF ( ic.GE.65 .AND. ic.LE.90 ) s(kk:kk) = CHAR(ic+32)
      ENDDO
      END SUBROUTINE LCASE

      LOGICAL FUNCTION SAMEKEY(a,b)
      IMPLICIT NONE
      CHARACTER*(*) a, b
      CHARACTER*64 aa, bb
      aa = a
      bb = b
      CALL LCASE(aa)
      CALL LCASE(bb)
      SAMEKEY = TRIM(ADJUSTL(aa)).EQ.TRIM(ADJUSTL(bb))
      END FUNCTION SAMEKEY

      _RL FUNCTION GETR(oKey,oVal,oN,key,defVal)
      IMPLICIT NONE
      CHARACTER*(*) oKey(*), oVal(*)
      INTEGER oN
      CHARACTER*(*) key
      _RL defVal
      INTEGER kk
      GETR = defVal
      DO kk = 1, oN
        IF ( SAMEKEY(oKey(kk),key) ) READ(oVal(kk),*) GETR
      ENDDO
      END FUNCTION GETR

      INTEGER FUNCTION GETI(oKey,oVal,oN,key,defVal)
      IMPLICIT NONE
      CHARACTER*(*) oKey(*), oVal(*)
      INTEGER oN
      CHARACTER*(*) key
      INTEGER defVal
      INTEGER kk
      GETI = defVal
      DO kk = 1, oN
        IF ( SAMEKEY(oKey(kk),key) ) READ(oVal(kk),*) GETI
      ENDDO
      END FUNCTION GETI

      LOGICAL FUNCTION GETL(oKey,oVal,oN,key,defVal)
      IMPLICIT NONE
      CHARACTER*(*) oKey(*), oVal(*)
      INTEGER oN
      CHARACTER*(*) key
      LOGICAL defVal
      INTEGER kk
      CHARACTER*8 sval
      GETL = defVal
      DO kk = 1, oN
        IF ( SAMEKEY(oKey(kk),key) ) THEN
          sval = ADJUSTL(oVal(kk))
          CALL LCASE(sval)
          IF ( sval(1:1).EQ.'t' .OR. sval(1:1).EQ.'1' ) THEN
            GETL = .TRUE.
          ELSEIF ( sval(1:1).EQ.'f' .OR. sval(1:1).EQ.'0' ) THEN
            GETL = .FALSE.
          ENDIF
        ENDIF
      ENDDO
      END FUNCTION GETL

      END PROGRAM THSICE_ORACLE
