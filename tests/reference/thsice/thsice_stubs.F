C     thsice_stubs.F -- stand-ins for external routines that
C     thsice_solve4temp.F / thsice_calc_thickn.F call (as plain
C     Fortran CALLs, not behind #ifdef) but that are never actually
C     exercised by the oracle driver:
C
C     - PRINT_ERROR: only reached if hIce<hIceMin for an icy grid
C       point (a case the driver avoids); just echoes the message.
C     - THSICE_GET_EXF / THSICE_GET_BULKF: only reached when
C       useEXF/useBulkForce are .TRUE.; the driver always calls
C       THSICE_SOLVE4TEMP with both .FALSE. (surface flux supplied
C       via flxExSW instead), so these bodies are dead code kept
C       only to satisfy the linker.

      SUBROUTINE PRINT_ERROR( msgBuf, myThid )
      IMPLICIT NONE
#include "EEPARAMS.h"
      CHARACTER*(MAX_LEN_MBUF) msgBuf
      INTEGER myThid
      WRITE(0,'(A)') TRIM(msgBuf)
      RETURN
      END

      SUBROUTINE THSICE_GET_EXF(
     I     bi, bj, k,
     I     iMin, iMax, jMin, jMax,
     I     iceFlag, hSnow, tSrfLoc,
     O     flx0exSW, dFlxdT, evap_0, dEvdT,
     I     myTime, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
      INTEGER bi, bj, k
      INTEGER iMin, iMax, jMin, jMax
      _RL iceFlag (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL hSnow   (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL tSrfLoc (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL flx0exSW(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dFlxdT  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL evap_0  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dEvdT   (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL myTime
      INTEGER myIter, myThid
      STOP 'ABNORMAL END: THSICE_GET_EXF stub called (useEXF not sup
     &ported by the oracle)'
      END

      SUBROUTINE THSICE_GET_BULKF(
     I     bi, bj,
     I     iMin, iMax, jMin, jMax,
     I     iceFlag, hSnow, tSrfLoc,
     O     flx0exSW, dFlxdT, evap_0, dEvdT,
     I     myTime, myIter, myThid )
      IMPLICIT NONE
#include "SIZE.h"
      INTEGER bi, bj
      INTEGER iMin, iMax, jMin, jMax
      _RL iceFlag (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL hSnow   (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL tSrfLoc (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL flx0exSW(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dFlxdT  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL evap_0  (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dEvdT   (1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL myTime
      INTEGER myIter, myThid
      STOP 'ABNORMAL END: THSICE_GET_BULKF stub called (useBulkForce
     & not supported by the oracle)'
      END
