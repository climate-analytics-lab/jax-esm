C     THSICE_OPTIONS.h -- minimal stub for the thsice standalone oracle.
C     ALLOW_THSICE is supplied on the compiler command line (-DALLOW_THSICE,
C     see build.sh). THSICE_FRACEN_POWERLAW is left defined to match the
C     upstream package default (real MITgcm/pkg/thsice/THSICE_OPTIONS.h
C     defines it unconditionally) so thsice_calc_thickn.F takes the same
C     code path as the production model. ALLOW_DBUG_THSICE is left
C     undefined per the task spec (avoids needing THSICE_DEBUG.h and
C     PRINT_ERROR/dBug plumbing for single-point debug prints).

#ifndef THSICE_OPTIONS_H
#define THSICE_OPTIONS_H

#ifdef ALLOW_THSICE
#define THSICE_FRACEN_POWERLAW
#undef  ALLOW_DBUG_THSICE
#undef  CHECK_ENERGY_CONSERV
#undef  THSICE_REGULARIZE_CALC_THICKN
#endif /* ALLOW_THSICE */

#endif /* THSICE_OPTIONS_H */
