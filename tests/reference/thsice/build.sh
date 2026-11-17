#!/bin/sh
# Build the thsice single-column oracle: the UNMODIFIED MITgcm pkg/thsice routines THSICE_SOLVE4TEMP and
# THSICE_CALC_THICKN, fetched from GitHub at a pinned commit, plus this directory's driver and minimal stub headers.
# MITgcm is MIT-licensed (Copyright (c) 2018 MITgcm Developers and Contributors); see ../../..//jem/components/slab/
# winton_seaice_model/NOTICE. Usage: [FC=gfortran] ./build.sh   -> ./thsice_oracle
set -e
FC=${FC:-gfortran}
MITGCM_COMMIT=${MITGCM_COMMIT:-d861cd501f21303825de860eb3caa0a8a7ae22f8}   # master, 2026-08; thsice unchanged since 4c7e765
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
mkdir -p upstream
for f in thsice_solve4temp.F thsice_calc_thickn.F THSICE_PARAMS.h THSICE_SIZE.h; do
  [ -s "upstream/$f" ] || curl -sfL "https://raw.githubusercontent.com/MITgcm/MITgcm/$MITGCM_COMMIT/pkg/thsice/$f" -o "upstream/$f"
  grep -qiE 'SUBROUTINE|COMMON|PARAMETER' "upstream/$f" || { echo "upstream/$f does not look like MITgcm Fortran (download failed?)" >&2; rm -f "upstream/$f"; exit 1; }
done
CPPFLAGS="-DALLOW_THSICE -I$HERE -I$HERE/upstream"
FFLAGS="-ffixed-line-length-132 -O0 -g"
# MITgcm writes double literals as "1. _d 0"; genmake2 rewrites them to "1.D0" after preprocessing (tools/set64bitConst.sh)
fix_d_const () { "$FC" -cpp $CPPFLAGS -P -E -ffixed-line-length-132 "$1" | sed -E 's/ *_d[[:space:]]+/D/g' > "$2"; }
fix_d_const upstream/thsice_solve4temp.F  thsice_solve4temp.f
fix_d_const upstream/thsice_calc_thickn.F thsice_calc_thickn.f
"$FC" $FFLAGS -c thsice_solve4temp.f -o thsice_solve4temp.o
"$FC" $FFLAGS -c thsice_calc_thickn.f -o thsice_calc_thickn.o
"$FC" -cpp $CPPFLAGS $FFLAGS -c thsice_stubs.F -o thsice_stubs.o
"$FC" -cpp $CPPFLAGS $FFLAGS -c thsice_oracle.F -o thsice_oracle.o
"$FC" -o thsice_oracle thsice_oracle.o thsice_solve4temp.o thsice_calc_thickn.o thsice_stubs.o
echo "Built: $HERE/thsice_oracle (MITgcm $MITGCM_COMMIT)"
