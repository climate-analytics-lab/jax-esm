from pathlib import Path
import jcm

# Prepare boundary file
boundary_file = (Path(jcm.__file__).parent / "data/bc/boundaries_daily_t31.nc").resolve()

if not boundary_file.exists():
    print("Boundary file %s does not exist. Need to produce it." % (str(boundary_file),))
    import subprocess, sys
    interpolation_file = boundary_file.parent / "interpolate.py"
    subprocess.run([sys.executable, str(interpolation_file), "31"], check=True)

if boundary_file.exists():
    print("Boundary file %s exists!" % (str(boundary_file), ) )
else:
    raise Exception("Something went wrong. The daily file is not generated. Please check.")
