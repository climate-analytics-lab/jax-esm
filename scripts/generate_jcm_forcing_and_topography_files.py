from pathlib import Path
import jcm

# Prepare boundary file
files_to_check = [
    (Path(jcm.__file__).parent / "data/bc/terrain_t31.nc").resolve(),
    (Path(jcm.__file__).parent / "data/bc/forcing_t31.nc").resolve(),
]
    
interpolation_code = (Path(jcm.__file__).parent / "data/bc/interpolate.py").resolve()

def get_files_exist(file_names):
    return [Path(file).exists() for file in file_names]

for file_name in files_to_check:
    print(f"Check file: {str(file_name)}")


if not all(get_files_exist(files_to_check)):
    print("Some files do not exist. Need to produce it.")

    import subprocess, sys
    try:
        result = subprocess.run([sys.executable, str(interpolation_code), "31"], check=True, capture_output=True, text=True,)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)
if all(get_files_exist(files_to_check)):
    print("All files exist!")
else:
    raise Exception("Something went wrong. The daily file is not generated. Please check.")
