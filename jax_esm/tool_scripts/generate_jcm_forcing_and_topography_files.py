from pathlib import Path
import jcm

def generate_jcm_forcing_and_topography_files(resolution:int=31):

    # Prepare boundary file
    files_to_check = dict(
        terrain = (Path(jcm.__file__).parent / f"data/bc/terrain_t{resolution:d}.nc").resolve(),
        forcing = (Path(jcm.__file__).parent / f"data/bc/forcing_t{resolution:d}.nc").resolve(),
    )
        
    interpolation_code = (Path(jcm.__file__).parent / "data/bc/interpolate.py").resolve()

    def get_files_exist(file_dict):
        return [Path(file).exists() for _, file in file_dict.items()]

    for _, file_name in files_to_check.items():
        print(f"Check file: {str(file_name)}")

    if not all(get_files_exist(files_to_check)):
        print("Some files do not exist. Need to produce it.")

        import subprocess
        import sys
        try:
            result = subprocess.run([sys.executable, str(interpolation_code), f"{resolution:d}"], check=True, capture_output=True, text=True,)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error output:", e.stderr)
    if all(get_files_exist(files_to_check)):
        print("All files exist!")
    else:
        raise Exception("Something went wrong. The daily file is not generated. Please check.")

    return files_to_check


if __name__ == "__main__":
    generate_jcm_forcing_and_topography_files() 
