from typing import Optional, Dict
import os
import shutil
import subprocess
import sys
from pathlib import Path

def generate_jcm_forcing_and_topography_files(
    resolution: int = 31,
    data_directory: Optional[Path] = None,
) -> Dict[str, Path]:

    import jcm

    if not (isinstance(data_directory, Path) or data_directory is None):
        raise TypeError("`data_directory` must be of type `Path` or `None`.")

    if data_directory is None:
        home_data_directory = os.environ.get("HOME", None)

        if home_data_directory is None:
            data_directory = Path.cwd()
        else:
            data_directory = Path(home_data_directory)

        data_directory = data_directory / ".cache/jcm"

        print(f'Using input data directory: "{str(data_directory)}".')

    raw_data_directory = Path(jcm.__file__).parent / "data/bc"

    # Prepare boundary file
    files_to_check = dict(
        terrain=(data_directory / f"terrain_t{resolution:d}.nc").resolve(),
        forcing=(data_directory / f"forcing_t{resolution:d}.nc").resolve(),
    )

    def check_if_file_exist(
        file_dict: Dict[str, Path], verbose: bool = True
    ) -> Dict[Path, bool]:

        file_status = {file: Path(file).exists() for _, file in file_dict.items()}

        if verbose:
            for file, result in file_status.items():
                print(
                    f"Check file: {str(file):s}...",
                    "found." if result else "not found.",
                )

        return file_status

    file_status = check_if_file_exist(files_to_check)

    if not all(list(file_status.values())):
        print("Some files are missing. Need to generate them.")

        data_directory.mkdir(parents=True, exist_ok=True)
        interpolation_code = (raw_data_directory / "interpolate.py").resolve()

        try:
            subprocess.run(
                [sys.executable, str(interpolation_code), f"{resolution:d}"],
                check=True,
                capture_output=True,
                text=True,
                cwd=data_directory,
            )
        except subprocess.CalledProcessError as e:
            print("Error output:", e.stderr)

        for destination_file in files_to_check.values():
            source_file = Path(raw_data_directory / destination_file.name)
            if not source_file.exists():
                raise FileNotFoundError(
                    f"File `{str(source_file)}` is not found. Please check."
                )

            print(f"Copying: {str(source_file):s} => {str(destination_file):s}")
            shutil.copy(source_file, destination_file)

        new_file_status = check_if_file_exist(files_to_check)
        if not all(list(new_file_status.values())):
            raise FileNotFoundError(
                "Something went wrong. The daily file is not generated. Please check."
            )

    return files_to_check

