import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_jcm_forcing_and_topography_files(
    resolution: int = 31,
    data_directory: Path | None = None,
) -> dict[str, Path]:

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

        logger.info('Using input data directory: "%s".', data_directory)

    raw_data_directory = Path(jcm.__file__).parent / "data/bc"

    # Prepare boundary file
    files_to_check = {
        "terrain": (data_directory / f"terrain_t{resolution:d}.nc").resolve(),
        "forcing": (data_directory / f"forcing_t{resolution:d}.nc").resolve(),
    }

    def check_if_file_exist(
        file_dict: dict[str, Path], verbose: bool = True
    ) -> dict[Path, bool]:

        file_status = {file: Path(file).exists() for _, file in file_dict.items()}

        if verbose:
            for file, result in file_status.items():
                logger.info(
                    "Check file: %s... %s",
                    file,
                    "found." if result else "not found.",
                )

        return file_status

    file_status = check_if_file_exist(files_to_check)

    if not all(list(file_status.values())):
        logger.info("Some files are missing. Need to generate them.")

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
            logger.error("Interpolation failed; its error output was: %s", e.stderr)

        new_file_status = check_if_file_exist(files_to_check)
        if not all(list(new_file_status.values())):
            raise FileNotFoundError(
                "Something went wrong. The daily file is not generated. Please check."
            )

    return files_to_check

