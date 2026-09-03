# JCM-Veros coupled run in a double-drake configuration

This simulation couples JCM and Veros in a double-drake configuration. This configuration does not need any regridding or vector rotation, as JCM and Veros share the same Gaussian lat-lon grid. The "land" is achieved here by "slab ocean model", and will be switched to slab land model in the next release.

Users can start by running `run.sh` to produce a 60 days of daily output.

1. `run.sh`: The bash-side main running file. Suitable for HPC job submission.
2. `main_forward.py`: The python-side main file invoked by `run.sh`.
3. `model_setup.py`: Generate coupled model.
4. `veros_case_setup.py`: The detail configuration file for veros.
