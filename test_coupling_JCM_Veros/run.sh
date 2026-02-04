#!/bin/bash

# Remember to
# pip3 install veros

rm acc.*.nc
python3 modify_jcm_terrain.py
python3 jem_JCM_Veros.py
