# CLEWS modeling


## Getting JAX-GCM and JAX-ESM

```
mkdir working_directory
cd working_directory

git clone -b jem_CLEWS https://github.com/climate-analytics-lab/jax-esm.git
git clone -b dev https://github.com/climate-analytics-lab/jax-gcm.git

```

For each package, install them and their dependencies.

```
cd jax-gcm
pip install -e .

cd ../jax-esm
pip install -e .

```

## Test if you have it

In python, inspect the package.

```
import jcm
print(jcm.__file__)

import jem
print(jem.__file__)

# You should see their paths consistent with where you download them.

```

## Run the model

```
cd notebooks/04_CLEWS
python main.py
```
