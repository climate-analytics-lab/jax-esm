#!/bin/bash

for f in $( ls *.ipynb ) ; do
    jupytext --from ipynb --to py $f
done
