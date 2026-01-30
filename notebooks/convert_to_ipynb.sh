#!/bin/bash

for f in $( ls *.py ) ; do
    jupytext --from py --to ipynb $f
done
