#!/bin/bash

# Serve the built HTML (docs/build/html) on http://localhost:8000, from any
# working directory.
python3 -m http.server -d "$(dirname "${BASH_SOURCE[0]}")/build/html"
