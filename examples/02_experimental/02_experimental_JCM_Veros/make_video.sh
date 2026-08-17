#!/bin/bash

ffmpeg -framerate 12 \
  -pattern_type glob \
  -i 'figures/frame-*.png' \
  -c:v libopenh264 \
  -b:v 12M \
  -pix_fmt yuv420p \
  output.mp4
