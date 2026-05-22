#!/bin/bash
source ~/spack/share/spack/setup-env.sh

# Clean previous logs before each build/run
rm -rf logs
mkdir -p logs

make clean && make
