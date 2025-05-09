#!/bin/bash

#=====================================================================#
# CONDA_SYSLIB could be needed to link libc.so.6 and libpthread.so.0
# export CONDA_SYSLIB=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64
#=====================================================================#


# Abort script on first error
set -e

# Need to tell where nvcc is
export CUDACXX=/usr/local/cuda-12.1/bin/nvcc
# export DETAIL_CAGRA_SOURCE_PATH=/nfs/home/daisy1212/anaconda3/envs/raft/include



PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

BUILD_TYPE=Release
BUILD_DIR=build/

RAFT_REPO_REL=""
EXTRA_CMAKE_ARGS=""

if [[ ${RAFT_REPO_REL} != "" ]]; then
  RAFT_REPO_PATH="`readlink -f \"${RAFT_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_raft_SOURCE=${RAFT_REPO_PATH}"
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

 # added for libthread
cmake \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DRAFT_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 ${EXTRA_CMAKE_ARGS} \
 ../

cmake  --build . -j${PARALLEL_LEVEL}
