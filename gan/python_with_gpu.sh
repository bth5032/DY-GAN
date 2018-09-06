#!/usr/bin/env bash

export KERAS_BACKEND=tensorflow
export CUDA_VISIBLE_DEVICES="1"
singularity exec --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest python $@
