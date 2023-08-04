#!/bin/bash
# pip install pytorch-fid
# working directory should be /.../Flow-Style-VTON/test
BASE_DIRS="results/eval_origin results/demo/try_on"
EVAL_DIRS="results/stylized-warp/try_on/la_muse.jpg results/stylized-before-warp/try_on/la_muse.jpg results/stylized-after-warp/try_on/la_muse.jpg"
for BASE in $BASE_DIRS
do
    for EVAL in $EVAL_DIRS
    do
        echo python -m pytorch_fid $BASE $EVAL --dims 192
        python -m pytorch_fid $BASE $EVAL --dims 192
    done
done