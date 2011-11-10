#!/bin/bash
if [ -z "$1" ] ; then
    echo 'need device'
    exit 1
fi
. ~/.VENV/base/bin/activate
THEANO_FLAGS=device=$1 ~/cvs/hyperopt/bin/hyperopt-mongo-worker
