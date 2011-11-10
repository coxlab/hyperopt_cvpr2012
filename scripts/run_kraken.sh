#!/bin/bash
if [ -z "$1" ] ; then
    echo 'need host arg'
    exit 1
fi
if [ -z "$2" ] ; then
    echo 'need host device'
    exit 1
fi
if [ -f go.sh ] ; then
    echo 'pls remove ./go.sh file'
fi
echo  "" > go.sh
echo '. .VENV/bin/activate' >> go.sh
echo 'export PYTHONPATH=~/cvs/PYTHONPATH:$PYTHONPATH' >> go.sh
echo 'export CUDA_ROOT=~/local/cuda' >> go.sh
echo "THEANO_FLAGS=device=$2 cvs/hyperopt/bin/hyperopt-mongo-worker"\
    >> go.sh
scp go.sh $1:go.sh
rm go.sh

if [ -z "$3" ] ; then
    echo "RUNNING WITHOUT PORT FORWARDING"
    echo ssh $1 "bash go.sh"
    ssh $1 "bash go.sh"
else
    echo "RUNNING WITH PORT FORWARDING"
    echo ssh -R $3:localhost:$3 $1 "bash go.sh"
    ssh -R $3:localhost:$3 $1 "bash go.sh"
fi
