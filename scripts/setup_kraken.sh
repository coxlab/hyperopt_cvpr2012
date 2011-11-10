#!/bin/bash
for DSA in id_dsa id_dsa.pub ; do
    scp kraken4:.ssh/$DSA $DSA
    scp $DSA kraken5:.ssh/$DSA
    scp $DSA kraken6:.ssh/$DSA
    rm $DSA
done

for NODE in kraken4 kraken5 kraken6 ; do
  ssh $NODE "mkdir -p .skdata/lfw"
  rsync -a ~/.skdata/lfw/ ${NODE}:.skdata/lfw/
  scp setup_kraken_remote.sh ${NODE}:setup.sh
  ssh $NODE ".  .VENV/bin/activate; sh setup.sh"
done


# WORKER:
# ssh -R27017:localhost:27017 kraken5 \
#    "THEANO_FLAGS=device=gpu0 cvs/hyperopt/bin/hyperopt-mongo-worker"
