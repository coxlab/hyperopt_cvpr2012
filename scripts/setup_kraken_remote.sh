#!/bin/bash

clone_or_pull () {
   cd $HOME
   DIR=$1
   SRC=$2
   if [ -d cvs/${DIR} ] ; then
      cd cvs/${DIR}
      git pull
      cd ../../
   else
      cd cvs
      git clone ${SRC}
      cd ../
   fi

   if [ -z "$3" ] ; then
      echo 'keeping default branch'
   else
        cd cvs/${DIR}
	if [ "$(git branch | grep \* | cut -d' ' -f 2)" = "$3" ] ; then
	      echo 'already on requested branch'
	else
	      echo 'checking out ' $3
	      git checkout -b "$3" "remotes/origin/$3"
        fi
	cd ../../
   fi
}


clone_or_pull Theano git://github.com/Theano/Theano.git
clone_or_pull hyperopt git://github.com/jaberg/hyperopt.git
clone_or_pull MonteTheano git://github.com/jaberg/MonteTheano.git
clone_or_pull scikit-data git://github.com/jaberg/scikit-data.git
clone_or_pull hyperopt_cvpr2012 git@github.com:coxlab/hyperopt_cvpr2012.git
clone_or_pull asgd git://github.com/jaberg/asgd.git
clone_or_pull pythor3 git@github.com:nsf-ri-ubicv/pythor3.git develop
clone_or_pull genson git://github.com/yamins81/genson.git feature/hyperopt

# -- PYTHONPATH
cd $HOME
mkdir -p cvs/PYTHONPATH
cd cvs/PYTHONPATH
rm ~/cvs/PYTHONPATH/*
ln -s ../Theano/theano
ln -s ../hyperopt/hyperopt
ln -s ../MonteTheano/montetheano
ln -s ../scikit-data/skdata
ln -s ../asgd/asgd
ln -s ../hyperopt_cvpr2012
ln -s ../pythor3/pythor3
ln -s ../genson/genson

cd $HOME
if python -c 'import bson' ; then
 echo 'Already have mongo'
else
 pip install pymongo
fi

if python -c 'import lockfile' ; then
 echo 'Already have lockfile'
else
 pip install lockfile
fi

if python -c 'import pyparsing' ; then
 echo 'Already have pyparsing'
else
 pip install pyparsing
fi

if python -c 'import codepy' ; then
 echo 'Already have codepy'
else
 pip install codepy
fi

if python -c 'import collections; collections.OrderedDict' ; then
 echo 'Already have OrderedDict'
else
    if python -c 'import ordereddict' ; then
     echo 'Already have ordereddict'
    else
     pip install -vUI ordereddict
    fi
fi

if python -c 'import PIL' ; then
 echo 'Already have PIL'
else
 pip install PIL
fi

if [ -d .theano/bindir ]; then
  echo 'Already have bindir'
else
  mkdir -p .theano/bindir
  ln -s $(which gcc-4.2) .theano/bindir/gcc
  ln -s $(which g++-4.2) .theano/bindir/g++
fi

  
