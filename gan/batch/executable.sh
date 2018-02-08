#!/usr/bin/env bash

OUTPUTDIR=${1}
ARGS=${@:2}

CMSSWVERSION=CMSSW_9_4_0_pre2
SCRAMARCH=slc6_amd64_gcc630


echo -e "\n--- begin header output ---\n" #                     <----- section division
echo "OUTPUTDIR: $OUTPUTDIR"
echo "CMSSWVERSION: $CMSSWVERSION"
echo "SCRAMARCH: $SCRAMARCH"
echo "ARGS: $ARGS"

echo "hostname: $(hostname)"
echo "uname -a: $(uname -a)"
echo "time: $(date +%s)"
echo "args: $@"

echo -e "\n--- end header output ---\n" #                       <----- section division

source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH=${SCRAMARCH}

eval `scramv1 project CMSSW $CMSSWVERSION`
cd $CMSSWVERSION
eval `scramv1 runtime -sh`
cd  ..

echo "before running: ls -lrth"
ls -lrth 

echo -e "\n--- begin running ---\n" #                           <----- section division


export KERAS_BACKEND=tensorflow
export USE_ONE_CPU=true
python gan.py ${ARGS}

echo -e "\n--- end running ---\n" #                             <----- section division

echo "after running: ls -lrth"
ls -lrth

echo "after running: ls -lrth progress/"
ls -lrth progress/

echo -e "\n--- begin copying output ---\n" #                    <----- section division
# echo "cp -rp progress/* ${OUTPUTDIR}/"
# cp -rp progress/* ${OUTPUTDIR}/
echo gfal-copy -p -f -t 4200 -r progress gsiftp://gftp.t2.ucsd.edu${OUTPUTDIR}
gfal-copy -p -f -t 4200 -r progress gsiftp://gftp.t2.ucsd.edu${OUTPUTDIR}
echo -e "\n--- end copying output ---\n" #                      <----- section division

