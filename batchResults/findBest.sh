#!/bin/bash

LogName=log.txt
RunKey=batchrun
LogDir=/nfs-7/userdata/bhashemi/DY-GAN/models/progress

mkdir logs/

ssh uaf-10 "ls ${LogDir}/*${RunKey}/${LogName}" > logs/lognames.txt
for logfile in `cat logs/lognames.txt`
do
  scp uaf-10:${logfile} logs/
  version=$(echo $(basename `dirname $logfile`) | sed 's/v\([0-9]*\)_.*/\1/g')
  mv logs/$LogName logs/${version}_raw.txt
done

cat logs/*raw.txt | grep MLLKSStatistic | sort -n -k6 > logs/score.txt
cat logs/*raw.txt | grep MLLKSStatistic |sort -n -k8 > logs/MLL.txt
cat logs/*raw.txt | grep MLLKSStatistic |sort -n -k12 > logs/MetPhi.txt
cat logs/*raw.txt | grep MLLKSStatistic |sort -n -k16 > logs/LepIso.txt

python computeRank.py