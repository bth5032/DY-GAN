#!/bin/bash
showTR () {
  scp uaf-10:${LogDir}/v${trial}*${RunKey}/plots*${epoch}.png results/${trial}_${epoch}.png 
  imgcat results/${trial}_${epoch}.png
}

if [[ ! -d results ]]; then
    mkdir results
fi

RunKey=batchrun
LogDir=/nfs-7/userdata/bhashemi/DY-GAN/models/progress

if [[ $# > 1 ]]
then
  trial=$1
  epoch=$2
  showTR
else
  infile=$1
  lnum=2
  answer=y
  while [[ $answer == 'y' ]]
  do
    line=`sed "${lnum}q;d" $infile`
    trial=`echo $line | awk '{print $1}'`
    epoch=`echo $line | awk '{print $2}'`
    echo "Trail: $trial Epoch: $epoch"
    showTR
    read -n1 -p "Show next? (y/n): " answer
    echo ""
    lnum=$(( lnum + 1 ))
  done
fi

