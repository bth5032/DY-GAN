#!/bin/bash
for model in $@
do
  epoch=${model/#*-/}
  trial=${model/%-*/}
  ./showResult.sh $trial $epoch
  cp results/${trial}_${epoch}.png ~/Documents/Work/Presentations/GAN/New\ Data\ Training/figs/models/
done
