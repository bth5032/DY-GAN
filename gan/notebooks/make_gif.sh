#!/usr/bin/env bash

convert -delay 30 -loop 0 `ls -1 $1/*.png | python natsort.py ` $1/animation.gif
echo "made animation.gif"
