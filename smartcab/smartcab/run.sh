#!/bin/sh
python agent.py > result.dat
#python agent2.py > result.dat
echo `grep reached result.dat | wc -l` *0.002*100 | bc
rm result.dat
