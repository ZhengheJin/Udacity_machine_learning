#!/bin/sh
#python agent.py > result.dat
python agent2.py > result.dat
echo `grep reached result.dat | wc -l`
rm result.dat
