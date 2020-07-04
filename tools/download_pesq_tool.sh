#!/usr/bin/env bash

echo "Start to download 2 PESQ packages, 
the calculation results of two packages are different.
Please decide which one is prefered to you."

cd metric/

echo "Start to download composite.zip"

wget https://ecs.utdallas.edu/loizou/speech/composite.zip

echo "Unzip composite.zip" 

unzip composite.zip -d composite

echo "Start to download "

wget https://www.routledge.com/downloads/K14513/K14513_CD_Files.zip

echo "Unzip K14513_CD_Files.zip"

unzip K14513_CD_Files.zip -d K14513_CD_Files