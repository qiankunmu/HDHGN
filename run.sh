#!/bin/sh

cd data
tar -zxvf Project_CodeNet_Java250.tar.gz
tar -zxvf Project_CodeNet_Python800.tar.gz
cd ..
python ProcessData.py
python vocab.py
cd trains
python trainHDHGN.py
python trainHDHGN_java.py