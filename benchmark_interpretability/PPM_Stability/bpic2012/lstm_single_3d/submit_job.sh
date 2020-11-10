#!/bin/bash -l
#PBS -N example
#PBS -l ncpus=2
#PBS -l ngpus=1
#PBS -l mem=3gb
#PBS -l walltime=45:00

cd $PBS_O_WORKDIR
echo "queue started"

module purge
echo "modules purged"

module load tensorflow/2.3.1-fosscuda-2019b-python-3.7.4
echo "tensorflow loaded"
 
python -m pip install --upgrade pip --user
echo "pip upgraded"
python -m pip install lime==0.2.0.1 --user
echo "lime installed"
python -m pip install shap==0.35.0 --user
echo "shap installed"
python -m pip install --upgrade pandas --user
echo "pandas upgraded"

echo "starting test"
python lstm_test.py
echo "test ended"
