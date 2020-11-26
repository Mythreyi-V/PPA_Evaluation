#!/bin/bash -l
#PBS -N bpic2012
#PBS -l ncpus=6
#PBS -l ngpus=2
#PBS -l mem=20gb
#PBS -l walltime=90:00:00

cd $PBS_O_WORKDIR
echo "queue started"

module purge
echo "modules purged"

module load cuda/10.1.243-gcc-8.3.0
module load cudnn/7.6.4.38-gcccuda-2019b
echo "cuda loaded"
module load tensorflow/2.3.1-fosscuda-2019b-python-3.7.4
echo "tensorflow loaded"
 
python3 -m pip install --upgrade pip --user
echo "pip upgraded"
python3 -m pip install lime==0.2.0.1 --user
echo "lime installed"
python3 -m pip install shap==0.35.0 --user
echo "shap installed"
python3 -m pip install keras --user
echo "keras installed"
python3 -m pip install tensorflow==2.2.0 --user
echo "tensorflow downgraded"
python3 -m pip install --upgrade pandas --user
echo "pandas upgraded"

echo "starting test"
#python3 lstm_test.py
#python3 lstm_test_stability.py
python3 lstm_fidelity_trial.py
echo "test ended"
