#!/bin/bash
echo '================================================================================'
echo 'Setting up environment...'

# Get required paths
path=`pwd`;
if [[ $path == *scripts ]]; then
    path=`dirname $path`;
fi
echo "Current path:      $path"
submodules="$path/submodules"
transformers="$submodules/transformers"
datasets="$submodules/datasets"
echo "Submodules path:   $submodules"
echo "Transformers path: $transformers"
echo "Datasets path:     $datasets"
requirements="$path/requirements.txt"
echo "Requirements file: $requirements"
echo '--------------------------------------------------------------------------------'

# init the submodules
echo 'Setting up submodules...'
git submodule update --init --recursive;
cd $transformers
git remote add upstream https://github.com/huggingface/transformers.git
cd $datasets
git remote add upstream https://github.com/huggingface/datasets.git
cd $path
echo '--------------------------------------------------------------------------------'

# set up environment
echo 'Setting up environment...'
tmp=`conda env create -f huggingface.yml`
conda_dir=`conda env list | grep huggingface | awk '{print $NF}'`
python_exe="$conda_dir/bin/python3"
pip_exe="$conda_dir/bin/pip"
$python_exe -m pip install -e $transformers
$python_exe -m pip install -e $datasets
echo '--------------------------------------------------------------------------------'

echo 'Testing environment...'
$python_exe tests/test_transformers_installation.py
echo '--------------------------------------------------------------------------------'

echo 'Done!'
echo 'Please activate the (huggingface) environment before running the project:'
echo 'conda activate huggingface'
echo '================================================================================'
