pip uninstall myquant

rm -rf build dist myquant.egg-info

MAX_JOBS=64 python setup.py install

python script/gemm_awq_test.py --algo 0

