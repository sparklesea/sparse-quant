pip uninstall quant

rm -rf build dist quant.egg-info

MAX_JOBS=64 python setup.py install

python script/gemm_awq_test.py --algo 0

# python script/test_wmma.py
