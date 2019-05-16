import pytest, os, os.path

def test_main_model():

    if os.path.isfile('./circuit/results/R.npy'):
        os.remove('./circuit/results/R.npy')

    os.system('python ./circuit/main.py -c ./circuit/test/config_test.ini')

    assert os.path.isfile('./circuit/results/R.npy')
