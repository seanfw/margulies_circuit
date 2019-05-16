import pytest
from circuit.phase import run_phase
import os, os.path

def test_run_phase():

    os.remove('./circuits/figures/phase1.png')
    os.remove('./circuits/figures/phase2.png')

    run_phase()

    assert os.path.isfile('./circuits/figures/phase1.png')
    assert os.path.isfile('./circuits/figures/phase2.png')
