import pytest
from circuit.phase import run_phase
import os, os.path

def test_run_phase():
    
    if os.path.isfile('./circuit/figures/phase1.png'):
        os.remove('./circuit/figures/phase1.png')
    if os.path.isfile('./circuit/figures/phase2.png'):
        os.remove('./circuit/figures/phase2.png')

    run_phase()

    assert os.path.isfile('./circuit/figures/phase1.png')
    assert os.path.isfile('./circuit/figures/phase2.png')
