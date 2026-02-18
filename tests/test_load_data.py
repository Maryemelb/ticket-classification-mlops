import pytest
from src.data.load_data import load_data

def test_load_data():
    data= load_data()
    assert data is not None