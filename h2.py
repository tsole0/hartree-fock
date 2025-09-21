# Test case for a H2 (molecular hydrogen) HF simulation
import numpy as np

from restricted_hf import NucConfig, Coord


geometry = [
    (1, Coord(0,0,0)),
    (1, Coord(1,1,1)),
]

H2 = NucConfig(geometry, 2, 'sto-3g')

