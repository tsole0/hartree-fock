import scipy
import numpy as np
import numpy.typing as npt

class Coord:
    """
    Class holding coordinates in 3-dimensional space.
    *If not needed later, shunt into Atom class.
    """
    def __init__(self, position: npt.NDArray[np.floating]) -> None:
        if position.shape != (3,):
            raise ValueError(f"Position must be in form (3,), got {position.shape}")
        self.pos = position

class Atom:
    """
    Defines atomic position and charge
    """
    def __init__(self, coord: Coord, charge: np.int16) -> None:
        self.charge = charge        #nuclear charge
        self.position = coord.pos

class BasisSet:
    """
    Defines the basis set to be used in Hartree-Fock calculations.
    NOTE: At present, only STO-3G is implemented.
    """

    _alpha_1g = np.array()

    def __init__(self, basis_set) -> None:
            basis_sets = {
            'sto-3g': lambda distance, orb_exponent: (2 * orb_exponent / np.pi)**(3/4) * np.exp(-orb_exponent*distance**2)
        }
            if basis_set not in basis_sets.keys:
                raise ValueError("Please provide a valid basis set:")
            else:
                self.basis_fxn = basis_sets[basis_set]
    
    @staticmethod
    def generate_alpha(slater_gamma):
        """
        Generate a Gaussian alpha coefficient from a corresponding Slater gamma coefficient,
        for use in STO-NG method.
        """
        return slater_gamma