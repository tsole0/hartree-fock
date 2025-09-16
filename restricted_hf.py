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
    :param basis_set: Name of basis set
    :param sto_gamma: The Slater-type orbital (STO) gamma value, if required by the choice
    of basis set
    """

    _basis_sets = {
            'sto-3g': {
                'fxn': lambda distance, orb_exponent: (2 * orb_exponent / np.pi)**(3/4) * np.exp(-orb_exponent*distance**2),
                'alpha_1g': np.array([0.109818, 0.405771, 2.22766]),
                'contraction_coefficients': np.array([0.444635, 0.535328, 0.154329]),
                       }
        }

    def __init__(self, basis_set, sto_gamma=None) -> None:
            if basis_set not in self._basis_sets.keys():
                raise ValueError("Please provide a valid basis set:")
            else:
                self.basis_fxn = self._basis_sets[basis_set]
    
    def generate_alpha(self, slater_gamma: np.floating) -> np.floating:
        """
        Generate a Gaussian alpha coefficient from a corresponding Slater gamma coefficient,
        for use in STO-NG method.
        """
        alphas = self.basis_fxn['alpha_1g']
        return alphas * slater_gamma**2