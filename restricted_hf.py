# Math imports
import scipy
import numpy as np
import numpy.typing as npt

#Utility imports
from typing import Union
from functools import partial

class Coord:
    """
    Class holding coordinates in 3-dimensional space and coordinate operations
    """
    def __init__(self, x: int, y: int, z: int) -> None:
        self.pos = np.array([x, y, z])
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Coord ({', '.join(str(element) for element in self.pos)})"
    
    def __sub__(self, other: "Coord") -> "Coord":
        if not isinstance(other, Coord):
            return NotImplemented
        return Coord(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __abs__(self) -> np.float64:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def __pow__(self, power: int) -> "Coord":
        return Coord(self.x**power, self.y**power, self.z**power)

    @property
    def position(self) -> npt.NDArray[np.floating]:
        return self._pos
    
    @position.setter
    def position(self, position: npt.NDArray[np.floating]) -> None:
        if position.shape != (3,):
            raise ValueError(f"Position must be in form (3,), got {position.shape}")
        self._pos = position

class BasisSet:
    """
    Defines the basis set to be used in Hartree-Fock calculations.
    NOTE: At present, only STO-3G is implemented.
    :param basis_set: Name of basis set
    :param sto_zeta: The Slater-type orbital (STO) zeta value, if required by the choice
    of basis set
    """

    _basis_sets = {
            'sto-3g': {
                'fxn': lambda r, orb_exponent, center: (2 * orb_exponent / np.pi)**(3/4) * np.exp(-orb_exponent*abs(r - center)**2),
                'alpha_1z': np.array([0.109818, 0.405771, 2.22766]),
                'contraction_coefficients': np.array([0.444635, 0.535328, 0.154329]),
                       }
        }

    def __init__(self, basis_set, sto_zeta=None) -> None:
            if basis_set not in self._basis_sets.keys():
                raise ValueError("Please provide a valid basis set:")
            else:
                self.basis_category = self._basis_sets[basis_set]
    
    @staticmethod
    def generate_alpha(alpha_1z: np.floating, sto_zeta: np.floating) -> np.floating:
        """
        Generate a Gaussian alpha coefficient from a corresponding Slater zeta coefficient,
        for use in STO-NG method.
        """
        return alpha_1z * sto_zeta**2

class NucConfig:
    """
    Defines nuclear configuration (position), charge, and calculates invariants based on this (core-shell Hamiltonian)
    :param: 
    """
    def __init__(self, geometry: list[tuple[int, Coord]], n_elec: int, basis: str) -> None:
        self.n_elec = n_elec
        self.geometry = geometry
        self.basis = BasisSet(basis)

        # Generate basis functions
        self.basis_fxns = [None] * len(geometry) # Probably a better way to do this, and doesn't account for larger basis sets
        for n in range(len(self.basis_fxns)):
            
        #Calculate core-shell Hamiltonian
        self.core_shell_mat = self.core_shell()

    @property
    def n_elec(self) -> int:
        return self._n_elec

    @n_elec.setter
    def n_elec(self, n_elec) -> None:
        if n_elec < 0:
            raise ValueError(f"Number of electrons cannot be negative. Provided {n_elec}.")
        self._n_elec = n_elec
    
    def core_shell(self) -> npt.NDArray[np.float64]:
        """
        Calculate the core-shell Hamiltonian
        """
        #Initialize matrices
        n_basis = len(self.geometry) # Probably a better way to do this, and doesn't account for larger basis sets
        self.kinetic_mat = np.zeros((n_basis, n_basis), dtype=np.float64)
        self.nuclear_attraction_mat = np.zeros((n_basis, n_basis), dtype=np.float64)

        core_shell_mat = self.kinetic_mat + self.nuclear_attraction_mat # Temporary method to combine?
        
        return core_shell_mat
    
    def eval_nuclear_attraction(self):
        """
        Calculate nuclear attraction matrix
        """
        pass

    class BasisFxn:
        """
        class that holds coefficients, functions, etc. for a single basis function
        """

        def __init__(self, basis_set: "BasisSet", nuc_charge: int) -> None:
            sto_zeta = {
                # Nuclear charge : slater zeta value
                # TODO: Fine-grained control over zeta value--pass in as param
                1: np.float64(1.24)
            }
            self.basis_fxn = basis_set.basis_category['fxn']

            # Transform the alpha values for zeta=1 into atom-specific values
            alpha_1z = basis_set.basis_category['alpha_1z']
            self.alphas = np.array(
                list(
                    map(
                        BasisSet.generate_alpha,
                        alpha_1z,
                        [sto_zeta[nuc_charge]] * len(alpha_1z)
                    )
                )
            )
            self.contraction_coeff = basis_set.basis_category['contraction_coeff']

