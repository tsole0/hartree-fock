import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto
from unrestricted_hf import UHF


# Define a molecule and run to see convergence energy (density matrix will be printed)
#Example: HO radical
#mole = gto.M(atom='O 0 0 0; H 1 0 0', basis='sto-3g', charge=0, spin=1) #Doublet

mole = gto.M()
mole.atom = '''
    C   0.000000   0.000000   0.000000;
    C   1.340000   0.000000   0.000000;
    C   2.680000   0.000000   0.000000;
    H  -0.540000   0.935000   0.000000;
    H  -0.540000  -0.935000   0.000000;
    H   1.880000   0.935000   0.000000;
    H   1.880000  -0.935000   0.000000;
    H   3.220000   0.935000   0.000000
''' # Allyl radical
mole.basis = '6-31g*' #need a better basis set
mole.charge = 0
mole.spin = 1 #One doublet as a radical
mole.build()

allyl = UHF(mole)
allyl.run()


try:
	print(allyl.P_alpha)
	print(allyl.P_beta)
except AttributeError:
	print("Density matrix P not available. Run may have failed to converge or did not set P.")
print("energy_output:", allyl.energy_output)

energy = np.asarray(allyl.energy_output, dtype=float)
n = energy.size
iters = np.arange(1, n + 1)
plt.plot(iters, energy, marker='o', linestyle='-', color='C0', label='SCF energy')
final = float(energy[-1])
plt.axhline(final, color='red', alpha=0.35, linestyle='--', linewidth=2, label=f'Converged energy = {final:.8f} Ha')
plt.xlabel('Iteration $n$')
plt.ylabel('Energy (Hartrees)')
plt.title(f'Energy convergence (n={n})')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
out_png = 'energy_convergence.png'
plt.savefig(out_png, dpi=200)
print(f"Saved plot to: {out_png}")
plt.show()