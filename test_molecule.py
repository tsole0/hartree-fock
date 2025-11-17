# Test case for a H2 (molecular hydrogen) HF simulation
import numpy as np
from pyscf import gto
import matplotlib.pyplot as plt
from restricted_hf import RHF


# Define a molecule and run to see convergence energy (density matrix will be printed)
#Example: H2O
mole = gto.M(atom='H 0 0 0; H 0 2.5 0; O 0.5 1.25 0', basis='sto-3g')
H2 = RHF(mole)
H2.run()


try:
	print(H2.P)
except AttributeError:
	print("Density matrix P not available. Run may have failed to converge or did not set P.")
print("energy_output:", H2.energy_output)

energy = np.asarray(H2.energy_output, dtype=float)
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
