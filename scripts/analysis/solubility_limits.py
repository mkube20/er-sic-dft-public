#!/usr/bin/env python
"""
solubility_limits.py: upper bounds on mu_O and mu_N set by SiO2 and Si3N4.

    d_mu_Si + 2 d_mu_O <= dH_f(SiO2),   3 d_mu_Si + 4 d_mu_N <= dH_f(Si3N4)

with d_mu measured from bulk Si and the O2 and N2 molecules, and d_mu_Si = 0
(Si-rich) or dH_f(SiC) (C-rich). dH_f(SiO2) and dH_f(Si3N4) are the NIST-JANAF
experimental values (Chase 1998); dH_f(SiC) is computed. Combining experimental
enthalpies with computed molecular references gives an uncertainty of about
0.5 eV per impurity atom. Reports the oxygen and nitrogen complexes at the bounds.

Input:   results/formation_energies/formation_energies_spin.json
Output:  results/formation_energies/solubility_limits.json
Usage:   python scripts/analysis/solubility_limits.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FE = REPO / "results/formation_energies/formation_energies_spin.json"
OUT = REPO / "results/formation_energies/solubility_limits.json"

KJ_PER_EV = 96.48533212
DH_SIO2 = -910.86 / KJ_PER_EV
DH_SI3N4 = -744.75 / KJ_PER_EV
HA = 27.211386245988

PARTNER = {"er_ko": ("O", 1), "er_ho": ("O", 1),
           "er_kn": ("N", 1), "er_hn": ("N", 1)}


def main():
    """Compute the O and N solubility limits and the complexes' formation energies at them.
    Used by: command line.
    Reads formation_energies_spin.json; writes solubility_limits.json.
    """
    data = json.loads(FE.read_text())
    mu = data["chemical_potentials_hartree"]
    dh_sic = (mu["C_rich"]["mu_Si"] - mu["Si_rich"]["mu_Si"]) * HA
    d_mu_si = {"Si_rich": 0.0, "C_rich": dh_sic}

    d_mu = {lim: {"O": (DH_SIO2 - d_mu_si[lim]) / 2.0,
                  "N": (DH_SI3N4 - 3.0 * d_mu_si[lim]) / 4.0}
            for lim in d_mu_si}

    fe = data["formation_energies_eV"]
    bounded = {}
    for cfg, (species, n) in PARTNER.items():
        bounded[cfg] = {lim: fe[cfg][lim] - n * d_mu[lim][species]
                        for lim in d_mu_si}

    out = {
        "method": "upper bounds on mu_O and mu_N from SiO2 and Si3N4 coexistence",
        "enthalpy_source": "NIST-JANAF (Chase 1998) for SiO2 and Si3N4; "
                           "this work's DFT value for SiC",
        "dH_f_eV": {"SiO2": DH_SIO2, "Si3N4": DH_SI3N4, "SiC": dh_sic},
        "d_mu_bound_eV": d_mu,
        "formation_energy_at_bound_eV": bounded,
        "caveat": "mixed experimental/DFT estimate; ~0.5 eV uncertainty per "
                  "impurity atom from PBE O2/N2 overbinding. Replace with DFT "
                  "SiO2 and Si3N4 references before quoting as final.",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Written: {OUT}")
    for lim in d_mu_si:
        print(f"  {lim:7}: d_mu_O <= {d_mu[lim]['O']:+.2f} eV, "
              f"d_mu_N <= {d_mu[lim]['N']:+.2f} eV")
    for cfg, v in bounded.items():
        print(f"  {cfg}: {fe[cfg]['Si_rich']:.3f} -> "
              f"{v['Si_rich']:.3f} (Si-rich) / {v['C_rich']:.3f} (C-rich)")


if __name__ == "__main__":
    main()
