"""
binding_energies.py: binding energies of the erbium complexes (Table 2).

    E_bind = E(Er_Si) + E(partner) - E(complex) - E(pristine)

from the total energies of the 128-atom cells on the quasi-cubic site. This is
equal to E_f(Er_Si) + E_f(partner) - E_f(complex), and every chemical potential
cancels.

Input:   data/supplementary_data/total_energies.csv (Supplementary Data 1)
Output:  results/formation_energies/binding_energies.json
Usage:   python scripts/analysis/binding_energies.py
"""

import argparse
import csv
import json
from pathlib import Path

HA_EV = 27.211386245988
COMPLEXES = {"er_ko": "o_c", "er_kn": "n_c", "er_kh": "h_i"}


def main():
    """Compute the binding energies of the oxygen, nitrogen and hydrogen complexes.
    Used by: command line.
    Reads --energies (CSV of total energies); writes --output (JSON, eV).
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energies", type=Path,
                    default=Path("data/supplementary_data/total_energies.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/formation_energies/binding_energies.json"))
    args = ap.parse_args()

    with open(args.energies, newline="") as f:
        e = {r["cell"]: float(r["etotal_Ha"]) for r in csv.DictReader(f)}

    out = {}
    for cplx, partner in COMPLEXES.items():
        ha = e["er_k"] + e[partner] - e[cplx] - e["pristine"]
        out[cplx] = round(ha * HA_EV, 4)
        print(f"{cplx}: E_bind = {ha * HA_EV:.3f} eV")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"binding_energies_eV": out}, indent=2) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
