"""
formation_energy.py: neutral and charged defect formation energies.

    E_f(q; E_F) = E_defect(q) - E_pristine - sum_i n_i mu_i + q (E_VBM + E_F) + E_corr(q)

Chemical potentials come from the reference cells in abinit/references/
(mu_Si + mu_C = mu_SiC between the Si-rich and C-rich limits; O, N and H from
their molecules; Er, Al, P and B from their elemental solids). Also provides the
energy and band-edge readers used by the other analysis scripts.

Output:  results/formation_energies/formation_energies_spin.json
Usage:   python scripts/analysis/formation_energy.py --cell production --spin --all-configs \
             --output results/formation_energies/formation_energies_spin.json
"""

import argparse
import json
import sys
import re
from pathlib import Path

HARTREE_TO_EV = 27.2114


def extract_total_energy(abinit_output_file, allow_scf_fallback=True):
    """Final total energy of an ABINIT run.
    Used by: main, charge_transition_levels, u_scan.
    Input: .abo path; returns: energy (Ha). Raises if the run did not finish.
    """
    abinit_output_file = Path(abinit_output_file)
    value = None
    with open(abinit_output_file) as f:
        for line in f:
            s = line.strip()
            if s.startswith("etotal"):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                    except ValueError:
                        pass
    if value is not None:
        return value
    if allow_scf_fallback:
        text = abinit_output_file.read_text(encoding="utf-8", errors="replace")
        etots = re.findall(r"^\s*ETOT\s+\d+\s+(-?\d+\.\d+(?:[eE][+-]?\d+)?)",
                           text, re.M)
        if etots:
            return float(etots[-1])
    raise ValueError(f"No 'etotal' (or ETOT) line found in {abinit_output_file}")


def _parse_eigenvalues_by_kpt(abo_path):
    """Parse the last eigenvalue block of an .abo file.
    Used by: extract_band_edges (fallback when no _GSR.nc).
    Input: .abo path; returns: list of {'eig', 'occ'} per k point (Ha).
    """
    lines = Path(abo_path).read_text(encoding="utf-8", errors="replace").splitlines()

    header_re = re.compile(r"Eigenvalues \(hartree\) for nkpt=\s*(\d+)")
    starts = [(i, header_re.search(l)) for i, l in enumerate(lines)
              if header_re.search(l)]
    if not starts:
        raise ValueError(f"{abo_path}: no 'Eigenvalues (hartree)' block found "
                         f"(needs prteig 1).")
    anchor, m = starts[-1]
    nkpt = int(m.group(1))

    num = re.compile(r"^[+-]?\d*\.\d")

    def read_floats(idx):
        """Read floats from consecutive numeric lines.
        Used by: _parse_eigenvalues_by_kpt.
        Input: line index; returns: (values, next index).
        """
        vals = []
        while idx < len(lines):
            toks = lines[idx].split()
            if toks and all(num.match(t) for t in toks):
                vals.extend(float(t) for t in toks)
                idx += 1
            else:
                break
        return vals, idx

    kpts = []
    i = anchor + 1
    while i < len(lines) and len(kpts) < nkpt:
        if lines[i].strip().startswith("kpt#"):
            eig, i = read_floats(i + 1)
            occ = []
            if i < len(lines) and "occupation numbers" in lines[i]:
                occ, i = read_floats(i + 1)
            kpts.append({"eig": eig, "occ": occ})
        else:
            i += 1
    return kpts


def _locate_gsr(abo_path):
    """Path of the _GSR.nc file that belongs to an .abo file.
    Used by: extract_band_edges.
    Input: .abo path; returns: path or None.
    """
    abo_path = Path(abo_path)
    base = abo_path.stem
    gsr = abo_path.parent.parent / f"{base}_run" / f"{base}o_GSR.nc"
    return gsr if gsr.exists() else None


def _band_edges_from_gsr(gsr_path, occ_threshold=1.0):
    """VBM, CBM and gap from all k points of a _GSR.nc file.
    Used by: extract_band_edges.
    Inputs: _GSR.nc path, occupation threshold; returns: (VBM Ha, CBM Ha, gap
    eV).
    """
    import h5py
    import numpy as np

    with h5py.File(gsr_path, "r") as h:
        eig = np.array(h["eigenvalues"])
        occ = np.array(h["occupations"])
    occupied = eig[occ > occ_threshold]
    empty = eig[occ < occ_threshold]
    if occupied.size == 0 or empty.size == 0:
        raise ValueError(f"{gsr_path}: no clean occupied/empty split "
                         f"(metallic, or wrong occ_threshold).")
    vbm = float(occupied.max())
    cbm = float(empty.min())
    return vbm, cbm, (cbm - vbm) * HARTREE_TO_EV


def extract_band_edges(pristine_abo_file, occ_threshold=1.0):
    """VBM, CBM and gap of the pristine supercell.
    Used by: extract_vbm.
    Inputs: pristine .abo path, occupation threshold; returns: (VBM Ha, CBM Ha,
    gap eV).
    """
    gsr = _locate_gsr(pristine_abo_file)
    if gsr is not None:
        return _band_edges_from_gsr(gsr, occ_threshold)

    content = Path(pristine_abo_file).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Eigenvalues \(hartree\) for nkpt=\s*(\d+)", content)
    nkpt_total = int(m.group(1)) if m else None
    kpts = _parse_eigenvalues_by_kpt(pristine_abo_file)
    if nkpt_total is not None and len(kpts) < nkpt_total:
        raise ValueError(
            f"{pristine_abo_file}: eigenvalues for only {len(kpts)} of "
            f"{nkpt_total} k-points are printed (default prtvol) and no _GSR.nc "
            f"was found next to it, so the band edges cannot be trusted. Pull the "
            f"run's _GSR.nc, rerun pristine with prtvol 1, or do an NSCF band run.")

    vbm, cbm = None, None
    for kp in kpts:
        for e, o in zip(kp["eig"], kp["occ"]):
            if o > occ_threshold:
                vbm = e if vbm is None else max(vbm, e)
            else:
                cbm = e if cbm is None else min(cbm, e)
    if vbm is None or cbm is None:
        raise ValueError(f"{pristine_abo_file}: could not identify VBM/CBM.")
    return vbm, cbm, (cbm - vbm) * HARTREE_TO_EV


def extract_vbm(pristine_abo_file):
    """VBM of the pristine supercell, the E_F = 0 reference.
    Used by: charge_transition_levels.
    Input: pristine .abo path; returns: VBM (Ha).
    """
    vbm, _cbm, _gap = extract_band_edges(pristine_abo_file)
    return vbm


def compute_chemical_potentials(refs):
    """Chemical potentials in the Si-rich and C-rich limits.
    Used by: main, charge_transition_levels.
    Input: dict of reference total energies (Ha); returns: {'Si_rich': {...},
    'C_rich': {...}}.
    """
    mu_SiC = refs["sic_bulk"]
    mu_Si_bulk = refs["si_bulk"]
    mu_C_graphite = refs["graphite"]
    mu_Er_metal = refs["er_metal"]
    mu_N = refs["n2_molecule"] / 2.0
    mu_O = refs["o2_molecule"] / 2.0
    mu_H = refs["h2_molecule"] / 2.0 if refs.get("h2_molecule") else None
    mu_Al = refs.get("al_metal")
    mu_P = refs.get("p_bulk")
    mu_B = refs.get("b_bulk")

    mu_Si_rich = mu_Si_bulk
    mu_C_rich_limit_from_Si_rich = mu_SiC - mu_Si_bulk

    mu_C_rich = mu_C_graphite
    mu_Si_at_C_rich = mu_SiC - mu_C_graphite

    return {
        "Si_rich": {"mu_Si": mu_Si_rich, "mu_C": mu_C_rich_limit_from_Si_rich,
                    "mu_Er": mu_Er_metal, "mu_N": mu_N, "mu_O": mu_O,
                    "mu_Al": mu_Al, "mu_P": mu_P, "mu_B": mu_B, "mu_H": mu_H},
        "C_rich":  {"mu_Si": mu_Si_at_C_rich, "mu_C": mu_C_rich,
                    "mu_Er": mu_Er_metal, "mu_N": mu_N, "mu_O": mu_O,
                    "mu_Al": mu_Al, "mu_P": mu_P, "mu_B": mu_B, "mu_H": mu_H},
    }


def formation_energy(E_defect, E_pristine, n_atoms_removed, mu_dict,
                      charge=0, e_fermi=0.0, e_vbm=0.0, e_corr=0.0):
    """Formation energy of a defect, neutral or charged.
    Used by: main, formation_energy_vs_efermi, charge_transition_levels.
    Inputs: energies (Ha), atoms removed, mu, charge, E_F, E_VBM, correction
    (eV); returns: E_f (eV).
    """
    E_f_ha = E_defect - E_pristine
    for species, n in n_atoms_removed.items():
        E_f_ha += n * mu_dict[f"mu_{species}"]

    E_f_eV = E_f_ha * HARTREE_TO_EV

    if charge != 0:
        E_f_eV += charge * (e_vbm * HARTREE_TO_EV + e_fermi) + e_corr

    return E_f_eV


def formation_energy_vs_efermi(E_defect_by_charge, E_pristine, n_atoms_removed,
                                mu_dict, e_vbm, e_gap, corrections=None, n_points=50):
    """E_f(q; E_F) of every charge state across the gap.
    Used by: charge_transition_levels.
    Inputs: energies by charge, stoichiometry, mu, E_VBM, gap, corrections;
    returns: dict of arrays.
    """
    if corrections is None:
        corrections = {q: 0.0 for q in E_defect_by_charge}

    e_fermi_grid = [i * e_gap / (n_points - 1) for i in range(n_points)]
    result = {}
    for q, E_defect in E_defect_by_charge.items():
        e_f_values = [
            formation_energy(E_defect, E_pristine, n_atoms_removed, mu_dict,
                              charge=q, e_fermi=ef, e_vbm=e_vbm,
                              e_corr=corrections.get(q, 0.0))
            for ef in e_fermi_grid
        ]
        result[q] = {"e_fermi": e_fermi_grid, "e_f": e_f_values}
    return result


DEFECT_STOICHIOMETRY = {
    "er_h":  {"Si": 1, "Er": -1},
    "er_k":  {"Si": 1, "Er": -1},
    "er_hc": {"C": 1, "Er": -1},
    "er_i":  {"Er": -1},
    "er_hv": {"Si": 1, "C": 1, "Er": -1},
    "er_kv": {"Si": 1, "C": 1, "Er": -1},
    "er_hn": {"Si": 1, "C": 1, "Er": -1, "N": -1},
    "er_ho": {"Si": 1, "C": 1, "Er": -1, "O": -1},
    "er_hp":  {"Si": 2, "Er": -1, "P": -1},
    "er_hal": {"Si": 2, "Er": -1, "Al": -1},
    "er_hb":  {"Si": 2, "Er": -1, "B": -1},
    "er_ko":  {"Si": 1, "C": 1, "Er": -1, "O": -1},
    "er_kn":  {"Si": 1, "C": 1, "Er": -1, "N": -1},
    "er_kp":  {"Si": 2, "Er": -1, "P": -1},
    "er_kal": {"Si": 2, "Er": -1, "Al": -1},
    "er_kb":  {"Si": 2, "Er": -1, "B": -1},
    "er_kc":  {"C": 1, "Er": -1},
    "er_kosi": {"Si": 2, "Er": -1, "O": -1},
    "er_knsi": {"Si": 2, "Er": -1, "N": -1},
    "er_kpc":  {"Si": 1, "C": 1, "Er": -1, "P": -1},
    "er_kalc": {"Si": 1, "C": 1, "Er": -1, "Al": -1},
    "er_kbc":  {"Si": 1, "C": 1, "Er": -1, "B": -1},
    "er_kvsi": {"Si": 2, "Er": -1},
    "er_kh": {"Si": 1, "Er": -1, "H": -1},
    "er_hh": {"Si": 1, "Er": -1, "H": -1},
    "er_hosi": {"Si": 2, "Er": -1, "O": -1},
    "er_hnsi": {"Si": 2, "Er": -1, "N": -1},
    "er_hpc":  {"Si": 1, "C": 1, "Er": -1, "P": -1},
    "er_halc": {"Si": 1, "C": 1, "Er": -1, "Al": -1},
    "er_hbc":  {"Si": 1, "C": 1, "Er": -1, "B": -1},
    "er_hvsi": {"Si": 2, "Er": -1},
    "n_c_only": {"C": 1, "N": -1},
    "o_c_only": {"C": 1, "O": -1},
    "h_i_only": {"H": -1},
}

SCREENING_CONFIGS = ["er_h", "er_k", "er_hc", "er_i", "er_hv", "er_kv", "er_hn", "er_ho"]

PRODUCTION_ALL = ["er_ho", "er_hn", "er_kv", "er_hv", "er_hc", "er_k", "er_h",
                  "er_i", "er_hp", "er_hal", "er_hb",
                  "er_ko", "er_kn", "er_kp", "er_kal", "er_kb", "er_kc",
                  "er_kosi", "er_knsi", "er_kpc", "er_kalc", "er_kbc", "er_kvsi",
                  "er_kh", "er_hh",
                  "er_hosi", "er_hnsi", "er_hpc", "er_halc", "er_hbc", "er_hvsi"]

REFERENCE_DIVISORS = {
    "sic_bulk": 4,
    "si_bulk": 2,
    "graphite": 4,
    "er_metal": 2,
    "n2_molecule": 1,
    "o2_molecule": 1,
    "h2_molecule": 1,
    "al_metal": 1,
    "p_bulk": 4,
    "b_bulk": 12,
}

RELAXED_IN_SCREENING = {"er_hc", "er_i"}

DOWNSELECTED = ["er_ho", "er_hn", "er_hv", "er_kv"]

SPIN_REFERENCES = {
    "er_metal": "references/er_metal_spin.abo",
}


def main():
    """Compute the neutral formation energies of the configurations.
    Used by: command line.
    Writes --output (JSON).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--abinit-results", type=Path, default=Path("abinit"),
                        help="Root directory containing ABINIT output files")
    parser.add_argument("--cell", choices=["small_cell", "production"],
                        default="production",
                        help="Which outputs to read (production: the 128-atom cells)")
    parser.add_argument("--output", type=Path,
                        default=Path("results/formation_energies/formation_energies.json"),
                        help="Where to write results JSON")
    parser.add_argument("--threshold", type=float, default=8.0,
                        help="report configurations below this formation energy (eV) "
                             "in both chemical-potential limits")
    parser.add_argument("--configs", nargs="+",
                        help="score exactly these configs instead of the default "
                             "set. Used for the isolated partner cells, which are "
                             "kept out of PRODUCTION_ALL on purpose.")
    parser.add_argument("--all-configs", action="store_true",
                        help="score all nineteen configurations (PRODUCTION_ALL); "
                             "configurations without a completed .abo are skipped")
    parser.add_argument("--spin", action="store_true",
                        help="read the spin-polarized calculations: the highest-numbered "
                             "<config>_relax<k>_spin.abo, pristine_supercell_gamma.abo "
                             "and the spin-polarized Er and O2 references. Required "
                             "for all reported results.")
    args = parser.parse_args()

    if args.spin and args.cell != "production":
        parser.error("--spin exists only for the production cell: the 2x2x1 "
                     "screening was never redone spin-polarized.")

    ref_files = {
        "sic_bulk":    "references/pristine_4hsic.abo",
        "si_bulk":     "references/si_bulk.abo",
        "graphite":    "references/graphite.abo",
        "er_metal":    "references/er_metal.abo",
        "n2_molecule": "references/n2_molecule.abo",
        "o2_molecule": "references/o2_molecule.abo",
        "h2_molecule": "references/h2_molecule.abo",
    }
    for key, path in (("al_metal", "references/al_metal.abo"),
                      ("p_bulk", "references/p_bulk.abo"),
                      ("b_bulk", "references/b_bulk.abo")):
        if (args.abinit_results / path).exists():
            ref_files[key] = path
    if args.spin:
        missing = []
        for key, spin_path in SPIN_REFERENCES.items():
            if (args.abinit_results / spin_path).exists():
                ref_files[key] = spin_path
            else:
                missing.append((key, spin_path))
        if missing:
            print("REFUSING to score --spin: open-shell reference(s) not yet "
                  "recomputed spin-polarized:", file=sys.stderr)
            for key, path in missing:
                print(f"    {key:12s} expected {args.abinit_results / path}", file=sys.stderr)
            print("  Both er_metal and O2 have open shells (Er 4f^11; O2 is a\n"
                  "  S = 1 triplet). Run them before quoting any number that\n"
                  "  depends on mu_Er or mu_O.",
                  file=sys.stderr)
            return 1

    refs = {}
    for key, path in ref_files.items():
        f = args.abinit_results / path
        if not f.exists():
            print(f"WARNING: reference {f} not found - species referenced to it "
                  f"will be skipped.", file=sys.stderr)
            continue
        refs[key] = extract_total_energy(f) / REFERENCE_DIVISORS[key]

    mu = compute_chemical_potentials(refs)

    if args.spin:
        pristine_file = "pristine_supercell_gamma.abo"
    elif args.cell == "production":
        pristine_file = "pristine_supercell.abo"
    else:
        pristine_file = "pristine_2x2x1.abo"
    E_pristine = extract_total_energy(args.abinit_results / args.cell / pristine_file)

    configs = (args.configs if args.configs
               else PRODUCTION_ALL if args.spin and args.all_configs
               else DOWNSELECTED if args.spin else SCREENING_CONFIGS)
    results = {}
    incomplete = []
    for config in configs:
        if args.spin:
            abo = None
            for k in range(9, 1, -1):
                cand = args.abinit_results / args.cell / f"{config}_relax{k}_spin.abo"
                if cand.exists():
                    abo = cand
                    break
            if abo is None:
                abo = args.abinit_results / args.cell / f"{config}_relax_spin.abo"
        else:
            suffix = "_relax.abo" if args.cell == "production" else "_2x2x1.abo"
            abo = args.abinit_results / args.cell / f"{config}{suffix}"
        if not abo.exists():
            print(f"WARNING: {abo} not found — skipping.")
            continue

        if "Calculation completed" not in abo.read_text(encoding="utf-8", errors="replace"):
            print(f"SKIPPING {config}: {abo.name} has no 'Calculation completed' — the "
                  f"run is unfinished, still running, or was killed.")
            incomplete.append(config)
            continue

        E_def = extract_total_energy(abo)
        stoich = DEFECT_STOICHIOMETRY[config]

        is_relaxed = (args.cell == "production") or (config in RELAXED_IN_SCREENING)
        m = re.search(r"_relax(\d)_spin\.abo$", abo.name)
        geom_pass = int(m.group(1)) if (args.spin and m) else 1

        results[config] = {
            "Si_rich": formation_energy(E_def, E_pristine, stoich, mu["Si_rich"]),
            "C_rich":  formation_energy(E_def, E_pristine, stoich, mu["C_rich"]),
            "relaxed": is_relaxed,
            "source_abo": abo.name,
            "geometry_pass": geom_pass if args.spin else None,
        }
        flag = " (relaxed)" if is_relaxed else " (single-point)"
        if args.spin:
            flag += f" pass{geom_pass}" if geom_pass > 1 else " pass1"
        print(f"{config:6s}  Si-rich = {results[config]['Si_rich']:+7.3f} eV   "
              f"C-rich = {results[config]['C_rich']:+7.3f} eV{flag}")

    def score(kv):
        """Mean of the Si-rich and C-rich formation energies.
        Used by: main.
        Input: (config, energies); returns: eV.
        """
        return 0.5 * (kv[1]["Si_rich"] + kv[1]["C_rich"])
    ranked = sorted(results.items(), key=score)
    print("\n--- Ranking (lowest first, mean of the two limits) ---")
    for config, e in ranked:
        flag = "relaxed" if e["relaxed"] else "single-pt"
        print(f"  {config:6s} {score((config, e)):+7.3f} eV   [{flag}]")

    def below_threshold(e):
        """Whether a configuration is below the threshold in both limits.
        Used by: main.
        Input: energies; returns: bool.
        """
        return max(e["Si_rich"], e["C_rich"]) < args.threshold
    selected = [c for c, e in ranked if below_threshold(e)]
    lowest = None
    if not ranked:
        print("No config in the requested set has a completed .abo "
              "yet - nothing to rank.")
    else:
        lowest = ranked[0][0]
        print(f"\nBelow {args.threshold:.1f} eV in both limits ({len(selected)}): {selected}")
        print(f"  lowest: {lowest}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "chemical_potentials_hartree": mu,
            "formation_energies_eV": results,
            "ranking": [c for c, _ in ranked],
            "threshold_eV": args.threshold,
            "selected": selected,
            "lowest": lowest,
        }, f, indent=2)

    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    sys.exit(main() or 0)
