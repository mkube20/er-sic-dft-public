"""
defect_levels.py: Kohn-Sham levels inside the gap of a defect cell.

For each level between the pristine VBM and CBM (all k points), reports the
occupation, the dominant orbital character from the projected DOS, the
fraction of projected weight on the defect complex (Er, partner and first
neighbors) and a symmetry label.

Input:   an ABINIT .abo and the run directory with its _GSR.nc and *_DOS_AT* files
Output:  results/defect_levels/<config>_<q>.json
Usage:   python scripts/postprocess/defect_levels.py --config er_ko --charge 0 \
             --eig abinit/er_ko_occprobe_spin_run/er_ko_occprobe_spin.abo \
             --e-vbm 7.6953 --e-cbm 9.9072 --pdos-dir abinit/er_ko_occprobe_spin_run \
             --output results/defect_levels/er_ko_q0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.2114


def _locate_gsr(abo_path, gsr_path=None):
    """Path of the _GSR.nc file that belongs to an .abo file.
    Used by: parse_eigenvalues_and_occupations, _gsr_structure.
    Inputs: .abo path, optional explicit path; returns: path or None.
    """
    if gsr_path is not None:
        gsr_path = Path(gsr_path)
        if not gsr_path.exists():
            raise FileNotFoundError(f"No such GSR file: {gsr_path}")
        return gsr_path

    abo_path = Path(abo_path)
    base = abo_path.stem
    run_dir = abo_path.parent.parent / f"{base}_run"

    plain = run_dir / f"{base}o_GSR.nc"
    if plain.exists():
        return plain

    dataset = sorted(run_dir.glob(f"{base}o_DS*_GSR.nc"))
    if len(dataset) == 1:
        return dataset[0]
    if len(dataset) > 1:
        names = "\n      ".join(p.name for p in dataset)
        raise ValueError(
            f"{run_dir} holds {len(dataset)} dataset GSR files and there is no "
            f"plain {base}o_GSR.nc, so which one is meant is ambiguous:\n"
            f"      {names}\n"
            f"    Pass one explicitly (--gsr). NOTE for the SOC ladder: DS5 of "
            f"er_ho_soc_2x2x1 is the DIVERGED rung — the full-strength result "
            f"lives in abinit/er_ho_soc_2x2x1_ds5_run/ instead.")
    return None


def parse_eigenvalues_and_occupations(abo_path: Path, gsr_path=None) -> dict:
    """Eigenvalues and occupations at all k points.
    Used by: main.
    Inputs: .abo path, optional _GSR.nc path; returns: dict (eV).
    """
    import h5py
    import numpy as np

    gsr = _locate_gsr(abo_path, gsr_path)
    if gsr is None:
        raise FileNotFoundError(
            f"No _GSR.nc found next to {abo_path} "
            f"(expected abinit/{Path(abo_path).stem}_run/{Path(abo_path).stem}o_GSR.nc "
            f"or a single {Path(abo_path).stem}o_DS<n>_GSR.nc). "
            f"Pull the run outputs first.")

    with h5py.File(gsr, "r") as h:
        eig = np.array(h["eigenvalues"])
        occ = np.array(h["occupations"])
        kpts = np.array(h["reduced_coordinates_of_kpoints"])
        wts = np.array(h["kpoint_weights"])
        fermie = float(np.array(h["fermie"]))

    return {
        "kpoints": kpts.tolist(),
        "kpoint_weights": wts.tolist(),
        "nsppol": int(eig.shape[0]),
        "eigenvalues_eV": (eig * HARTREE_TO_EV).tolist(),
        "occupations": occ.tolist(),
        "fermie_eV": fermie * HARTREE_TO_EV,
    }


def find_gap_levels(eig_data: dict, e_vbm_eV: float, e_cbm_eV: float,
                    occ_full: float = None) -> list:
    """Kohn-Sham levels between the pristine VBM and CBM.
    Used by: main.
    Inputs: eigenvalue data, VBM and CBM (eV), full-occupation value; returns:
    list of levels.
    """
    if occ_full is None:
        occ_full = 2.0 if eig_data["nsppol"] == 1 else 1.0
    hi, lo = 0.75 * occ_full, 0.25 * occ_full

    levels = []
    eig = eig_data["eigenvalues_eV"]
    occ = eig_data["occupations"]
    for isp in range(eig_data["nsppol"]):
        for ik, kco in enumerate(eig_data["kpoints"]):
            for ib, (e, o) in enumerate(zip(eig[isp][ik], occ[isp][ik])):
                if e_vbm_eV < e < e_cbm_eV:
                    status = ("occupied" if o >= hi else
                              "unoccupied" if o <= lo else "partial")
                    levels.append({
                        "energy_eV": e,
                        "energy_above_vbm_eV": e - e_vbm_eV,
                        "occupation": o,
                        "occ_status": status,
                        "band_index": ib,
                        "spin": isp,
                        "kpoint_index": ik,
                        "kpoint": kco,
                    })
    levels.sort(key=lambda d: d["energy_eV"])
    return levels


def assign_symmetry_labels(gap_levels: list, point_group: str,
                           tol_eV: float = 1e-3) -> list:
    """Label gap levels a1 or e by degeneracy.
    Used by: main.
    Inputs: levels, point group, tolerance (eV); returns: the labeled levels.
    """
    c3v = point_group.lower().replace(" ", "") in ("c3v", "3m")
    from collections import defaultdict
    buckets = defaultdict(list)
    for lvl in gap_levels:
        buckets[(lvl["spin"], lvl["kpoint_index"])].append(lvl)

    for group in buckets.values():
        group.sort(key=lambda d: d["energy_eV"])
        i = 0
        while i < len(group):
            j = i + 1
            while j < len(group) and abs(group[j]["energy_eV"] - group[i]["energy_eV"]) <= tol_eV:
                j += 1
            deg = j - i
            if c3v:
                label = "e" if deg == 2 else ("a1" if deg == 1 else f"deg{deg}?")
            else:
                label = "a"
            for k in range(i, j):
                group[k]["symmetry_label"] = label
                group[k]["degeneracy"] = deg
                if not c3v:
                    group[k]["symmetry_note"] = (
                        f"point group {point_group}: C3v a1/e labels do not apply")
            i = j
    return gap_levels


def check_local_symmetry(relaxed_structure, defect_index: int,
                         shell_radius: float = 3.2) -> str:
    """Point group of the relaxed defect and its first neighbors.
    Used by: main.
    Inputs: structure, defect index, shell radius (Angstrom); returns:
    Schoenflies symbol.
    """
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    site = relaxed_structure[defect_index]
    neighbors = relaxed_structure.get_neighbors(site, r=shell_radius)
    species = [site.specie] + [nb.specie for nb in neighbors]
    coords = [site.coords] + [nb.coords for nb in neighbors]
    origin = site.coords
    coords = [c - origin for c in coords]
    mol = Molecule(species, coords)
    return PointGroupAnalyzer(mol, tolerance=0.2).sch_symbol


def _gsr_structure(abo_path):
    """Structure stored in the _GSR.nc file of a run.
    Used by: _species_order, _defect_complex_indices, plot_structure.
    Input: .abo path; returns: pymatgen Structure.
    """
    from abipy.abilab import abiopen
    with abiopen(str(_locate_gsr(abo_path))) as gsr:
        return gsr.structure


def _species_order(abo_path):
    """Element symbol of each atom, in input order.
    Used by: main.
    Input: .abo path; returns: list of symbols.
    """
    return [s.specie.symbol for s in _gsr_structure(abo_path)]


def _defect_complex_indices(abo_path, shell_cutoff_ang: float = 2.6) -> list:
    """Atoms of the defect complex: non-host atoms and their first neighbors.
    Used by: main.
    Inputs: .abo path, neighbor cutoff (Angstrom); returns: 1-based atom
    indices.
    """
    struct = _gsr_structure(abo_path)
    host = {"Si", "C"}
    core = [i for i, s in enumerate(struct) if s.specie.symbol not in host]
    if not core:
        raise ValueError(f"{abo_path}: no non-host (non-Si/C) atom found — "
                         "is this a pristine cell?")
    idx = set(core)
    for i in core:
        for nb in struct.get_neighbors(struct[i], shell_cutoff_ang):
            idx.add(nb.index)
    return sorted(int(i) + 1 for i in idx)


def _weights_at(pdos: dict, energy_eV: float, sigma_eV: float = 0.10):
    """Gaussian-weighted projected DOS at one energy.
    Used by: localization_from_pdos, orbital_character.
    Inputs: projected DOS, energy, width (eV); returns: (weights by atom and
    channel, total).
    """
    e = pdos["energies_eV"]
    w = np.exp(-0.5 * ((e - energy_eV) / sigma_eV) ** 2)
    out = {}
    for species, channels in pdos["by_species"].items():
        for lname, vals in channels.items():
            out[(species, lname)] = float(np.sum(vals * w))
    total = sum(out.values())
    return out, total


def localization_from_pdos(gap_level: dict, pdos: dict, defect_atoms: list,
                           sigma_eV: float = 0.10) -> dict:
    """Fraction of a level's projected weight on the defect complex.
    Used by: main.
    Inputs: level, projected DOS, defect atoms, width (eV); returns: dict.
    """
    weights, total = _weights_at(pdos, gap_level["energy_eV"], sigma_eV)
    if total <= 0:
        return {"localization": None, "note": "no projected weight at this energy"}

    defect_species = {pdos["by_atom"][i]["species"] for i in defect_atoms
                      if i in pdos["by_atom"]}
    e = pdos["energies_eV"]
    win = np.exp(-0.5 * ((e - gap_level["energy_eV"]) / sigma_eV) ** 2)
    defect_w = 0.0
    for i in defect_atoms:
        if i not in pdos["by_atom"]:
            raise KeyError(f"atom index {i} has no DOS_AT file in the parsed set")
        for vals in pdos["by_atom"][i]["l"].values():
            defect_w += float(np.sum(vals * win))

    return {"localization": defect_w / total,
            "defect_atoms": list(defect_atoms),
            "defect_species": sorted(defect_species),
            "sigma_eV": sigma_eV,
            "method": "PAW-sphere projected weight fraction (not a real-space IPR)"}


def orbital_character(gap_level: dict, pdos: dict, sigma_eV: float = 0.10) -> dict:
    """Orbital character of a level from the projected DOS.
    Used by: main.
    Inputs: level, projected DOS, width (eV); returns: dict with the weights
    and the dominant channel.
    """
    weights, total = _weights_at(pdos, gap_level["energy_eV"], sigma_eV)
    if total <= 0:
        return {"character": {}, "dominant": None,
                "note": "no projected weight at this energy"}
    frac = {f"{s} {l}": w / total for (s, l), w in weights.items() if w > 0}
    frac = dict(sorted(frac.items(), key=lambda kv: -kv[1]))
    return {"character": frac,
            "dominant": next(iter(frac), None),
            "sigma_eV": sigma_eV}


def main():
    """Analyze the gap levels of one defect cell.
    Used by: command line.
    Writes results/defect_levels/<config>_<q>.json.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="defect configuration, e.g. er_ko")
    parser.add_argument("--charge", type=int, default=0, help="Charge state analyzed")
    parser.add_argument("--eig", type=Path, required=True,
                        help="ABINIT .abo whose sibling _GSR.nc holds the eigenvalues")
    parser.add_argument("--e-vbm", type=float, required=True,
                        help="Host VBM (eV) on the DEFECT cell's aligned energy scale")
    parser.add_argument("--e-cbm", type=float, required=True,
                        help="Host CBM (eV) on the DEFECT cell's aligned energy scale")
    parser.add_argument("--point-group", default="C3v",
                        help="Local point group for a1/e labels (default C3v; "
                             "pass the check_local_symmetry result for the relaxed cell)")
    parser.add_argument("--pdos-dir", type=Path,
                        help="Run directory holding the prtdos-3 *_DOS_AT* files "
                             "(usually the same dir as --eig). Enables orbital "
                             "character and the localization fraction.")
    parser.add_argument("--defect-atoms", type=int, nargs="+",
                        help="1-based atom indices forming the defect complex. "
                             "Default: every non-Si/C atom plus its first shell.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eig_data = parse_eigenvalues_and_occupations(args.eig)
    gap_levels = find_gap_levels(eig_data, args.e_vbm, args.e_cbm)
    gap_levels = assign_symmetry_labels(gap_levels, args.point_group)

    print(f"--- Gap levels for {args.config} (q={args.charge:+d}), "
          f"point group {args.point_group} ---")
    if not gap_levels:
        print("  (no states found strictly inside the VBM/CBM window)")
    for lvl in gap_levels:
        print(f"  E = {lvl['energy_above_vbm_eV']:.3f} eV above VBM  "
              f"[{lvl['occ_status']}, occ={lvl['occupation']:.3f}]  "
              f"{lvl.get('symmetry_label','?')} (deg {lvl.get('degeneracy','?')})  "
              f"band {lvl['band_index']} kpt {lvl['kpoint_index']}")
    pdos = None
    if args.pdos_dir:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from pdos_io import parse_projected_dos

        species = _species_order(args.eig)
        pdos = parse_projected_dos(args.pdos_dir, species)
        defect_atoms = args.defect_atoms or _defect_complex_indices(args.eig)
        print(f"  (defect complex = atoms {defect_atoms})")
        for lvl in gap_levels:
            lvl["orbital_character"] = orbital_character(lvl, pdos)
            lvl["localization"] = localization_from_pdos(lvl, pdos, defect_atoms)
            loc = lvl["localization"].get("localization")
            dom = lvl["orbital_character"]["dominant"]
            loc_s = f"{loc:.2f}" if loc is not None else "n/a"
            print(f"    E={lvl['energy_above_vbm_eV']:+.3f} eV: dominant {dom}, "
                  f"localization {loc_s}")
    else:
        print("  (no --pdos-dir given: orbital character and localization skipped)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": args.config,
            "charge": args.charge,
            "point_group": args.point_group,
            "e_vbm_eV": args.e_vbm,
            "e_cbm_eV": args.e_cbm,
            "pdos_source": str(args.pdos_dir) if args.pdos_dir else None,
            "gap_levels": gap_levels,
        }, f, indent=2)
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
