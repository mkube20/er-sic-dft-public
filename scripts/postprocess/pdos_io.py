"""
pdos_io.py: readers for ABINIT ground-state (_GSR.nc) and projected-DOS
(prtdos 3, *_DOS_AT<nnnn>) output, shared by the plotting and analysis scripts.
"""

import re
from pathlib import Path

import numpy as np

HARTREE = 27.211386245988
HARTREE_TO_EV = HARTREE
L_NAMES = ["s", "p", "d", "f", "g"]


def gsr_band_edges(path):
    """k points, VBM, CBM and gap from a _GSR.nc file.
    Used by: plot_pdos.
    Input: _GSR.nc path; returns: dict (eV).
    """
    import h5py
    with h5py.File(path, "r") as h:
        e = np.array(h["eigenvalues"]) * HARTREE
        o = np.array(h["occupations"])
        k = np.array(h["reduced_coordinates_of_kpoints"])
    occ = o > 0.5
    return dict(k=k, vbm=e[occ].max(), cbm=e[~occ].min(),
                gap=e[~occ].min() - e[occ].max())


def read_spin_dos(path, l=3):
    """One angular-momentum channel of a projected-DOS file, per spin.
    Used by: plot_pdos, plot_u_and_comparison.
    Inputs: _DOS_AT file, l; returns: list of (energy eV, DOS states/eV) per
    spin.
    """
    txt = Path(path).read_text(errors="replace").splitlines()
    marks = [i for i, s in enumerate(txt)
             if "Spin-up DOS" in s or "Spin-dn DOS" in s or "Spin-down DOS" in s]
    if not marks:
        marks = [0]
    bounds = marks + [len(txt)]
    out = []
    for n in range(len(marks)):
        rows = []
        for s in txt[bounds[n]:bounds[n + 1]]:
            t = s.strip()
            if not t or t.startswith("#"):
                continue
            v = t.split()
            if len(v) >= 11:
                rows.append([float(x) for x in v])
        if rows:
            a = np.array(rows)
            out.append((a[:, 0] * HARTREE, a[:, 1 + l] / HARTREE))
    return out


def parse_abinit_dos(dos_file: Path) -> dict:
    """All channels of one projected-DOS file.
    Used by: parse_projected_dos.
    Input: _DOS_AT file; returns: dict of energies (eV) and DOS by channel.
    """
    lines = Path(dos_file).read_text(encoding="utf-8", errors="replace").splitlines()
    header = [ln for ln in lines if ln.lstrip().startswith("#")]
    data_rows = [ln.split() for ln in lines
                 if ln.strip() and not ln.lstrip().startswith("#")]
    if not data_rows:
        raise ValueError(f"{dos_file}: no numeric rows found — is this a prtdos-3 file?")

    arr = np.array([[float(x) for x in row] for row in data_rows])
    ncol = arr.shape[1]

    col_line = None
    for ln in reversed(header):
        if "l=" in ln or "energy" in ln.lower():
            col_line = ln
            break
    if col_line is None:
        raise ValueError(
            f"{dos_file}: could not find a column-describing header line.\n"
            f"Header was:\n" + "\n".join(header) +
            "\nRefusing to guess column order — inspect the file and extend "
            "parse_abinit_dos for this ABINIT output variant.")

    head = re.split(r"\(\s*integral", col_line, maxsplit=1)[0]
    l_indices = [int(m) for m in re.findall(r"l\s*=\s*(\d+)", head)]
    if not l_indices:
        raise ValueError(
            f"{dos_file}: header line has no 'l=<n>' tokens before the integral "
            f"marker, so the l-channels cannot be identified:\n  {col_line}\n"
            "Refusing to guess — extend parse_abinit_dos for this variant.")
    if len(l_indices) > ncol - 1:
        raise ValueError(
            f"{dos_file}: header names {len(l_indices)} l-channels but the file "
            f"has only {ncol - 1} non-energy columns:\n  {col_line}")

    out_l = {}
    for k, l in enumerate(l_indices):
        name = L_NAMES[l] if l < len(L_NAMES) else f"l{l}"
        out_l[name] = arr[:, 1 + k]

    energies = arr[:, 0] * HARTREE_TO_EV
    total = np.sum([v for v in out_l.values()], axis=0)
    return {"energies_eV": energies, "l": out_l, "total": total, "header": header}


def parse_projected_dos(run_dir: Path, structure_species) -> dict:
    """Projected DOS of every atom in a run directory.
    Used by: defect_levels.
    Inputs: run directory, element symbols in atom order; returns: dict by atom
    and by species.
    """
    files = sorted(Path(run_dir).glob("*_DOS_AT*"))
    if not files:
        raise FileNotFoundError(
            f"no *_DOS_AT* files in {run_dir} — the run needs `prtdos 3`, and "
            "the pull must not exclude *_DOS*.")

    by_atom, energies = {}, None
    for f in files:
        m = re.search(r"_DOS_AT(\d+)", f.name)
        if not m:
            continue
        idx = int(m.group(1))
        d = parse_abinit_dos(f)
        if energies is None:
            energies = d["energies_eV"]
        elif len(energies) != len(d["energies_eV"]):
            raise ValueError(f"{f}: energy grid length differs from the first DOS file")
        if idx > len(structure_species):
            raise ValueError(
                f"{f}: atom index {idx} exceeds the {len(structure_species)} atoms "
                "supplied — structure_species must match the run's atom order.")
        by_atom[idx] = {"species": structure_species[idx - 1], "l": d["l"]}

    by_species = {}
    for info in by_atom.values():
        acc = by_species.setdefault(info["species"], {})
        for lname, vals in info["l"].items():
            acc[lname] = acc.get(lname, 0.0) + vals
    return {"energies_eV": energies, "by_atom": by_atom, "by_species": by_species}
