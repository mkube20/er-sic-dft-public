"""
build_supercell.py: build the 128-atom 4x4x1 supercell of a defect configuration.

Writes structures/<config>/ (CIF, ABINIT positions file and metadata) from the
PBE-relaxed 4H-SiC unit cell, with the defect placed at the site nearest the
cell center.

Usage:   python scripts/setup/build_supercell.py --config er_ko
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter


Z_NUMBER = {"Si": 14, "C": 6, "Er": 68, "N": 7, "O": 8,
            "Al": 13, "B": 5, "P": 15, "H": 1}
TYPE_RANK = {"Si": 0, "C": 1, "Er": 2, "N": 3, "O": 4,
             "Al": 5, "B": 6, "P": 7, "H": 8}

ANGSTROM_TO_BOHR = 1.8897259886


CONFIG_REGISTRY = {
    "pristine": {"site": None, "host_species": None, "interstitial": None,
                 "vacancy": None, "impurity": None,
                 "description": "No defect — reference structure"},
    "er_h": {"site": "h", "host_species": "Si", "interstitial": None,
             "vacancy": None, "impurity": None,
             "description": "Er substitutes Si at the hexagonal site"},
    "er_k": {"site": "k", "host_species": "Si", "interstitial": None,
             "vacancy": None, "impurity": None,
             "description": "Er substitutes Si at the quasi-cubic site"},
    "er_hc": {"site": "h", "host_species": "C", "interstitial": None,
              "vacancy": None, "impurity": None,
              "description": "Er substitutes C at the hexagonal site"},
    "er_i": {"site": None, "host_species": None, "interstitial": "T_Si",
             "vacancy": None, "impurity": None,
             "description": "Er as a tetrahedral interstitial near the Si sublattice"},
    "er_hv": {"site": "h", "host_species": "Si", "interstitial": None,
              "vacancy": "C", "impurity": None,
              "description": "Er_Si(h) + adjacent C vacancy"},
    "er_kv": {"site": "k", "host_species": "Si", "interstitial": None,
              "vacancy": "C", "impurity": None,
              "description": "Er_Si(k) + adjacent C vacancy"},
    "er_hn": {"site": "h", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": "N",
              "description": "Er_Si(h) + adjacent N substituting C (N_C pair)"},
    "er_ho": {"site": "h", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": "O",
              "description": "Er_Si(h) + adjacent O substituting C (O_C pair)"},

    "er_hp": {"site": "h", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": "P", "impurity_host": "Si",
              "description": "Er_Si(h) + adjacent P substituting Si (P_Si pair)"},
    "er_hal": {"site": "h", "host_species": "Si", "interstitial": None,
               "vacancy": None, "impurity": "Al", "impurity_host": "Si",
               "description": "Er_Si(h) + adjacent Al substituting Si (Al_Si pair)"},
    "er_hb": {"site": "h", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": "B", "impurity_host": "Si",
              "description": "Er_Si(h) + adjacent B substituting Si (B_Si pair)"},

    "er_ko": {"site": "k", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": "O",
              "description": "Er_Si(k) + adjacent O substituting C (O_C pair)"},
    "er_kn": {"site": "k", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": "N",
              "description": "Er_Si(k) + adjacent N substituting C (N_C pair)"},
    "er_kp": {"site": "k", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": "P", "impurity_host": "Si",
              "description": "Er_Si(k) + adjacent P substituting Si (P_Si pair)"},
    "er_kal": {"site": "k", "host_species": "Si", "interstitial": None,
               "vacancy": None, "impurity": "Al", "impurity_host": "Si",
               "description": "Er_Si(k) + adjacent Al substituting Si (Al_Si pair)"},
    "er_kb": {"site": "k", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": "B", "impurity_host": "Si",
              "description": "Er_Si(k) + adjacent B substituting Si (B_Si pair)"},

    "er_kc": {"site": "k", "host_species": "C", "interstitial": None,
              "vacancy": None, "impurity": None,
              "description": "Er substitutes C at the quasi-cubic site"},

    "er_kosi": {"site": "k", "host_species": "Si", "interstitial": None,
                "vacancy": None, "impurity": "O", "impurity_host": "Si",
                "description": "Er_Si(k) + adjacent O substituting Si (O_Si pair)"},
    "er_knsi": {"site": "k", "host_species": "Si", "interstitial": None,
                "vacancy": None, "impurity": "N", "impurity_host": "Si",
                "description": "Er_Si(k) + adjacent N substituting Si (N_Si pair)"},
    "er_kpc": {"site": "k", "host_species": "Si", "interstitial": None,
               "vacancy": None, "impurity": "P", "impurity_host": "C",
               "description": "Er_Si(k) + adjacent P substituting C (P_C pair)"},
    "er_kalc": {"site": "k", "host_species": "Si", "interstitial": None,
                "vacancy": None, "impurity": "Al", "impurity_host": "C",
                "description": "Er_Si(k) + adjacent Al substituting C (Al_C pair)"},
    "er_kbc": {"site": "k", "host_species": "Si", "interstitial": None,
               "vacancy": None, "impurity": "B", "impurity_host": "C",
               "description": "Er_Si(k) + adjacent B substituting C (B_C pair)"},
    "er_kvsi": {"site": "k", "host_species": "Si", "interstitial": None,
                "vacancy": "Si", "impurity": None,
                "description": "Er_Si(k) + adjacent Si vacancy"},

    "er_hosi": {"site": "h", "host_species": "Si", "interstitial": None,
                "vacancy": None, "impurity": "O", "impurity_host": "Si",
                "description": "Er_Si(h) + adjacent O substituting Si (O_Si pair)"},
    "er_hnsi": {"site": "h", "host_species": "Si", "interstitial": None,
                "vacancy": None, "impurity": "N", "impurity_host": "Si",
                "description": "Er_Si(h) + adjacent N substituting Si (N_Si pair)"},
    "er_hpc": {"site": "h", "host_species": "Si", "interstitial": None,
               "vacancy": None, "impurity": "P", "impurity_host": "C",
               "description": "Er_Si(h) + adjacent P substituting C (P_C pair)"},
    "er_halc": {"site": "h", "host_species": "Si", "interstitial": None,
                "vacancy": None, "impurity": "Al", "impurity_host": "C",
                "description": "Er_Si(h) + adjacent Al substituting C (Al_C pair)"},
    "er_hbc": {"site": "h", "host_species": "Si", "interstitial": None,
               "vacancy": None, "impurity": "B", "impurity_host": "C",
               "description": "Er_Si(h) + adjacent B substituting C (B_C pair)"},
    "er_hvsi": {"site": "h", "host_species": "Si", "interstitial": None,
                "vacancy": "Si", "impurity": None,
                "description": "Er_Si(h) + adjacent Si vacancy"},

    "n_c_only": {"site": "k", "host_species": "C", "interstitial": None,
                 "vacancy": None, "impurity": "N", "standalone": True,
                 "description": "Isolated N substituting C (no Er) - binding reference"},
    "o_c_only": {"site": "k", "host_species": "C", "interstitial": None,
                 "vacancy": None, "impurity": "O", "standalone": True,
                 "description": "Isolated O substituting C (no Er) - binding reference"},
    "h_i_only": {"site": "k", "host_species": None, "interstitial": None,
                 "vacancy": None, "impurity": None, "standalone": True,
                 "hydrogen": True,
                 "description": "Isolated interstitial H bonded to C (no Er) - binding reference"},

    "er_kh": {"site": "k", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": None, "hydrogen": True,
              "description": "Er_Si(k) + H bonded to a neighboring C (AB_C site)"},
    "er_hh": {"site": "h", "host_species": "Si", "interstitial": None,
              "vacancy": None, "impurity": None, "hydrogen": True,
              "description": "Er_Si(h) + H bonded to a neighboring C (AB_C site)"},
}


def parse_args():
    """Parse the command-line arguments.
    Used by: main.
    Returns: argparse namespace.
    """
    p = argparse.ArgumentParser(description="Build Er:4H-SiC supercell")
    p.add_argument("--config", required=True, choices=list(CONFIG_REGISTRY),
                   help="Defect configuration to build")
    p.add_argument("--supercell", nargs=3, type=int, default=[4, 4, 1],
                   help="Supercell multiplicity nx ny nz")
    p.add_argument("--charge", type=int, default=0,
                   help="Intended charge state (bookkeeping only — see module docstring)")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent.parent.parent / "structures",
                   help="Output directory root")
    return p.parse_args()


def build_pristine_4hsic():
    """4H-SiC unit cell with h and k site labels.
    Used by: build_defect_structure.
    Returns: pymatgen Structure (8 atoms).
    """
    a, c = 3.09, 10.13
    lattice = Lattice.from_parameters(a, a, c, 90.0, 90.0, 120.0)

    A = (0.0, 0.0)
    B = (1.0 / 3.0, 2.0 / 3.0)
    C = (2.0 / 3.0, 1.0 / 3.0)
    u = 3.0 / 16.0

    bilayers = [
        (A, 0.00, "k"),
        (B, 0.25, "h"),
        (A, 0.50, "k"),
        (C, 0.75, "h"),
    ]

    species, coords, labels = [], [], []
    for (lx, ly), z_si, hk in bilayers:
        species.append("Si")
        coords.append([lx, ly, z_si])
        labels.append(f"Si_{hk}")
        species.append("C")
        coords.append([lx, ly, z_si + u])
        labels.append(f"C_{hk}")

    return Structure(lattice, species, coords,
                     site_properties={"site_label": labels})


def _cell_center_cart(structure):
    """Cartesian coordinates of the cell center.
    Used by: substitute_er, add_hydrogen_standalone.
    Input: structure; returns: array (Angstrom).
    """
    return structure.lattice.get_cartesian_coords(np.array([0.5, 0.5, 0.5]))


def substitute_er(structure, site_type, host_species="Si", species="Er"):
    """Replace the central host atom of a sublattice with Er.
    Used by: build_defect_structure.
    Inputs: structure, site type, host element, new element; returns:
    structure.
    """
    target_label = f"{host_species}_{site_type}"
    labels = list(structure.site_properties["site_label"])
    center = _cell_center_cart(structure)

    best_idx, best_dist = None, None
    for i, site in enumerate(structure):
        if labels[i] != target_label:
            continue
        d = float(np.linalg.norm(site.coords - center))
        if best_dist is None or d < best_dist:
            best_dist, best_idx = d, i

    if best_idx is None:
        raise ValueError(
            f"No site labeled '{target_label}' found — cannot substitute Er. "
            f"Available labels: {sorted(set(labels))}"
        )

    structure.replace(best_idx, species,
                      properties={"site_label": f"{species}_{site_type}"})
    return structure


def add_interstitial_er(structure, interstitial_site="T_Si"):
    """Insert an Er atom at a tetrahedral interstitial site.
    Used by: build_defect_structure.
    Inputs: structure, site; returns: structure.
    """
    n = 12
    host_frac = structure.frac_coords

    best = None
    best_any = None
    for i in range(n):
        for j in range(n):
            for k in range(n):
                frac = np.array([i / n, j / n, k / n])
                dists = structure.lattice.get_all_distances(
                    frac.reshape(1, -1), host_frac)[0]
                order = np.argsort(dists)
                min_d = float(dists[order[0]])
                nn4 = [structure[int(o)].specie.symbol for o in order[:4]]
                si_frac = sum(s == "Si" for s in nn4) / 4.0

                if best_any is None or min_d > best_any[0]:
                    best_any = (min_d, si_frac, frac)
                if si_frac >= 0.5:
                    if best is None or min_d > best[0]:
                        best = (min_d, si_frac, frac)

    chosen = best if best is not None else best_any
    min_d, si_frac, frac = chosen
    if best is None:
        print(f"  WARNING: no Si-majority void found for '{interstitial_site}'; "
              f"using largest void instead (nn Si fraction {chosen[1]:.2f}).")
    if min_d < 1.5:
        print(f"  WARNING: interstitial site is close to a host atom "
              f"({min_d:.2f} Ang) - inspect in VESTA before running.")

    structure.append("Er", frac, coords_are_cartesian=False,
                     properties={"site_label": "Er_i"})
    print(f"  Interstitial Er placed at frac {frac.round(4).tolist()}, "
          f"min host distance {min_d:.2f} Ang, nn Si fraction {si_frac:.2f}")
    return structure


def add_vacancy(structure, er_index, vacancy_species="C"):
    """Remove the nearest neighbor of the given element to Er.
    Used by: build_defect_structure.
    Inputs: structure, Er index, element; returns: structure.
    """
    er_site = structure[er_index]
    neighbors = structure.get_neighbors(er_site, r=3.2)
    cands = [nb for nb in neighbors if nb.specie.symbol == vacancy_species]
    if not cands:
        raise ValueError(
            f"No {vacancy_species} neighbor within 3.2 Å of Er — "
            f"cannot create vacancy complex."
        )
    cands.sort(key=lambda nb: nb.nn_distance)
    structure.remove_sites([cands[0].index])
    return structure


def add_impurity_pair(structure, er_index, impurity_species, host_species="C"):
    """Replace the nearest host neighbor of Er with an impurity.
    Used by: build_defect_structure.
    Inputs: structure, Er index, impurity, host element; returns: structure.
    """
    radius = 3.2 if host_species == "C" else 3.8
    er_site = structure[er_index]
    neighbors = structure.get_neighbors(er_site, r=radius)
    cands = [nb for nb in neighbors if nb.specie.symbol == host_species]
    if not cands:
        raise ValueError(
            f"No {host_species} neighbor within {radius} Å of Er — cannot "
            f"form Er-{impurity_species} pair."
        )
    cands.sort(key=lambda nb: nb.nn_distance)
    structure.replace(cands[0].index, impurity_species,
                      properties={"site_label": f"{impurity_species}_{host_species}"})
    return structure


def add_hydrogen_at_c_neighbor(structure, er_index, bond_ang=1.10):
    """Add an interstitial H bonded to a carbon neighbor of Er.
    Used by: build_defect_structure.
    Inputs: structure, Er index, C-H bond length (Angstrom); returns:
    structure.
    """
    er = structure[er_index]
    nbrs = [nb for nb in structure.get_neighbors(er, r=3.2)
            if nb.specie.symbol == "C"]
    if not nbrs:
        raise ValueError("No carbon neighbor of Er — cannot place H.")
    nbrs.sort(key=lambda nb: nb.nn_distance)
    c_site = nbrs[0]

    d = c_site.coords - er.coords
    d = d / np.linalg.norm(d)
    h_cart = c_site.coords + bond_ang * d

    structure.append("H", h_cart, coords_are_cartesian=True,
                     properties={"site_label": "H_i"})
    return structure


def add_hydrogen_standalone(structure, site_type, bond_ang=1.10):
    """Add an interstitial H bonded to a carbon in a cell without Er.
    Used by: build_defect_structure.
    Inputs: structure, site type, C-H bond length (Angstrom); returns:
    structure.
    """
    target = f"C_{site_type}"
    labels = structure.site_properties["site_label"]
    center = _cell_center_cart(structure)
    cands = [i for i, l in enumerate(labels) if l == target]
    if not cands:
        raise ValueError(f"no site labeled {target}")
    c_index = min(cands,
                  key=lambda i: float(np.linalg.norm(structure[i].coords - center)))
    c_site = structure[c_index]
    si = [nb for nb in structure.get_neighbors(c_site, r=2.6)
          if nb.specie.symbol == "Si"]
    if not si:
        raise ValueError("carbon has no silicon neighbor")
    si.sort(key=lambda nb: nb.nn_distance)
    d = c_site.coords - si[0].coords
    d = d / np.linalg.norm(d)
    structure.append("H", c_site.coords + bond_ang * d,
                     coords_are_cartesian=True,
                     properties={"site_label": "H_i"})
    return structure


def write_abinit_positions(structure, output_path):
    """Write the typat and xred block of an ABINIT input.
    Used by: main.
    Inputs: structure, output path.
    """
    present = sorted({s.symbol for s in structure.species},
                     key=lambda s: TYPE_RANK[s])
    type_index = {s: i + 1 for i, s in enumerate(present)}
    typat = [type_index[site.specie.symbol] for site in structure]
    znucl = [Z_NUMBER[s] for s in present]

    lines = [
        "# Auto-generated by build_supercell.py - DO NOT hand-edit.",
        "# The main .abi file must declare matching ntypat/znucl/natom; this",
        "# file provides the active typat and xred blocks (include this file).",
        f"# ntypat {len(present)}",
        f"# znucl  {' '.join(str(z) for z in znucl)}   # {' '.join(present)}",
        f"# natom  {len(structure)}",
        "",
        "typat " + " ".join(str(t) for t in typat),
        "xred",
    ]
    for site in structure:
        f = site.frac_coords
        lines.append(f"   {f[0]:.8f}  {f[1]:.8f}  {f[2]:.8f}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_defect_structure(config, supercell_dims):
    """Build the supercell of one configuration.
    Used by: main.
    Inputs: config, supercell dimensions; returns: pymatgen Structure.
    """
    spec = CONFIG_REGISTRY[config]

    supercell = build_pristine_4hsic()
    supercell.make_supercell(supercell_dims)

    if config == "pristine":
        return supercell

    if spec.get("standalone"):
        if spec.get("hydrogen"):
            return add_hydrogen_standalone(supercell, spec["site"])
        return substitute_er(supercell, spec["site"], spec["host_species"],
                             species=spec["impurity"])

    if spec["interstitial"] is not None:
        structure = add_interstitial_er(supercell, spec["interstitial"])
    else:
        structure = substitute_er(supercell, spec["site"], spec["host_species"])

    er_idx = [i for i, s in enumerate(structure) if s.specie.symbol == "Er"][0]

    if spec["vacancy"] is not None:
        structure = add_vacancy(structure, er_idx, spec["vacancy"])

    if spec["impurity"] is not None:
        structure = add_impurity_pair(structure, er_idx, spec["impurity"],
                                      spec.get("impurity_host", "C"))

    if spec.get("hydrogen"):
        structure = add_hydrogen_at_c_neighbor(structure, er_idx)

    return structure


def main():
    """Write the structure files of one configuration.
    Used by: command line.
    Writes structures/<config>/.
    """
    args = parse_args()

    structure = build_defect_structure(args.config, args.supercell)

    out_dir = args.output_dir / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = args.supercell
    tag = f"{args.config}_{nx}x{ny}x{nz}"

    CifWriter(structure).write_file(str(out_dir / f"{tag}.cif"))

    write_abinit_positions(structure, out_dir / f"positions_{args.config}.abi")

    metadata = {
        "config": args.config,
        "description": CONFIG_REGISTRY[args.config]["description"],
        "supercell": args.supercell,
        "intended_charge": args.charge,
        "natom": len(structure),
        "lattice_a_bohr": structure.lattice.a * ANGSTROM_TO_BOHR,
        "lattice_c_bohr": structure.lattice.c * ANGSTROM_TO_BOHR,
        "composition": structure.composition.formula,
    }
    with open(out_dir / "structure_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built {tag} (q={args.charge:+d}): {metadata['natom']} atoms, "
          f"{metadata['composition']} -> {out_dir}")


if __name__ == "__main__":
    main()
