#!/usr/bin/env python
"""
make_si_data.py: Supplementary Table 1 and Supplementary Data 1.

Reads the total energy, charge and magnetization of every reported cell from
its ABINIT output and writes the relaxed structures as CIF files.

Output:  data/supplementary_data/total_energies.csv,
         data/supplementary_data/structures/<cell>.cif,
         data/supplementary_table_energies.tex
Usage:   python scripts/postprocess/make_si_data.py
"""
import csv
import glob
import re
from pathlib import Path

from abipy.abilab import abiopen

REPO = Path(__file__).resolve().parents[2]
AB = REPO / "abinit"
OUTDIR = REPO / "data/supplementary_data"
TEX = REPO / "data/supplementary_table_energies.tex"
HA = 27.211386245988

CAPTION = (r"\textbf{Absolute total energies of all cells used in this work.} "
           r"ABINIT total energies in Ha, computed with the settings of "
           r"Supplementary Table~4. The formation energies, binding energies, and "
           r"transition levels in the main text are obtained from these values "
           r"and the chemical potentials of Supplementary Table~3; the "
           r"finite-size corrections for charged cells are given in the main "
           r"text. $q$ is the net charge of the cell, and the moment is the "
           r"total cell magnetization in $\mu_{\rm B}$ (blank for "
           r"spin-unpolarized cells). Rows labeled 4H-SiC are pristine cells of 4H "
           r"silicon carbide (4H-SiC). The density-of-states (DOS) statics reproduce "
           r"the energies of their relaxations at the same geometry to within "
           r"0.1~$\mu$Ha. Supplementary Data~1 gives the same values together "
           r"with the output file from which each was read.")

P, R = "production", "references"
CELLS = [
    ("er_ko", r"Er$_{\rm Si}(k)$+O$_{\rm C}$", "screen", "er_ko_relax_spin", P, True),
    ("er_ho", r"Er$_{\rm Si}(h)$+O$_{\rm C}$", "screen", "er_ho_relax3_spin", P, True),
    ("er_kn", r"Er$_{\rm Si}(k)$+N$_{\rm C}$", "screen", "er_kn_relax_spin", P, True),
    ("er_hn", r"Er$_{\rm Si}(h)$+N$_{\rm C}$", "screen", "er_hn_relax_spin", P, True),
    ("er_kp", r"Er$_{\rm Si}(k)$+P$_{\rm Si}$", "screen", "er_kp_relax_spin", P, True),
    ("er_hp", r"Er$_{\rm Si}(h)$+P$_{\rm Si}$", "screen", "er_hp_relax_spin", P, True),
    ("er_kh", r"Er$_{\rm Si}(k)$+H$_i$", "screen", "er_kh_relax_spin", P, True),
    ("er_hh", r"Er$_{\rm Si}(h)$+H$_i$", "screen", "er_hh_relax_spin", P, True),
    ("er_kv", r"Er$_{\rm Si}(k)$+V$_{\rm C}$", "screen", "er_kv_relax_spin", P, True),
    ("er_hv", r"Er$_{\rm Si}(h)$+V$_{\rm C}$", "screen", "er_hv_relax_spin", P, True),
    ("er_k", r"Er$_{\rm Si}(k)$", "screen", "er_k_relax_spin", P, True),
    ("er_h", r"Er$_{\rm Si}(h)$", "screen", "er_h_relax_spin", P, True),
    ("er_kal", r"Er$_{\rm Si}(k)$+Al$_{\rm Si}$", "screen", "er_kal_relax_spin", P, True),
    ("er_hal", r"Er$_{\rm Si}(h)$+Al$_{\rm Si}$", "screen", "er_hal_relax_spin", P, True),
    ("er_kb", r"Er$_{\rm Si}(k)$+B$_{\rm Si}$", "screen", "er_kb_relax_spin", P, True),
    ("er_hb", r"Er$_{\rm Si}(h)$+B$_{\rm Si}$", "screen", "er_hb_relax_spin", P, True),
    ("er_kc", r"Er$_{\rm C}(k)$", "screen", "er_kc_relax_spin", P, True),
    ("er_hc", r"Er$_{\rm C}(h)$", "screen", "er_hc_relax_spin", P, True),
    ("er_i", r"Er$_i$", "screen", "er_i_relax_spin", P, True),
    ("pristine", "4H-SiC, 128 atoms", "host", "pristine_supercell_gamma", P, True),
    ("o_c", r"O$_{\rm C}$ (no Er)", "partner", "o_c_only_relax_spin", P, True),
    ("n_c", r"N$_{\rm C}$ (no Er)", "partner", "n_c_only_relax_spin", P, True),
    ("h_i", r"H$_i$ (no Er)", "partner", "h_i_only_relax_spin", P, True),
    ("er_ko_qp2", r"Er$_{\rm Si}(k)$+O$_{\rm C}$, $q=+2$", "charged", "er_ko_relax_qp2_spin", P, True),
    ("er_ko_qp1", r"Er$_{\rm Si}(k)$+O$_{\rm C}$, $q=+1$", "charged", "er_ko_relax_qp1_spin", P, True),
    ("er_ko_qm1", r"Er$_{\rm Si}(k)$+O$_{\rm C}$, $q=-1$", "charged", "er_ko_relax_qm1_spin", P, True),
    ("er_ko_dos", r"Er$_{\rm Si}(k)$+O$_{\rm C}$, DOS static", "static", "er_ko_occprobe_spin", P, False),
    ("er_k_dos", r"Er$_{\rm Si}(k)$, DOS static", "static", "er_k_occprobe_spin", P, False),
    ("ref_sic", "4H-SiC unit cell", "reference", "pristine_4hsic", R, True),
    ("ref_si", "Si (diamond)", "reference", "si_bulk", R, True),
    ("ref_c", "graphite", "reference", "graphite", R, True),
    ("ref_er", "Er metal (hcp, spin-polarized)", "reference", "er_metal_spin", R, True),
    ("ref_o2", r"O$_2$ molecule (triplet)", "reference", "o2_molecule", R, True),
    ("ref_n2", r"N$_2$ molecule", "reference", "n2_molecule", R, True),
    ("ref_h2", r"H$_2$ molecule", "reference", "h2_molecule", R, True),
    ("ref_al", "Al (fcc)", "reference", "al_metal", R, True),
    ("ref_p", "P (black)", "reference", "p_bulk", R, True),
    ("ref_b", r"B ($\alpha$-rhombohedral)", "reference", "b_bulk", R, True),
]


def last(pattern, text, cast=float):
    """Last regular-expression match in a text.
    Used by: read_abo.
    Inputs: pattern, text, type; returns: value or None.
    """
    hits = re.findall(pattern, text)
    return cast(hits[-1]) if hits else None


def read_abo(path):
    """Total energy, atom count, charge and magnetization of a run.
    Used by: main.
    Input: .abo path; returns: dict.
    """
    t = Path(path).read_text(errors="ignore")
    return {
        "etotal_Ha": last(r"\n\s*etotal\d*\s+(-?\d+\.\d+E[+-]\d+)", t),
        "natom": last(r"\n\s*natom\s+(\d+)", t, int),
        "charge": last(r"\n\s*(?:cell)?charge\s+(-?\d+\.?\d*(?:E[+-]\d+)?)", t) or 0.0,
        "magnetization_muB": last(r"Magnetization \(Bohr magneton\)=\s*(-?\d+\.\d+E[+-]\d+)", t),
    }


def find_abo(stem, sub):
    """Path of a run's .abo file.
    Used by: main.
    Inputs: run name, subdirectory; returns: path.
    """
    for cand in (AB / sub / f"{stem}.abo", AB / f"{stem}_run" / f"{stem}.abo"):
        if cand.exists():
            return cand
    raise FileNotFoundError(stem)


def find_gsr(stem):
    """Path of a run's _GSR.nc file.
    Used by: main.
    Input: run name; returns: path or None.
    """
    hits = sorted(glob.glob(str(AB / f"{stem}_run" / "*_GSR.nc"))) or \
        sorted(glob.glob(str(AB / "references" / f"{stem}*_GSR.nc")))
    return hits[-1] if hits else None


def main():
    """Write Supplementary Table 1 and Supplementary Data 1.
    Used by: command line.
    Writes total_energies.csv, the CIF files and the LaTeX table.
    """
    (OUTDIR / "structures").mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for cid, label, group, stem, sub, want_struct in CELLS:
        abo = find_abo(stem, sub)
        info = read_abo(abo)
        struct_file = ""
        if want_struct:
            gsr = find_gsr(stem)
            if gsr:
                with abiopen(gsr) as g:
                    st = g.structure
                struct_file = f"structures/{cid}.cif"
                st.to(filename=str(OUTDIR / struct_file))
            else:
                missing.append(stem)
        rows.append(dict(cell=cid, label=label, group=group, natom=info["natom"],
                         charge=int(round(info["charge"])),
                         etotal_Ha=info["etotal_Ha"],
                         etotal_eV=info["etotal_Ha"] * HA,
                         magnetization_muB=info["magnetization_muB"],
                         structure=struct_file,
                         source=str(abo.relative_to(REPO)).replace("\\", "/")))

    with open(OUTDIR / "total_energies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "label"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "label"})

    lines = []
    group_title = {"screen": "Neutral configurations of the screen (relaxed)",
                   "host": "Host", "partner": "Isolated partner defects (relaxed)",
                   "charged": "Charged cells (relaxed)",
                   "static": "Density-of-states statics (relaxed geometry)",
                   "reference": "Chemical-potential references"}
    current = None
    for r in rows:
        if r["group"] != current:
            current = r["group"]
            lines.append(r"\midrule" if lines else "")
            lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{group_title[current]}}}}}\\")
        mag = "" if r["magnetization_muB"] is None else f"{r['magnetization_muB']:.3f}"
        q = f"{r['charge']:+d}" if r["charge"] else "0"
        lines.append(f"{r['label']} & {r['natom']} & {q} & "
                     f"{r['etotal_Ha']:.8f} & {mag}\\\\")
    head = [r"\begin{small}", r"\begin{longtable}{lrrrr}",
            r"\caption{" + CAPTION + r"}", r"\label{tab:energies}\\",
            r"\toprule",
            r"Cell & Atoms & $q$ & $E_{\rm tot}$ (Ha) & Moment ($\mu_{\rm B}$)\\",
            r"\midrule", r"\endfirsthead",
            r"\multicolumn{5}{l}{\small\textit{Supplementary Table~1, continued}}\\",
            r"\toprule",
            r"Cell & Atoms & $q$ & $E_{\rm tot}$ (Ha) & Moment ($\mu_{\rm B}$)\\",
            r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    tail = [r"\end{longtable}", r"\end{small}"]
    TEX.write_text("\n".join(head + [l for l in lines if l] + tail) + "\n",
                   encoding="utf-8")

    print(f"{len(rows)} cells -> {OUTDIR / 'total_energies.csv'}")
    print(f"structures written: {sum(1 for r in rows if r['structure'])}")
    if missing:
        print("NO GSR (structure not written):", ", ".join(missing))


if __name__ == "__main__":
    main()
