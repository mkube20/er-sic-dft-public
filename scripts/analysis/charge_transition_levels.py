"""
charge_transition_levels.py: stable charge states and transition levels.

Builds E_f(q; E_F) for each charge state from the ABINIT outputs, applies the
finite-size corrections, finds the lower envelope across the gap and solves for
the crossing points eps(q/q') in closed form. --relaxed uses the relaxed charged
cells; otherwise the charged cells are single points at the neutral geometry
(vertical levels).

Output:  results/transition_levels/<config>_transition_levels.json
Usage:   python scripts/analysis/charge_transition_levels.py --config er_ko --from-abo \
             --relaxed --spin --gap 2.212
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formation_energy as fe


def _as_int_charge(k):
    """Convert a charge key ('1', '+1', '-2' or int) to int.
    Used by: lower_envelope.
    Input: key; returns: int.
    """
    return int(str(k).replace("+", ""))


def lower_envelope(charge_data):
    """Find the charge states that are lowest in energy over the Fermi-level sweep.
    Used by: transition_diagram, main.
    Input: sweep from formation_energy_vs_efermi; returns: list of (q, E_F
    start, E_F end) segments.
    """
    charges = [_as_int_charge(k) for k in charge_data]
    grids = [charge_data[k]["e_fermi"] for k in charge_data]
    efs = grids[0]
    n = len(efs)
    ef_arrays = {c: charge_data[k]["e_f"] for c, k in zip(charges, charge_data)}

    segments = []
    cur_q, seg_start = None, efs[0]
    for i in range(n):
        best_q = min(charges, key=lambda c: (ef_arrays[c][i], c))
        if cur_q is None:
            cur_q, seg_start = best_q, efs[i]
        elif best_q != cur_q:
            segments.append((cur_q, seg_start, efs[i]))
            cur_q, seg_start = best_q, efs[i]
    segments.append((cur_q, seg_start, efs[-1]))
    return segments


def transition_level(charge_q, charge_qprime, E_defect_by_charge, E_pristine,
                     n_atoms_removed, mu_dict, e_vbm, corr_q=0.0, corr_qprime=0.0):
    """Solve the transition level eps(q/q') in closed form.
    Used by: transition_diagram.
    Inputs: the two charges, total energies, stoichiometry, chemical
    potentials, E_VBM and corrections; returns: eps (eV above the VBM).
    """
    if charge_q == charge_qprime:
        raise ValueError("transition_level needs two distinct charge states.")
    ef_q0 = fe.formation_energy(E_defect_by_charge[charge_q], E_pristine,
                                n_atoms_removed, mu_dict, charge=charge_q,
                                e_fermi=0.0, e_vbm=e_vbm, e_corr=corr_q)
    ef_qp0 = fe.formation_energy(E_defect_by_charge[charge_qprime], E_pristine,
                                 n_atoms_removed, mu_dict, charge=charge_qprime,
                                 e_fermi=0.0, e_vbm=e_vbm, e_corr=corr_qprime)
    return (ef_q0 - ef_qp0) / (charge_qprime - charge_q)


def transition_diagram(E_defect_by_charge, E_pristine, n_atoms_removed, mu_dict,
                       e_vbm, e_gap, corrections=None, n_points=501):
    """Stable charge states, transition levels and the E_f(q; E_F) sweep for one configuration.
    Used by: main.
    Inputs: energies by charge, pristine energy, stoichiometry, mu, E_VBM, gap,
    corrections; returns: dict.
    """
    if corrections is None:
        corrections = {q: 0.0 for q in E_defect_by_charge}

    sweep = fe.formation_energy_vs_efermi(
        E_defect_by_charge, E_pristine, n_atoms_removed, mu_dict, e_vbm, e_gap,
        corrections=corrections, n_points=n_points)

    envelope = lower_envelope(sweep)

    transitions = []
    for i in range(len(envelope) - 1):
        q_lo, _, _ = envelope[i]
        q_hi, _, _ = envelope[i + 1]
        eps = transition_level(q_lo, q_hi, E_defect_by_charge, E_pristine,
                               n_atoms_removed, mu_dict, e_vbm,
                               corrections.get(q_lo, 0.0), corrections.get(q_hi, 0.0))
        transitions.append({"q_lo": q_lo, "q_hi": q_hi, "epsilon_eV": eps})

    stable = [q for q, _, _ in envelope]
    return {"stable_states": stable, "transitions": transitions, "sweep": sweep}


def _charge_tag(q):
    """File tag of a charge state: q0, qp1, qm1, ...
    Used by: load_energies_from_abo.
    Input: q (int); returns: str.
    """
    return f"q{'p' if q > 0 else 'm'}{abs(q)}"


def _neutral_abo(config, prod_dir, spin):
    """Path of the neutral relaxation used as the q = 0 reference.
    Used by: load_energies_from_abo.
    Inputs: config, production directory, spin flag; returns: (path, relaxation
    pass).
    """
    if not spin:
        return prod_dir / f"{config}_relax.abo", None
    pass2 = prod_dir / f"{config}_relax2_spin.abo"
    if pass2.exists():
        return pass2, 2
    return prod_dir / f"{config}_relax_spin.abo", 1


def load_energies_from_abo(config, prod_dir, ref_dir_root, charges=(-2, -1, 1, 2),
                           relaxed=False, spin=False):
    """Read every energy a transition-level calculation needs from the ABINIT outputs.
    Used by: main.
    Inputs: config, directories, charges, relaxed/spin flags; returns:
    energies, E_pristine, E_VBM, mu, stoichiometry, sources.
    """
    prod_dir = Path(prod_dir)
    sfx = "_spin" if spin else ""

    neutral_abo, neutral_pass = _neutral_abo(config, prod_dir, spin)
    if not neutral_abo.exists():
        raise FileNotFoundError(
            f"{neutral_abo} not found — the q=0 reference for "
            f"{'the spin-polarized' if spin else 'the nsppol-1'} campaign is missing.")
    E_by_q = {0: fe.extract_total_energy(neutral_abo)}
    for q in charges:
        name = (f"{config}_relax_{_charge_tag(q)}{sfx}.abo" if relaxed
                else f"{config}_{_charge_tag(q)}{sfx}.abo")
        abo = prod_dir / name
        if not abo.exists():
            print(f"  NOTE: {abo.name} not found (still running?) — skipping q={q:+d}. "
                  f"Envelope is PARTIAL.")
            continue
        E_by_q[q] = fe.extract_total_energy(abo)

    pristine = prod_dir / ("pristine_supercell_gamma.abo" if spin
                           else "pristine_supercell.abo")
    E_pristine = fe.extract_total_energy(pristine)
    e_vbm = fe.extract_vbm(pristine)

    ref_files = {
        "sic_bulk":    "references/pristine_4hsic.abo",
        "si_bulk":     "references/si_bulk.abo",
        "graphite":    "references/graphite.abo",
        "er_metal":    "references/er_metal.abo",
        "n2_molecule": "references/n2_molecule.abo",
        "o2_molecule": "references/o2_molecule.abo",
    }
    if spin:
        missing = [(k, p) for k, p in fe.SPIN_REFERENCES.items()
                   if not (Path(ref_dir_root) / p).exists()]
        if missing:
            raise FileNotFoundError(
                "REFUSING to score --spin: open-shell reference(s) not recomputed "
                "spin-polarized: " + ", ".join(f"{k} ({p})" for k, p in missing))
        ref_files.update(fe.SPIN_REFERENCES)
    refs = {k: fe.extract_total_energy(Path(ref_dir_root) / v) / fe.REFERENCE_DIVISORS[k]
            for k, v in ref_files.items()}
    mu = fe.compute_chemical_potentials(refs)
    stoich = fe.DEFECT_STOICHIOMETRY[config]
    return E_by_q, E_pristine, e_vbm, mu, stoich, {
        "neutral_abo": neutral_abo.name, "neutral_geometry_pass": neutral_pass,
        "pristine_abo": pristine.name, "spin": spin}


def main():
    """Compute and write the transition levels of one configuration.
    Used by: command line.
    Writes results/transition_levels/<config>_transition_levels.json.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config name (e.g. er_ho); selects stoichiometry and files")
    parser.add_argument("--from-abo", action="store_true",
                        help="Read energies directly from the ABINIT .abo outputs")
    parser.add_argument("--relaxed", action="store_true",
                        help="use the charged relaxations "
                             "(<config>_relax_<tag>.abo) instead of the vertical single-points")
    parser.add_argument("--charges", nargs="+", type=int, default=[-2, -1, 1, 2],
                        help="Charge states to include (default -2 -1 1 2; pass the "
                             "subset you actually ran)")
    parser.add_argument("--spin", action="store_true",
                        help="read the spin-polarized calculations: "
                             "<config>_relax2_spin.abo (else _relax_spin.abo) for q=0, "
                             "<config>[_relax]_<tag>_spin.abo for the charged states, "
                             "pristine_supercell_gamma.abo and the spin-polarized "
                             "references. Required for all reported results.")
    parser.add_argument("--corrections", type=Path,
                        help="Freysoldt corrections JSON (from freysoldt_correction.py "
                             "--config ...); applies E_corr(q) to make the levels "
                             "publication-grade (Freysoldt-corrected).")
    parser.add_argument("--abinit-root", type=Path, default=Path("abinit"),
                        help="Root holding production/ and references/ (default: abinit)")
    parser.add_argument("--input", type=Path,
                        help="Alternatively, a pre-built formation_energy_vs_efermi grid JSON")
    parser.add_argument("--gap", type=float, default=2.212,
                        help="Gap (eV) for the E_F axis. Default 2.51 = DFT PBE+U gap "
                             "from the pristine GSR; use 3.23 for the experimental gap.")
    parser.add_argument("--limit", choices=["Si_rich", "C_rich"], default="Si_rich",
                        help="Chemical-potential limit for --from-abo (levels vs VBM are "
                             "the same in both for stoichiometric defects; matters otherwise)")
    parser.add_argument("--output", type=Path, default=Path("results/transition_levels"),
                        help="Output directory for the CTL table + figure data")
    args = parser.parse_args()

    if args.from_abo:
        E_by_q, E_pristine, e_vbm, mu, stoich, provenance = load_energies_from_abo(
            args.config, args.abinit_root / "production", args.abinit_root,
            charges=tuple(args.charges), relaxed=args.relaxed, spin=args.spin)
        print(f"Sources: q=0 {provenance['neutral_abo']}"
              + (f" (geometry pass {provenance['neutral_geometry_pass']})"
                 if provenance["neutral_geometry_pass"] else "")
              + f", pristine {provenance['pristine_abo']}, "
              + ("spin-polarized campaign" if args.spin else "nsppol-1 campaign"))
        corrections = None
        corrected = False
        if args.corrections:
            cdata = json.loads(args.corrections.read_text())["corrections_eV"]
            corrections = {int(str(k).replace("+", "")): v for k, v in cdata.items()}
            corrections.setdefault(0, 0.0)
            corrected = True
            print(f"Applying Freysoldt corrections (eV): {corrections}")
        result = transition_diagram(E_by_q, E_pristine, stoich, mu[args.limit],
                                    e_vbm, args.gap, corrections=corrections)
    elif args.input:
        with open(args.input) as f:
            sweep = json.load(f)
        envelope = lower_envelope(sweep)
        result = {"stable_states": [q for q, _, _ in envelope],
                  "transitions": [{"q_lo": envelope[i][0], "q_hi": envelope[i + 1][0],
                                   "epsilon_eV": envelope[i][2]}
                                  for i in range(len(envelope) - 1)],
                  "sweep": sweep}
        corrected = False
    else:
        parser.error("give either --from-abo or --input <grid.json>")

    print(f"--- {args.config}: stable charge states (lower envelope) ---")
    print(f"    {result['stable_states']}   "
          f"[{'UNCORRECTED — vertical screen, no Freysoldt term' if not corrected else 'corrected'}]")
    if not result["transitions"]:
        print("    Only one charge state is ever stable in the gap "
              "(no transition level in range).")
    for t in result["transitions"]:
        print(f"    epsilon({t['q_lo']:+d}/{t['q_hi']:+d}) = "
              f"{t['epsilon_eV']:.3f} eV above VBM")

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / f"{args.config}_transition_levels.json"
    payload = {"config": args.config, "gap_eV": args.gap, "corrected": corrected,
               "geometry": ("relaxed" if getattr(args, "relaxed", False) else "vertical"),
               "stable_charge_states": result["stable_states"],
               "transition_levels": result["transitions"]}
    if args.from_abo:
        payload["sources"] = provenance
    if args.from_abo:
        payload["sweep"] = result["sweep"]
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten: {out_file}")


if __name__ == "__main__":
    main()
