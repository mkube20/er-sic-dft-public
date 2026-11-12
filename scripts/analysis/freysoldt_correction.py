"""
freysoldt_correction.py: planar-average Freysoldt (FNV) correction, used as a
cross-check of kumagai_correction.py and for the Makov-Payne image energy.

Reads ABINIT potentials (prtpot 1) with abipy and calls
pymatgen.analysis.defects.corrections.freysoldt. In the 4x4x1 cell the planar
alignment has no bulk-like plateau, so the reported corrections are those of
kumagai_correction.py.

Usage:   python scripts/analysis/freysoldt_correction.py --config er_ko --image-only
"""

from pathlib import Path

HARTREE_TO_EV = 27.211386245988

_WARNED_NO_VXC = set()


def _read_field(nc_path):
    """Read an ABINIT NetCDF potential file.
    Used by: abinit_pot_to_locpot.
    Input: path; returns: (grid in eV, pymatgen Structure).
    """
    from abipy.abilab import abiopen
    from pymatgen.core import Structure

    with abiopen(str(nc_path)) as pf:
        field = pf.field
        datar = field.datar
        struct = field.structure

    grid = datar[0].real * HARTREE_TO_EV
    return grid, Structure.from_dict(struct.as_dict())


def abinit_pot_to_locpot(pot_path, vxc_path=None):
    """Electrostatic potential of an ABINIT run as a pymatgen Locpot.
    Used by: the correction functions in this module.
    Inputs: _POT.nc path, optional _VXC.nc path; returns: Locpot (eV).
    """
    from pymatgen.io.vasp.outputs import Locpot

    grid, structure = _read_field(pot_path)
    if vxc_path is None:
        name = Path(pot_path).name
        if name not in _WARNED_NO_VXC:
            _WARNED_NO_VXC.add(name)
            print(f"  WARNING: no VXC alongside {name} — using the TOTAL KS "
                  "potential. The alignment term will be contaminated by V_xc; "
                  "rerun with `prtvxc 1`. (See this function's docstring.)")
        return Locpot(structure, {"total": grid})

    vxc, _ = _read_field(vxc_path)
    if vxc.shape != grid.shape:
        raise ValueError(f"POT grid {grid.shape} != VXC grid {vxc.shape} for {pot_path}")
    return Locpot(structure, {"total": grid - vxc})


def _vxc_for(pot_path):
    """Path of the _VXC.nc file next to a _POT.nc file.
    Used by: the correction functions in this module.
    Input: _POT.nc path; returns: path or None.
    """
    p = Path(pot_path)
    vxc = p.with_name(p.name.replace("_POT.nc", "_VXC.nc"))
    return vxc if vxc.exists() else None


def freysoldt_correction(q, dielectric, defect_pot_path, bulk_pot_path,
                         defect_frac_coords=None, lattice=None):
    """Freysoldt correction for one charged defect state.
    Used by: the correction functions in this module.
    Inputs: q, dielectric constant, defect and bulk potential paths, defect
    position; returns: (correction eV, result).
    """
    from pymatgen.analysis.defects.corrections.freysoldt import get_freysoldt_correction

    defect_locpot = abinit_pot_to_locpot(defect_pot_path, _vxc_for(defect_pot_path))
    bulk_locpot = abinit_pot_to_locpot(bulk_pot_path, _vxc_for(bulk_pot_path))
    result = get_freysoldt_correction(
        q=q, dielectric=dielectric,
        defect_locpot=defect_locpot, bulk_locpot=bulk_locpot,
        defect_frac_coords=defect_frac_coords, lattice=lattice)
    return float(result.correction_energy), result


def _charge_tag(q):
    """File tag of a charge state: q0, qp1, qm1, ...
    Used by: the correction functions in this module.
    Input: q (int); returns: str.
    """
    if q == 0:
        return "q0"
    return f"q{'p' if q > 0 else 'm'}{abs(q)}"


def image_only_corrections(config, charges, prod_dir=Path("abinit/production"),
                           dielectric=9.7):
    """Image-charge (Makov-Payne) term for each charge state, without alignment.
    Used by: main (--image-only).
    Inputs: config, charges, production directory, dielectric constant;
    returns: {q: eV}.
    """
    prod_dir = Path(prod_dir)
    ref = _locate_pot_nc(f"{config}_align_q0", prod_dir)
    out = {}
    for q in charges:
        if q == 0:
            continue
        dp = _locate_pot_nc(f"{config}_align_{_charge_tag(q)}", prod_dir)
        if dp is None or ref is None:
            print(f"  NOTE: alignment-set potentials missing for q={q:+d} — skipped.")
            continue
        d_locpot = abinit_pot_to_locpot(dp, _vxc_for(dp))
        fc = defect_frac_coords_of(d_locpot)
        _, res = freysoldt_correction(q, dielectric, dp, ref, defect_frac_coords=fc)
        e_img = float(res.metadata["electrostatic"])
        out[q] = e_img
        print(f"  q={q:+d}: E_image = {e_img:+.4f} eV   "
              f"(alignment {float(res.metadata['mean_alignments']):+.3f} eV OMITTED "
              f"— see docstring)")
    if out:
        print("\n  q^2 check on the image term (this part IS reliable):")
        for q, e in sorted(out.items()):
            print(f"    q={q:+d}: E_image/q^2 = {e / (q * q):+.4f} eV")
    return out


def corrections_from_alignment_set(config, charges, prod_dir=Path("abinit/production"),
                                   dielectric=9.7):
    """Freysoldt corrections from the single points at the neutral geometry.
    Used by: main (--alignment-set).
    Inputs: config, charges, production directory, dielectric constant;
    returns: ({q: eV}, sanity flag).
    """
    prod_dir = Path(prod_dir)
    ref_base = f"{config}_align_q0"
    ref = _locate_pot_nc(ref_base, prod_dir)
    if ref is None:
        raise FileNotFoundError(
            f"{ref_base}o_POT.nc not found — the neutral common-geometry run is "
            "the reference every other charge state is measured against. Run "
            "./run_alignment.sh and pull before using --alignment-set.")
    ref_vxc = _vxc_for(ref)
    if ref_vxc is None:
        raise FileNotFoundError(
            f"{ref_base}o_VXC.nc not found — the electrostatic potential is "
            "V_POT - V_XC, so both components are required (prtvxc 1).")
    ref_locpot = abinit_pot_to_locpot(ref, ref_vxc)

    out = {}
    for q in charges:
        if q == 0:
            continue
        base = f"{config}_align_{_charge_tag(q)}"
        dp = _locate_pot_nc(base, prod_dir)
        if dp is None:
            print(f"  NOTE: {base}o_POT.nc not found — skipping q={q:+d}.")
            continue
        d_locpot = abinit_pot_to_locpot(dp, _vxc_for(dp))
        fc = defect_frac_coords_of(d_locpot)
        e_corr, res = freysoldt_correction(q, dielectric, dp, ref,
                                           defect_frac_coords=fc)
        out[q] = e_corr
        al = res.metadata["alignments"]
        spread = max(al.values()) - min(al.values())
        print(f"  q={q:+d}: E_corr = {e_corr:+.3f} eV "
              f"(image {float(res.metadata['electrostatic']):+.3f}, "
              f"alignment {float(res.metadata['mean_alignments']):+.3f}, "
              f"axis spread {spread:.3f})")
    return out, _report_sanity(out)


def _run_completed(base, prod_dir):
    """Whether an ABINIT run finished.
    Used by: _locate_pot_nc.
    Inputs: run name, production directory; returns: (bool, reason).
    """
    abo = Path(prod_dir).parent / f"{base}_run" / f"{base}.abo"
    if not abo.exists():
        return False, f"{abo.name} not found"
    text = abo.read_text(encoding="utf-8", errors="replace")
    if "Calculation completed" not in text:
        err = abo.parent / "__ABI_MPIABORTFILE__"
        detail = ""
        if err.exists():
            msg = [l.strip() for l in err.read_text(encoding="utf-8",
                                                    errors="replace").splitlines()
                   if l.strip() and not l.strip().startswith(("---", "...", "src_",
                                                             "mpi_rank", "message"))]
            if msg:
                detail = " — ABINIT said: " + " ".join(msg[:3])
        return False, f"{abo.name} does not report 'Calculation completed'{detail}"
    return True, ""


def _locate_pot_nc(base, prod_dir):
    """Path of a finished run's _POT.nc file.
    Used by: the correction functions in this module.
    Inputs: run name, production directory; returns: path or None.
    """
    ok, why = _run_completed(base, prod_dir)
    p = Path(prod_dir).parent / f"{base}_run" / f"{base}o_POT.nc"
    if not p.exists():
        return None
    if not ok:
        raise RuntimeError(
            f"{p} exists but its run did not complete ({why}).\n"
            "This is almost certainly a STALE file left behind by an earlier "
            "attempt: `pull` overwrites but never deletes. Delete the run "
            "directory's .nc files and rerun the job — do not correct against it.")
    return p


def defect_frac_coords_of(locpot, species="Er"):
    """Fractional coordinates of the defect atom.
    Used by: the correction functions in this module.
    Inputs: Locpot, element symbol; returns: [x, y, z].
    """
    sites = [s for s in locpot.structure if s.specie.symbol == species]
    if len(sites) != 1:
        raise ValueError(f"expected exactly one {species} site, found {len(sites)}")
    return list(sites[0].frac_coords)


def corrections_for_config(config, charges, prod_dir=Path("abinit/production"),
                           dielectric=9.7):
    """Freysoldt corrections from the single points at each relaxed geometry.
    Used by: main.
    Inputs: config, charges, production directory, dielectric constant;
    returns: ({q: eV}, sanity flag).
    """
    prod_dir = Path(prod_dir)
    bulk = _locate_pot_nc("pristine_supercell_prtpot", prod_dir)
    if bulk is None:
        raise FileNotFoundError(
            "bulk pristine_supercell_prtpoto_POT.nc not found — run "
            "pristine_supercell_prtpot.abi (./run_freysoldt_pot.sh) and pull.")
    out = {}
    for q in charges:
        if q == 0:
            continue
        base = f"{config}_pot_{_charge_tag(q)}"
        defect = _locate_pot_nc(base, prod_dir)
        if defect is None:
            print(f"  NOTE: {base}o_POT.nc not found — skipping q={q:+d}.")
            continue
        if _vxc_for(defect) is None:
            print(f"  WARNING: {base}o_VXC.nc missing — this run predates the "
                  "`prtvxc 1` fix and its alignment term is not trustworthy.")
        fc = defect_frac_coords_of(abinit_pot_to_locpot(defect, _vxc_for(defect)))
        e_corr, res = freysoldt_correction(q, dielectric, defect, bulk,
                                           defect_frac_coords=fc)
        out[q] = e_corr
        elec = float(res.metadata["electrostatic"])
        align = float(res.metadata["mean_alignments"])
        print(f"  q={q:+d}: E_corr = {e_corr:+.3f} eV "
              f"(image {elec:+.3f}, alignment {align:+.3f})")
    return out, _report_sanity(out)


def _report_sanity(corr):
    """Check that the corrections are positive and scale roughly as q^2.
    Used by: corrections_for_config, corrections_from_alignment_set.
    Input: {q: eV}; returns: bool.
    """
    if not corr:
        return False
    print("\n  --- sanity check ---")
    ok = True
    scaled = {}
    for q, e in sorted(corr.items()):
        scaled[q] = e / (q * q)
        flag = "" if e > 0 else "   <-- NEGATIVE, expected > 0"
        if e <= 0:
            ok = False
        print(f"    q={q:+d}: E_corr={e:+.3f} eV   E_corr/q^2 = {scaled[q]:+.3f}{flag}")
    if len(scaled) > 1:
        vals = list(scaled.values())
        spread = (max(vals) - min(vals)) / max(abs(v) for v in vals)
        print(f"    spread in E_corr/q^2: {spread*100:.0f}%")
        if spread > 0.5:
            ok = False
            print("      <-- q^2 scaling badly violated (>50% spread)")
    if ok:
        print("  PASS: corrections are positive and scale ~q^2.")
    else:
        print("  FAIL: do NOT use these corrections. Most likely causes, in order:\n"
              "    1. the runs lack `prtvxc 1`, so the alignment still contains V_xc\n"
              "       (see abinit_pot_to_locpot docstring);\n"
              "    2. a POT came from an SCF that did not converge to the same\n"
              "       electronic state as the relaxation it is meant to correct.")
    return ok


def _self_test():
    """Test the correction on a synthetic point-charge potential.
    Used by: main (when no --config is given).
    Returns: correction (eV).
    """
    import numpy as np
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.vasp.outputs import Locpot
    from pymatgen.analysis.defects.corrections.freysoldt import get_freysoldt_correction

    a = 10.0
    lat = Lattice.cubic(a)
    struct = Structure(lat, ["Si"], [[0.5, 0.5, 0.5]])
    n = 48
    bulk = np.zeros((n, n, n))
    ax = np.linspace(0, 1, n, endpoint=False)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    defect = 0.5 * np.exp(-r2 / (2 * 0.02))
    bulk_lp = Locpot(struct, {"total": bulk})
    def_lp = Locpot(struct, {"total": defect})
    res = get_freysoldt_correction(q=1, dielectric=9.7, defect_locpot=def_lp,
                                   bulk_locpot=bulk_lp,
                                   defect_frac_coords=[0.5, 0.5, 0.5])
    e = float(res.correction_energy)
    print(f"synthetic Freysoldt call OK: correction_energy = {e:.4f} eV")
    assert np.isfinite(e), "correction is not finite"
    print("metadata keys:", list(res.metadata.keys())[:6])
    return e


def main():
    """Compute and write Freysoldt corrections.
    Used by: command line.
    Writes results/transition_levels/<config>_freysoldt.json.
    """
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="Config, e.g. er_ho. If omitted, run the self-test.")
    ap.add_argument("--charges", nargs="+", type=int, default=[-2, -1, 1, 2])
    ap.add_argument("--prod-dir", type=Path, default=Path("abinit/production"))
    ap.add_argument("--dielectric", type=float, default=9.7,
                    help="Static eps_0 of 4H-SiC (Patrick & Choyke 1970)")
    ap.add_argument("--output", type=Path,
                    default=Path("results/transition_levels"))
    ap.add_argument("--image-only", action="store_true",
                    help="Write ONLY the Madelung image term as the correction, omitting "
                         "the potential alignment. This is the defensible fallback for "
                         "this supercell — see image_only_corrections() for why, and note "
                         "that the omitted alignment is a stated uncertainty, not zero.")
    ap.add_argument("--alignment-set", action="store_true",
                    help="Use the <config>_align_<tag> runs (all charge states at the "
                         "COMMON neutral geometry) and take the NEUTRAL DEFECT cell as "
                         "the alignment reference instead of the pristine cell. This is "
                         "the well-conditioned route — see corrections_from_alignment_set.")
    ap.add_argument("--force", action="store_true",
                    help="write the JSON even if the positivity/q^2 sanity check fails")
    args = ap.parse_args()

    if not args.config:
        _self_test()
        return

    if args.image_only:
        print(f"--- Image-term-only corrections for {args.config} "
              f"(eps={args.dielectric}); ALIGNMENT OMITTED ---")
        corr = image_only_corrections(args.config, args.charges, args.prod_dir,
                                      args.dielectric)
        args.output.mkdir(parents=True, exist_ok=True)
        out = args.output / f"{args.config}_freysoldt.json"
        out.write_text(json.dumps({
            "config": args.config,
            "dielectric": args.dielectric,
            "scheme": "image-term only (Madelung); potential alignment OMITTED",
            "alignment_omitted": True,
            "caveat": ("The supercell (4x4x1, one 4H repeat along c) cannot determine "
                       "the FNV alignment; see image_only_corrections(). eps(+1/-1) is "
                       "unaffected by this correction since both states have |q|=1. "
                       "Both levels move -1.0 eV per 1 eV of the omitted alignment."),
            "sanity_passed": True,
            "corrections_eV": {str(q): e for q, e in corr.items()},
        }, indent=2))
        print(f"\nWritten: {out}")
        return

    if args.alignment_set:
        print(f"--- Freysoldt corrections for {args.config} (eps={args.dielectric}), "
              "COMMON-GEOMETRY alignment set ---")
        corr, passed = corrections_from_alignment_set(
            args.config, args.charges, args.prod_dir, args.dielectric)
    else:
        print(f"--- Freysoldt corrections for {args.config} (eps={args.dielectric}) ---")
        corr, passed = corrections_for_config(args.config, args.charges, args.prod_dir,
                                              args.dielectric)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{args.config}_freysoldt.json"
    if not passed and not args.force:
        if out.exists():
            out.unlink()
            print(f"\nRemoved stale {out} (previous corrections did not pass).")
        print("NOT written — sanity check failed. Fix the runs and re-run; pass "
              "--force only if you deliberately want the failing numbers on disk.")
        raise SystemExit(1)
    out.write_text(json.dumps({"config": args.config, "dielectric": args.dielectric,
                               "sanity_passed": bool(passed),
                               "corrections_eV": {str(q): e for q, e in corr.items()}},
                              indent=2))
    print(f"\nWritten: {out}")
    print("Apply with: charge_transition_levels.py --from-abo --relaxed --config "
          f"{args.config} --charges {' '.join(map(str, args.charges))} "
          f"--corrections {out}")
    print("(The positivity / q^2 sanity check ran above — if it printed FAIL, the "
          "numbers in this JSON must not be used.)")


if __name__ == "__main__":
    main()
