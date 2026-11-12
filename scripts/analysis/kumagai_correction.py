"""
kumagai_correction.py: Kumagai-Oba finite-size correction for charged defects.

Uses the anisotropic static dielectric tensor (eps_perp = 9.66, eps_par = 10.03;
Patrick and Choyke, 6H-SiC) and the atomic-site electrostatic potentials of the
charged and neutral cells at a common geometry. Sites outside the sphere
inscribed in the supercell are sampled, and the site-to-site scatter of the
alignment is reported. Kumagai and Oba, Phys. Rev. B 89, 195205 (2014).

Input:   abinit/<config>_align_<q>_spin_run/*_POT.nc and the pristine potential
         (abinit/pristine_supercell_prtpot_run/); these files are not in the
         repository because of their size.
Output:  results/transition_levels/<config>_kumagai.json
Usage:   python scripts/analysis/kumagai_correction.py --config er_ko --spin --charges 2 1 -1
"""

import argparse
import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

HA_EV = 27.211386245988
BOHR_ANG = 0.529177210903

EPS_PERP, EPS_PARA = 9.66, 10.03


def _field(path, name):
    """Read one variable from an ABINIT NetCDF file.
    Used by: electrostatic_potential.
    Inputs: path, variable name; returns: array.
    """
    ds = nc.Dataset(path)
    v = np.array(ds.variables[name][0, :, :, :, 0])
    ds.close()
    return v


def _geometry(path):
    """Atomic positions, lattice vectors and charges from an ABINIT NetCDF file.
    Used by: main.
    Input: path; returns: (xred, primitive vectors, Z per atom).
    """
    ds = nc.Dataset(path)
    xred = np.array(ds.variables["reduced_atom_positions"][:])
    prim = np.array(ds.variables["primitive_vectors"][:])
    znucl = np.array(ds.variables["atomic_numbers"][:])
    species = np.array(ds.variables["atom_species"][:])
    ds.close()
    z_per_atom = np.array([int(znucl[s - 1]) for s in species])
    return xred, prim, z_per_atom


def pot_path(run_dir, base, kind):
    """Path of <base>o_<kind>.nc in a run directory, with or without the _DS1_ infix.
    Used by: electrostatic_potential, main.
    Inputs: run directory, run name, kind (POT or VXC); returns: path.
    """
    d = Path(run_dir)
    for name in (f"{base}o_{kind}.nc", f"{base}o_DS1_{kind}.nc"):
        if (d / name).exists():
            return d / name
    return d / f"{base}o_{kind}.nc"


def electrostatic_potential(run_dir, base):
    """Electrostatic potential of a run, phi = -(v_trial - v_xc).
    Used by: main.
    Inputs: run directory, run name; returns: grid (eV).
    """
    pot = _field(pot_path(run_dir, base, "POT"), "vtrial")
    vxc = _field(pot_path(run_dir, base, "VXC"), "exchange_correlation_potential")
    return -(pot - vxc) * HA_EV


def sample_at_sites(field, xred):
    """Trilinear interpolation of a periodic grid at atomic sites.
    Used by: main.
    Inputs: grid, fractional coordinates; returns: values at the sites.
    """
    n3, n2, n1 = field.shape
    n = np.array([n1, n2, n3])
    g = xred * n
    lo = np.floor(g).astype(int)
    frac = g - lo
    out = np.zeros(len(xred))
    for c1 in (0, 1):
        for c2 in (0, 1):
            for c3 in (0, 1):
                w = (np.where(c1, frac[:, 0], 1 - frac[:, 0])
                     * np.where(c2, frac[:, 1], 1 - frac[:, 1])
                     * np.where(c3, frac[:, 2], 1 - frac[:, 2]))
                i1 = (lo[:, 0] + c1) % n1
                i2 = (lo[:, 1] + c2) % n2
                i3 = (lo[:, 2] + c3) % n3
                out += w * field[i3, i2, i1]
    return out


def ewald_anisotropic(rvecs, eps, gamma=None, nreal=3, nrecip=8):
    """Ewald potential of a periodic point charge in an anisotropic dielectric.
    Used by: main.
    Inputs: lattice vectors (bohr), dielectric tensor, Ewald parameters;
    returns: (V(r) function, self term).
    """
    A = np.asarray(rvecs, float)
    omega = abs(np.linalg.det(A))
    B = 2 * np.pi * np.linalg.inv(A).T
    eps = np.asarray(eps, float)
    eps_inv = np.linalg.inv(eps)
    det_eps = np.linalg.det(eps)

    if gamma is None:
        gamma = np.pi ** 0.5 / omega ** (1.0 / 3.0)

    rng_r = range(-nreal, nreal + 1)
    Rs = np.array([i * A[0] + j * A[1] + k * A[2]
                   for i in rng_r for j in rng_r for k in rng_r])
    rng_g = range(-nrecip, nrecip + 1)
    Gs = np.array([i * B[0] + j * B[1] + k * B[2]
                   for i in rng_g for j in rng_g for k in rng_g])
    gnorm2 = np.einsum("ij,jk,ik->i", Gs, eps, Gs)
    keep = gnorm2 > 1e-12
    Gs, gnorm2 = Gs[keep], gnorm2[keep]
    gfac = 4 * np.pi / omega * np.exp(-gnorm2 / (4 * gamma ** 2)) / gnorm2

    def V(r):
        """Point-charge potential at a position.
        Used by: ewald_anisotropic.
        Input: Cartesian position (bohr); returns: potential (Ha/e).
        """
        r = np.atleast_2d(np.asarray(r, float))
        d = r[:, None, :] + Rs[None, :, :]
        s = np.sqrt(np.einsum("pri,ij,prj->pr", d, eps_inv, d))
        s = np.where(s < 1e-10, np.inf, s)
        real = (np.erfc(gamma * s) / s).sum(axis=1) / np.sqrt(det_eps) \
            if hasattr(np, "erfc") else \
            (_erfc(gamma * s) / s).sum(axis=1) / np.sqrt(det_eps)
        recip = (gfac[None, :] * np.cos(r @ Gs.T)).sum(axis=1)
        return real + recip - np.pi / (omega * gamma ** 2)

    d0 = Rs
    s0 = np.sqrt(np.einsum("ri,ij,rj->r", d0, eps_inv, d0))
    nz = s0 > 1e-10
    real0 = (_erfc(gamma * s0[nz]) / s0[nz]).sum() / np.sqrt(det_eps)
    real0 += -2 * gamma / np.sqrt(np.pi * det_eps)
    recip0 = gfac.sum()
    V_self = real0 + recip0 - np.pi / (omega * gamma ** 2)
    return V, V_self


def _erfc(x):
    """Complementary error function, vectorized.
    Used by: ewald_anisotropic.
    Input: array; returns: array.
    """
    from math import erfc as _e
    return np.vectorize(_e)(x)


def inscribed_radius(rvecs):
    """Radius of the largest sphere inside the cell.
    Used by: main.
    Input: lattice vectors (bohr); returns: radius (bohr).
    """
    A = np.asarray(rvecs, float)
    out = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        n = np.cross(A[j], A[k])
        n /= np.linalg.norm(n)
        out.append(abs(A[i] @ n) / 2)
    return min(out)


def main():
    """Compute and write the Kumagai-Oba corrections of one configuration.
    Used by: command line.
    Writes results/transition_levels/<config>_kumagai.json.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="er_ko")
    ap.add_argument("--spin", action="store_true",
                    help="read the SPIN-POLARIZED alignment set "
                         "(<config>_align_q*_spin) instead of the nsppol-1 one")
    ap.add_argument("--charges", nargs="+", type=int, default=[2, 1, -1])
    ap.add_argument("--abinit-root", type=Path, default=Path("abinit"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/transition_levels"))
    args = ap.parse_args()

    tag = {2: "qp2", 1: "qp1", -1: "qm1", -2: "qm2"}
    root = args.abinit_root
    sfx = "_spin" if args.spin else ""
    ref_base = f"{args.config}_align_q0{sfx}"
    ref_dir = root / f"{ref_base}_run"
    ref_pot = pot_path(ref_dir, ref_base, "POT")
    if not ref_pot.exists():
        print(f"missing neutral reference potential: {ref_pot}", file=sys.stderr)
        return 1

    xred, prim, _ = _geometry(ref_pot)
    phi0 = electrostatic_potential(ref_dir, ref_base)

    eps_aniso = np.diag([EPS_PERP, EPS_PERP, EPS_PARA])
    rmax = inscribed_radius(prim)

    _, _, z_per_atom = _geometry(ref_pot)
    er_idx = int(np.where(z_per_atom == 68)[0][0])
    center = xred[er_idx]

    dfrac = xred - center
    dfrac -= np.round(dfrac)
    dcart = dfrac @ prim
    dist = np.linalg.norm(dcart, axis=1)
    sampling = dist > rmax

    print(f"--- Kumagai-Oba (eFNV) corrections for {args.config}"
          f"{' (spin-polarized set)' if args.spin else ''} ---")
    print(f"  dielectric tensor: diag({EPS_PERP}, {EPS_PERP}, {EPS_PARA})  "
          f"(4H-SiC uniaxial; freysoldt_correction.py uses scalar 9.7)")
    print(f"  inscribed-sphere radius R_max = {rmax:.3f} Bohr "
          f"({rmax*BOHR_ANG:.2f} Ang)")
    print(f"  sampling sites (d > R_max): {sampling.sum()} of {len(xred)}\n")

    _, vself_iso = ewald_anisotropic(prim, np.eye(3) * 9.7)
    e_pc_iso = -0.5 * vself_iso * HA_EV
    print(f"  VALIDATION isotropic eps=9.7: lattice term at |q|=1 = "
          f"{e_pc_iso:+.4f} eV  (freysoldt_correction.py image term: +0.1915 eV)")

    Vfun, vself = ewald_anisotropic(prim, eps_aniso)
    e_pc_1 = -0.5 * vself * HA_EV
    print(f"  anisotropic tensor:          lattice term at |q|=1 = "
          f"{e_pc_1:+.4f} eV\n")

    results = {}
    for q in args.charges:
        base = f"{args.config}_align_{tag[q]}{sfx}"
        rundir = root / f"{base}_run"
        if not pot_path(rundir, base, "POT").exists():
            print(f"  q={q:+d}: no alignment-set potential — skipped.")
            continue
        dV = electrostatic_potential(rundir, base) - phi0
        dV_sites = sample_at_sites(dV, xred)

        v_pc = q * Vfun(dcart) * HA_EV

        resid = dV_sites - v_pc
        align = float(resid[sampling].mean())
        scatter = float(resid[sampling].std())
        e_pc = q ** 2 * e_pc_1
        e_corr = e_pc - q * align

        results[q] = {"E_pc_eV": e_pc, "alignment_eV": align,
                      "alignment_scatter_eV": scatter, "E_corr_eV": e_corr,
                      "n_sampling_sites": int(sampling.sum())}
        print(f"  q={q:+d}:  E_pc = {e_pc:+.4f}   dV_align = {align:+.4f} "
              f"+/- {scatter:.4f}   E_corr = {e_corr:+.4f} eV")

    if results:
        print("\n  q^2 check on the lattice term (must be exact):")
        for q, r in results.items():
            print(f"    q={q:+d}: E_pc/q^2 = {r['E_pc_eV']/q**2:+.4f} eV")

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{args.config}_kumagai.json"
    out.write_text(json.dumps({
        "config": args.config,
        "method": "Kumagai-Oba eFNV, anisotropic dielectric tensor, "
                  "neutral-defect-cell reference at common geometry",
        "dielectric_tensor": [[EPS_PERP, 0, 0], [0, EPS_PERP, 0], [0, 0, EPS_PARA]],
        "R_max_bohr": rmax,
        "alignment_set": f"{args.config}_align_q*{sfx}",
        "provenance_warning": (
            None if args.spin else
            "alignment set is nsppol 1 on the nsppol-1 relaxed geometry; "
            "rerun with --spin if a spin-polarized set exists"),
        "corrections_eV": {str(q): r["E_corr_eV"] for q, r in results.items()},
        "detail": {str(q): r for q, r in results.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
