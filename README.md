# Erbium defects in 4H-SiC: first-principles data and analysis

This repository contains the calculation inputs and outputs, analysis code and
processed results for the manuscript

> M. Kuban and E. Towe, *Site Preference and Electronic Properties of Erbium
> Defects in 4H-SiC for Quantum Emitters* (submitted).

Nineteen erbium configurations in 4H-SiC were computed with density functional
theory (PBE+U, spin-polarized, explicit Er 4f shell) in 128-atom supercells.
The study reports formation energies, impurity solubility limits, binding
energies, charge-transition levels and projected densities of states.

## Contents

| Path | Contents |
| --- | --- |
| `abinit/production/` | ABINIT inputs (`.abi`) and main outputs (`.abo`) of all defect and pristine supercell calculations, and the atomic-position files they include |
| `abinit/references/` | Reference cells for the chemical potentials and the HSE06 band gap of the primitive cell |
| `abinit/<name>_run/` | Ground-state files (`_GSR.nc`) of each calculation |
| `abinit/common_parameters*.abi` | Settings shared by all PBE+U inputs and by the HSE06 inputs |
| `data/supplementary_data/` | Supplementary Data 1: relaxed structures (CIF) and absolute total energies of all reported cells |
| `data/pdos/` | Projected densities of states of the cells shown in the figures (compressed) |
| `results/` | Processed results (JSON) and the figures of the manuscript |
| `scripts/analysis/` | Formation energies, solubility limits, finite-size corrections, transition levels, U sensitivity |
| `scripts/postprocess/` | Gap-level analysis, plotting and Supplementary Data generation |
| `scripts/setup/build_supercell.py` | Construction of the defect supercells |
| `structures/` | Initial (unrelaxed) supercells of each configuration |
| `pseudopotentials/` | JTH v2.0 PAW datasets (GPL-3.0; see below) |
| `slurm/run_abinit.slurm` | Example batch script for running an input |

## Calculations

- ABINIT 10.4.7; JTH v2.0 PAW datasets; PBE exchange-correlation.
- Plane-wave cutoff 22 Ha, PAW fine-grid cutoff 44 Ha, Fermi-Dirac smearing 0.005 Ha.
- Hubbard U = 7.21 eV (0.265 Ha), J = 0, on Er 4f; collinear spin polarization; no spin-orbit coupling.
- 4x4x1 supercells (128 atoms) with a Gamma-centered 2x2x2 k mesh; a = 3.090 Å, c = 10.131 Å.
- Relaxations to a maximum force of 5e-5 Ha/bohr; SCF converged to a potential residual of 1e-10.
- Kumagai-Oba finite-size corrections for charged cells.

The manuscript's Methods section gives the full details.

### File naming

Input files are named `<config>_<calculation>.abi`.

| Config | Defect | Config | Defect |
| --- | --- | --- | --- |
| `er_h`, `er_k` | Er_Si on the h or k site | `er_hc`, `er_kc` | Er_C on the h or k site |
| `er_i` | Interstitial Er | `er_{h,k}o`, `er_{h,k}n` | Er_Si + O_C, Er_Si + N_C |
| `er_{h,k}p`, `er_{h,k}al`, `er_{h,k}b` | Er_Si + P_Si, Al_Si, B_Si | `er_{h,k}h` | Er_Si + interstitial H |
| `er_{h,k}v` | Er_Si + V_C | `o_c_only`, `n_c_only`, `h_i_only` | Partner defect without Er |

| Calculation | Meaning |
| --- | --- |
| `relax_spin` | Neutral relaxation. For `er_ho`, `relax2` and `relax3` are later passes; `relax3` reached the lowest-energy 4f occupation solution. |
| `relax_q{p2,p1,m1}_spin` | Relaxation in charge state +2, +1 or -1 |
| `align_q*_spin` | Single point at the neutral geometry that writes the potential for the finite-size correction |
| `pot_q*_spin` | Single point at the relaxed geometry of that charge state, with projected DOS |
| `occprobe_spin` | Neutral single point started from the relaxation's 4f occupation matrix, with projected DOS |
| `u{5,6,8}_q*_spin` | Single point at U = 5, 6 or 8 eV |
| `q{p2,p1,m1}_spin` | Vertical charge state at the neutral relaxed geometry |

Each input begins with a one-line description of the calculation.

## Reproducing the results

The analysis requires Python 3.10 or later and the packages in
`requirements.txt`:

```bash
pip install -r requirements.txt
bash scripts/make_figures.sh
```

`make_figures.sh` extracts the projected densities of states, recomputes the
formation energies, solubility limits and charge-transition levels from the
ABINIT outputs, and redraws the figures in `results/figures/`.

| Manuscript item | Script |
| --- | --- |
| Table 1, Fig. 2 | `formation_energy.py`, `solubility_limits.py`, `plot_production_formation_energies.py` |
| Table 2 | `binding_energies.py` |
| Fig. 3 | `plot_structure.py --view-block 2`; rendered in VESTA (`results/figures/er_ko_view_2x2x1.vesta`) |
| Fig. 4, Supplementary Fig. 4 | `plot_u_and_comparison.py`, `u_scan.py` |
| Fig. 5, Supplementary Figs. 3 and 5, Supplementary Table 2 | `charge_transition_levels.py`, `plot_transition_levels.py` |
| Supplementary Figs. 1 and 2 | `plot_pdos.py`, `defect_levels.py`, `plot_defect_levels.py` |
| Supplementary Table 1, Supplementary Data 1 | `make_si_data.py` |

The finite-size corrections in `results/transition_levels/*_kumagai.json` were
computed with `kumagai_correction.py` from the electrostatic-potential files of
the `align` calculations. Those files (about 0.4 GB) are not included here
because of their size.

To rerun a calculation, submit its input from the repository root, for example

```bash
sbatch --job-name=er_ko_relax_spin --export=INPUT=abinit/production/er_ko_relax_spin.abi slurm/run_abinit.slurm
```

The 128-atom calculations used two 128-core nodes of Bridges-2 at the Pittsburgh
Supercomputing Center.

## Pseudopotentials

`pseudopotentials/` contains unmodified PBE datasets from the JTH v2.0 table
(F. Jollet, M. Torrent and N. Holzwarth), renamed to their element symbols. They
are distributed under the GNU General Public License v3.0
(`pseudopotentials/LICENSE`) and are available from
https://www.abinit.org/atomic_data/paw/ and
https://github.com/abinit/paw_jth_datasets. Please cite F. Jollet *et al.*,
*Comput. Phys. Commun.* **185**, 1246 (2014).

## Citation

If you use these data or scripts, please cite the manuscript above (reference to
be added on publication).

## License

The code and data in this repository are released under the MIT License
(`LICENSE`). The PAW datasets in `pseudopotentials/` are distributed under
GPL-3.0.
