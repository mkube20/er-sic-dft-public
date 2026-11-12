#!/bin/bash
# Regenerate the analysis results and manuscript figures from the ABINIT
# outputs in this repository. Run from the repository root.
set -e

# Projected densities of states (compressed) -> abinit/<run>/
for f in data/pdos/*.tar.xz; do tar -xJf "$f" -C abinit; done

# Formation energies and solubility limits (Table 1, Fig. 2)
python scripts/analysis/formation_energy.py --spin --all-configs \
    --output results/formation_energies/formation_energies_spin.json
python scripts/analysis/solubility_limits.py
python scripts/analysis/binding_energies.py
python scripts/postprocess/plot_production_formation_energies.py

# Relaxed charge-transition levels (Fig. 5, Supplementary Fig. 3)
for c in er_ko er_ho; do
    python scripts/analysis/charge_transition_levels.py --config "$c" --from-abo \
        --relaxed --spin --charges 2 1 -1 --gap 2.212 \
        --corrections "results/transition_levels/${c}_kumagai.json"
done
# Vertical levels (Supplementary Table 2, Supplementary Fig. 5)
for c in er_hn er_kv er_hv; do
    python scripts/analysis/charge_transition_levels.py --config "$c" --from-abo \
        --spin --charges 2 1 -1 --gap 2.212 \
        --corrections results/transition_levels/er_ho_kumagai.json
done
for c in er_ko er_ho er_hn er_kv er_hv; do
    python scripts/postprocess/plot_transition_levels.py \
        --input "results/transition_levels/${c}_transition_levels.json" \
        --output "results/figures/${c}_transition_diagram.pdf"
done

# Hubbard-U sensitivity (Supplementary Fig. 4) and Er 4f/5d PDOS (Fig. 4)
python scripts/analysis/u_scan.py --config er_ko
python scripts/postprocess/plot_u_and_comparison.py --left er_k --right er_ko

# Gap levels (Supplementary Fig. 2) and full PDOS (Supplementary Fig. 1)
python scripts/postprocess/defect_levels.py --config er_ko --charge 0 \
    --eig abinit/production/er_ko_occprobe_spin.abo --e-vbm 7.6953 --e-cbm 9.9072 \
    --pdos-dir abinit/er_ko_occprobe_spin_run --output results/defect_levels/er_ko_q0.json
for q in qp1 qm1; do
    charge=$([ "$q" = qp1 ] && echo 1 || echo -1)
    python scripts/postprocess/defect_levels.py --config er_ko --charge "$charge" \
        --eig "abinit/production/er_ko_pot_${q}_spin.abo" --e-vbm 7.6953 --e-cbm 9.9072 \
        --pdos-dir "abinit/er_ko_pot_${q}_spin_run" --output "results/defect_levels/er_ko_${q}.json"
done
python scripts/postprocess/plot_defect_levels.py --levels results/defect_levels/er_ko_qp1.json \
    results/defect_levels/er_ko_q0.json results/defect_levels/er_ko_qm1.json \
    --output results/figures/er_ko_defect_levels.pdf
python scripts/postprocess/plot_pdos.py --run abinit/er_ko_occprobe_spin_run \
    --stem er_ko_occprobe_spino_DS1_DOS_AT --er 75 --output results/figures/er_ko_pdos.pdf
