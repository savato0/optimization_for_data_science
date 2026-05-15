**1. Generazione dati**

Eseguire i notebook in ordine:

jupyter nbconvert --to notebook --execute generate_matrices.ipynb --inplace
jupyter nbconvert --to notebook --execute generate_vectors.ipynb --inplace

Generare punti iniziali:

python scripts/generate_initial_points.py

Controllare gli scenari x_u:

python scripts/summarize_targets.py private/data/dim_n100_k10

## **2. Frank-Wolfe su tutti gli x0**

python scripts/run_fw_experiments.py private/data/dim_n100_k10 \
  --case Q_well:q_well_sc1 \
  --case Q_well:q_well_sc2 \
  --case Q_well:q_well_sc3 \
  --case Q_ill:q_ill_sc1 \
  --case Q_ill:q_ill_sc2 \
  --case Q_ill:q_ill_sc3 \
  --x0-file private/data/dim_n100_k10/initial_points.npz \
  --all-x0 \
  --max-iter 10000 \
  --tol-gap 1e-6 \
  --quiet \
  --include-solution \
  --output private/data/dim_n100_k10/fw_all_x0_results.json

## **3. Summary FW**

Tutto, salvato su file:

python scripts/summarize_fw_results.py \
  private/data/dim_n100_k10/fw_all_x0_results.json \
  --targets-file private/data/dim_n100_k10/targets.npz \
  --output private/data/dim_n100_k10/summaries/fw_summary_all.txt

Un certo caso e un certo vertice, esecuzione completa, salvata su file

python scripts/summarize_fw_results.py \
  private/data/dim_n100_k10/fw_all_x0_results.json \
  --case Q_well:q_well_sc2 \
  --x0-key vertex_0 \
  --targets-file private/data/dim_n100_k10/targets.npz \
  --output private/data/dim_n100_k10/summaries/fw_result_q_well_sc2_vertex_0.txt \
  --history

## **4. Gurobi baseline**

python scripts/run_gurobi_baseline.py private/data/dim_n100_k10 \
  --case Q_well:q_well_sc1 \
  --case Q_well:q_well_sc2 \
  --case Q_well:q_well_sc3 \
  --case Q_ill:q_ill_sc1 \
  --case Q_ill:q_ill_sc2 \
  --case Q_ill:q_ill_sc3 \
  --output private/data/dim_n100_k10/gurobi_baseline.json

## **5. Confronto FW già salvato vs Gurobi**

Tutto, salvato su file:

python scripts/summarize_fw_results.py \
  private/data/dim_n100_k10/fw_all_x0_results.json \
  --targets-file private/data/dim_n100_k10/targets.npz \
  --gurobi-file private/data/dim_n100_k10/gurobi_baseline.json \
  --output private/data/dim_n100_k10/fw_vs_gurobi_summary_all.txt

Solo un x0, esempio vertex_9:

python scripts/summarize_fw_results.py \
  private/data/dim_n100_k10/fw_all_x0_results.json \
  --x0-key vertex_9 \
  --targets-file private/data/dim_n100_k10/targets.npz \
  --gurobi-file private/data/dim_n100_k10/gurobi_baseline.json \
  --output private/data/dim_n100_k10/fw_vs_gurobi_vertex_9.txt

Solo una coppia Q:q:

python scripts/summarize_fw_results.py \
  private/data/dim_n100_k10/fw_all_x0_results.json \
  --case Q_well:q_well_sc2 \
  --targets-file private/data/dim_n100_k10/targets.npz \
  --gurobi-file private/data/dim_n100_k10/gurobi_baseline.json

Una coppia e un solo x0:

python scripts/summarize_fw_results.py \
  private/data/dim_n100_k10/fw_all_x0_results.json \
  --case Q_well:q_well_sc2 \
  --x0-key vertex_9 \
  --targets-file private/data/dim_n100_k10/targets.npz \
  --gurobi-file private/data/dim_n100_k10/gurobi_baseline.json \
  --output private/data/dim_n100_k10/fw_vs_gurobi_Q_well_sc2_vertex_9.txt

## **6. History FW**

History di un solo x0 su una coppia:

python scripts/summarize_fw_results.py \
  private/data/dim_n100_k10/fw_all_x0_results.json \
  --case Q_well:q_well_sc2 \
  --history-for vertex_9 \
  --output private/data/dim_n100_k10/history_Q_well_sc2_vertex_9.txt

History in CSV:

python scripts/summarize_fw_results.py \
  private/data/dim_n100_k10/fw_all_x0_results.json \
  --case Q_well:q_well_sc2 \
  --history-for vertex_9 \
  --csv private/data/dim_n100_k10/history_Q_well_sc2_vertex_9.csv

## **Note rapide**

- run_fw_experiments.py esegue solo Frank-Wolfe.
- run_gurobi_baseline.py esegue solo Gurobi.
- summarize_fw_results.py --gurobi-file confronta FW già salvato con Gurobi, senza rieseguire FW.
- Per avere distance_to_xu, FW deve essere stato lanciato con --include-solution.
- Se non passi --case o --x0-key, il summary mostra tutto.