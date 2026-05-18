# Comandi Globali

Flusso per eseguire tutti i casi e tutti i punti iniziali.

Esempio sotto per:

```text
private/dim_n10000_k100
```

Cambiare la cartella se si lavora su un'altra dimensione.

## 1. Generazione Dati

Eseguire i notebook in ordine:

```bash
jupyter nbconvert --to notebook --execute generate_matrices.ipynb --inplace
jupyter nbconvert --to notebook --execute generate_vectors.ipynb --inplace
```

Generare i punti iniziali:

```bash
python scripts/generate_initial_points.py
```

Controllare gli scenari `x_u`:

```bash
python scripts/summarize_targets.py private/dim_n10000_k100
```

## 2. Frank-Wolfe Su Tutti I Casi

Esegue FW su tutte le coppie `Q:q` e su tutti i punti iniziali presenti in `initial_points.npz`.

```bash
python scripts/run_fw_experiments.py private/dim_n10000_k100 \
  --case Q_well:q_well_sc1 \
  --case Q_well:q_well_sc2 \
  --case Q_well:q_well_sc3 \
  --case Q_ill:q_ill_sc1 \
  --case Q_ill:q_ill_sc2 \
  --case Q_ill:q_ill_sc3 \
  --x0-file private/dim_n10000_k100/data/initial_points.npz \
  --all-x0 \
  --max-iter 10000 \
  --tol-gap 1e-6 \
  --tol-rel-gap 1e-6 \
  --quiet \
  --include-solution \
  --overwrite \
  --output private/dim_n10000_k100/results/fw_all_x0_results.json
```

## 3. Summary FW Globale

Summary completo:

```bash
python scripts/summarize_fw_results.py \
  private/dim_n10000_k100/results/fw_all_x0_results.json \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --output private/dim_n10000_k100/summaries/fw_summary_all.txt
```

History di una singola esecuzione già contenuta nel file completo:

```bash
python scripts/summarize_fw_results.py \
  private/dim_n10000_k100/results/fw_all_x0_results.json \
  --case Q_well:q_well_sc1 \
  --x0-key vertex_0 \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --output private/dim_n10000_k100/summaries/fw_result_q_well_sc1_vertex_0.txt \
  --history
```

## 4. Baseline Gurobi Globale

Esegue Gurobi su tutte le coppie `Q:q`.

```bash
python scripts/run_gurobi_baseline.py private/dim_n10000_k100 \
  --case Q_well:q_well_sc1 \
  --case Q_well:q_well_sc2 \
  --case Q_well:q_well_sc3 \
  --case Q_ill:q_ill_sc1 \
  --case Q_ill:q_ill_sc2 \
  --case Q_ill:q_ill_sc3 \
  --include-solution \
  --output private/dim_n10000_k100/results/gurobi_baseline.json
```

Se `gurobi_baseline.json` esiste già e vuoi rigenerarlo includendo le soluzioni complete:

```bash
python scripts/run_gurobi_baseline.py private/dim_n10000_k100 \
  --case Q_well:q_well_sc1 \
  --case Q_well:q_well_sc2 \
  --case Q_well:q_well_sc3 \
  --case Q_ill:q_ill_sc1 \
  --case Q_ill:q_ill_sc2 \
  --case Q_ill:q_ill_sc3 \
  --include-solution \
  --overwrite \
  --output private/dim_n10000_k100/results/gurobi_baseline.json
```

## 5. Confronto FW Salvato Vs Gurobi

Confronto completo:

```bash
python scripts/summarize_fw_results.py \
  private/dim_n10000_k100/results/fw_all_x0_results.json \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --gurobi-file private/dim_n10000_k100/results/gurobi_baseline.json \
  --output private/dim_n10000_k100/summaries/fw_vs_gurobi_summary_all.txt
```

Solo una coppia `Q:q`:

```bash
python scripts/summarize_fw_results.py \
  private/dim_n10000_k100/results/fw_all_x0_results.json \
  --case Q_well:q_well_sc2 \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --gurobi-file private/dim_n10000_k100/results/gurobi_baseline.json \
  --output private/dim_n10000_k100/summaries/fw_vs_gurobi_q_well_sc2.txt
```

Una coppia `Q:q` e un solo `x0`:

```bash
python scripts/summarize_fw_results.py \
  private/dim_n10000_k100/results/fw_all_x0_results.json \
  --case Q_well:q_well_sc2 \
  --x0-key vertex_0 \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --gurobi-file private/dim_n10000_k100/results/gurobi_baseline.json \
  --output private/dim_n10000_k100/summaries/fw_vs_gurobi_q_well_sc2_vertex_0.txt
```

## Note

- `run_fw_experiments.py` esegue solo Frank-Wolfe.
- `run_gurobi_baseline.py` esegue solo Gurobi.
- `summarize_fw_results.py --gurobi-file` confronta risultati già salvati, senza rieseguire FW.
- `--overwrite` su FW/Gurobi cancella il file di output indicato e ricomincia da zero.
- Senza `--overwrite`, FW/Gurobi fanno resume e saltano i record già presenti.
- Per avere `distance_to_xu`, FW deve essere stato lanciato con `--include-solution`.
- `--tol-gap` è assoluta; `--tol-rel-gap` usa `fw_gap / max(1, abs(objective))`.
