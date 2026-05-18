# Comandi Caso Singolo

Flusso per eseguire un solo caso `Q:q` e un solo punto iniziale `x0`, con trace proiettata in 2D.

Esempio sotto per:

```text
private/dim_n10000_k100
Q_well:q_well_sc2
vertex_0
```

Cambiare cartella, caso e `x0-key` se si lavora su un'altra esecuzione.

## 1. Baseline Gurobi Del Caso

Genera un file Gurobi separato con soluzione completa `x_gurobi`.

Usare un output separato evita di sovrascrivere il baseline globale.

```bash
python scripts/run_gurobi_baseline.py private/dim_n10000_k100 \
  --case Q_well:q_well_sc2 \
  --include-solution \
  --overwrite \
  --output private/dim_n10000_k100/results/gurobi_q_well_sc2_with_solution.json
```

## 2. Frank-Wolfe Del Caso

Esegue FW su un solo punto iniziale e salva anche la trace proiettata in 2D.

```bash
python scripts/run_fw_experiments.py private/dim_n10000_k100 \
  --case Q_well:q_well_sc2 \
  --x0-file private/dim_n10000_k100/data/initial_points.npz \
  --x0-key vertex_0 \
  --max-iter 10000 \
  --tol-gap 1e-6 \
  --tol-rel-gap 1e-6 \
  --quiet \
  --include-solution \
  --save-projected-trace \
  --trace-every 10 \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --gurobi-file private/dim_n10000_k100/results/gurobi_q_well_sc2_with_solution.json \
  --overwrite \
  --output private/dim_n10000_k100/results/fw_q_well_sc2_vertex_0.json
```

## 3. Summary Del Caso

Summary finale:

```bash
python scripts/summarize_fw_results.py \
  private/dim_n10000_k100/results/fw_q_well_sc2_vertex_0.json \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --gurobi-file private/dim_n10000_k100/results/gurobi_q_well_sc2_with_solution.json \
  --output private/dim_n10000_k100/summaries/fw_q_well_sc2_vertex_0_summary.txt
```

History completa:

```bash
python scripts/summarize_fw_results.py \
  private/dim_n10000_k100/results/fw_q_well_sc2_vertex_0.json \
  --case Q_well:q_well_sc2 \
  --x0-key vertex_0 \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --output private/dim_n10000_k100/summaries/fw_q_well_sc2_vertex_0_history.txt \
  --history
```

## 4. Grafico Interattivo

Visualizzare la trace interattiva:

```bash
python scripts/plot_fw_trace.py private/dim_n10000_k100/results/traces/fw_trace_Q_well_q_well_sc2_vertex_0.json
```

## Variante Senza Gurobi

Se non vuoi usare Gurobi come riferimento, puoi generare la trace usando solo `x_u`.

```bash
python scripts/run_fw_experiments.py private/dim_n10000_k100 \
  --case Q_well:q_well_sc2 \
  --x0-file private/dim_n10000_k100/data/initial_points.npz \
  --x0-key vertex_0 \
  --max-iter 10000 \
  --tol-gap 1e-6 \
  --tol-rel-gap 1e-8 \
  --quiet \
  --include-solution \
  --save-projected-trace \
  --trace-every 10 \
  --targets-file private/dim_n10000_k100/data/targets.npz \
  --overwrite \
  --output private/dim_n10000_k100/results/fw_q_well_sc2_vertex_0.json
```

## Note

- Per vedere `x_gurobi` nella trace, Gurobi deve essere stato lanciato con `--include-solution`.
- Se un file Gurobi esiste già ma era stato creato senza soluzioni, rigeneralo con `--overwrite` oppure usa un nuovo file di output.
- `--save-projected-trace` salva file leggeri in `results/traces` senza inserire i vettori `x_t` completi nel JSON principale.
- `--trace-every 10` salva un punto ogni 10 iterazioni; usare `--trace-every 1` per problemi piccoli se vuoi vedere ogni iterazione.
- `--tol-gap` è assoluta; `--tol-rel-gap` usa `fw_gap / max(1, abs(objective))`.
