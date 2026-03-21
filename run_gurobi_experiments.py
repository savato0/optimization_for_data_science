import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import json
from datetime import datetime

# ==========================================
# 1. SETUP INIZIALE
# ==========================================
n = int(1e4)
k = int(1e2)

# set folder name based on n
n_readable = f"{n:.0e}".replace(".0", "").replace("+0", "")
experiment_dim = "dim_" + n_readable
folder = "./private/data/" + experiment_dim + "/"

# check if folder exists
if not os.path.exists(folder):
    raise FileNotFoundError(f"Folder {folder} does not exist. Please run generate_matrices.ipynb and generate_vectors.ipynb first to generate the matrices.")

# check if n/k has no remainder
if n % k == 0:
    m = n // k
    sizes = [n // k] * k
else: 
    raise ValueError("n must be divisible by k")

print(f"n: {n}, k: {k}, m: {m}")
print(f"Folder: {folder}\n")

# Crea la cartella logs se non esiste
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logs_folder = folder + "logs" + timestamp + "/"
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

# ==========================================
# 2. DEFINISCI I CASI DA TESTARE
# ==========================================
cases = [
    # {"matrix": "Q_well", "vector": "q_well_sc1"}
    # ,{"matrix": "Q_well", "vector": "q_well_sc2"}
    # ,{"matrix": "Q_well", "vector": "q_well_sc3"}
    {"matrix": "Q_ill", "vector": "q_ill_sc1"}
    ,{"matrix": "Q_ill", "vector": "q_ill_sc2"}
    ,{"matrix": "Q_ill", "vector": "q_ill_sc3"}
]

methods = [0, 1, 2]  # 0=Primal Active Set, 1=Dual Active Set, 2=Barrier

# Dizionario per memorizzare i risultati
results = []

# ==========================================
# 3. CICLO PRINCIPALE
# ==========================================
total_runs = len(cases) * len(methods)
current_run = 0

for case in cases:
    # Carica Q e q
    matrix_name = case["matrix"]
    vector_name = case["vector"]
    
    print(f"Caricamento {matrix_name} e {vector_name}...", end=" ")
    Q = np.load(folder + "matrices.npz")[matrix_name]
    q = np.load(folder + "vectors.npz")[vector_name]
    print("✓")
    
    for method in methods:
        current_run += 1
        print(f"\n[{current_run}/{total_runs}] Esecuzione: {matrix_name} + {vector_name}, Metodo {method}")
        print("-" * 60)
        
        try:
            # ==========================================
            # 4. CREAZIONE E CONFIGURAZIONE DEL MODELLO
            # ==========================================
            model = gp.Model("proj_33")
            x = model.addMVar(shape=n, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
            obiettivo = 0.5 * (x.T @ Q @ x) + (q.T @ x)
            model.setObjective(obiettivo, GRB.MINIMIZE)
            
            # Costruiamo il vettore b di dimensione K, tutto di 1
            b = np.ones(k)
            
            # Costruiamo la matrice A (K righe, n colonne)
            A = np.repeat(np.eye(k), m, axis=1)
            
            # Convertiamo A in matrice sparsa per ottimizzare la memoria
            A_sparse = sp.csr_matrix(A)
            
            # Aggiungiamo il blocco di vincoli Ax = b
            model.addConstr(A_sparse @ x == b, name="vincoli_simplessi")
            
            # Imposta il metodo
            model.Params.Method = method
            
            # Imposta il logfile
            log_filename = f"{matrix_name}_{vector_name}_metodo_{method}_gurobi.log"
            log_path = logs_folder + log_filename
            model.setParam("LogFile", log_path)
            
            # ==========================================
            # 5. RISOLUZIONE
            # ==========================================
            start_time = datetime.now()
            model.optimize()
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            
            # ==========================================
            # 6. ESTRAZIONE DEI RISULTATI
            # ==========================================
            if model.Status == GRB.OPTIMAL:
                valore_ottimo = model.ObjVal
                runtime = model.Runtime
                
                result = {
                    "matrix": matrix_name,
                    "vector": vector_name,
                    "method": method,
                    "status": "OPTIMAL",
                    "objective_value": float(valore_ottimo),
                    "gurobi_runtime": float(runtime),
                    "elapsed_time": float(elapsed_time),
                    "log_file": log_filename
                }
                
                print(f"✓ OTTIMALE")
                print(f"  Valore ottimo: {valore_ottimo:.6e}")
                print(f"  Tempo Gurobi: {runtime:.4f}s")
                print(f"  Tempo totale: {elapsed_time:.4f}s")
                
            else:
                result = {
                    "matrix": matrix_name,
                    "vector": vector_name,
                    "method": method,
                    "status": "NOT_OPTIMAL",
                    "status_code": model.Status,
                    "elapsed_time": float(elapsed_time),
                    "log_file": log_filename
                }
                
                print(f"✗ NON OTTIMALE (Status: {model.Status})")
            
            results.append(result)
            
        except Exception as e:
            print(f"✗ ERRORE: {str(e)}")
            result = {
                "matrix": matrix_name,
                "vector": vector_name,
                "method": method,
                "status": "ERROR",
                "error": str(e)
            }
            results.append(result)

# ==========================================
# 7. SALVA I RISULTATI
# ==========================================
print("\n" + "=" * 60)
print("RIEPILOGO RISULTATI")
print("=" * 60)

results_file = folder + "gurobi_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nRisultati salvati in: {results_file}")

# Stampa un riepilogo
print("\nRISULTATI OTTIMI:")
optimal_count = 0
for r in results:
    if r["status"] == "OPTIMAL":
        optimal_count += 1
        print(f"  {r['matrix']:6s} + {r['vector']:12s}, Metodo {r['method']}: {r['objective_value']:.6e}")

print(f"\nTotale esecuzioni: {len(results)}")
print(f"Esecuzioni ottimali: {optimal_count}")
print(f"Esecuzioni fallite: {len(results) - optimal_count}")
