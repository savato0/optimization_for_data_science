# 🚀 Guida alla Gestione dell'Ambiente (Conda)

Questo repository utilizza un file `environment.yml` per gestire in modo centralizzato tutte le dipendenze del progetto. Questo file combina sia i pacchetti gestiti da **Conda** che quelli installati tramite **pip**.

---

## 🛠 1. Setup Iniziale (Prima volta)

Se è la prima volta che cloni il repository, segui questi passaggi per creare l'ambiente:

1. **Crea l'ambiente** partendo dal file di configurazione:
   ```bash
   conda env create -f environment.yml
   ```

2. **Attiva l'ambiente**:
   ```bash
   conda activate o4ds
   ```

---

## 🔄 2. Aggiornare l'ambiente (Dopo un git pull)

Se un altro collaboratore ha aggiunto nuove librerie e il file `environment.yml` è cambiato, sincronizza il tuo ambiente locale con questo comando:

```bash
conda env update -f environment.yml --prune
```

> [!TIP]
> Il flag `--prune` è importante perché rimuove dal tuo ambiente locale i pacchetti che sono stati eliminati dal file YAML.

---

## ➕ 3. Aggiungere nuove librerie al progetto

Se devi installare una nuova libreria per lo sviluppo, segui rigorosamente questo flusso per non "sporcare" il progetto:

1. **Installa la libreria nell'ambiente attivo**:
   * Prova prima con Conda: `conda install nome_pacchetto`
   * Se non presente sui canali Conda: `pip install nome_pacchetto`

2. **Aggiorna il file environment.yml**:
   Invece di scriverlo a mano, esporta lo stato attuale dell'ambiente per includere le versioni corrette:
   ```bash
   conda env export --no-builds > environment.yml
   ```

3. **Invia le modifiche**:
   Fai il commit del file `environment.yml` aggiornato e caricalo sul repository.

---

## ⚠️ Regole e Buone Pratiche

* **Verifica l'ambiente**: Prima di installare qualsiasi cosa, assicurati che il nome dell'ambiente sia visibile tra parentesi nel terminale.
* **Evita pip freeze**: Non creare file `requirements.txt` separati. Usiamo solo il file `.yml` per gestire tutto in un unico posto.
* **Check del percorso Pip**: Se hai dubbi su quale pip stai usando, digita `which pip` (Mac/Linux) o `where pip` (Windows). Deve puntare alla cartella del tuo ambiente Conda attuale.
* **Pulizia**: Se l'ambiente diventa instabile o troppo pesante, puoi eliminarlo e ricrearlo da zero:
  ```bash
  conda deactivate
  conda env remove -n o4ds
  conda env create -f environment.yml
  ```
