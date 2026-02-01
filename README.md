# 🚀 Environment Management Guide (Conda)

This repository uses an `environment.yml` file to centrally manage all project dependencies. This file combines both **Conda** packages and those installed via **pip**.

---

## 🛠 1. Initial Setup (First time)

If this is your first time cloning the repository, follow these steps to create the environment:

1. **Create the environment** from the configuration file:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment**:
   ```bash
   conda activate o4ds
   ```

---

## 🔄 2. Updating the Environment (After a git pull)

If another collaborator has added new libraries and the `environment.yml` file has changed, synchronize your local environment with this command:

```bash
conda env update -f environment.yml --prune
```

> [!TIP]
> The `--prune` flag is important because it removes packages from your local environment that have been deleted from the YAML file.

---

## ➕ 3. Adding New Libraries to the Project

If you need to install a new library for development, strictly follow this workflow to keep the project "clean":

1. **Install the library in the active environment**:
   * Try Conda first: `conda install package_name`
   * If not available on Conda channels: `pip install package_name`

2. **Update the environment.yml file**:
   Instead of writing it manually, export the current state of the environment to include the correct versions:
   ```bash
   conda env export --no-builds > environment.yml
   ```

3. **Submit changes**:
   Commit the updated `environment.yml` file and push it to the repository.

---

## ⚠️ Rules and Best Practices

* **Verify the environment**: Before installing anything, make sure the environment name is visible in parentheses in your terminal.
* **Avoid `pip freeze`**: Do not create separate `requirements.txt` files. We use only the `.yml` file to manage everything in one place.
* **Check Pip path**: If in doubt about which pip you are using, type `which pip` (Mac/Linux) or `where pip` (Windows). It must point to your current Conda environment folder.
* **Cleanup**: If the environment becomes unstable or too heavy, you can delete it and recreate it from scratch:
  ```bash
  conda deactivate
  conda env remove -n o4ds
  conda env create -f environment.yml
  ```
