# Conda One Pager for Persona0

## Goal
Set up one clean Conda environment and one matching Jupyter kernel for this project.

Project folder:
- /Users/claudiogonzalez/Documents/MacKU/ItoLab/Persona0

Recommended names:
- Environment: persona0
- Kernel name: persona0
- Kernel display name: Python (Persona0)

## 1) Create the environment (one time)
Run from any folder:

conda create -n persona0 python=3.11 -y

## 2) Activate and install essentials

conda activate persona0
python -m pip install --upgrade pip
python -m pip install jupyter ipykernel

Optional common data packages:

python -m pip install numpy pandas matplotlib scikit-learn

## 3) Register notebook kernel

python -m ipykernel install --user --name persona0 --display-name "Python (Persona0)"

## 4) Verify setup

python -c "import sys; print(sys.executable)"
python -m jupyter kernelspec list

Expected:
- Python path points to the persona0 env.
- A kernel named persona0 appears.

## 5) Use in VS Code
1. Open Persona0 project folder.
2. Select interpreter: persona0.
3. Open notebook and select kernel: Python (Persona0).

## 6) Daily usage
Start work:

conda activate persona0

Install new package later:

conda activate persona0
python -m pip install PACKAGE_NAME

## 7) Export environment for reproducibility
From active persona0 env:

conda env export --no-builds > environment.yml

Recreate on another machine:

conda env create -f environment.yml

## 8) Update or remove
Update packages:

conda update --all -n persona0

Remove only this environment:

conda deactivate
conda env remove -n persona0

Remove kernel (if needed):

jupyter kernelspec remove persona0

## 9) Quick troubleshooting
Issue: jupyter command not found
- Use python module form from active env:
  python -m jupyter kernelspec list

Issue: wrong kernel appears
- Re-select Python (Persona0) in notebook kernel picker.
- Reload VS Code window if stale entries remain.

Issue: duplicate kernels
- Keep the one you use and remove extras with:
  jupyter kernelspec list
  jupyter kernelspec remove KERNEL_NAME

## 10) Best-practice rule
- One project = one environment + one kernel.
- Avoid doing project work in base.
