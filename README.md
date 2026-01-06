# WDBC Conformal Prediction (Binary Classification)

Reproducible conformal prediction experiments for reliable binary classification on the UCI WDBC dataset.
Methods include Split Conformal, Mondrian (class-conditional), and Adaptive Prediction Sets (APS),
with evaluation focused on coverage–efficiency trade-offs.

This repository is designed to demonstrate reproducible experimentation and clear evaluation for data science and machine learning.

Dataset: UCI Breast Cancer Wisconsin Diagnostic (WDBC). The dataset file `wdbc.data` is NOT included in this repo.

README (Windows) — Step-by-step to reproduce WDBC conformal results
===============================================================

This folder contains:
  - wdbc_conformal_2x2.py   (the script)
You also need:
  - wdbc.data              (WDBC dataset file from UCI)

Goal:
  Run ONE command to generate an output folder (out_wdbc/) containing:
  - summary CSV
  - LaTeX table (.tex) for the report
  - 4 PNG plots
  - metadata/how-to files

---------------------------------------------------------------
1) Put files in one folder
---------------------------------------------------------------
Create a folder, e.g.
  C:\wdbc_project\

Put inside:
  - wdbc_conformal_2x2.py
  - (this file) README_WINDOWS.txt
  - wdbc.data   (recommended: put it in the same folder)

If you keep wdbc.data somewhere else, note its full path (we need it later).

---------------------------------------------------------------
2) Open PowerShell in this folder
---------------------------------------------------------------
Option A (easy):
  - Open File Explorer
  - Go to C:\wdbc_project\
  - Click the address bar, type: powershell
  - Press Enter

A PowerShell window should open already inside this folder.

---------------------------------------------------------------
3) Check Python is installed
---------------------------------------------------------------
In PowerShell, run:
  python --version

If that does NOT work, try:
  py --version

If both do not work, install Python:
  - Download and install from python.org
  - IMPORTANT during install: tick "Add Python to PATH"
Then close PowerShell, reopen it, and try again.

---------------------------------------------------------------
4) Create and activate a virtual environment (recommended)
---------------------------------------------------------------
In PowerShell (inside the project folder), run:

  python -m venv .venv

Activate it:

  .venv\Scripts\Activate.ps1

If you get a permission error, run this ONCE:

  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then activate again:

  .venv\Scripts\Activate.ps1

After activation, you usually see "(.venv)" at the start of the command line.

---------------------------------------------------------------
5) Install required packages
---------------------------------------------------------------
Still in PowerShell with venv activated, run:

  python -m pip install -U pip
  python -m pip install numpy pandas matplotlib scikit-learn

---------------------------------------------------------------
6) Run the script (one command)
---------------------------------------------------------------

Option A (recommended): wdbc.data is in the SAME folder as the script
Run:

  python wdbc_conformal_2x2.py --data-path "wdbc.data" --outdir out_wdbc --overwrite

Option B: wdbc.data is somewhere else (use full path)
Example:

  python wdbc_conformal_2x2.py --data-path "C:\Users\YOURNAME\Desktop\wdbc.data" --outdir out_wdbc --overwrite

Note:
- Always keep quotes around the path, especially if there are spaces.

---------------------------------------------------------------
7) What success looks like
---------------------------------------------------------------
After the script finishes, you should see a new folder:

  out_wdbc\

Typical key outputs inside out_wdbc\ include:
  - wdbc_summary.csv
  - wdbc_table_2x2_key.tex
  - wdbc_coverage_curve.png
  - wdbc_efficiency_curve.png
  - wdbc_empty_curve.png
  - wdbc_fallback_curve.png
  - metadata.json
  - HOWTO_REPRODUCE.txt

Open PNG files to view plots; open CSV to inspect metrics.

---------------------------------------------------------------
8) Troubleshooting (common issues)
---------------------------------------------------------------

A) "python is not recognized"
- Try:
    py --version
- If py works, use "py" instead of "python", e.g.
    py wdbc_conformal_2x2.py --data-path "wdbc.data" --outdir out_wdbc --overwrite
- If neither works: install Python and tick "Add Python to PATH"

B) "No module named numpy/pandas/sklearn"
- You did not install packages in the active environment.
- Make sure you activated venv (Step 4), then run Step 5 again.

C) "File not found" for wdbc.data
- Check the filename is exactly: wdbc.data
- Check the path in --data-path is correct
- Put wdbc.data in the same folder and use Option A

D) PowerShell permission error when activating venv
Run once:
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Then:
  .venv\Scripts\Activate.ps1

---------------------------------------------------------------
9) Optional: deactivate the environment
---------------------------------------------------------------
When finished, run:
  deactivate
