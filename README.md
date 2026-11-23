# Pt-Mn1-NC — reconstructed tabular analysis

Companion materials for *Holo-Oxygen Activation Enables Ultra-Sensitive and Potential-Dependent Electrochemiluminescence Detection of PFAS*.

`ml_general.py` was reconstructed after the manuscript analysis from the available JSON templates and software tests. Its design follows those files. The historical implementation and the equivalence of outputs on the manuscript inputs remain unverified.

## System requirements and installation

Use Python 3 with NumPy, pandas, openpyxl and scikit-learn. From the repository root, create an isolated environment and install the dependencies:

```sh
python -m venv .venv
```

Activate it with `./.venv/Scripts/Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in a POSIX shell. Then run:

```sh
python -m pip install -r requirements.txt
```

Verified on Windows 11 (build 26300) with Python 3.12.14, NumPy 2.5.5, pandas 3.0.1, openpyxl 3.1.5 and scikit-learn 1.9.1. Typical clean installation time and cross-platform compatibility have yet to be measured. The tabular demo uses CPU-based estimators and requires no non-standard hardware.

## Synthetic software demo

`demo/synthetic_demo.csv` contains 30 artificial rows across three arbitrary classes. The numbers are software test inputs. They contain no experimental PFAS measurements.

```sh
python ml_general.py --config demo/config_synthetic.json
```

The configuration specifies stratified five-fold outer evaluation and four-fold inner SVM parameter selection. Outputs are written to `demo/synthetic_results/synthetic/`. Expected outputs include `summary.json` with aggregate metrics, `predictions.csv` with one outer-fold prediction per sample per classifier, `cv_partitions.csv`, `inner_partitions.csv`, `fold_parameters.json`, `input_data.csv` and `results.xlsx`. Choose a fresh `output_root` for another run. The script refuses to overwrite an existing dataset output directory.

To run the existing general tests:

```sh
python -m unittest discover -s tests -p test_general.py
```

All 10 general tests passed in the environment above. One timed invocation of the synthetic demo took approximately 2.3 seconds, including Python startup. The run produced all seven listed files and 30 prediction rows. This is a single-machine observation.

## Using your own data

Copy the relevant template and set paths and column names for your data. Relative paths are resolved from the configuration file's directory. Use `python ml_general.py --config path/to/config.json --check-only` to read and validate inputs without fitting models. Use a fresh output path for the subsequent analysis.

| Template | Input |
| --- | --- |
| `config_rows.json` | CSV or Excel sheet with one sample per row. Map sample IDs, labels, preparation groups and numeric features. |
| `config_spectra_columns.json` | A spectral matrix with one sample per column, plus metadata matching sample IDs to classes and preparation groups. |
| `config_origin_pca.json` | Original template for Origin projects. For this implementation, export the worksheet to CSV or XLSX and update the source type, source path and column names in a copy. Existing score columns are passed in as supplied. |

The tabular implementation supports LDA and SVM, with fold-local preprocessing for raw features. When a preparation-group column is present, the split keeps each group together in both validation loops. Fold counts are validated and are never reduced automatically. Binary LDA has a single discriminant score. Discriminant scores fitted in different folds need not be on a common coordinate system.

The original templates and tests are retained. Native Origin project I/O, the Origin integration test, and automatic figure generation are outside the scope of this reconstruction. The tabular outputs include LDA score columns where the selected solver supports them.

## Manuscript reproduction and license

Reproducing the manuscript requires the actual input files, their mappings to the figures and tables, and a verified historical analysis workflow. The synthetic demo is a software example only. The code license is pending confirmation by the rights holders.
