# Pt-Mn1-NC

Code submission materials for *Holo-Oxygen Activation Enables Ultra-Sensitive and Potential-Dependent Electrochemiluminescence Detection of PFAS*.

## Source and current status

The [GitHub repository](https://github.com/Zhangyiming135/Pt-Mn1-NC) contains three analysis configuration templates and two software test scripts. These files were reviewed at commit `86b564603d69e33bd2891ae3d4cd17cea5faae30`. Both scripts require the original analysis entry point, `ml_general.py`. The reviewed commit omits this file. A runnable installation, demo, and reproduction workflow are pending the addition of the original source and any companion files it needs.

## System requirements and installation

- Python 3. The supported Python versions, tested operating systems, and hardware requirements for the analysis remain unverified.
- `requirements.txt` lists the dependencies visible in the software tests, including the Excel reader used by the test suite. The full analysis dependencies and compatible versions require verification against the original source.
- The optional Origin integration test is written for Windows with Origin and Windows PowerShell. The two Origin-capable templates contain machine-specific paths for `Origin64.exe` and `Interop.Origin.dll`. Set those paths for the local Origin installation when using these templates with Origin. Compatibility with particular Origin versions remains unverified.

From the repository root, install the currently identified dependencies in a dedicated Python environment:

```sh
python -m venv .venv
```

Activate it with `./.venv/Scripts/Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in a POSIX shell. Then run:

```sh
python -m pip install -r requirements.txt
```

Typical installation time has yet to be measured for the complete software.

## Demo and expected outputs

The tests generate synthetic software fixtures with arbitrary numeric features and class labels. These fixtures contain no PFAS measurements. Their outputs are for software verification only.

Once the original module and its dependencies are available, place `ml_general.py` at the repository root and place `test_general.py` and `origin_smoke_test.py` in a `tests/` subdirectory. Both scripts locate the module relative to that layout. Run the general tests from the repository root:

```sh
python -m unittest discover -s tests -p test_general.py
```

The test suite creates temporary inputs and checks outputs such as `predictions.csv`, `cv_partitions.csv`, `inner_partitions.csv`, `fold_parameters.json`, and `results.xlsx`. Its assertions cover prediction counts, training-fold preprocessing, group separation, and input handling. A successful test run ends with `OK`.

On a Windows machine configured for Origin, the separate integration script generates its own synthetic CSV and JSON configuration, runs Origin export and readback checks, and writes `integration_test_record.json` upon success. The output path must be new:

```powershell
python tests/origin_smoke_test.py --output origin_software_test_run
```

An end-to-end demo with a standalone dataset has yet to be verified. Typical test and demo runtimes have yet to be measured.

## Instructions for use with your own data

Start with a copy of the appropriate template. Set `output_root` to a new results directory and replace the example paths and identifiers with those for your data.

| Template | Input and fields to check |
| --- | --- |
| `config_rows.json` | One sample per row. Set the workbook and worksheet, sample ID, class label, preparation group, and numeric feature columns. The example tasks are species, concentration, and mixture ratio. Check the wavelength list and document how it was selected. |
| `config_spectra_columns.json` | One sample per column. Set the wavelength axis, sample columns, and metadata workbook and worksheet. Match each sample column to its metadata identifier and supply its true class and preparation group. The template omits explicit cross-validation settings. |
| `config_origin_pca.json` | Existing PCA scores from an Origin `.opju` worksheet. Check the project path, worksheet, score columns `I` and `J`, and class-label column `K`. Document how the input scores were computed, including which samples were used to fit the PCA. |

The row and Origin PCA templates specify five outer folds, four inner folds, and seed `20261009`. The row template explicitly selects LDA and RBF SVM and specifies standard scaling and SVM search grids. Use sufficient samples and independent preparation groups per class for the configured folds. Keep identifiers, class labels, and group fields separate from the numeric feature columns.

The CLI invocation documented by the integration script is `python ml_general.py --config path/to/config.json`. The same script uses `--check-only` for input checks and `--origin` for Origin export. These commands require the original module and have yet to be verified with the public repository contents.

## Reproducing manuscript results

Reproduction requires the original analysis source revision, the experimental input files or their access instructions, and the exact input mappings and settings used for each reported result. The repository templates contain example paths and column mappings. The tests verify software behavior on synthetic inputs only. Manuscript-specific run instructions remain pending.

## License

The reviewed repository contains no license file. The code license is pending confirmation by the rights holders.
