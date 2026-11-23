"""Reconstructed tabular analysis. This implementation postdates the manuscript."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_table(source, base):
    path = base / source["path"]
    if source.get("type", "table") != "table":
        raise ValueError("This reconstruction supports tabular inputs only. Export Origin worksheets to CSV or XLSX first.")
    header = source.get("header_row", 0)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=header, float_precision="round_trip")
    return pd.read_excel(path, sheet_name=source.get("sheet", 0), header=header)


def input_matrix(dataset, frame, source, base):
    label_column = dataset["label_column"]
    group_column = dataset.get("preparation_group_column")
    id_column = dataset.get("sample_id_column")
    if dataset.get("orientation", "samples_in_rows") == "samples_in_columns":
        metadata = dataset["metadata"]
        id_column = metadata["sample_column"]
        samples = dataset["sample_columns"]
        if len(set(samples)) != len(samples):
            raise ValueError("Repeated sample IDs")
        spectral = frame.set_index(dataset["feature_axis_column"])[samples].T
        spectral.columns = spectral.columns.map(str)
        if not spectral.columns.is_unique:
            raise ValueError("Repeated feature-axis values")
        spectral.index.name = id_column
        records = read_table(metadata, base)
        records[id_column] = records[id_column].astype("string")
        columns = [id_column, label_column] + ([group_column] if group_column else [])
        frame = spectral.reset_index().merge(records[columns], on=id_column, how="left", validate="one_to_one")
    ids = frame[id_column] if id_column else pd.Series(np.arange(1, len(frame) + 1))
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Repeated sample IDs or missing sample IDs")
    excluded = {id_column, label_column, group_column}
    features = dataset.get("feature_columns", [c for c in frame if c not in excluded])
    if excluded.intersection(features):
        raise ValueError("Identifiers, labels and preparation groups must not be classifier features")
    if not features or len(features) != len(set(features)):
        raise ValueError("Specify unique numeric feature columns")
    labels = frame[label_column]
    groups = frame[group_column] if group_column else None
    if labels.isna().any() or (groups is not None and groups.isna().any()):
        raise ValueError("Missing classes or preparation groups")
    return ids.to_numpy(), labels.astype(str).to_numpy(), groups.to_numpy() if groups is not None else None, frame[features].apply(pd.to_numeric)


def cv_splits(x, y, groups, folds, seed):
    classes, counts = np.unique(y, return_counts=True)
    if folds < 2 or len(classes) < 2 or min(counts) < folds:
        raise ValueError("insufficient samples per class for configured folds")
    if groups is None:
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    else:
        if any(len(np.unique(groups[y == label])) < folds for label in classes):
            raise ValueError("insufficient preparation groups per class for configured folds")
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y, groups))
    for train, validation in splits:
        if len(np.unique(y[train])) != len(classes) or len(np.unique(y[validation])) != len(classes):
            raise ValueError("insufficient class coverage in a configured fold")
    return splits


def pipeline(config, existing, estimator):
    settings = config.get("preprocessing", {})
    steps = []
    if settings.get("missing", "error") == "median":
        steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
    if not existing:
        if settings.get("scale", "standard") == "standard":
            steps.append(("scale", StandardScaler()))
        if settings.get("select_k") is not None:
            steps.append(("select", SelectKBest(k=settings["select_k"])))
        if settings.get("pca_components") is not None:
            steps.append(("pca", PCA(n_components=settings["pca_components"], random_state=config.get("random_seed", 20261009))))
    return Pipeline(steps + [("model", estimator)])


def partition_rows(ids, groups, split, outer_fold, inner_fold=None):
    for role, indices in zip(("train", "validation"), split):
        for index in indices:
            row = {"Outer_Fold": outer_fold, "Role": role, "Sample_ID": ids[index]}
            if groups is not None:
                row["Preparation_Group"] = groups[index]
            if inner_fold is not None:
                row["Inner_Fold"] = inner_fold
            yield row


def analyze(dataset, config, config_path, output_root, check_only=False, origin=None, powershell=None):
    source = dataset["source"]
    base = Path(config_path).resolve().parent
    frame = read_table(source, base)
    ids, y, groups, features = input_matrix(dataset, frame, source, base)
    x = features.to_numpy(dtype=float)
    settings = config.get("preprocessing", {})
    missing = settings.get("missing", "error")
    if missing not in ("error", "median"):
        raise ValueError("Unsupported missing-input policy")
    if np.isinf(x).any() or (missing == "error" and np.isnan(x).any()):
        raise ValueError("Missing numeric inputs or non-finite inputs")
    existing = dataset.get("input_kind") == "existing_pca"
    if existing and x.shape[1] != 2:
        raise ValueError("existing_pca expects two score columns")
    algorithms = config.get("algorithms", ["LDA", "SVM"])
    if not algorithms or set(algorithms) - {"LDA", "SVM"} or len(set(algorithms)) != len(algorithms):
        raise ValueError("Select unique algorithms from LDA and SVM")
    key = dataset["key"]
    if not key or key in (".", "..") or "/" in key or "\\" in key or ":" in key:
        raise ValueError("Dataset key must be a single directory name")
    output = Path(output_root) / key
    output.mkdir(parents=True, exist_ok=False)
    inputs = features.assign(Sample_ID=ids, True_Class=y)
    if groups is not None:
        inputs["Preparation_Group"] = groups
    inputs.to_csv(output / "input_data.csv", index=False)
    summary = {"n_samples": len(y), "checked_only": check_only, "algorithms": algorithms}
    if check_only:
        save_json(output / "summary.json", summary)
        return summary

    folds = config.get("validation", {})
    seed = config.get("random_seed", 20261009)
    outer = cv_splits(x, y, groups, folds.get("outer_folds", 5), seed)
    outer_rows, inner_rows, parameters = [], [], []
    predictions = pd.DataFrame({"Sample_ID": ids, "True_Class": y})
    if existing:
        predictions[["PC1", "PC2"]] = x
    if "LDA" in algorithms:
        predictions[["LD1", "LD2"]] = np.nan
    for algorithm in algorithms:
        predictions[f"{algorithm}_Prediction"] = pd.Series(index=predictions.index, dtype=object)
    svm = config.get("svm", {})
    lda = config.get("lda", {})
    for fold, (train, validation) in enumerate(outer, 1):
        train_groups = groups[train] if groups is not None else None
        inner = cv_splits(x[train], y[train], train_groups, folds.get("inner_folds", 4), seed)
        outer_rows.extend(partition_rows(ids, groups, (train, validation), fold))
        for inner_fold, split in enumerate(inner, 1):
            inner_rows.extend(partition_rows(ids[train], train_groups, split, fold, inner_fold))
        record = {"outer_fold": fold}
        for algorithm in algorithms:
            if algorithm == "LDA":
                estimator = LinearDiscriminantAnalysis(solver=lda.get("solver", "svd"), shrinkage=lda.get("shrinkage"))
                fitted = pipeline(config, existing, estimator).fit(x[train], y[train])
                if estimator.solver != "lsqr":
                    scores = fitted.transform(x[validation])
                    predictions.loc[validation, ["LD1", "LD2"][:scores.shape[1]]] = scores[:, :2]
            else:
                estimator = SVC(kernel=svm.get("kernel", "rbf"), random_state=seed)
                grid = {"model__C": svm.get("C_grid", [.1, 1, 10, 100]),
                        "model__gamma": svm.get("gamma_grid", ["scale", .001, .01, .1, 1])}
                search = GridSearchCV(pipeline(config, existing, estimator), grid, cv=inner, scoring="accuracy", error_score="raise")
                search.fit(x[train], y[train])
                fitted = search.best_estimator_
                record["svm_best_parameters"] = search.best_params_
            predictions.loc[validation, f"{algorithm}_Prediction"] = fitted.predict(x[validation])
            steps = fitted.named_steps
            if "impute" in steps:
                record["median_values"] = steps["impute"].statistics_.tolist()
            if "scale" in steps:
                record["scale_mean"] = steps["scale"].mean_.tolist()
            if "select" in steps:
                record["selected_features"] = features.columns[steps["select"].get_support()].tolist()
        parameters.append(record)

    classes = sorted(set(y))
    summary.update(classes=classes, metrics={}, plots=[])
    if "LDA" in algorithms:
        summary["plots"].append({"mode": "LDA", "one_dimensional": len(classes) == 2,
                                 "y_column": "Group_ID" if len(classes) == 2 else "LD2"})
    for algorithm in algorithms:
        predicted = predictions[f"{algorithm}_Prediction"]
        correct = predictions[f"{algorithm}_Correct"] = predicted == y
        summary["metrics"][algorithm] = {"correct": int(correct.sum()), "accuracy": float(correct.mean()),
                                           "confusion": confusion_matrix(y, predicted, labels=classes).tolist()}
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(output / "cv_partitions.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output / "inner_partitions.csv", index=False)
    save_json(output / "fold_parameters.json", parameters)
    with pd.ExcelWriter(output / "results.xlsx", engine="openpyxl") as workbook:
        inputs.to_excel(workbook, sheet_name="InputData", index=False)
        predictions.to_excel(workbook, sheet_name="Predictions", index=False)
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--origin", action="store_true")
    args = parser.parse_args()
    if args.origin:
        parser.error("Native Origin integration is unavailable in the tabular reconstruction.")
    config_path = args.config.resolve()
    config = load_json(config_path)
    output_root = config_path.parent / config["output_root"]
    for dataset in config["datasets"]:
        analyze(dataset, config, config_path, output_root, args.check_only)


if __name__ == "__main__":
    main()
