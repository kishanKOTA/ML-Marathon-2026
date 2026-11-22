"""Score a predictions CSV from inference.py against the shared split's labels.

Input:  predictions CSV (inference.py, one row per GT-box crop) + data/processed/split_tagged.csv;
        optionally baseline prediction CSVs (columns: filename, age_pred and/or antler_pred, using
        the team label order) from the two separate ResNet18 models
Output: printed report + CSVs next to the predictions file (<stem>_eval_*.csv):
        - per-head accuracy, F1 on class 1 (team metric), macro-F1: raw head argmax, after the
          constraint, and for each baseline
        - confusion matrix: true combined class vs predicted label (incl. adult_undetermined)
        - breakdowns by bbox_touches_edge (proxy for "head may be cut off"; no head-visible label
          exists), day_night, month, and camera
        - needs_review / no_detection / undetermined rates
        - calibration per head: expected calibration error + reliability table

Run on val while tuning; run on test once, at the end.

Usage (from the repo root):
    uv run python -m models.dual_resnet18.evaluate logs/runs/dual_resnet18/default_val_predictions.csv
    uv run python -m models.dual_resnet18.evaluate preds.csv --baseline age_preds.csv --baseline antler_preds.csv
"""

import argparse  # CLI flags
from pathlib import Path  # filesystem paths

import numpy as np  # calibration bins
import pandas as pd  # tables
from sklearn.metrics import accuracy_score, f1_score  # per-head metrics

from models.dual_resnet18.constants import AGE_MAP, ANTLER_MAP, COMBINED_NAMES  # label conventions
from models.dual_resnet18.dataset import ROOT, load_config  # paths

N_BINS = 10
BREAKDOWNS = ('bbox_touches_edge', 'day_night', 'month', 'camera_location_seq_no')


def head_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Accuracy, F1 on class 1 (team metric), and macro-F1.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        {'acc', 'f1', 'macro_f1'}.
    """
    return {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS
) -> tuple[float, pd.DataFrame]:
    """ECE: bin predictions by confidence; average |accuracy - confidence| weighted by bin size.

    Guo et al. 2017, "On Calibration of Modern Neural Networks", Eq. 3.

    Args:
        confidence: Probability of the predicted class, per sample.
        correct: Whether the prediction was right, per sample.
        n_bins: Number of equal-width bins on [0, 1].

    Returns:
        (ECE, reliability table with bin edges, count, mean confidence, accuracy).
    """
    bins = np.clip((confidence * n_bins).astype(int), 0, n_bins - 1)
    table = (
        pd.DataFrame({'bin': bins, 'confidence': confidence, 'correct': correct.astype(float)})
        .groupby('bin')
        .agg(n=('correct', 'size'), mean_confidence=('confidence', 'mean'), accuracy=('correct', 'mean'))
        .reset_index()
    )
    table['bin_low'] = table['bin'] / n_bins
    ece = float((table['n'] * (table['accuracy'] - table['mean_confidence']).abs()).sum() / len(confidence))
    return ece, table[['bin_low', 'n', 'mean_confidence', 'accuracy']]


def true_combined(df: pd.DataFrame) -> pd.Series:
    """Combined ground-truth class name per row (e.g. 'adult_antlered')."""
    return pd.Series(
        [COMBINED_NAMES[(a, g)] for a, g in zip(df['antler_true'], df['age_true'], strict=True)], index=df.index
    )


def breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group size, final per-head F1/accuracy, and flag rates.

    Args:
        df: Merged predictions + labels.
        by: Column to group by.

    Returns:
        One row per group value.
    """
    rows = []
    for value, g in df.groupby(by):
        age, ant = head_metrics(g['age_true'], g['age_pred']), head_metrics(g['antler_true'], g['antler_pred'])
        rows.append(
            {
                by: value,
                'n': len(g),
                'n_young': int(g['age_true'].sum()),
                'n_antlered': int(g['antler_true'].sum()),
                'age_f1': age['f1'],
                'age_acc': age['acc'],
                'antler_f1': ant['f1'],
                'antler_acc': ant['acc'],
                'needs_review': g['needs_review'].mean(),
                'undetermined': (g['antler_status'] == 'undetermined').mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('predictions', type=Path, help='CSV from inference.py')
    parser.add_argument('--config', default="default.toml", help='file in models/dual_resnet18/configs/')
    parser.add_argument('--baseline', type=Path, action='append', default=[], help='baseline predictions CSV')
    args = parser.parse_args()

    config = load_config(name=args.config)
    labels = pd.read_csv(ROOT / config['paths']['split'])
    labels['age_true'] = labels['age_status'].map(AGE_MAP)
    labels['antler_true'] = labels['antler_status'].map(ANTLER_MAP)
    preds = pd.read_csv(args.predictions)
    assert preds['filename'].is_unique, 'expected one crop per image (GT boxes); got duplicates'
    df = preds.merge(labels, on='filename', how='inner', suffixes=('', '_csv'))
    print(f"{len(df)} of {len(preds)} predictions have labels ({df['split'].unique().tolist()})")

    df['age_raw'] = (df['p_age_young'] > df['p_age_adult']).astype(int)
    df['antler_raw'] = (df['p_antlered'] > df['p_antlerless']).astype(int)
    summary = {
        ('dual', 'raw'): (
            head_metrics(df['age_true'], df['age_raw']),
            head_metrics(df['antler_true'], df['antler_raw']),
        ),
        ('dual', 'constrained'): (
            head_metrics(df['age_true'], df['age_pred']),
            head_metrics(df['antler_true'], df['antler_pred']),
        ),
    }
    for path in args.baseline:
        base = df[['filename', 'age_true', 'antler_true']].merge(pd.read_csv(path), on='filename', how='inner')
        age = head_metrics(base['age_true'], base['age_pred']) if 'age_pred' in base else {}
        ant = head_metrics(base['antler_true'], base['antler_pred']) if 'antler_pred' in base else {}
        summary[(path.stem, f'n={len(base)}')] = (age, ant)
    heads = pd.DataFrame(
        [
            {'model': m, 'variant': v}
            | {f'age_{k}': x for k, x in a.items()}
            | {f'antler_{k}': x for k, x in t.items()}
            for (m, v), (a, t) in summary.items()
        ]
    )

    confusion = pd.crosstab(true_combined(df), df['label'], rownames=['true'], colnames=['pred'])
    rates = pd.Series(
        {
            'needs_review': df['needs_review'].mean(),
            'no_detection': df['no_detection'].mean(),
            'undetermined': (df['antler_status'] == 'undetermined').mean(),
        }
    )
    age_ece, age_rel = expected_calibration_error(
        confidence=df[['p_age_adult', 'p_age_young']].max(axis=1).to_numpy(),
        correct=(df['age_raw'] == df['age_true']).to_numpy(),
    )
    ant_ece, ant_rel = expected_calibration_error(
        confidence=df[['p_antlerless', 'p_antlered']].max(axis=1).to_numpy(),
        correct=(df['antler_raw'] == df['antler_true']).to_numpy(),
    )

    stem = args.predictions.parent / f"{args.predictions.stem}_eval"
    heads.to_csv(f"{stem}_heads.csv", index=False)
    confusion.to_csv(f"{stem}_confusion.csv")
    pd.concat([age_rel.assign(head='age'), ant_rel.assign(head='antler')]).to_csv(
        f"{stem}_calibration.csv", index=False
    )
    tables = {by: breakdown(df=df, by=by) for by in BREAKDOWNS}
    for by, table in tables.items():
        table.to_csv(f"{stem}_by_{by}.csv", index=False)

    pd.set_option('display.width', 160)
    print('\nPer-head metrics (F1 on class 1: young / antlered)\n', heads.round(3).to_string(index=False))
    print('\nConfusion (rows true, cols predicted)\n', confusion.to_string())
    print('\nRates\n', rates.round(3).to_string())
    print(f"\nECE: age {age_ece:.3f}, antler {ant_ece:.3f}")
    for by in ('bbox_touches_edge', 'day_night', 'month'):
        print(f"\nBy {by}\n", tables[by].round(3).to_string(index=False))
    print(f"\nWrote {stem}_*.csv (camera breakdown only in CSV)")


if __name__ == '__main__':
    main()
