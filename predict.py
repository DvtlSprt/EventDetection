"""
Rugby Event Detection — Inference Script
=========================================

Loads a trained XGBoost model and runs event predictions on new game data.

Usage:
    python predict.py

Input:
    models/xgb_model_<timestamp>.json     - Trained XGBoost model (latest)
    models/label_encoder_<timestamp>.pkl  - Label encoder
    models/feature_cols_<timestamp>.pkl   - Feature column names
    models/thresholds_<timestamp>.pkl     - Per-class confidence thresholds
    models/metadata_<timestamp>.pkl       - Training metadata
    Predict/predict*.csv                  - Game CSVs to run predictions on

Output:
    predictions_output/<game_name>_predictions.csv  - Predicted events per game
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import xgboost as xgb
from scipy.signal import find_peaks
import glob


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_latest_model():
    """Load the most recent trained model."""
    models_folder = Path("models")

    model_files = sorted(models_folder.glob("xgb_model_*.json"))
    if not model_files:
        raise FileNotFoundError("No trained models found in models/ folder")

    latest_model = model_files[-1]
    timestamp = latest_model.stem.replace("xgb_model_", "")

    print(f"Loading model from timestamp: {timestamp}")

    clf = xgb.XGBClassifier()
    clf.load_model(str(latest_model))

    le_path = models_folder / f"label_encoder_{timestamp}.pkl"
    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    features_path = models_folder / f"feature_cols_{timestamp}.pkl"
    with open(features_path, 'rb') as f:
        feature_cols = pickle.load(f)

    thresholds_path = models_folder / f"thresholds_{timestamp}.pkl"
    with open(thresholds_path, 'rb') as f:
        thresholds = pickle.load(f)

    metadata_path = models_folder / f"metadata_{timestamp}.pkl"
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f"✓ Model loaded successfully!")
    print(f"  - Training precision: {metadata['precision']:.2%}")
    print(f"  - Training recall: {metadata['recall']:.2%}")
    print(f"  - Training F1: {metadata['f1']:.2%}")
    print(f"  - Features: {metadata['n_features']}")

    return clf, le, feature_cols, thresholds, metadata


# ─── Feature Engineering (must match train.py exactly) ───────────────────────

def normalize_positions(df):
    df = df.copy()
    df['x'] = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())
    df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
    return df


def extract_features(df):
    """Extract features from position data."""
    df = df.copy()
    if 'dx' not in df.columns:
        df['dx'] = 0
    if 'dy' not in df.columns:
        df['dy'] = 0

    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['theta'] = np.arctan2(df['dy'], df['dx'])

    if 'team' not in df.columns:
        return pd.DataFrame()

    df = df[df['team'].notna() & df['team'].isin([1, 2])]
    if len(df) == 0:
        return pd.DataFrame()

    df = df.sort_values(['frame', 'team'])

    df['y_rank_low'] = df.groupby(['frame', 'team'])['y'].rank(method='first', ascending=True)
    df['y_rank_high'] = df.groupby(['frame', 'team'])['y'].rank(method='first', ascending=False)
    df['x_rank_low'] = df.groupby(['frame', 'team'])['x'].rank(method='first', ascending=True)
    df['x_rank_high'] = df.groupby(['frame', 'team'])['x'].rank(method='first', ascending=False)

    df['y_low3'] = df['y'].where(df['y_rank_low'] <= 3)
    df['y_high3'] = df['y'].where(df['y_rank_high'] <= 3)
    df['x_low3'] = df['x'].where(df['x_rank_low'] <= 3)
    df['x_high3'] = df['x'].where(df['x_rank_high'] <= 3)

    agg = df.groupby(['frame', 'team']).agg(
        x_mean=('x', 'mean'),
        y_mean=('y', 'mean'),
        x_std=('x', 'std'),
        y_std=('y', 'std'),
        x_max=('x', 'max'),
        x_min=('x', 'min'),
        y_max=('y', 'max'),
        y_min=('y', 'min'),
        y_low3_min=('y_low3', 'min'),
        y_low3_max=('y_low3', 'max'),
        y_high3_min=('y_high3', 'min'),
        y_high3_max=('y_high3', 'max'),
        x_low3_min=('x_low3', 'min'),
        x_low3_max=('x_low3', 'max'),
        x_high3_min=('x_high3', 'min'),
        x_high3_max=('x_high3', 'max'),
        dx_mean=('dx', 'mean'),
        dx_std=('dx', 'std'),
        dy_mean=('dy', 'mean'),
        dy_std=('dy', 'std'),
        speed_mean=('speed', 'mean'),
        speed_std=('speed', 'std'),
        speed_max=('speed', 'max'),
        speed_min=('speed', 'min'),
        theta_mean=('theta', 'mean'),
        theta_std=('theta', 'std'),
        count=('x', 'count')
    ).reset_index()

    agg['width_x'] = agg['x_max'] - agg['x_min']
    agg['depth_y'] = agg['y_max'] - agg['y_min']
    agg['compression'] = agg['width_x'] / (agg['depth_y'] + 1)
    agg['front_x'] = agg['x_max']
    agg['compactness'] = agg['count'] / (agg['width_x'] * agg['depth_y'] + 1)

    white = agg[agg['team'] == 1].set_index('frame').drop(columns='team').add_suffix('_white')
    black = agg[agg['team'] == 2].set_index('frame').drop(columns='team').add_suffix('_black')
    features = white.join(black, how='outer').reset_index()

    features['front_player_sep'] = (features['front_x_black'] - features['front_x_white']).abs()
    features['speed_differential'] = (features['speed_mean_black'] - features['speed_mean_white']).abs()
    features['dx_diff'] = (features['dx_mean_black'] - features['dx_mean_white']).abs()
    features['dy_diff'] = (features['dy_mean_black'] - features['dy_mean_white']).abs()
    features['compression_diff'] = (features['compression_black'] - features['compression_white']).abs()
    features['team_sep'] = np.sqrt(
        (features['x_mean_black'] - features['x_mean_white'])**2 +
        (features['y_mean_black'] - features['y_mean_white'])**2
    )
    features['avg_team_speed'] = (
        features['speed_mean_black'].fillna(0) +
        features['speed_mean_white'].fillna(0)
    ) / 2

    features['theta_change_white'] = features['theta_mean_white'].diff()
    features['theta_change_black'] = features['theta_mean_black'].diff()
    features['acceleration'] = features['avg_team_speed'].diff()
    features['jerk'] = features['acceleration'].diff()

    features['near_left_touchline'] = ((features['x_mean_white'] < 0.1) | (features['x_mean_black'] < 0.1)).astype(int)
    features['near_right_touchline'] = ((features['x_mean_white'] > 0.9) | (features['x_mean_black'] > 0.9)).astype(int)
    features['near_tryline_top'] = ((features['y_mean_white'] < 0.1) | (features['y_mean_black'] < 0.1)).astype(int)
    features['near_tryline_bottom'] = ((features['y_mean_white'] > 0.9) | (features['y_mean_black'] > 0.9)).astype(int)

    features['in_tryzone'] = (
        (features['y_mean_white'] < 0.07) | (features['y_mean_black'] < 0.07) |
        (features['y_mean_white'] > 0.93) | (features['y_mean_black'] > 0.93)
    ).astype(int)

    features['scrum_score'] = (
        (1 / (features['width_x_white'] + 0.01)) +
        (1 / (features['width_x_black'] + 0.01))
    ) * (1 / (features['team_sep'] + 0.01)) * (1 / (features['avg_team_speed'] + 0.01))

    features['lineout_score'] = (
        (1 / (features['width_x_white'] + 0.01)) +
        (1 / (features['width_x_black'] + 0.01))
    ) * (features['depth_y_white'].fillna(0) + features['depth_y_black'].fillna(0))

    features['try_score'] = features['in_tryzone'] * (1 / (features['avg_team_speed'] + 0.01))

    features['kick_score'] = (
        (((features['speed_mean_white'] < 0.2) | (features['speed_mean_black'] < 0.2)).astype(int)) *
        (((features['width_x_white'] > 0.3) | (features['width_x_black'] > 0.3)).astype(int))
    )

    WINDOWS = [25, 100, 250, 500]
    BASE_COLS = [
        'avg_team_speed', 'team_sep',
        'compactness_white', 'compactness_black',
        'scrum_score', 'lineout_score',
        'try_score', 'acceleration'
    ]

    new_features = {}

    for col in BASE_COLS:
        s = features[col].astype(float)
        for w in WINDOWS:
            new_features[f'{col}_roll_center_mean_{w}'] = s.rolling(2*w+1, center=True, min_periods=1).mean().values
            new_features[f'{col}_roll_center_std_{w}'] = s.rolling(2*w+1, center=True, min_periods=1).std().values
            new_features[f'{col}_roll_center_max_{w}'] = s.rolling(2*w+1, center=True, min_periods=1).max().values
            new_features[f'{col}_roll_center_min_{w}'] = s.rolling(2*w+1, center=True, min_periods=1).min().values
            new_features[f'{col}_lag_pos_{w}'] = s.shift(w).bfill().values
            new_features[f'{col}_lag_neg_{w}'] = s.shift(-w).ffill().values

    VEL_DIFFS = [25, 100, 250, 500]
    for col in ['avg_team_speed', 'team_sep', 'compactness_white', 'compactness_black']:
        s = features[col].astype(float)
        for d in VEL_DIFFS:
            new_features[f'{col}_vel_center_{d}'] = (s.shift(-d).ffill() - s.shift(d).bfill()).values
            new_features[f'{col}_vel_forward_{d}'] = (s.shift(-d).ffill() - s).values
            new_features[f'{col}_vel_backward_{d}'] = (s - s.shift(d).bfill()).values

    new_features['sudden_stop'] = (
        (features['avg_team_speed'].shift(1).fillna(0) > 0.5) &
        (features['avg_team_speed'] < 0.15)
    ).astype(int).values

    new_features['accel_1s'] = features['avg_team_speed'].diff(25).values
    new_features['explosive_acceleration'] = (features['avg_team_speed'].diff(25) > 0.25).astype(int).values
    new_features['hard_deceleration'] = (features['avg_team_speed'].diff(25) < -0.25).astype(int).values

    new_features['formation_disruption'] = (
        features['x_std_white'].diff().abs() +
        features['x_std_black'].diff().abs()
    ).values

    new_features['momentum_change'] = (
        features['avg_team_speed'].diff().abs() +
        features['team_sep'].diff().abs()
    ).values

    features = pd.concat([features, pd.DataFrame(new_features, index=features.index)], axis=1)
    return features


def build_temporal_features(feats_df, feature_cols, window=60):
    """Add temporal window statistics for selected features."""
    df = feats_df.copy()
    arr = df[feature_cols].values
    N, F = arr.shape

    feats = {}

    def roll_stat(func, name):
        rolled = np.zeros_like(arr)
        for i in range(N):
            left = max(0, i - window)
            right = min(N, i + window + 1)
            rolled[i] = func(arr[left:right], axis=0)
        for j, col in enumerate(feature_cols):
            feats[f"{col}_{name}"] = rolled[:, j]

    roll_stat(np.mean, "mean")
    roll_stat(np.std,  "std")
    roll_stat(np.max,  "max")

    delta_forward = np.zeros_like(arr)
    delta_forward[:-1] = arr[1:] - arr[:-1]

    for j, col in enumerate(feature_cols):
        feats[f"{col}_dfwd"] = delta_forward[:, j]

    df_out = pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    return df_out


def detect_event_peaks_adaptive_conf(frame_feats, clf, feature_cols, le, thresholds, min_distance_default=200):
    """Detect event peaks using adaptive confidence thresholds."""
    X = frame_feats[feature_cols].fillna(0)
    probs = clf.predict_proba(X)
    inv = {i: lab for i, lab in enumerate(le.classes_)}
    peak_rows = []

    for cls_idx, cls_label in inv.items():
        if cls_label == "OpenPlay":
            continue

        min_conf = thresholds.get(cls_label, 0.5)
        class_prob = probs[:, cls_idx]

        smoothed = pd.Series(class_prob).rolling(7, center=True, min_periods=1).mean()

        peaks, _ = find_peaks(
            smoothed,
            height=min_conf,
            distance=min_distance_default
        )

        for p in peaks:
            peak_rows.append({
                "frame": frame_feats.iloc[p]["frame"],
                "event_smooth": cls_label,
                "confidence": class_prob[p]
            })

    return pd.DataFrame(peak_rows)


def post_filter_predictions(predictions, min_frame_gap=300):
    """Remove duplicate predictions that are too close together."""
    if len(predictions) == 0:
        return predictions

    filtered = predictions.sort_values(['event_smooth', 'frame'])
    to_keep = []

    for event_type in filtered['event_smooth'].unique():
        event_preds = filtered[filtered['event_smooth'] == event_type]
        if len(event_preds) == 0:
            continue

        kept_indices = [event_preds.index[0]]
        last_kept_frame = event_preds.iloc[0]['frame']

        for idx, row in event_preds.iloc[1:].iterrows():
            if row['frame'] - last_kept_frame > min_frame_gap:
                kept_indices.append(idx)
                last_kept_frame = row['frame']

        to_keep.extend(kept_indices)

    return filtered.loc[to_keep].sort_values('frame').reset_index(drop=True)


def reverse_features(feats):
    """Reverse feature order for backward pass."""
    rev = feats.copy().sort_values('frame', ascending=False)
    rev['frame'] = np.sort(rev['frame'].values)
    return rev


# ─── Prediction Pipeline ──────────────────────────────────────────────────────

def predict_game(game_df, clf, le, feature_cols, thresholds, game_name, min_frame_gap=500):
    """Run full prediction pipeline on a game."""
    print(f"\n{'='*60}")
    print(f"Processing: {game_name}")
    print(f"{'='*60}")

    game_df = normalize_positions(game_df)

    if 'dx' not in game_df.columns or 'dy' not in game_df.columns:
        if 'player' in game_df.columns:
            game_df = game_df.sort_values(['player', 'frame'])
            game_df['dx'] = game_df.groupby('player')['x'].diff().fillna(0)
            game_df['dy'] = game_df.groupby('player')['y'].diff().fillna(0)
        else:
            game_df = game_df.sort_values('frame')
            game_df['dx'] = game_df['x'].diff().fillna(0)
            game_df['dy'] = game_df['y'].diff().fillna(0)

    print("  ✓ Data normalized")

    base_feats = extract_features(game_df)
    print(f"  ✓ Base features extracted ({len(base_feats)} frames)")

    exclude_cols = ['frame', 'event', 'sample_weight', 'team', 'event_team']
    base_feature_cols = [col for col in base_feats.columns if col not in exclude_cols]

    game_feats = build_temporal_features(base_feats, base_feature_cols, window=60)
    print(f"  ✓ Temporal features added")

    preds_forward = detect_event_peaks_adaptive_conf(
        frame_feats=game_feats,
        clf=clf,
        feature_cols=feature_cols,
        le=le,
        thresholds=thresholds,
        min_distance_default=200
    )

    game_feats_rev = reverse_features(game_feats)
    preds_backward = detect_event_peaks_adaptive_conf(
        frame_feats=game_feats_rev,
        clf=clf,
        feature_cols=feature_cols,
        le=le,
        thresholds=thresholds,
        min_distance_default=200
    )
    preds_backward = preds_backward.sort_values('frame')

    merged = preds_forward.copy()
    if 'event_smooth' in merged.columns:
        merged['event_smooth'] = np.where(
            preds_forward['event_smooth'].isna() | (preds_forward['event_smooth'] == -1),
            preds_backward['event_smooth'],
            preds_forward['event_smooth']
        )
    if 'confidence' in merged.columns:
        merged['confidence'] = np.maximum(preds_forward['confidence'], preds_backward['confidence'])

    predictions = merged
    predictions = post_filter_predictions(predictions, min_frame_gap=min_frame_gap)

    print(f"  ✓ Generated {len(predictions)} predictions")

    return predictions


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_predictions_timeline(predictions, game_name):
    """Visualize predictions on a timeline."""
    if len(predictions) == 0:
        print(f"No predictions to plot for {game_name}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))

    event_types = sorted(predictions['event_smooth'].unique())
    event_to_y = {event: idx for idx, event in enumerate(event_types)}

    colors = plt.cm.tab20(np.linspace(0, 1, len(event_types)))
    color_map = dict(zip(event_types, colors))

    ax.scatter(
        predictions["frame"],
        predictions["event_smooth"].map(event_to_y),
        s=150, marker='o',
        c=predictions["event_smooth"].map(color_map),
        edgecolors="black", linewidths=1.5, alpha=0.8
    )

    ax.set_title(f"Predicted Events - {game_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Event Type', fontsize=12)
    ax.set_yticks(range(len(event_types)))
    ax.set_yticklabels(event_types)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load the model
    clf, le, feature_cols, thresholds, metadata = load_latest_model()

    predict_folder = Path("Predict")
    predict_files = sorted(predict_folder.glob("predict*.csv"))

    if not predict_files:
        print("No prediction files found in Predict/ folder")
        return

    print(f"\nFound {len(predict_files)} files to predict")

    output_folder = Path("predictions_output")
    output_folder.mkdir(exist_ok=True)

    all_predictions = {}

    for predict_file in predict_files:
        game_name = predict_file.stem

        game_df = pd.read_csv(predict_file)

        if 'team' in game_df.columns and game_df['team'].dtype == 'object':
            game_df['team'] = game_df['team'].map({'white': 1, 'black': 2})

        predictions = predict_game(
            game_df=game_df,
            clf=clf,
            le=le,
            feature_cols=feature_cols,
            thresholds=thresholds,
            game_name=game_name,
            min_frame_gap=500
        )

        output_path = output_folder / f"{game_name}_predictions.csv"
        predictions.to_csv(output_path, index=False)
        print(f"  ✓ Saved to: {output_path}")

        all_predictions[game_name] = predictions

    print(f"\n{'='*60}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*60}")
    for game_name, preds in all_predictions.items():
        print(f"\n{game_name}:")
        print(f"  Total predictions: {len(preds)}")
        if len(preds) > 0:
            event_counts = preds['event_smooth'].value_counts()
            for event, count in event_counts.items():
                print(f"    - {event}: {count}")

    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    for game_name, preds in all_predictions.items():
        plot_predictions_timeline(preds, game_name)


if __name__ == '__main__':
    main()
