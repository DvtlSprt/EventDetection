"""
EventDetection Training Pipeline
=================================

Trains an XGBoost model for rugby event detection from player tracking data.

Usage:
    python train.py

Input:
    Train/game*.csv           - Player tracking data (frame, x, y, dx, dy, team)
    Train/RugbyEvents*.csv    - Ground truth events (Event, Frame_Start, Frame_End, Team)
    Evaluate/game*.csv        - Evaluation tracking data
    Evaluate/RugbyEvents*.csv - Evaluation ground truth events

Output:
    models/xgb_model_<timestamp>.json     - Trained XGBoost model
    models/label_encoder_<timestamp>.pkl  - Label encoder
    models/feature_cols_<timestamp>.pkl   - Feature column names
    models/thresholds_<timestamp>.pkl     - Per-class confidence thresholds
    models/metadata_<timestamp>.pkl       - Training metadata
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import xgboost as xgb
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import joblib
import shutil
import os
import re

# ─── Configuration ────────────────────────────────────────────────────────────

TRAIN_FOLDER = "Train"
EVAL_FOLDER = "Evaluate"
MODELS_FOLDER = "models"
PREDICTIONS_FOLDER = "predictions"
GAME_DATA_CACHE = "Game_data"

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_games_from_folder(folder_path):
    """
    Load (positions, events) pairs from a folder.
    Returns: list of (game_df, events_df, game_id)
    """
    folder = Path(folder_path)
    games = []

    game_files = sorted(folder.glob("game*.csv"))

    for game_file in game_files:
        match = re.search(r"game(\d+)", game_file.stem)
        if not match:
            continue

        game_id = match.group(1)
        events_file = folder / f"RugbyEvents{game_id}.csv"

        if not events_file.exists():
            print(f"⚠️ Missing events for game {game_id}, skipping")
            continue

        game_df = pd.read_csv(game_file)
        events_df = pd.read_csv(events_file)

        games.append((game_df, events_df, game_id))

    return games


def load_game(df):
    df = df.drop(columns=['team'], errors='ignore')
    df['x_med'] = df.groupby('frame')['x'].transform('median')
    df['team'] = (df['x'] >= df['x_med']).astype(int) + 1
    return df


def process_game(game_df, events_df):
    game_df = load_game(game_df)

    upsampled_rows = []
    frame_indices = sorted(game_df['frame'].astype(int).unique())

    for i in range(len(frame_indices)):
        cur = int(frame_indices[i])
        nxt = int(frame_indices[i+1]) if i+1 < len(frame_indices) else cur + 1
        current_frame_data = game_df[game_df['frame'] == cur]

        for f in range(cur, nxt):
            temp = current_frame_data.copy()
            temp['frame'] = f
            upsampled_rows.append(temp)

    game_upsampled = pd.concat(upsampled_rows, ignore_index=True)

    events_df = events_df.rename(columns={
        'Event': 'event',
        'Frame_Start': 'frame_start',
        'Frame_End': 'frame_end',
        'Team': 'event_team'
    })

    expanded_events = []
    for _, row in events_df.iterrows():
        expanded_events.extend([
            {'frame': f, 'event': row['event'], 'event_team': row['event_team']}
            for f in range(row['frame_start'], row['frame_end'] + 1)
        ])

    expanded_events = pd.DataFrame(expanded_events)
    game_data = game_upsampled.merge(expanded_events, on='frame', how='left')
    return game_data.loc[:, ~game_data.columns.duplicated()]


def load_or_process_game(game_df, events_df, game_id, cache_folder=GAME_DATA_CACHE):
    """Load cached game data if exists, otherwise process and save."""
    cache_path = Path(cache_folder) / f"game_data{game_id}.csv"

    if cache_path.exists():
        print(f"✓ Loading cached game_data{game_id}.csv")
        return pd.read_csv(cache_path)

    print(f"Processing game {game_id}...")
    df = process_game(game_df, events_df)

    Path(cache_folder).mkdir(exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"✓ Saved game_data{game_id}.csv")

    return df


# ─── Helper Functions ─────────────────────────────────────────────────────────

def normalize_positions(df):
    df = df.copy()
    df['x'] = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())
    df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
    df['event'] = df['event'].replace('Kick', np.nan)
    return df


def create_class_weights(training_data):
    """Assign sample weights to balance classes."""
    class_counts = training_data['event'].value_counts()
    total = len(training_data)
    class_weights = {cls: total/count for cls, count in class_counts.items()}

    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

    training_data['sample_weight'] = training_data['event'].apply(lambda x: class_weights[x])
    return training_data


def detect_event_peaks_adaptive_conf(frame_feats, clf, feature_cols, le, thresholds, min_distance_default=200):
    X = frame_feats[feature_cols].fillna(0)
    probs = clf.predict_proba(X)
    inv = {i: lab for i, lab in enumerate(le.classes_)}
    peak_rows = []

    for cls_idx, cls_label in inv.items():
        if cls_label == "OpenPlay":
            continue

        # Adaptive confidence threshold
        min_conf = thresholds.get(cls_label, 0.5)
        class_prob = probs[:, cls_idx]

        # Smooth signal
        smoothed = pd.Series(class_prob).rolling(7, center=True, min_periods=1).mean()

        # Peak detection
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


def build_event_intervals(event_df):
    """Convert frame-level events to intervals."""
    event_df = event_df.sort_values('frame').reset_index(drop=True)
    intervals = []
    prev_event = None
    start_frame = None

    for i, row in event_df.iterrows():
        if row['event'] != prev_event:
            if prev_event is not None:
                intervals.append({
                    'event': prev_event,
                    'start_frame': start_frame,
                    'end_frame': event_df.iloc[i-1]['frame']
                })
            prev_event = row['event']
            start_frame = row['frame']

    if prev_event is not None:
        intervals.append({
            'event': prev_event,
            'start_frame': start_frame,
            'end_frame': event_df.iloc[-1]['frame']
        })

    return pd.DataFrame(intervals)


def evaluate_predictions(predicted_events, ground_truth_intervals, method_name="Model"):
    if len(predicted_events) == 0:
        print(f"{method_name}: No events predicted!")
        return None

    def check_correct(row):
        pred_frame = row['frame']
        pred_event = row['event_smooth']

        matching = ground_truth_intervals[
            (ground_truth_intervals['event'] == pred_event) &
            (ground_truth_intervals['start_frame'] <= pred_frame) &
            (ground_truth_intervals['end_frame'] >= pred_frame)
        ]
        return len(matching) > 0

    predicted_events['correct'] = predicted_events.apply(check_correct, axis=1)

    precision = predicted_events['correct'].mean() if len(predicted_events) > 0 else 0

    recall_results = []
    for event in ground_truth_intervals['event'].unique():
        actual_intervals = ground_truth_intervals[ground_truth_intervals['event'] == event]
        detected = 0
        for _, interval in actual_intervals.iterrows():
            any_pred = predicted_events[
                (predicted_events['event_smooth'] == event) &
                (predicted_events['frame'] >= interval['start_frame']) &
                (predicted_events['frame'] <= interval['end_frame'])
            ]
            if len(any_pred) > 0:
                detected += 1
        recall = detected / len(actual_intervals) if len(actual_intervals) > 0 else 0
        recall_results.append({'event': event, 'recall': recall, 'detected': detected, 'total': len(actual_intervals)})

    recall_df = pd.DataFrame(recall_results)
    avg_recall = recall_df['recall'].mean()
    f1 = 2 * (precision * avg_recall) / (precision + avg_recall) if (precision + avg_recall) > 0 else 0

    print(f"\n{method_name}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {avg_recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    print(f"  Predictions: {len(predicted_events)}")

    return {
        'method': method_name,
        'precision': precision,
        'recall': avg_recall,
        'f1': f1,
        'predictions': predicted_events,
        'recall_by_event': recall_df
    }


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(df):
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


def build_temporal_features(feats_df, feature_cols, window=100):
    """
    Add temporal window statistics for selected features.
    Use sparingly to avoid feature explosion.
    """
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


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_comparison_timeline(predictions, ground_truth_raw):
    """Timeline visualization of predictions vs ground truth."""
    fig, ax = plt.subplots(1, 1, figsize=(22, 8))

    if 'is_peak' in ground_truth_raw.columns:
        gt_peaks = ground_truth_raw[ground_truth_raw['is_peak'] == True]
    else:
        gt_peaks = ground_truth_raw.sort_values('frame')
        gt_peaks = gt_peaks[
            (gt_peaks['event'] != gt_peaks['event'].shift(1)) |
            (gt_peaks['frame'] != gt_peaks['frame'].shift(1))
        ]

    event_types = sorted(set(gt_peaks['event'].unique()) |
                         set(predictions['event_smooth'].unique()))
    event_to_y = {event: idx for idx, event in enumerate(event_types)}

    colors = plt.cm.tab20(np.linspace(0, 1, len(event_types)))
    color_map = dict(zip(event_types, colors))

    ax.scatter(
        gt_peaks['frame'],
        gt_peaks['event'].map(event_to_y),
        s=200, marker='|', linewidths=4,
        c=gt_peaks['event'].map(color_map),
        zorder=2, label='Ground Truth', alpha=0.9
    )

    ax.scatter(
        predictions["frame"],
        predictions["event_smooth"].map(event_to_y) + 0.15,
        s=100, marker='o',
        c=predictions["event_smooth"].map(color_map),
        edgecolors="black", linewidths=1.5, alpha=0.8,
        label='Predictions', zorder=3
    )

    ax.set_title("Event Detection Timeline", fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Event Type', fontsize=12)
    ax.set_yticks(range(len(event_types)))
    ax.set_yticklabels(event_types)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(results1, results2):
    """Compare performance metrics between two models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = [results1['method'], results2['method']]
    metrics = ['precision', 'recall', 'f1']
    metric_names = ['Precision', 'Recall', 'F1 Score']

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results1[metric], results2[metric]]
        bars = axes[idx].bar(methods, values, color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[idx].set_ylabel(name, fontsize=12, fontweight='bold')
        axes[idx].set_ylim(0, 1)
        axes[idx].set_title(name, fontsize=13, fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold'
            )

    plt.tight_layout()
    plt.show()


def plot_feature_importance(clf, feature_cols, top_n=20):
    """Plot top feature importances."""
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ─── Training ─────────────────────────────────────────────────────────────────

def reverse_features(feats):
    rev = feats.copy().sort_values('frame', ascending=False)
    rev['frame'] = np.sort(rev['frame'].values)
    return rev


def train_and_evaluate(train_df, train_feats, test_df, test_feats, feature_cols, name,
                       thresholds,
                       min_frame_gap=300):

    print(f"\n{'='*80}")
    print(name)
    print(f"{'='*80}")

    gt_train = train_df[train_df['event'].notna()][['frame', 'event']].drop_duplicates('frame')
    gt_test = test_df[test_df['event'].notna()][['frame', 'event']].drop_duplicates('frame')

    training_data = train_feats.merge(gt_train, on='frame', how='inner')

    available_nonevent = train_feats[~train_feats['frame'].isin(gt_train['frame'])]
    n_nonevents = min(5000, len(available_nonevent))
    nonevent = available_nonevent.sample(n=n_nonevents, random_state=42)
    nonevent["event"] = "NonEvent"

    training_full = pd.concat([training_data, nonevent], ignore_index=True)
    training_full = create_class_weights(training_full)

    le = LabelEncoder()
    y_train = le.fit_transform(training_full['event'])
    X_train = training_full[feature_cols].fillna(0)
    sample_weights = training_full['sample_weight'].values

    clf = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        min_child_weight=10,
        gamma=0.3,
        reg_alpha=0.1,
        reg_lambda=3.0,
        max_delta_step=1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )

    clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=1)

    print("\n✓ XGBoost model trained")

    preds_forward = detect_event_peaks_adaptive_conf(
        frame_feats=test_feats,
        clf=clf,
        feature_cols=feature_cols,
        le=le,
        thresholds=thresholds,
        min_distance_default=200
    )

    test_feats_rev = reverse_features(test_feats)

    preds_backward = detect_event_peaks_adaptive_conf(
        frame_feats=test_feats_rev,
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

    predictions['event_smooth'] = predictions['event_smooth'].apply(
        lambda x: le.classes_[x] if isinstance(x, int) else x
    )

    predictions = post_filter_predictions(predictions, min_frame_gap=min_frame_gap)

    print(f"✓ Generated {len(predictions)} predictions after post-filtering")

    gt_test_intervals = build_event_intervals(gt_test)
    results = evaluate_predictions(predictions, gt_test_intervals, method_name=name)

    results['ground_truth_raw'] = gt_test
    results['clf'] = clf
    results['label_encoder'] = le

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load training games
    print("Loading training games...")
    train_games = load_games_from_folder(TRAIN_FOLDER)
    print(f"✓ Loaded {len(train_games)} training games")

    train_dfs = []
    for game_df, events_df, game_id in train_games:
        df = load_or_process_game(game_df, events_df, game_id)
        train_dfs.append(df)

    # Load evaluation games
    print("Loading evaluation games...")
    eval_games = load_games_from_folder(EVAL_FOLDER)
    print(f"✓ Loaded {len(eval_games)} evaluation games")

    eval_dfs = []
    for game_df, events_df, game_id in eval_games:
        df = load_or_process_game(game_df, events_df, game_id)
        eval_dfs.append(df)

    # Combine all game data
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(eval_dfs, ignore_index=True)

    train_df = normalize_positions(train_df)
    test_df = normalize_positions(test_df)

    # Extract features
    print("\nExtracting features...")
    train_feats = extract_features(train_df)
    test_feats = extract_features(test_df)

    exclude_cols = ['frame', 'event', 'sample_weight', 'team', 'event_team']
    base_feature_cols = [col for col in train_feats.columns if col not in exclude_cols]

    train_feats = build_temporal_features(train_feats, base_feature_cols, window=100)
    test_feats = build_temporal_features(test_feats, base_feature_cols, window=100)

    feature_cols = [col for col in train_feats.columns if col not in exclude_cols]

    thresholds = {
        "KickRestart": 0.3,
        "Scrum": 0.3,
        "Lineout": 0.3,
        "Try": 0.3,
        "Turnover": 0.25,
        "CollectKick": 0.25,
        "Ruck": 0.3,
    }

    results = train_and_evaluate(
        train_df=train_df,
        train_feats=train_feats,
        test_df=test_df,
        test_feats=test_feats,
        feature_cols=feature_cols,
        name="XGBoost Event Detector",
        thresholds=thresholds,
        min_frame_gap=300
    )

    # Save model
    clf = results['clf']
    le = results['label_encoder']

    Path(MODELS_FOLDER).mkdir(exist_ok=True)

    # Remove old model
    old_model_path = Path(MODELS_FOLDER) / "event_model_latest.joblib"
    if old_model_path.exists():
        os.remove(old_model_path)
        print(f"🗑️ Deleted old model: event_model_latest.joblib")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = Path(MODELS_FOLDER) / f"xgb_model_{timestamp}.json"
    clf.save_model(str(model_path))

    le_path = Path(MODELS_FOLDER) / f"label_encoder_{timestamp}.pkl"
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)

    features_path = Path(MODELS_FOLDER) / f"feature_cols_{timestamp}.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)

    thresholds_path = Path(MODELS_FOLDER) / f"thresholds_{timestamp}.pkl"
    with open(thresholds_path, 'wb') as f:
        pickle.dump(thresholds, f)

    metadata = {
        'timestamp': timestamp,
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'n_features': len(feature_cols),
        'train_games': len(train_games),
        'eval_games': len(eval_games),
    }
    metadata_path = Path(MODELS_FOLDER) / f"metadata_{timestamp}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n✓ Model saved: {model_path}")

    # Visualize results
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    plot_comparison_timeline(
        predictions=results['predictions'],
        ground_truth_raw=results['ground_truth_raw']
    )

    plot_feature_importance(clf=clf, feature_cols=feature_cols, top_n=20)


if __name__ == '__main__':
    main()
