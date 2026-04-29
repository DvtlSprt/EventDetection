#!/usr/bin/env python3
"""
Setup Training Data for XGBoost Event Detection
================================================

This script converts the real All Blacks v Wales rugby match data
(located in Myfiles/xboostfiles/) into the format expected by
EventDetection_Pipeline.ipynb.

It places:
- Train/game1.csv + Train/RugbyEvents1.csv  (from chunk_008)
- Evaluate/game2.csv + Evaluate/RugbyEvents2.csv (from chunk_009)

The XY CSV format expected:
  timestamp, frame, object_type, object_id, team, team_conf, team_src, x, y, team_possession, note

The Events CSV format expected:
  frameIdx, timeStart, Event, teamIn, x (length)m, y (width)

Output game CSV format (for EventDetection_Pipeline.ipynb):
  frame, x, y, dx, dy, team

Output RugbyEvents CSV format:
  Event, Frame_Start, Frame_End, Team

Usage:
    python setup_training_data.py

Requirements:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Event type mapping: pipeline events -> notebook expected events
EVENT_MAPPING = {
    'Scrum': 'Scrum',
    'Lineout': 'Lineout',
    'Turnover': 'Turnover',
    'KickRestart': 'KickRestart',
    'CollectKick': 'CollectKick',
    'Ruck': 'Ruck',
    'Stop': 'OpenPlay',
    'Halftime': 'Halftime',
}


def compute_velocities(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dx, dy velocity columns per track.
    Input: DataFrame with columns [frame, object_id, team, x, y]
    Output: DataFrame with columns [frame, x, y, dx, dy, team]
    """
    print("    🔄 Computing per-track velocities (dx, dy)...")

    players = players_df.copy()

    # Sort by track and frame for velocity computation
    players = players.sort_values(['object_id', 'frame'])

    # Compute dx, dy per track
    players['dx'] = players.groupby('object_id')['x'].diff()
    players['dy'] = players.groupby('object_id')['y'].diff()

    # Fill NaN (first frame per track) with 0
    players['dx'] = players['dx'].fillna(0.0)
    players['dy'] = players['dy'].fillna(0.0)

    # Handle outliers (jumps > 20m/frame are likely track switches)
    speed = np.sqrt(players['dx']**2 + players['dy']**2)
    outlier_mask = speed > 20.0
    players.loc[outlier_mask, 'dx'] = 0.0
    players.loc[outlier_mask, 'dy'] = 0.0

    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        print(f"    ⚠️  Clamped {outlier_count} velocity outliers (>20m/frame)")

    print(f"    ✓ Computed velocities for {len(players)} player detections")
    return players


def create_game_csv(xy_df: pd.DataFrame, output_path: str) -> int:
    """
    Create game.csv in notebook format: frame, x, y, dx, dy, team
    
    Returns the minimum frame number (offset) from the original data.
    """
    print(f"\n    📝 Creating game CSV: {output_path}")

    # Filter to only player rows (exclude ball, referee, meta)
    players = xy_df[
        (xy_df['object_type'] == 'player') &
        (xy_df['team'].isin([1.0, 2.0]))
    ].copy()

    print(f"    ℹ️  Player rows: {len(players)} (from {len(xy_df)} total)")

    if len(players) == 0:
        print("    ❌ ERROR: No player rows found! Check data format.")
        return 0

    # Cast team to int
    players['team'] = players['team'].astype(int)

    # Compute velocities
    players = compute_velocities(players)

    # Get minimum frame for offset
    min_frame = int(players['frame'].min())

    # Reset frame numbers to start from 0
    players['frame'] = players['frame'] - min_frame

    # Select output columns
    game_df = players[['frame', 'x', 'y', 'dx', 'dy', 'team']].copy()
    game_df['frame'] = game_df['frame'].astype(int)

    # Sort by frame
    game_df = game_df.sort_values(['frame', 'team']).reset_index(drop=True)

    # Save
    game_df.to_csv(output_path, index=False)

    print(f"    ✓ Saved {len(game_df)} rows to {output_path}")
    print(f"      Frames: {game_df['frame'].min()} → {game_df['frame'].max()}")
    print(f"      Team 1: {(game_df['team'] == 1).sum()} detections")
    print(f"      Team 2: {(game_df['team'] == 2).sum()} detections")

    return min_frame


def generate_openplay_events(fps: float, existing_events: list,
                              event_duration_frames: int, max_frame: int) -> list:
    """Generate OpenPlay (background) samples every 5 seconds where no event."""
    sample_interval = int(5 * fps)
    candidate_frames = list(range(0, max_frame, sample_interval))

    event_frames = {e['Frame_Start'] for e in existing_events}
    min_distance = int(10 * fps)

    openplay_events = []
    for frame_start in candidate_frames:
        nearest = min([abs(frame_start - ef) for ef in event_frames], default=9999)
        if nearest > min_distance:
            openplay_events.append({
                'Event': 'OpenPlay',
                'Frame_Start': frame_start,
                'Frame_End': frame_start + event_duration_frames,
                'Team': 0
            })

    return openplay_events


def create_rugbyevent_csv(events_df: pd.DataFrame, output_path: str,
                           frame_offset: int = 0, fps: float = 30.0,
                           event_duration_s: float = 3.0) -> None:
    """
    Create RugbyEvents.csv in notebook format: Event, Frame_Start, Frame_End, Team
    
    Args:
        events_df: DataFrame with events.
        output_path: Where to save the CSV.
        frame_offset: Minimum frame number from the XY data (to normalize frames to 0).
        fps: Frames per second.
        event_duration_s: Duration of each event window in seconds.
    """
    print(f"\n    📝 Creating rugby events CSV: {output_path}")

    event_duration_frames = int(event_duration_s * fps)
    events = []

    for _, row in events_df.iterrows():
        event_type = row['Event']
        frame_start = int(row['frameIdx']) - frame_offset
        team_code = str(row.get('teamIn', '1'))

        # Skip events that fall before the start of the XY data
        if frame_start < 0:
            continue

        # Map team code to team number
        # Format is like "L2", "R2", "L", "R"
        if team_code.startswith('L'):
            team = 1  # Left team = Team 1
        elif team_code.startswith('R'):
            team = 2  # Right team = Team 2
        else:
            try:
                team = int(team_code[0])
            except (ValueError, IndexError):
                team = 1

        # Map to notebook event class
        mapped_event = EVENT_MAPPING.get(event_type)

        if mapped_event:
            events.append({
                'Event': mapped_event,
                'Frame_Start': frame_start,
                'Frame_End': frame_start + event_duration_frames,
                'Team': team
            })

    print(f"    ✓ Mapped {len(events)} auto-tagged events")

    if not events:
        print("    ❌ WARNING: No events mapped! Check data format and frame_offset.")
        return

    # Find the max frame from the game for OpenPlay sampling
    max_frame = max(e['Frame_End'] for e in events)

    # Add OpenPlay background samples
    openplay_events = generate_openplay_events(fps, events, event_duration_frames, max_frame)
    events.extend(openplay_events)
    print(f"    ✓ Added {len(openplay_events)} OpenPlay samples")

    # Convert to DataFrame and sort
    events_df_out = pd.DataFrame(events)
    events_df_out = events_df_out.sort_values('Frame_Start').drop_duplicates(subset=['Frame_Start'])

    # Save
    events_df_out.to_csv(output_path, index=False)

    print(f"    ✓ Saved {len(events_df_out)} events to {output_path}")
    print(f"      Event distribution:")
    for event_type, count in events_df_out['Event'].value_counts().items():
        print(f"        {event_type}: {count}")


def process_chunk(xy_path: Path, events_path: Path,
                  game_csv_path: Path, rugby_events_csv_path: Path,
                  fps: float = 30.0) -> None:
    """Process a single chunk of game data."""
    print(f"\n    📂 Loading XY data from: {xy_path.name}")
    xy_df = pd.read_csv(xy_path)
    print(f"       ✓ Loaded {len(xy_df)} total rows")
    print(f"       Columns: {xy_df.columns.tolist()}")

    print(f"\n    📂 Loading events from: {events_path.name}")
    events_df = pd.read_csv(events_path)
    print(f"       ✓ Loaded {len(events_df)} events")
    print(f"       Event types: {events_df['Event'].value_counts().to_dict()}")

    # Create game CSV and get frame offset
    frame_offset = create_game_csv(xy_df, str(game_csv_path))

    # Create rugby events CSV with correct offset
    create_rugbyevent_csv(events_df, str(rugby_events_csv_path),
                           frame_offset=frame_offset, fps=fps)


def main():
    print("=" * 60)
    print("🎯 Setting Up XGBoost Training Data")
    print("   All Blacks v Wales Cardiff 2025")
    print("=" * 60)

    # Base directory (where this script lives)
    base_dir = Path(__file__).parent
    xboost_dir = base_dir / "Myfiles" / "xboostfiles"
    train_dir = base_dir / "Train"
    eval_dir = base_dir / "Evaluate"

    # Ensure output directories exist
    train_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)

    # Find available chunk files
    chunk_008_xy = xboost_dir / "All_Blacks_v_Wales_Cardiff_2025_FULL_GAME_chunk_008_9f3ffb_xy.csv"
    chunk_008_events = xboost_dir / "All_Blacks_v_Wales_Cardiff_2025_FULL_GAME_chunk_008_9f3ffb_events.csv"
    chunk_009_xy = xboost_dir / "All_Blacks_v_Wales_Cardiff_2025_FULL_GAME_chunk_009_9f3ffb_xy.csv"
    chunk_009_events = xboost_dir / "All_Blacks_v_Wales_Cardiff_2025_FULL_GAME_chunk_009_9f3ffb_events.csv"

    # Check files exist
    missing = []
    for f in [chunk_008_xy, chunk_008_events, chunk_009_xy, chunk_009_events]:
        if not f.exists():
            missing.append(str(f))

    if missing:
        print("\n❌ Missing files:")
        for f in missing:
            print(f"   {f}")
        print("\nPlease ensure the xboostfiles directory contains both chunks.")
        return

    print("\n✓ Found all required data files")
    print(f"  Source: {xboost_dir}")
    print(f"  Training output: {train_dir}")
    print(f"  Evaluation output: {eval_dir}")

    # Process chunk_008 -> Train (game1)
    print("\n" + "=" * 60)
    print("📊 Processing TRAINING data")
    print("   chunk_008 → Train/game1.csv + Train/RugbyEvents1.csv")
    print("=" * 60)
    process_chunk(
        xy_path=chunk_008_xy,
        events_path=chunk_008_events,
        game_csv_path=train_dir / "game1.csv",
        rugby_events_csv_path=train_dir / "RugbyEvents1.csv"
    )

    # Process chunk_009 -> Evaluate (game2)
    print("\n" + "=" * 60)
    print("📊 Processing EVALUATION data")
    print("   chunk_009 → Evaluate/game2.csv + Evaluate/RugbyEvents2.csv")
    print("=" * 60)
    process_chunk(
        xy_path=chunk_009_xy,
        events_path=chunk_009_events,
        game_csv_path=eval_dir / "game2.csv",
        rugby_events_csv_path=eval_dir / "RugbyEvents2.csv"
    )

    print("\n" + "=" * 60)
    print("✅ COMPLETE! Training files are ready:")
    print(f"   📄 Train/game1.csv")
    print(f"   📄 Train/RugbyEvents1.csv")
    print(f"   📄 Evaluate/game2.csv")
    print(f"   📄 Evaluate/RugbyEvents2.csv")
    print()
    print("🚀 Next Step:")
    print("   Open and run EventDetection_Pipeline.ipynb")
    print("   The model will be saved to models/event_model_latest.joblib")
    print("=" * 60)


if __name__ == '__main__':
    main()
