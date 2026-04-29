#!/usr/bin/env python3
"""
Data Preparation Script for XGBoost Event Detection
====================================================

This script transforms your existing pipeline outputs into the format expected
by the EventDetection_Pipeline.ipynb notebook.

INPUT FILES (from your pipeline):
- *_xy.csv: Player/ball tracking data with team assignments
- *_events.csv: Rule-based auto-tagged events

OUTPUT FILES (for XGBoost notebook):
- game1.csv / game2.csv: Per-frame player tracking with dx/dy velocities
- rugbyevent1.csv / rugbyevent2.csv: Ground-truth event labels

Usage:
    python prepare_xgboost_data.py --input-xy your_game_xy.csv --input-events your_game_events.csv --output-prefix game1
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class XGBoostDataPreparation:
    """Convert pipeline outputs to XGBoost training format"""
    
    # Event type mapping: your events -> notebook expected events
    EVENT_MAPPING = {
        'Scrum': 'Scrum',
        'Lineout': 'Lineout',
        'Turnover': 'Turnover',
        'KickRestart': 'KickRestart',
        'CollectKick': 'CollectKick',
        'Ruck': 'Ruck',
        'Stop': 'OpenPlay',  # Map Stop to OpenPlay (background class)
        'Halftime': 'Halftime',
    }
    
    # Notebook expects these event classes
    EXPECTED_EVENTS = ['Scrum', 'Lineout', 'Turnover', 'KickRestart', 
                       'Ruck', 'Try', 'OpenPlay', 'Halftime']
    
    def __init__(self, xy_csv_path: str, events_csv_path: str):
        """
        Initialize data preparation
        
        Args:
            xy_csv_path: Path to *_xy.csv from pipeline
            events_csv_path: Path to *_events.csv from pipeline
        """
        self.xy_path = Path(xy_csv_path)
        self.events_path = Path(events_csv_path)
        
        # Load data
        print(f"📂 Loading data from:")
        print(f"   XY: {self.xy_path}")
        print(f"   Events: {self.events_path}")
        
        self.xy_df = pd.read_csv(self.xy_path)
        self.events_df = pd.read_csv(self.events_path)
        
        print(f"✅ Loaded {len(self.xy_df)} tracking rows, {len(self.events_df)} events")
    
    def compute_velocities(self) -> pd.DataFrame:
        """
        Compute dx, dy velocity columns per track
        
        The notebook expects: frame, x, y, dx, dy, team
        
        Returns:
            DataFrame with velocity columns added
        """
        print("\n🔄 Computing per-track velocities (dx, dy)...")
        
        df = self.xy_df.copy()
        
        # Filter to players only (exclude ball and referees)
        players = df[
            (df['object_type'] == 'player') & 
            (df['team'].isin([1, 2]))  # Only teams 1 and 2
        ].copy()
        
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
            print(f"   ⚠️  Clamped {outlier_count} velocity outliers (>20m/frame)")
        
        print(f"✅ Computed velocities for {len(players)} player detections")
        
        return players
    
    def create_game_csv(self, output_path: str) -> None:
        """
        Create game.csv in notebook format: frame, x, y, dx, dy, team
        
        Args:
            output_path: Where to save game CSV
        """
        print(f"\n📝 Creating game CSV: {output_path}")
        
        players = self.compute_velocities()
        
        # Select and rename columns to match notebook format
        game_df = players[['frame', 'x', 'y', 'dx', 'dy', 'team']].copy()
        
        # Ensure correct dtypes
        game_df['frame'] = game_df['frame'].astype(int)
        game_df['team'] = game_df['team'].astype(int)
        
        # Save
        game_df.to_csv(output_path, index=False)
        
        print(f"✅ Saved {len(game_df)} rows to {output_path}")
        print(f"   Frames: {game_df['frame'].min()} → {game_df['frame'].max()}")
        print(f"   Team 1: {(game_df['team'] == 1).sum()} detections")
        print(f"   Team 2: {(game_df['team'] == 2).sum()} detections")
    
    def create_rugbyevent_csv(self, output_path: str, fps: float = 30.0, event_duration_s: float = 3.0) -> None:
        """
        Create RugbyEvents.csv in notebook format: Event, Frame_Start, Frame_End, Team
        
        Maps your auto-tagged events to notebook event classes.
        Adds synthetic 'Try' events near tryline for training.
        
        Args:
            output_path: Where to save event CSV
            fps: Frames per second
            event_duration_s: Duration of each event in seconds (default 3s)
        """
        print(f"\n📝 Creating rugby events CSV: {output_path}")
        
        event_duration_frames = int(event_duration_s * fps)
        events = []
        
        # 1. Map existing auto-tagged events
        for _, row in self.events_df.iterrows():
            event_type = row['Event']
            frame_start = int(row['frameIdx'])
            team_code = row.get('teamIn', 'L')  # L or R from your pipeline
            
            # Map team code to team number (1 or 2)
            team = 1 if team_code == 'L' else 2
            
            # Map to notebook event class
            mapped_event = self.EVENT_MAPPING.get(event_type)
            
            if mapped_event:
                events.append({
                    'Event': mapped_event,
                    'Frame_Start': frame_start,
                    'Frame_End': frame_start + event_duration_frames,
                    'Team': team
                })
        
        print(f"   ✓ Mapped {len(events)} auto-tagged events")
        
        # 2. Add synthetic Try events (near tryline + low ball speed)
        try_events = self._generate_try_events(fps, event_duration_frames)
        events.extend(try_events)
        print(f"   ✓ Added {len(try_events)} synthetic Try events")
        
        # 3. Add OpenPlay background samples (every 5 seconds where no event)
        openplay_events = self._generate_openplay_events(fps, events, event_duration_frames)
        events.extend(openplay_events)
        print(f"   ✓ Added {len(openplay_events)} OpenPlay samples")
        
        # Convert to DataFrame and sort
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values('Frame_Start').drop_duplicates(subset=['Frame_Start'])
        
        # Save
        events_df.to_csv(output_path, index=False)
        
        print(f"✅ Saved {len(events_df)} events to {output_path}")
        print(f"   Event distribution:")
        for event_type, count in events_df['Event'].value_counts().items():
            print(f"      {event_type}: {count}")
    
    def _generate_try_events(self, fps: float, event_duration_frames: int) -> list:
        """
        Generate synthetic Try events based on ball position near tryline
        
        Try criteria:
        - Ball in tryzone (last 5m of pitch length)
        - Ball speed very low (< 0.5 m/s)
        - Possession stable
        
        Returns:
            List of Try event dicts with Event, Frame_Start, Frame_End, Team
        """
        # Get ball tracking
        ball = self.xy_df[self.xy_df['object_type'] == 'ball'].copy()
        ball = ball.sort_values('frame')
        
        # Compute ball speed
        ball['dx'] = ball['x'].diff().fillna(0)
        ball['dy'] = ball['y'].diff().fillna(0)
        ball['speed'] = np.sqrt(ball['dx']**2 + ball['dy']**2) * fps  # m/s
        
        # Identify tryzone (assuming pitch length ~100m, tryzones are x<5 or x>95)
        in_tryzone = (ball['x'] < 5) | (ball['x'] > 95)
        ball_slow = ball['speed'] < 0.5
        
        try_candidates = ball[in_tryzone & ball_slow].copy()
        
        # Determine which team scored based on tryzone location
        try_candidates['try_team'] = try_candidates['x'].apply(lambda x: 1 if x < 5 else 2)
        
        # Sample Try events (avoid clustering)
        try_events = []
        min_gap_frames = int(10 * fps)  # 10 seconds between tries
        
        last_try_frame = -999999
        for _, row in try_candidates.iterrows():
            frame_start = int(row['frame'])
            if frame_start - last_try_frame > min_gap_frames:
                try_events.append({
                    'Event': 'Try',
                    'Frame_Start': frame_start,
                    'Frame_End': frame_start + event_duration_frames,
                    'Team': int(row['try_team'])
                })
                last_try_frame = frame_start
        
        return try_events
    
    def _generate_openplay_events(self, fps: float, existing_events: list, event_duration_frames: int) -> list:
        """
        Generate OpenPlay (background) samples
        
        Sample every 5 seconds where no other event exists within 10 seconds
        
        Args:
            fps: Frames per second
            existing_events: List of existing event dicts
            event_duration_frames: Duration of each event in frames
            
        Returns:
            List of OpenPlay event dicts with Event, Frame_Start, Frame_End, Team
        """
        # Get frame range
        max_frame = self.xy_df['frame'].max()
        
        # Sample frames every 5 seconds
        sample_interval = int(5 * fps)
        candidate_frames = list(range(0, max_frame, sample_interval))
        
        # Existing event frames (use Frame_Start)
        event_frames = {e['Frame_Start'] for e in existing_events}
        
        # Filter: only keep frames far from existing events
        min_distance = int(10 * fps)  # 10 seconds
        
        openplay_events = []
        for frame_start in candidate_frames:
            # Check distance to nearest event
            nearest = min([abs(frame_start - ef) for ef in event_frames], default=9999)
            if nearest > min_distance:
                openplay_events.append({
                    'Event': 'OpenPlay',
                    'Frame_Start': frame_start,
                    'Frame_End': frame_start + event_duration_frames,
                    'Team': 0  # No specific team for OpenPlay
                })
        
        return openplay_events
    
    def prepare_all(self, output_prefix: str, fps: float = 30.0) -> None:
        """
        Create both game.csv and RugbyEvents.csv
        
        Args:
            output_prefix: Prefix for output files (e.g., 'game1' or 'game2')
            fps: Video frames per second
        """
        print(f"\n{'='*60}")
        print(f"🎯 Preparing XGBoost training data: {output_prefix}")
        print(f"{'='*60}")
        
        game_path = f"{output_prefix}.csv"
        
        # Extract game number from prefix (e.g., 'game1' -> '1')
        import re
        match = re.search(r'(\d+)', output_prefix)
        game_num = match.group(1) if match else '1'
        events_path = f"RugbyEvents{game_num}.csv"
        
        self.create_game_csv(game_path)
        self.create_rugbyevent_csv(events_path, fps)
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETE! Training files ready:")
        print(f"   📄 {game_path}")
        print(f"   📄 {events_path}")
        print(f"{'='*60}")


def split_into_train_eval(
    xy_csv_path: str, 
    events_csv_path: str, 
    train_ratio: float = 0.7,
    fps: float = 30.0
) -> None:
    """
    Split a single game into train (game1) and eval (game2) sets
    
    Args:
        xy_csv_path: Path to full game XY CSV
        events_csv_path: Path to full game events CSV
        train_ratio: Fraction to use for training (default 0.7 = 70%)
        fps: Frames per second
    """
    print("\n🔀 Splitting game into train/eval sets...")
    
    # Load full data
    xy_df = pd.read_csv(xy_csv_path)
    events_df = pd.read_csv(events_csv_path)
    
    # Find split point
    max_frame = xy_df['frame'].max()
    split_frame = int(max_frame * train_ratio)
    
    print(f"   Split at frame {split_frame} ({train_ratio*100:.0f}% train)")
    
    # Split XY data
    xy_train = xy_df[xy_df['frame'] <= split_frame]
    xy_eval = xy_df[xy_df['frame'] > split_frame]
    
    # Adjust eval frames to start from 0
    xy_eval = xy_eval.copy()
    xy_eval['frame'] = xy_eval['frame'] - split_frame - 1
    
    # Split events data
    events_train = events_df[events_df['frameIdx'] <= split_frame]
    events_eval = events_df[events_df['frameIdx'] > split_frame]
    
    events_eval = events_eval.copy()
    events_eval['frameIdx'] = events_eval['frameIdx'] - split_frame - 1
    
    # Save temporary split files
    xy_train_path = '/tmp/train_xy.csv'
    xy_eval_path = '/tmp/eval_xy.csv'
    events_train_path = '/tmp/train_events.csv'
    events_eval_path = '/tmp/eval_events.csv'
    
    xy_train.to_csv(xy_train_path, index=False)
    xy_eval.to_csv(xy_eval_path, index=False)
    events_train.to_csv(events_train_path, index=False)
    events_eval.to_csv(events_eval_path, index=False)
    
    # Create train set
    prep_train = XGBoostDataPreparation(xy_train_path, events_train_path)
    prep_train.prepare_all('game1', fps=fps)
    
    # Create eval set
    prep_eval = XGBoostDataPreparation(xy_eval_path, events_eval_path)
    prep_eval.prepare_all('game2', fps=fps)
    
    print("\n✅ Train/eval split complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare XGBoost training data from pipeline outputs'
    )
    
    parser.add_argument(
        '--input-xy',
        required=True,
        help='Path to *_xy.csv file from pipeline'
    )
    
    parser.add_argument(
        '--input-events',
        required=True,
        help='Path to *_events.csv file from pipeline'
    )
    
    parser.add_argument(
        '--output-prefix',
        default='game1',
        help='Output file prefix (default: game1)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Video frames per second (default: 30.0)'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split into train (game1) and eval (game2) sets'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='If --split, ratio for training set (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    if args.split:
        # Split mode: create both game1 and game2
        split_into_train_eval(
            args.input_xy,
            args.input_events,
            args.train_ratio,
            args.fps
        )
    else:
        # Single output mode
        prep = XGBoostDataPreparation(args.input_xy, args.input_events)
        prep.prepare_all(args.output_prefix, args.fps)


if __name__ == '__main__':
    main()
