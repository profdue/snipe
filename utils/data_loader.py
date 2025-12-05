import json
import pandas as pd
import os

def load_teams_data(filepath='data/teams.json'):
    """Load team statistics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_matches_data(filepath='data/matches.csv'):
    """Load upcoming matches from CSV"""
    return pd.read_csv(filepath)

def calculate_team_stats_from_tables():
    """Helper function to calculate GPG/GA PG from your tables"""
    # You can use this to populate teams.json
    pass
