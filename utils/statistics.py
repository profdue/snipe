"""
Data loading utilities for league and match data
"""

import pandas as pd
import os
import json

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
    def get_available_leagues(self):
        """Get list of available leagues"""
        leagues_path = os.path.join(self.data_dir, 'leagues')
        
        if not os.path.exists(leagues_path):
            return []
        
        leagues = []
        for item in os.listdir(leagues_path):
            item_path = os.path.join(leagues_path, item)
            if os.path.isdir(item_path):
                leagues.append(item)
        
        return sorted(leagues)
    
    def load_league_teams(self, league_name):
        """Load team data for a league"""
        league_path = os.path.join(self.data_dir, 'leagues', league_name)
        teams_file = os.path.join(league_path, 'teams.csv')
        
        if not os.path.exists(teams_file):
            raise FileNotFoundError(f"Teams file not found: {teams_file}")
        
        df = pd.read_csv(teams_file)
        return df
    
    def load_upcoming_matches(self, league_name):
        """Load upcoming matches for a league"""
        league_path = os.path.join(self.data_dir, 'leagues', league_name)
        matches_file = os.path.join(league_path, 'upcoming.csv')
        
        if not os.path.exists(matches_file):
            return pd.DataFrame()
        
        df = pd.read_csv(matches_file)
        return df
    
    def load_historical_matches(self, league_name):
        """Load historical matches for a league"""
        league_path = os.path.join(self.data_dir, 'leagues', league_name)
        matches_file = os.path.join(league_path, 'matches.csv')
        
        if not os.path.exists(matches_file):
            return pd.DataFrame()
        
        df = pd.read_csv(matches_file)
        return df
    
    def get_team_details(self, league_name, team_name):
        """Get detailed statistics for a specific team"""
        teams_df = self.load_league_teams(league_name)
        team_data = teams_df[teams_df['team_name'] == team_name]
        
        if team_data.empty:
            return None
        
        team_stats = team_data.iloc[0].to_dict()
        
        # Calculate derived statistics
        home_matches = team_stats['home_wins'] + team_stats['home_draws'] + team_stats['home_losses']
        away_matches = team_stats['away_wins'] + team_stats['away_draws'] + team_stats['away_losses']
        
        if home_matches > 0:
            team_stats['home_gpg'] = team_stats['home_goals_for'] / home_matches
            team_stats['home_gapg'] = team_stats['home_goals_against'] / home_matches
            team_stats['home_win_rate'] = team_stats['home_wins'] / home_matches
        else:
            team_stats['home_gpg'] = 0
            team_stats['home_gapg'] = 0
            team_stats['home_win_rate'] = 0
        
        if away_matches > 0:
            team_stats['away_gpg'] = team_stats['away_goals_for'] / away_matches
            team_stats['away_gapg'] = team_stats['away_goals_against'] / away_matches
            team_stats['away_win_rate'] = team_stats['away_wins'] / away_matches
        else:
            team_stats['away_gpg'] = 0
            team_stats['away_gapg'] = 0
            team_stats['away_win_rate'] = 0
        
        # Overall statistics
        total_matches = team_stats['matches_played']
        if total_matches > 0:
            team_stats['overall_gpg'] = (team_stats['home_goals_for'] + team_stats['away_goals_for']) / total_matches
            team_stats['overall_gapg'] = (team_stats['home_goals_against'] + team_stats['away_goals_against']) / total_matches
        
        return team_stats