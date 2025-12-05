"""
Phantom Predictor v4.3 - Statistical Prediction Engine
Uses Poisson distribution with xG integration and Bayesian shrinkage
"""

import numpy as np
from scipy.stats import poisson, norm
import pandas as pd

class PhantomPredictor:
    def __init__(self, league_data):
        """
        Initialize predictor with league data
        
        Args:
            league_data: DataFrame with team statistics
        """
        self.league_data = league_data
        self.league_stats = self._calculate_league_statistics()
        
    def _calculate_league_statistics(self):
        """Calculate league-wide statistics for normalization"""
        total_home_matches = len(self.league_data)
        total_away_matches = len(self.league_data)
        
        avg_home_goals = self.league_data['home_goals_for'].sum() / total_home_matches
        avg_away_goals = self.league_data['away_goals_for'].sum() / total_away_matches
        avg_home_conceded = self.league_data['home_goals_against'].sum() / total_home_matches
        avg_away_conceded = self.league_data['away_goals_against'].sum() / total_away_matches
        
        return {
            'avg_home_goals': avg_home_goals,
            'avg_away_goals': avg_away_goals,
            'avg_home_conceded': avg_home_conceded,
            'avg_away_conceded': avg_away_conceded,
            'total_matches': total_home_matches + total_away_matches
        }
    
    def calculate_team_strengths(self, team_name, is_home=True):
        """
        Calculate attack and defense strength for a team
        Uses Bayesian shrinkage to regress towards mean
        """
        team_data = self.league_data[self.league_data['team_name'] == team_name]
        
        if team_data.empty:
            return 1.0, 1.0  # League average
        
        team_data = team_data.iloc[0]
        
        if is_home:
            matches = team_data['matches_played']
            goals_for = team_data['home_goals_for']
            goals_against = team_data['home_goals_against']
            
            # Bayesian shrinkage: weight between team performance and league average
            shrinkage_factor = min(1.0, matches / 10)  # More matches = less shrinkage
            
            attack_strength = (
                (goals_for / matches) / self.league_stats['avg_home_goals'] * shrinkage_factor +
                1.0 * (1 - shrinkage_factor)
            )
            
            defense_strength = (
                (goals_against / matches) / self.league_stats['avg_away_conceded'] * shrinkage_factor +
                1.0 * (1 - shrinkage_factor)
            )
        else:
            matches = team_data['matches_played']
            goals_for = team_data['away_goals_for']
            goals_against = team_data['away_goals_against']
            
            shrinkage_factor = min(1.0, matches / 10)
            
            attack_strength = (
                (goals_for / matches) / self.league_stats['avg_away_goals'] * shrinkage_factor +
                1.0 * (1 - shrinkage_factor)
            )
            
            defense_strength = (
                (goals_against / matches) / self.league_stats['avg_home_conceded'] * shrinkage_factor +
                1.0 * (1 - shrinkage_factor)
            )
        
        return attack_strength, defense_strength
    
    def calculate_expected_goals(self, home_team, away_team, home_xg=None, away_xg=None):
        """
        Calculate expected goals using statistical model with xG integration (60/40)
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_xg: Optional xG value for home team
            away_xg: Optional xG value for away team
            
        Returns:
            Tuple of (home_expected_goals, away_expected_goals)
        """
        # Get team strengths
        home_attack, home_defense = self.calculate_team_strengths(home_team, is_home=True)
        away_attack, away_defense = self.calculate_team_strengths(away_team, is_home=False)
        
        # Statistical model expected goals
        stat_home_goals = home_attack * away_defense * self.league_stats['avg_home_goals']
        stat_away_goals = away_attack * home_defense * self.league_stats['avg_away_goals']
        
        if home_xg is not None and away_xg is not None:
            # Blend statistical model with xG (60% xG, 40% statistical model)
            home_exp_goals = home_xg * 0.6 + stat_home_goals * 0.4
            away_exp_goals = away_xg * 0.6 + stat_away_goals * 0.4
        else:
            # Use statistical model only
            home_exp_goals = stat_home_goals
            away_exp_goals = stat_away_goals
        
        return home_exp_goals, away_exp_goals
    
    def calculate_match_probabilities(self, home_exp_goals, away_exp_goals):
        """
        Calculate match outcome probabilities using Poisson distribution
        
        Returns:
            Dictionary with probabilities for all outcomes
        """
        max_goals = 8  # Reasonable maximum for Poisson calculation
        
        # Initialize probabilities
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        over_15_prob = 0
        over_25_prob = 0
        over_35_prob = 0
        btts_prob = 0
        
        # Calculate all possible scorelines
        for i in range(max_goals):
            for j in range(max_goals):
                prob = poisson.pmf(i, home_exp_goals) * poisson.pmf(j, away_exp_goals)
                
                # Match outcome
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
                
                # Over/under markets
                total_goals = i + j
                if total_goals > 1.5:
                    over_15_prob += prob
                if total_goals > 2.5:
                    over_25_prob += prob
                if total_goals > 3.5:
                    over_35_prob += prob
                
                # Both teams to score
                if i > 0 and j > 0:
                    btts_prob += prob
        
        # Calculate clean sheet probabilities
        home_clean_sheet = poisson.pmf(0, away_exp_goals)
        away_clean_sheet = poisson.pmf(0, home_exp_goals)
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'over_15': over_15_prob,
            'over_25': over_25_prob,
            'over_35': over_35_prob,
            'under_15': 1 - over_15_prob,
            'under_25': 1 - over_25_prob,
            'under_35': 1 - over_35_prob,
            'btts_yes': btts_prob,
            'btts_no': 1 - btts_prob,
            'home_clean_sheet': home_clean_sheet,
            'away_clean_sheet': away_clean_sheet
        }
    
    def get_most_likely_scorelines(self, home_exp_goals, away_exp_goals, n=3):
        """
        Get n most likely scorelines
        
        Returns:
            List of tuples (home_goals, away_goals, probability)
        """
        max_goals = 5
        scorelines = []
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = poisson.pmf(i, home_exp_goals) * poisson.pmf(j, away_exp_goals)
                scorelines.append(((i, j), prob))
        
        # Sort by probability and return top n
        scorelines.sort(key=lambda x: x[1], reverse=True)
        return scorelines[:n]
    
    def predict_match(self, home_team, away_team, home_xg=None, away_xg=None):
        """
        Main prediction function for a match
        
        Returns:
            Dictionary with all predictions and statistics
        """
        # Calculate expected goals
        home_exp_goals, away_exp_goals = self.calculate_expected_goals(
            home_team, away_team, home_xg, away_xg
        )
        
        # Calculate probabilities
        probabilities = self.calculate_match_probabilities(home_exp_goals, away_exp_goals)
        
        # Get most likely scorelines
        likely_scorelines = self.get_most_likely_scorelines(home_exp_goals, away_exp_goals, n=3)
        
        # Calculate form advantage
        home_form_score = self._calculate_form_score(home_team, is_home=True)
        away_form_score = self._calculate_form_score(away_team, is_home=False)
        form_advantage = home_form_score - away_form_score
        
        return {
            'teams': {'home': home_team, 'away': away_team},
            'expected_goals': {'home': home_exp_goals, 'away': away_exp_goals},
            'probabilities': probabilities,
            'likely_scorelines': likely_scorelines,
            'form_scores': {'home': home_form_score, 'away': away_form_score},
            'form_advantage': form_advantage,
            'total_expected_goals': home_exp_goals + away_exp_goals
        }
    
    def _calculate_form_score(self, team_name, is_home=True):
        """Calculate form score from last 5 matches string"""
        team_data = self.league_data[self.league_data['team_name'] == team_name]
        
        if team_data.empty:
            return 0.5
        
        form_string = team_data.iloc[0]['form_last_5']
        
        if not isinstance(form_string, str) or len(form_string) != 5:
            return 0.5
        
        # Convert form to numerical score
        form_map = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        score = 0
        
        for result in form_string:
            score += form_map.get(result, 0.5)
        
        return score / 5
    
    def calculate_confidence_score(self, probability, market_odds):
        """
        Calculate confidence score based on probability vs market odds
        
        Returns:
            Confidence score from 0 to 1
        """
        if market_odds <= 1.0:
            return 0.0
        
        implied_probability = 1 / market_odds
        edge = probability - implied_probability
        
        if edge <= 0:
            return 0.0
        
        # Confidence increases with edge size, capped at reasonable level
        confidence = min(edge * 10, 0.9)  # Scale edge to confidence
        
        return confidence