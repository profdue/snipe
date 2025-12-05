import numpy as np
from scipy.stats import poisson

class OverUnderPredictor:
    def __init__(self):
        self.over_threshold = 1.5  # Goals per game for Over prediction
        self.under_threshold_defense = 1.0  # Goals against per game for strong defense
        self.under_threshold_attack = 1.5  # Goals per game for weak attack
        
    def predict_over_under(self, home_stats, away_stats, is_home=True):
        """
        Predict Over/Under 2.5 goals with confidence level
        
        Args:
            home_stats: Dictionary of home team statistics
            away_stats: Dictionary of away team statistics
            is_home: Boolean indicating if home_stats is for home team
        
        Returns:
            Dictionary with prediction, confidence, probability, and explanation
        """
        # Get relevant stats based on home/away
        if is_home:
            home_gpg = home_stats['home_gpg']
            home_gapg = home_stats['home_gapg']
            away_gpg = away_stats['away_gpg']
            away_gapg = away_stats['away_gapg']
        else:
            home_gpg = home_stats['away_gpg']
            home_gapg = home_stats['away_gapg']
            away_gpg = away_stats['home_gpg']
            away_gapg = away_stats['home_gapg']
        
        # Use hybrid stats for prediction
        home_attack = home_stats['gpg_hybrid']
        home_defense = home_stats['gapg_hybrid']
        away_attack = away_stats['gpg_hybrid']
        away_defense = away_stats['gapg_hybrid']
        
        # Calculate expected goals using Poisson distribution
        lambda_home = (home_attack * away_defense) / 2.5  # Normalized to league average
        lambda_away = (away_attack * home_defense) / 2.5
        
        # Adjust for home advantage
        if is_home:
            lambda_home *= 1.2
            lambda_away *= 0.9
        
        # Calculate probabilities
        prob_0_goals = poisson.pmf(0, lambda_home + lambda_away)
        prob_1_goal = poisson.pmf(1, lambda_home + lambda_away)
        prob_2_goals = poisson.pmf(2, lambda_home + lambda_away)
        
        prob_under_25 = prob_0_goals + prob_1_goal + prob_2_goals
        prob_over_25 = 1 - prob_under_25
        
        # Apply prediction rules
        prediction, confidence, explanation = self._apply_rules(
            home_stats, away_stats, prob_over_25, prob_under_25
        )
        
        # Expected total goals
        expected_goals = lambda_home + lambda_away
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability': prob_over_25 if prediction == 'Over 2.5' else prob_under_25,
            'expected_goals': expected_goals,
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'explanation': explanation
        }
    
    def _apply_rules(self, home_stats, away_stats, prob_over, prob_under):
        """
        Apply the 5-rule system for predictions
        """
        # Rule 1: High Confidence Over
        if (home_stats['gpg_last10'] > self.over_threshold and 
            away_stats['gpg_last10'] > self.over_threshold and
            home_stats['gpg_hybrid'] > self.over_threshold and
            away_stats['gpg_hybrid'] > self.over_threshold):
            return 'Over 2.5', 'High', (
                "High Confidence Over: Both teams average >1.5 goals per game "
                "in last 10 matches and hybrid metrics"
            )
        
        # Rule 2: High Confidence Under
        if (home_stats['gapg_last10'] < self.under_threshold_defense and 
            away_stats['gpg_last10'] < self.under_threshold_attack and
            home_stats['gapg_hybrid'] < self.under_threshold_defense and
            away_stats['gpg_hybrid'] < self.under_threshold_attack):
            return 'Under 2.5', 'High', (
                "High Confidence Under: Strong defense (<1.0 GA/game) vs "
                "weak attack (<1.5 G/game) in both recent and hybrid metrics"
            )
        
        # Rule 3: Moderate Confidence Over (last 5 only)
        if (home_stats['gpg_hybrid'] > self.over_threshold and 
            away_stats['gpg_hybrid'] > self.over_threshold):
            return 'Over 2.5', 'Moderate', (
                "Moderate Confidence Over: Both teams show strong attacking "
                "form in recent/hybrid metrics (>1.5 goals per game)"
            )
        
        # Rule 4: Moderate Confidence Under (last 5 only)
        if (home_stats['gapg_hybrid'] < self.under_threshold_defense and 
            away_stats['gpg_hybrid'] < self.under_threshold_attack):
            return 'Under 2.5', 'Moderate', (
                "Moderate Confidence Under: Recent defensive strength vs "
                "recent attacking weakness suggests low scoring"
            )
        
        # Rule 5: No clear edge
        if prob_over > 0.55:
            return 'Over 2.5', 'Low', "Slight statistical edge for Over based on Poisson model"
        elif prob_under > 0.55:
            return 'Under 2.5', 'Low', "Slight statistical edge for Under based on Poisson model"
        else:
            return 'No Bet', 'None', "No clear statistical edge - recommend avoiding this market"
