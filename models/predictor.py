import numpy as np
import math

class OverUnderPredictor:
    def __init__(self):
        self.over_threshold = 1.5
        self.under_threshold_defense = 1.0
        self.under_threshold_attack = 1.5
        
    def poisson_pmf(self, k, lambd):
        """Calculate Poisson probability mass function without scipy"""
        return (lambd ** k * math.exp(-lambd)) / math.factorial(k)
    
    def predict_over_under(self, home_stats, away_stats, is_home=True):
        """
        Predict Over/Under 2.5 goals with enhanced last 5/last 10 logic
        
        Args:
            home_stats: Dictionary of home team statistics
            away_stats: Dictionary of away team statistics
            is_home: Boolean indicating if home_stats is for home team
        
        Returns:
            Dictionary with prediction, confidence, probability, and explanation
        """
        # Extract all relevant stats
        stats = self._extract_stats_for_prediction(home_stats, away_stats, is_home)
        
        # Apply the 5-rule system
        prediction, confidence, rule_number = self._apply_five_rules(stats)
        
        # Calculate Poisson probabilities for expected goals
        expected_goals, prob_over, prob_under = self._calculate_poisson_probabilities(stats)
        
        # If no clear rule matches but Poisson suggests edge
        if prediction == "No Bet" and max(prob_over, prob_under) > 0.55:
            if prob_over > prob_under:
                prediction = "Over 2.5"
                confidence = "Low"
                rule_number = 5
            else:
                prediction = "Under 2.5"
                confidence = "Low"
                rule_number = 5
        
        explanation = self._generate_explanation(prediction, confidence, rule_number, stats)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability': prob_over if prediction == "Over 2.5" else prob_under,
            'expected_goals': expected_goals,
            'rule_number': rule_number,
            'explanation': explanation,
            'stats_used': stats
        }
    
    def _extract_stats_for_prediction(self, home_stats, away_stats, is_home):
        """Extract and organize all relevant statistics for prediction"""
        stats = {}
        
        # For home team (playing at home)
        stats['home_last5_gpg'] = home_stats.get('last5_home_gpg', home_stats.get('home_gpg', 0))
        stats['home_last5_gapg'] = home_stats.get('last5_home_gapg', home_stats.get('home_gapg', 0))
        stats['home_last10_gpg'] = home_stats.get('last10_home_gpg', stats['home_last5_gpg'])
        stats['home_last10_gapg'] = home_stats.get('last10_home_gapg', stats['home_last5_gapg'])
        
        # For away team (playing away)
        stats['away_last5_gpg'] = away_stats.get('last5_away_gpg', away_stats.get('away_gpg', 0))
        stats['away_last5_gapg'] = away_stats.get('last5_away_gapg', away_stats.get('away_gapg', 0))
        stats['away_last10_gpg'] = away_stats.get('last10_away_gpg', stats['away_last5_gpg'])
        stats['away_last10_gapg'] = away_stats.get('last10_away_gapg', stats['away_last5_gapg'])
        
        # Hybrid metrics (60% actual, 40% xG) - using overall stats
        home_gpg_last10 = home_stats.get('gpg_last10', 0)
        away_gpg_last10 = away_stats.get('gpg_last10', 0)
        home_gapg_last10 = home_stats.get('gapg_last10', 0)
        away_gapg_last10 = away_stats.get('gapg_last10', 0)
        
        stats['home_hybrid_gpg'] = 0.6 * home_gpg_last10 + 0.4 * home_stats.get('avg_xg_for', home_gpg_last10)
        stats['home_hybrid_gapg'] = 0.6 * home_gapg_last10 + 0.4 * home_stats.get('avg_xg_against', home_gapg_last10)
        stats['away_hybrid_gpg'] = 0.6 * away_gpg_last10 + 0.4 * away_stats.get('avg_xg_for', away_gpg_last10)
        stats['away_hybrid_gapg'] = 0.6 * away_gapg_last10 + 0.4 * away_stats.get('avg_xg_against', away_gapg_last10)
        
        return stats
    
    def _apply_five_rules(self, stats):
        """
        Apply the 5-rule prediction system:
        1. High Confidence Over: Both teams >1.5 GPG (Last 10 & 5)
        2. High Confidence Under: Defense <1.0 GA PG vs Attack <1.5 GPG (Last 10 & 5)
        3. Moderate Confidence Over: Both teams >1.5 GPG (Last 5 only)
        4. Moderate Confidence Under: Defense <1.0 GA PG vs Attack <1.5 GPG (Last 5 only)
        5. No clear statistical edge
        """
        
        # Rule 1: High Confidence Over
        home_meets_last10 = stats['home_last10_gpg'] > self.over_threshold
        away_meets_last10 = stats['away_last10_gpg'] > self.over_threshold
        home_meets_last5 = stats['home_last5_gpg'] > self.over_threshold
        away_meets_last5 = stats['away_last5_gpg'] > self.over_threshold
        
        if home_meets_last10 and away_meets_last10 and home_meets_last5 and away_meets_last5:
            return "Over 2.5", "High", 1
        
        # Rule 2: High Confidence Under
        home_def_last10 = stats['home_last10_gapg'] < self.under_threshold_defense
        away_att_last10 = stats['away_last10_gpg'] < self.under_threshold_attack
        home_def_last5 = stats['home_last5_gapg'] < self.under_threshold_defense
        away_att_last5 = stats['away_last5_gpg'] < self.under_threshold_attack
        
        if home_def_last10 and away_att_last10 and home_def_last5 and away_att_last5:
            return "Under 2.5", "High", 2
        
        # Rule 3: Moderate Confidence Over (Last 5 only)
        if home_meets_last5 and away_meets_last5:
            return "Over 2.5", "Moderate", 3
        
        # Rule 4: Moderate Confidence Under (Last 5 only)
        if home_def_last5 and away_att_last5:
            return "Under 2.5", "Moderate", 4
        
        # Rule 5: No clear edge
        return "No Bet", "None", 5
    
    def _calculate_poisson_probabilities(self, stats):
        """Calculate expected goals and probabilities using Poisson distribution"""
        
        # Use weighted average: 40% last10, 40% last5, 20% hybrid
        home_attack = (0.4 * stats['home_last10_gpg'] + 
                      0.4 * stats['home_last5_gpg'] + 
                      0.2 * stats['home_hybrid_gpg'])
        
        away_defense = (0.4 * stats['away_last10_gapg'] + 
                      0.4 * stats['away_last5_gapg'] + 
                      0.2 * stats['away_hybrid_gapg'])
        
        away_attack = (0.4 * stats['away_last10_gpg'] + 
                     0.4 * stats['away_last5_gpg'] + 
                     0.2 * stats['away_hybrid_gpg'])
        
        home_defense = (0.4 * stats['home_last10_gapg'] + 
                      0.4 * stats['home_last5_gapg'] + 
                      0.2 * stats['home_hybrid_gapg'])
        
        # Calculate expected goals with home advantage
        lambda_home = home_attack * (away_defense / 1.5) * 1.2  # Normalize to league average
        lambda_away = away_attack * (home_defense / 1.5) * 0.9
        
        expected_goals = lambda_home + lambda_away
        
        # Calculate Poisson probabilities
        total_lambda = lambda_home + lambda_away
        
        # Probability of 0, 1, or 2 goals (Under 2.5)
        prob_0 = self.poisson_pmf(0, total_lambda)
        prob_1 = self.poisson_pmf(1, total_lambda)
        prob_2 = self.poisson_pmf(2, total_lambda)
        prob_under = prob_0 + prob_1 + prob_2
        
        # Probability of Over 2.5 goals
        prob_over = 1 - prob_under
        
        return expected_goals, prob_over, prob_under
    
    def _generate_explanation(self, prediction, confidence, rule_number, stats):
        """Generate detailed explanation for the prediction"""
        
        explanations = {
            1: f"High Confidence Over: Both teams average >1.5 goals per game in their last 10 AND last 5 matches. "
                f"Home team: {stats['home_last10_gpg']:.2f} GPG (last10), {stats['home_last5_gpg']:.2f} GPG (last5). "
                f"Away team: {stats['away_last10_gpg']:.2f} GPG (last10), {stats['away_last5_gpg']:.2f} GPG (last5).",
            
            2: f"High Confidence Under: Strong defense (<1.0 GA/game) vs weak attack (<1.5 G/game) in both recent periods. "
                f"Home defense: {stats['home_last10_gapg']:.2f} GApg (last10), {stats['home_last5_gapg']:.2f} GApg (last5). "
                f"Away attack: {stats['away_last10_gpg']:.2f} GPG (last10), {stats['away_last5_gpg']:.2f} GPG (last5).",
            
            3: f"Moderate Confidence Over: Both teams show strong attacking form in last 5 matches (>1.5 goals per game). "
                f"Home: {stats['home_last5_gpg']:.2f} GPG (last5). Away: {stats['away_last5_gpg']:.2f} GPG (last5).",
            
            4: f"Moderate Confidence Under: Recent defensive strength vs recent attacking weakness suggests low scoring. "
                f"Home defense (last5): {stats['home_last5_gapg']:.2f} GApg. Away attack (last5): {stats['away_last5_gpg']:.2f} GPG.",
            
            5: f"Low Confidence {prediction}: Slight statistical edge based on Poisson model. Expected goals: {(stats['home_last5_gpg'] + stats['away_last5_gpg'])/2:.2f}. "
                f"No strong rule match but model suggests {prediction.lower()}."
        }
        
        if rule_number == 5 and prediction == "No Bet":
            return f"No clear statistical edge. Teams don't meet any established criteria. " \
                   f"Home: {stats['home_last5_gpg']:.2f} GPG, {stats['home_last5_gapg']:.2f} GApg. " \
                   f"Away: {stats['away_last5_gpg']:.2f} GPG, {stats['away_last5_gapg']:.2f} GApg."
        
        return explanations.get(rule_number, f"{confidence} confidence {prediction} based on statistical analysis.")
