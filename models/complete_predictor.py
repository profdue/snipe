import numpy as np
import math
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

# Import your existing KellyCriterion
try:
    from models.staking import KellyCriterion
    USE_CUSTOM_STAKING = True
except ImportError:
    USE_CUSTOM_STAKING = False

@dataclass
class TeamStats:
    """Container for team statistics from CSV"""
    team_name: str
    matches_played: int
    home_wins: int
    home_draws: int
    home_losses: int
    home_goals_for: int
    home_goals_against: int
    away_wins: int
    away_draws: int
    away_losses: int
    away_goals_for: int
    away_goals_against: int
    home_xg: Optional[float]
    away_xg: Optional[float]
    avg_xg_for: float
    avg_xg_against: float
    form_last_5: str
    attack_strength: float
    defense_strength: float
    last5_home_gpg: float
    last5_home_gapg: float
    last5_away_gpg: float
    last5_away_gapg: float
    last10_home_gpg: float
    last10_home_gapg: float
    last10_away_gpg: float
    last10_away_gapg: float

class CompletePhantomPredictor:
    """
    Complete v5.3 Football Predictor with:
    - Mathematically correct Poisson probabilities
    - Rule-based system with validation
    - Proper edge calculations
    - Fixed staking for NO BET scenarios
    """
    
    def __init__(self, bankroll: float = 1000.0, min_confidence: float = 0.60):
        self.bankroll = bankroll
        self.min_confidence = min_confidence
        
        # Initialize Kelly Criterion
        if USE_CUSTOM_STAKING:
            self.kelly = KellyCriterion(fraction=0.5)
        else:
            self.kelly = self._create_builtin_kelly()
        
        # League context
        self.league_context = {
            'premier_league': {'avg_gpg': 2.8, 'avg_gapg': 2.8, 'home_advantage': 0.15},
            'la_liga': {'avg_gpg': 2.6, 'avg_gapg': 2.6, 'home_advantage': 0.18},
            'bundesliga': {'avg_gpg': 3.1, 'avg_gapg': 3.1, 'home_advantage': 0.12},
            'serie_a': {'avg_gpg': 2.5, 'avg_gapg': 2.5, 'home_advantage': 0.20},
            'ligue_1': {'avg_gpg': 2.7, 'avg_gapg': 2.7, 'home_advantage': 0.16},
            'default': {'avg_gpg': 2.7, 'avg_gapg': 2.7, 'home_advantage': 0.16}
        }
        
        # Rule thresholds
        self.over_threshold = 1.5
        self.under_threshold_defense = 1.0
        self.under_threshold_attack = 1.5
        
        # Minimum expected goals for Over bet
        self.min_over_goals = 2.6
        self.max_under_goals = 2.4
        
        # Betting parameters
        self.max_stake_pct = 0.05
        
        # Explanation templates
        self._init_explanation_templates()
    
    def _create_builtin_kelly(self):
        """Create built-in Kelly calculator if custom one not available"""
        class BuiltInKelly:
            def __init__(self, fraction=0.5):
                self.fraction = fraction
            
            def calculate_stake(self, probability, odds, bankroll, max_percent=0.05, min_probability=0.0):
                q = 1 - probability
                b = odds - 1
                
                if b <= 0 or probability <= 0:
                    kelly_fraction = 0
                else:
                    kelly_fraction = (b * p - q) / b
                
                # Apply fractional Kelly
                kelly_fraction *= self.fraction
                kelly_fraction = max(0, kelly_fraction)
                
                # Calculate stake
                stake_amount = bankroll * kelly_fraction
                
                # Apply maximum stake limit
                max_stake = bankroll * max_percent
                stake_amount = min(stake_amount, max_stake)
                
                # Calculate expected value
                expected_value = (probability * (stake_amount * (odds - 1))) - (q * stake_amount)
                
                # Determine risk level
                if kelly_fraction > 0.1:
                    risk_level = "High"
                elif kelly_fraction > 0.05:
                    risk_level = "Medium"
                elif kelly_fraction > 0:
                    risk_level = "Low"
                else:
                    risk_level = "No Bet"
                
                # Calculate edge
                implied_prob = 1 / odds if odds > 0 else 0
                edge = probability - implied_prob
                
                # Determine value rating
                if edge > 0.1:
                    value_rating = "Excellent"
                elif edge > 0.05:
                    value_rating = "Good"
                elif edge > 0.02:
                    value_rating = "Fair"
                else:
                    value_rating = "Poor"
                
                return {
                    'stake_amount': stake_amount,
                    'stake_percent': stake_amount / bankroll,
                    'kelly_fraction': kelly_fraction,
                    'expected_value': expected_value,
                    'risk_level': risk_level,
                    'edge_percent': edge * 100,
                    'value_rating': value_rating,
                    'implied_probability': implied_prob,
                    'true_probability': probability,
                    'max_stake_limit': max_stake
                }
        
        return BuiltInKelly(fraction=0.5)
    
    def _init_explanation_templates(self):
        """Initialize explanation templates"""
        self.explanation_templates = {
            1: "High Confidence {prediction}: Rule 1 triggered. Both teams >1.5 GPG in last 5 & 10. Expected goals: {total_goals:.2f}. Poisson P({prediction}): {poisson_prob:.1%}. Edge: {edge:.1f}%.",
            2: "High Confidence {prediction}: Rule 2 triggered. Strong defense vs weak attack. Expected goals: {total_goals:.2f}. Poisson P({prediction}): {poisson_prob:.1%}. Edge: {edge:.1f}%.",
            3: "Moderate Confidence {prediction}: Rule 3 triggered. Last 5 attacking form. Expected goals: {total_goals:.2f}. Poisson P({prediction}): {poisson_prob:.1%}. Edge: {edge:.1f}%.",
            4: "Moderate Confidence {prediction}: Rule 4 triggered. Last 5 defensive strength. Expected goals: {total_goals:.2f}. Poisson P({prediction}): {poisson_prob:.1%}. Edge: {edge:.1f}%.",
            5: "Low Confidence {prediction}: Rule 5 triggered. xG-based edge. Expected goals: {total_goals:.2f}. Poisson P({prediction}): {poisson_prob:.1%}. Edge: {edge:.1f}%.",
            'no_bet': "No Bet: Rule {rule} triggered but expected goals ({total_goals:.2f}) don't support {prediction}. Poisson P({prediction}): {poisson_prob:.1%}. Market edge: {edge:.1f}%.",
            'no_edge': "No Bet: Rule {rule} triggered but no market edge. Model: {model_prob:.1%} vs Market: {market_prob:.1%}. Edge: {edge:.1f}%."
        }
    
    def poisson_pmf(self, k: int, lambd: float) -> float:
        """Calculate Poisson probability mass function"""
        return (lambd ** k * math.exp(-lambd)) / math.factorial(k)
    
    def calculate_poisson_probability(self, expected_goals: float, prediction: str) -> float:
        """Calculate correct Poisson probability for Over/Under 2.5"""
        # Calculate probability of 0, 1, or 2 goals
        prob_under = sum(self.poisson_pmf(k, expected_goals) for k in range(3))
        prob_over = 1 - prob_under
        
        return prob_over if prediction == "Over 2.5" else prob_under
    
    def _calculate_form_momentum(self, last5_value: float, last10_value: float) -> Tuple[str, float]:
        """Detect if team is improving or declining"""
        if last10_value == 0:
            return "stable", 1.0
            
        ratio = last5_value / last10_value
        
        if ratio > 1.15:
            return "improving", 1.1
        elif ratio < 0.85:
            return "declining", 0.9
        return "stable", 1.0
    
    def _prepare_stats_for_prediction(self, home_stats: dict, away_stats: dict, 
                                    league: str = "default") -> Dict:
        """Prepare statistics for prediction"""
        context = self.league_context.get(league, self.league_context['default'])
        
        # Get raw stats
        home_last10 = home_stats.get('last10_home_gpg', home_stats.get('last10_gpg', 1.5))
        home_last5 = home_stats.get('last5_home_gpg', home_stats.get('last5_gpg', 1.5))
        home_last5_gapg = home_stats.get('last5_home_gapg', home_stats.get('gapg_last10', 1.5))
        
        away_last10 = away_stats.get('last10_away_gpg', away_stats.get('last10_gpg', 1.5))
        away_last5 = away_stats.get('last5_away_gpg', away_stats.get('last5_gpg', 1.5))
        away_last5_gapg = away_stats.get('last5_away_gapg', away_stats.get('gapg_last10', 1.5))
        
        # Calculate momentum
        home_momentum, home_momentum_mult = self._calculate_form_momentum(home_last5, home_last10)
        away_momentum, away_momentum_mult = self._calculate_form_momentum(away_last5, away_last10)
        
        # Apply momentum to create final estimates
        home_attack = home_last5 * home_momentum_mult
        away_attack = away_last5 * away_momentum_mult
        
        # For defense, inverse momentum if improving (better defense = lower GApg)
        if home_momentum == "improving":
            home_defense_mult = 0.9
        elif home_momentum == "declining":
            home_defense_mult = 1.1
        else:
            home_defense_mult = 1.0
            
        if away_momentum == "improving":
            away_defense_mult = 0.9
        elif away_momentum == "declining":
            away_defense_mult = 1.1
        else:
            away_defense_mult = 1.0
            
        home_defense = home_stats.get('last10_home_gapg', home_stats.get('gapg_last10', 1.5)) * home_defense_mult
        away_defense = away_stats.get('last10_away_gapg', away_stats.get('gapg_last10', 1.5)) * away_defense_mult
        
        # Calculate expected goals
        lambda_home = home_attack * (away_defense / context['avg_gapg']) * (1 + context['home_advantage'])
        lambda_away = away_attack * (home_defense / context['avg_gapg'])
        expected_goals = lambda_home + lambda_away
        
        return {
            'home_attack': home_attack,
            'away_attack': away_attack,
            'home_defense': home_defense,
            'away_defense': away_defense,
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'home_last5': home_last5,
            'home_last10': home_last10,
            'away_last5': away_last5,
            'away_last10': away_last10,
            'home_last5_gapg': home_last5_gapg,
            'away_last5_gapg': away_last5_gapg,
            'expected_goals': expected_goals,
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'league_context': context
        }
    
    def _apply_rules(self, stats: Dict) -> Tuple[str, str, int]:
        """
        Apply the 5-rule system
        Returns: (prediction, confidence, rule_number)
        """
        # Rule 1: High Confidence Over
        if (stats['home_last10'] > self.over_threshold and stats['away_last10'] > self.over_threshold and
            stats['home_last5'] > self.over_threshold and stats['away_last5'] > self.over_threshold):
            return "Over 2.5", "High", 1
        
        # Rule 2: High Confidence Under
        if (stats['home_defense'] < self.under_threshold_defense and 
            stats['away_attack'] < self.under_threshold_attack and
            stats['home_last5_gapg'] < self.under_threshold_defense and 
            stats['away_last5'] < self.under_threshold_attack):
            return "Under 2.5", "High", 2
        
        # Rule 3: Moderate Confidence Over
        if stats['home_last5'] > self.over_threshold and stats['away_last5'] > self.over_threshold:
            return "Over 2.5", "Moderate", 3
        
        # Rule 4: Moderate Confidence Under
        if stats['home_last5_gapg'] < self.under_threshold_defense and stats['away_last5'] < self.under_threshold_attack:
            return "Under 2.5", "Moderate", 4
        
        # Rule 5: No clear rule matches
        return "No Bet", "None", 5
    
    def _validate_prediction(self, prediction: str, expected_goals: float, 
                           poisson_prob: float, market_odds: float) -> Tuple[str, str, float]:
        """
        Validate if prediction makes mathematical sense
        Returns: (final_prediction, confidence, probability)
        """
        # Check if expected goals support the prediction
        if prediction == "Over 2.5" and expected_goals < self.min_over_goals:
            return "No Bet", "None", poisson_prob
        
        if prediction == "Under 2.5" and expected_goals > self.max_under_goals:
            return "No Bet", "None", poisson_prob
        
        # Use Poisson probability as the true probability
        true_probability = poisson_prob
        
        # Check for edge
        implied_prob = 1 / market_odds if market_odds > 0 else 0
        edge = true_probability - implied_prob
        
        # If no edge, return No Bet
        if edge <= 0:
            return "No Bet", "None", true_probability
        
        return prediction, "High" if edge > 0.05 else "Moderate", true_probability
    
    def predict_with_staking(self, home_stats: dict, away_stats: dict,
                           market_odds: dict, league: str = "default", 
                           bankroll: float = None) -> Dict:
        """
        Main prediction method with mathematically correct probabilities
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        # Extract odds
        over_odds = market_odds.get('over_25', 1.85)
        under_odds = market_odds.get('under_25', 1.95)
        
        # Prepare stats
        stats = self._prepare_stats_for_prediction(home_stats, away_stats, league)
        expected_goals = stats['expected_goals']
        
        # Apply rules
        prediction, confidence, rule_number = self._apply_rules(stats)
        
        # Calculate Poisson probability
        poisson_prob = self.calculate_poisson_probability(expected_goals, prediction)
        
        # Determine market odds for this prediction
        market_odd = over_odds if prediction == "Over 2.5" else under_odds
        
        # Validate prediction
        final_prediction, final_confidence, final_probability = self._validate_prediction(
            prediction, expected_goals, poisson_prob, market_odd
        )
        
        # Calculate stake using Kelly calculator (works for both BET and NO BET)
        staking_result = self.kelly.calculate_stake(
            probability=final_probability,
            odds=market_odd,
            bankroll=bankroll,
            min_probability=self.min_confidence
        )
        
        # Convert staking result to dictionary
        if hasattr(staking_result, 'stake_amount'):  # If using custom Kelly class
            staking_info = {
                'stake_amount': staking_result.stake_amount,
                'stake_percent': staking_result.stake_percent,
                'kelly_fraction': staking_result.kelly_fraction,
                'expected_value': staking_result.expected_value,
                'risk_level': staking_result.risk_level,
                'edge_percent': staking_result.edge_percent,
                'value_rating': staking_result.value_rating,
                'implied_probability': staking_result.implied_probability,
                'true_probability': staking_result.true_probability
            }
        else:  # If using built-in dictionary
            staking_info = staking_result
        
        # If No Bet, create explanation
        if final_prediction == "No Bet":
            # Determine why no bet
            implied_prob = 1 / market_odd if market_odd > 0 else 0
            edge = final_probability - implied_prob
            
            if rule_number == 5:  # No rule matched
                explanation = f"No rule triggered. Expected goals: {expected_goals:.2f}. Poisson P(Over): {self.calculate_poisson_probability(expected_goals, 'Over 2.5'):.1%}."
            elif edge <= 0:  # No edge
                explanation = self.explanation_templates['no_edge'].format(
                    rule=rule_number,
                    model_prob=final_probability,
                    market_prob=implied_prob,
                    edge=edge*100
                )
            else:  # Expected goals don't support
                explanation = self.explanation_templates['no_bet'].format(
                    rule=rule_number,
                    total_goals=expected_goals,
                    prediction=prediction,
                    poisson_prob=poisson_prob,
                    edge=edge*100
                )
            
            return {
                'prediction': 'NO BET',
                'confidence': 'None',
                'probability': final_probability,
                'expected_goals': expected_goals,
                'rule_number': rule_number,
                'explanation': explanation,
                'staking_info': staking_info,  # This now has ALL required keys
                'market_odds': market_odd,
                'poisson_details': {
                    'lambda_home': stats['lambda_home'],
                    'lambda_away': stats['lambda_away'],
                    'expected_home_goals': stats['lambda_home'],
                    'expected_away_goals': stats['lambda_away'],
                    'home_momentum': stats['home_momentum'],
                    'away_momentum': stats['away_momentum']
                },
                'stats_analysis': {
                    'home_attack': stats['home_attack'],
                    'home_defense': stats['home_defense'],
                    'away_attack': stats['away_attack'],
                    'away_defense': stats['away_defense'],
                    'home_momentum': stats['home_momentum'],
                    'away_momentum': stats['away_momentum']
                }
            }
        
        # If it's a bet, create bet explanation
        explanation = self.explanation_templates[rule_number].format(
            prediction=final_prediction,
            total_goals=expected_goals,
            poisson_prob=final_probability,
            edge=staking_info['edge_percent']
        )
        
        # Return complete prediction for betting scenario
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'probability': final_probability,
            'expected_goals': expected_goals,
            'rule_number': rule_number,
            'explanation': explanation,
            'staking_info': staking_info,
            'market_odds': market_odd,
            'poisson_details': {
                'lambda_home': stats['lambda_home'],
                'lambda_away': stats['lambda_away'],
                'expected_home_goals': stats['lambda_home'],
                'expected_away_goals': stats['lambda_away'],
                'home_momentum': stats['home_momentum'],
                'away_momentum': stats['away_momentum']
            },
            'stats_analysis': {
                'home_attack': stats['home_attack'],
                'home_defense': stats['home_defense'],
                'away_attack': stats['away_attack'],
                'away_defense': stats['away_defense'],
                'home_momentum': stats['home_momentum'],
                'away_momentum': stats['away_momentum']
            }
        }
    
    # Keep compatibility method
    def predict_match(self, home_stats: dict, away_stats: dict,
                     over_odds: float, under_odds: float, 
                     league: str = "default", bankroll: float = None) -> Dict:
        """Alternative method that accepts separate odds"""
        market_odds = {'over_25': over_odds, 'under_25': under_odds}
        return self.predict_with_staking(home_stats, away_stats, market_odds, league, bankroll)
