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
    Complete v5.2 Football Predictor with CORRECT probability calculations
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
        
        # Kalman Filter parameters
        self.kalman_params = {
            'process_variance': 0.08,
            'measurement_variance_last5': 0.25,
            'measurement_variance_last10': 0.12,
            'initial_variance': 0.5
        }
        
        # FIXED: Better Bayesian parameters
        self.prior_over = 0.50  # Start with 50/50
        self.prior_under = 0.50
        
        # FIXED: Wider likelihood distributions
        self.likelihoods = {
            'over': {
                'last10_mean': 2.8,   # Higher mean for Bundesliga
                'last10_std': 0.5     # Wider distribution
            },
            'under': {
                'last10_mean': 2.0,   # Higher mean
                'last10_std': 0.5     # Wider distribution
            }
        }
        
        # Thresholds
        self.over_threshold = 1.5
        self.under_threshold_defense = 1.0
        self.under_threshold_attack = 1.5
        
        # Validation thresholds
        self.min_over_goals = 2.5  # Must be >= 2.5 for Over!
        self.max_under_goals = 2.5  # Must be <= 2.5 for Under!
        
        # Betting parameters
        self.max_stake_pct = 0.05
        
        # Explanation templates
        self._init_explanation_templates()
    
    def _create_builtin_kelly(self):
        """Create built-in Kelly calculator"""
        class BuiltInKelly:
            def __init__(self, fraction=0.5):
                self.fraction = fraction
            
            def calculate_stake(self, probability, odds, bankroll, max_percent=0.05):
                q = 1 - probability
                b = odds - 1
                
                if b <= 0 or probability <= 0:
                    kelly_fraction = 0
                else:
                    kelly_fraction = (probability * b - q) / b
                
                kelly_fraction *= self.fraction
                kelly_fraction = max(0, kelly_fraction)
                
                stake_amount = bankroll * kelly_fraction
                max_stake = bankroll * max_percent
                stake_amount = min(stake_amount, max_stake)
                
                expected_value = (probability * (stake_amount * (odds - 1))) - (q * stake_amount)
                
                if kelly_fraction > 0.1:
                    risk_level = "High"
                elif kelly_fraction > 0.05:
                    risk_level = "Medium"
                elif kelly_fraction > 0:
                    risk_level = "Low"
                else:
                    risk_level = "No Bet"
                
                implied_prob = 1 / odds if odds > 0 else 0
                edge = probability - implied_prob
                
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
            1: (
                "High Confidence {prediction}: Both teams >1.5 GPG in last 10 & last 5. "
                "Expected goals: {total_goals:.2f}. Poisson P(Over): {poisson_prob:.1%}. "
                "Edge: {edge:.1f}%."
            ),
            2: (
                "High Confidence {prediction}: Defense <1.0 GApg vs Attack <1.5 GPG. "
                "Expected goals: {total_goals:.2f}. Poisson P(Under): {poisson_prob:.1%}. "
                "Edge: {edge:.1f}%."
            ),
            3: (
                "Moderate Confidence {prediction}: Last 5 attacking form. "
                "Expected goals: {total_goals:.2f}. Poisson P(Over): {poisson_prob:.1%}. "
                "Edge: {edge:.1f}%."
            ),
            4: (
                "Moderate Confidence {prediction}: Last 5 defense vs attack mismatch. "
                "Expected goals: {total_goals:.2f}. Poisson P(Under): {poisson_prob:.1%}. "
                "Edge: {edge:.1f}%."
            ),
            5: (
                "Low Confidence {prediction}: xG-based edge. "
                "Expected goals: {total_goals:.2f}. Poisson P: {poisson_prob:.1%}. "
                "Edge: {edge:.1f}%."
            )
        }
    
    def poisson_pmf(self, k: int, lambd: float) -> float:
        """Calculate Poisson probability mass function"""
        return (lambd ** k * math.exp(-lambd)) / math.factorial(k)
    
    def normal_pdf(self, x: float, mean: float, std: float) -> float:
        """Normal probability density function"""
        if std <= 0:
            return 0.0
        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)
    
    def calculate_poisson_probability(self, expected_goals: float) -> float:
        """Calculate P(Over 2.5) from Poisson distribution"""
        prob_under = sum(self.poisson_pmf(k, expected_goals) for k in range(3))
        return 1 - prob_under
    
    def kalman_update(self, prior_mean: float, prior_var: float, 
                     measurement: float, measurement_var: float) -> Tuple[float, float]:
        """Kalman Filter update step"""
        if measurement_var <= 0:
            measurement_var = 0.01
            
        kalman_gain = prior_var / (prior_var + measurement_var)
        updated_mean = prior_mean + kalman_gain * (measurement - prior_mean)
        updated_var = (1 - kalman_gain) * prior_var
        
        return updated_mean, updated_var
    
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
        """Prepare statistics with Kalman optimal fusion"""
        context = self.league_context.get(league, self.league_context['default'])
        
        prior_mean = context['avg_gpg']
        prior_var = self.kalman_params['initial_variance']
        prior_var += self.kalman_params['process_variance']
        
        # Home team stats
        home_last10 = home_stats.get('last10_home_gpg', home_stats.get('last10_gpg', 1.5))
        home_last5 = home_stats.get('last5_home_gpg', home_stats.get('last5_gpg', 1.5))
        home_last5_gapg = home_stats.get('last5_home_gapg', home_stats.get('gapg_last10', 1.5))
        
        # Away team stats
        away_last10 = away_stats.get('last10_away_gpg', away_stats.get('last10_gpg', 1.5))
        away_last5 = away_stats.get('last5_away_gpg', away_stats.get('last5_gpg', 1.5))
        away_last5_gapg = away_stats.get('last5_away_gapg', away_stats.get('gapg_last10', 1.5))
        
        # Kalman updates
        home_posterior, home_var = self.kalman_update(
            prior_mean, prior_var, home_last10, self.kalman_params['measurement_variance_last10']
        )
        home_final, home_final_var = self.kalman_update(
            home_posterior, home_var, home_last5, self.kalman_params['measurement_variance_last5']
        )
        
        away_posterior, away_var = self.kalman_update(
            prior_mean, prior_var, away_last10, self.kalman_params['measurement_variance_last10']
        )
        away_final, away_final_var = self.kalman_update(
            away_posterior, away_var, away_last5, self.kalman_params['measurement_variance_last5']
        )
        
        # Momentum
        home_momentum, home_momentum_mult = self._calculate_form_momentum(home_last5, home_last10)
        away_momentum, away_momentum_mult = self._calculate_form_momentum(away_last5, away_last10)
        
        # Apply momentum
        home_final_adj = home_final * home_momentum_mult
        away_final_adj = away_final * away_momentum_mult
        
        # xG hybrid
        home_xg = home_stats.get('avg_xg_for', home_stats.get('gpg_last10', 1.5))
        away_xg = away_stats.get('avg_xg_for', away_stats.get('gpg_last10', 1.5))
        
        home_xg_hybrid = 0.6 * home_final_adj + 0.4 * home_xg
        away_xg_hybrid = 0.6 * away_final_adj + 0.4 * away_xg
        
        # Final estimates
        home_attack_final = 0.5 * home_final_adj + 0.5 * home_xg_hybrid
        away_attack_final = 0.5 * away_final_adj + 0.5 * away_xg_hybrid
        
        # Defense estimates
        home_defense_final = home_stats.get('last10_home_gapg', home_stats.get('gapg_last10', 1.5)) * 0.6 + home_stats.get('avg_xg_against', 1.5) * 0.4
        away_defense_final = away_stats.get('last10_away_gapg', away_stats.get('gapg_last10', 1.5)) * 0.6 + away_stats.get('avg_xg_against', 1.5) * 0.4
        
        return {
            'home_attack': home_attack_final,
            'away_attack': away_attack_final,
            'home_defense': home_defense_final,
            'away_defense': away_defense_final,
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'home_last5': home_last5,
            'home_last10': home_last10,
            'away_last5': away_last5,
            'away_last10': away_last10,
            'home_last5_gapg': home_last5_gapg,
            'away_last5_gapg': away_last5_gapg,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'league_context': context,
            'kalman_home': home_final,
            'kalman_away': away_final
        }
    
    def _apply_five_rules_with_validation(self, stats: Dict, expected_goals: float) -> Tuple[str, str, int, float]:
        """
        Enhanced 5-rule system with validation
        """
        # Rule 1: High Confidence Over
        rule1_condition = (
            stats['home_last10'] > self.over_threshold and
            stats['away_last10'] > self.over_threshold and
            stats['home_last5'] > self.over_threshold and
            stats['away_last5'] > self.over_threshold
        )
        
        if rule1_condition:
            # FIXED: Must have expected goals >= 2.5 for Over!
            if expected_goals >= self.min_over_goals:
                confidence_boost = 0.05 if (stats['home_momentum'] == "improving" or 
                                           stats['away_momentum'] == "improving") else 0
                return "Over 2.5", "High", 1, confidence_boost
            else:
                return "No Bet", "None", 1, 0
        
        # Rule 2: High Confidence Under
        rule2_condition = (
            stats['home_defense'] < self.under_threshold_defense and
            stats['away_attack'] < self.under_threshold_attack and
            stats['home_last5_gapg'] < self.under_threshold_defense and
            stats['away_last5'] < self.under_threshold_attack
        )
        
        if rule2_condition:
            if expected_goals <= self.max_under_goals:
                confidence_boost = 0.05 if (stats['home_momentum'] == "improving" or 
                                           stats['away_momentum'] == "declining") else 0
                return "Under 2.5", "High", 2, confidence_boost
            else:
                return "No Bet", "None", 2, 0
        
        # Rule 3: Moderate Over
        rule3_condition = (
            stats['home_last5'] > self.over_threshold and
            stats['away_last5'] > self.over_threshold
        )
        
        if rule3_condition:
            if expected_goals >= self.min_over_goals:
                momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                         stats['away_momentum'] == "improving") else 0
                return "Over 2.5", "Moderate", 3, momentum_bonus
            else:
                return "No Bet", "None", 3, 0
        
        # Rule 4: Moderate Under
        rule4_condition = (
            stats['home_last5_gapg'] < self.under_threshold_defense and
            stats['away_last5'] < self.under_threshold_attack
        )
        
        if rule4_condition:
            if expected_goals <= self.max_under_goals:
                momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                         stats['away_momentum'] == "declining") else 0
                return "Under 2.5", "Moderate", 4, momentum_bonus
            else:
                return "No Bet", "None", 4, 0
        
        # Rule 5: xG-based
        xg_avg = (stats['home_xg'] + stats['away_xg']) / 2
        
        if xg_avg > 1.8 and expected_goals >= 2.5:
            return "Over 2.5", "Low", 5, 0
        elif xg_avg < 1.2 and expected_goals <= 2.5:
            return "Under 2.5", "Low", 5, 0
        
        return "No Bet", "None", 5, 0
    
    def _calculate_expected_goals(self, stats: Dict) -> Tuple[float, Dict]:
        """Calculate expected goals using proper Poisson formula"""
        context = stats['league_context']
        
        # Proper Poisson expected goals
        lambda_home = (stats['home_attack'] * 
                      (stats['away_defense'] / context['avg_gapg']) * 
                      (1 + context['home_advantage']))
        
        lambda_away = (stats['away_attack'] * 
                      (stats['home_defense'] / context['avg_gapg']))
        
        expected_goals = lambda_home + lambda_away
        
        return expected_goals, {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'home_momentum': stats['home_momentum'],
            'away_momentum': stats['away_momentum'],
            'expected_home_goals': lambda_home,
            'expected_away_goals': lambda_away
        }
    
    def predict_with_staking(self, home_stats: dict, away_stats: dict,
                           market_odds: dict, league: str = "default", 
                           bankroll: float = None) -> Dict:
        """
        CORRECTED prediction pipeline
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        over_odds = market_odds.get('over_25', 1.85)
        under_odds = market_odds.get('under_25', 1.95)
        
        # Prepare stats
        stats = self._prepare_stats_for_prediction(home_stats, away_stats, league)
        
        # Calculate expected goals
        expected_goals, poisson_details = self._calculate_expected_goals(stats)
        
        # Calculate TRUE Poisson probability
        poisson_prob_over = self.calculate_poisson_probability(expected_goals)
        poisson_prob_under = 1 - poisson_prob_over
        
        # Apply rules with validation
        prediction, confidence, rule_number, confidence_boost = self._apply_five_rules_with_validation(
            stats, expected_goals
        )
        
        # Determine final probability (USE POISSON, not inflated Bayesian!)
        if prediction == "Over 2.5":
            base_probability = poisson_prob_over
            market_odd = over_odds
        elif prediction == "Under 2.5":
            base_probability = poisson_prob_under
            market_odd = under_odds
        else:  # No Bet
            explanation = f"No Bet: Expected goals {expected_goals:.2f} doesn't support any clear prediction."
            return {
                'prediction': 'NO BET',
                'confidence': 'None',
                'probability': max(poisson_prob_over, poisson_prob_under),
                'expected_goals': expected_goals,
                'rule_number': rule_number,
                'explanation': explanation,
                'staking_info': {
                    'stake_amount': 0.0,
                    'stake_percent': 0.0,
                    'edge_percent': 0.0,
                    'expected_value': 0.0,
                    'risk_level': 'No Bet',
                    'value_rating': 'None'
                },
                'market_odds': over_odds if poisson_prob_over > poisson_prob_under else under_odds,
                'poisson_details': poisson_details,
                'stats_analysis': {
                    'home_attack': stats['home_attack'],
                    'home_defense': stats['home_defense'],
                    'away_attack': stats['away_attack'],
                    'away_defense': stats['away_defense'],
                    'home_momentum': stats['home_momentum'],
                    'away_momentum': stats['away_momentum']
                }
            }
        
        # Apply small confidence boost
        final_probability = min(0.95, base_probability + confidence_boost)
        
        # Calculate stake
        staking_info = self._calculate_stake(final_probability, market_odd, confidence, bankroll)
        
        # Generate explanation
        explanation = self._generate_explanation(
            prediction, confidence, rule_number, stats, 
            final_probability, staking_info['edge_percent'],
            expected_goals, poisson_prob_over if prediction == "Over 2.5" else poisson_prob_under
        )
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability': final_probability,
            'expected_goals': expected_goals,
            'rule_number': rule_number,
            'explanation': explanation,
            'staking_info': staking_info,
            'market_odds': market_odd,
            'poisson_details': poisson_details,
            'stats_analysis': {
                'home_attack': stats['home_attack'],
                'home_defense': stats['home_defense'],
                'away_attack': stats['away_attack'],
                'away_defense': stats['away_defense'],
                'home_momentum': stats['home_momentum'],
                'away_momentum': stats['away_momentum']
            }
        }
    
    def _calculate_stake(self, probability: float, odds: float, confidence: str, 
                        bankroll: float) -> Dict:
        """Calculate stake using Kelly"""
        confidence_multipliers = {"High": 1.0, "Moderate": 0.7, "Low": 0.4, "None": 0.0}
        confidence_mult = confidence_multipliers.get(confidence, 0.5)
        
        if confidence_mult == 0 or probability < self.min_confidence:
            return {
                'stake_amount': 0.0,
                'stake_percent': 0.0,
                'kelly_fraction': 0.0,
                'expected_value': 0.0,
                'risk_level': 'No Bet',
                'edge_percent': 0.0,
                'value_rating': 'None',
                'implied_probability': 1 / odds if odds > 0 else 0.0
            }
        
        staking_result = self.kelly.calculate_stake(
            probability=probability,
            odds=odds,
            bankroll=bankroll,
            max_percent=self.max_stake_pct
        )
        
        return staking_result
    
    def _generate_explanation(self, prediction: str, confidence: str, rule_number: int, 
                            stats: Dict, probability: float, edge: float,
                            expected_goals: float, poisson_prob: float) -> str:
        """Generate explanation"""
        
        template_vars = {
            'prediction': prediction,
            'total_goals': expected_goals,
            'poisson_prob': poisson_prob,
            'edge': edge,
            'probability': probability,
        }
        
        template = self.explanation_templates.get(rule_number)
        if template:
            return template.format(**template_vars)
        
        return f"{confidence} confidence {prediction}. Expected goals: {expected_goals:.2f}, P={probability:.1%}, Edge: {edge:.1f}%"
