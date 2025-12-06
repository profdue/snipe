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
    Complete v5.1 Football Predictor with:
    - Kalman Filter for optimal Last5/Last10 fusion
    - Bayesian probability calculation
    - Hybrid decision system with rule-validation
    - Integrated Kelly staking
    """
    
    def __init__(self, bankroll: float = 1000.0, min_confidence: float = 0.60):
        self.bankroll = bankroll
        self.min_confidence = min_confidence
        
        # Initialize Kelly Criterion (use custom or built-in)
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
        
        # Bayesian parameters
        self.prior_over = 0.47
        self.prior_under = 0.53
        
        # Likelihood parameters (empirical)
        self.likelihoods = {
            'over': {
                'last5_mean': 2.15,
                'last5_std': 0.35,
                'last10_mean': 2.05,
                'last10_std': 0.28
            },
            'under': {
                'last5_mean': 1.25,
                'last5_std': 0.25,
                'last10_mean': 1.35,
                'last10_std': 0.20
            }
        }
        
        # Thresholds (adjusted for Kalman output)
        self.over_threshold = 1.5
        self.under_threshold_defense = 1.0
        self.under_threshold_attack = 1.5
        
        # Validation thresholds
        self.min_over_goals = 2.3  # Minimum expected goals for Over bet
        self.max_under_goals = 2.7  # Maximum expected goals for Under bet
        
        # Betting parameters
        self.max_stake_pct = 0.05
        
        # Explanation templates
        self._init_explanation_templates()
    
    def _create_builtin_kelly(self):
        """Create built-in Kelly calculator if custom one not available"""
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
            1: (
                "High Confidence {prediction}: Both teams average >1.5 goals per game in adjusted "
                "last 10 AND last 5 metrics. Expected goals: {total_goals:.2f}. "
                "Kalman fusion: Last5: {last5_contribution:.2f}, Last10: {last10_contribution:.2f}. "
                "Edge: {edge:.1f}%."
            ),
            2: (
                "High Confidence {prediction}: Strong defense (<1.0 GApg) vs weak attack "
                "(<1.5 GPG) in both periods. Expected goals: {total_goals:.2f}. "
                "Defense: {home_defense:.2f} GApg vs Attack: {away_attack:.2f} GPG. "
                "Edge: {edge:.1f}%."
            ),
            3: (
                "Moderate Confidence {prediction}: Strong attacking form in last 5 matches (>1.5 GPG). "
                "Home: {home_last5:.2f} GPG ({home_momentum}). Away: {away_last5:.2f} GPG ({away_momentum}). "
                "Expected goals: {total_goals:.2f}. Edge: {edge:.1f}%."
            ),
            4: (
                "Moderate Confidence {prediction}: Recent defensive strength vs attacking weakness. "
                "Home defense (last5): {home_last5_gapg:.2f} GApg. "
                "Away attack (last5): {away_last5:.2f} GPG. "
                "Expected goals: {total_goals:.2f}. Edge: {edge:.1f}%."
            ),
            5: (
                "Low Confidence {prediction}: xG-based prediction with slight statistical edge. "
                "Expected goals: {total_goals:.2f}. xG advantage: {xg_avg:.2f}. "
                "Edge: {edge:.1f}%."
            ),
            'no_bet': (
                "No Bet: Rule triggered {prediction} but expected goals ({total_goals:.2f}) "
                "don't support the prediction. Required: {required_condition}. "
                "Insufficient statistical edge."
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
    
    def kalman_update(self, prior_mean: float, prior_var: float, 
                     measurement: float, measurement_var: float) -> Tuple[float, float]:
        """Kalman Filter update step"""
        if measurement_var <= 0:
            measurement_var = 0.01
            
        # Kalman gain
        kalman_gain = prior_var / (prior_var + measurement_var)
        
        # Update estimate
        updated_mean = prior_mean + kalman_gain * (measurement - prior_mean)
        
        # Update uncertainty
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
        
        # Kalman optimal fusion of Last5 and Last10
        prior_mean = context['avg_gpg']
        prior_var = self.kalman_params['initial_variance']
        
        # Process noise
        prior_var += self.kalman_params['process_variance']
        
        # Home team: Fusion of home stats
        home_last10 = home_stats.get('last10_home_gpg', home_stats.get('last10_gpg', 1.5))
        home_last5 = home_stats.get('last5_home_gpg', home_stats.get('last5_gpg', 1.5))
        home_last5_gapg = home_stats.get('last5_home_gapg', home_stats.get('gapg_last10', 1.5))
        
        # Away team: Fusion of away stats
        away_last10 = away_stats.get('last10_away_gpg', away_stats.get('last10_gpg', 1.5))
        away_last5 = away_stats.get('last5_away_gpg', away_stats.get('last5_gpg', 1.5))
        away_last5_gapg = away_stats.get('last5_away_gapg', away_stats.get('gapg_last10', 1.5))
        
        # Apply Kalman filter to home stats
        home_posterior, home_var = self.kalman_update(
            prior_mean, prior_var,
            home_last10, self.kalman_params['measurement_variance_last10']
        )
        home_final, home_final_var = self.kalman_update(
            home_posterior, home_var,
            home_last5, self.kalman_params['measurement_variance_last5']
        )
        
        # Apply Kalman filter to away stats
        away_posterior, away_var = self.kalman_update(
            prior_mean, prior_var,
            away_last10, self.kalman_params['measurement_variance_last10']
        )
        away_final, away_final_var = self.kalman_update(
            away_posterior, away_var,
            away_last5, self.kalman_params['measurement_variance_last5']
        )
        
        # Calculate momentum
        home_momentum, home_momentum_mult = self._calculate_form_momentum(
            home_last5, home_last10
        )
        away_momentum, away_momentum_mult = self._calculate_form_momentum(
            away_last5, away_last10
        )
        
        # Apply momentum adjustments
        home_final_adj = home_final * home_momentum_mult
        away_final_adj = away_final * away_momentum_mult
        
        # xG hybrid (60/40)
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
            'home_momentum_mult': home_momentum_mult,
            'away_momentum_mult': away_momentum_mult,
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
            'kalman_away': away_final,
            'kalman_home_var': home_final_var,
            'kalman_away_var': away_final_var
        }
    
    def _apply_five_rules_with_validation(self, stats: Dict, expected_goals: float) -> Tuple[str, str, int, float]:
        """
        Enhanced 5-rule system with Kalman optimal estimates AND VALIDATION
        """
        # Extract Kalman optimal estimates
        home_attack = stats['home_attack']
        away_attack = stats['away_attack']
        home_defense = stats['home_defense']
        away_defense = stats['away_defense']
        
        # Rule 1: High Confidence Over (Both Kalman estimates > threshold)
        rule1_condition = (
            stats['home_last10'] > self.over_threshold and
            stats['away_last10'] > self.over_threshold and
            stats['home_last5'] > self.over_threshold and
            stats['away_last5'] > self.over_threshold
        )
        
        if rule1_condition:
            # VALIDATION: Expected goals must support Over prediction
            if expected_goals >= self.min_over_goals:
                confidence_boost = 0.05 if (stats['home_momentum'] == "improving" or 
                                           stats['away_momentum'] == "improving") else 0
                return "Over 2.5", "High", 1, confidence_boost
            else:
                return "No Bet", "None", 1, 0
        
        # Rule 2: High Confidence Under (Strong defense vs weak attack)
        rule2_condition = (
            home_defense < self.under_threshold_defense and
            away_attack < self.under_threshold_attack and
            stats['home_last5_gapg'] < self.under_threshold_defense and
            stats['away_last5'] < self.under_threshold_attack
        )
        
        if rule2_condition:
            # VALIDATION: Expected goals must support Under prediction
            if expected_goals <= self.max_under_goals:
                confidence_boost = 0.05 if (stats['home_momentum'] == "improving" or 
                                           stats['away_momentum'] == "declining") else 0
                return "Under 2.5", "High", 2, confidence_boost
            else:
                return "No Bet", "None", 2, 0
        
        # Rule 3: Moderate Confidence Over (Last 5 only)
        rule3_condition = (
            stats['home_last5'] > self.over_threshold and
            stats['away_last5'] > self.over_threshold
        )
        
        if rule3_condition:
            # VALIDATION: Expected goals must support Over prediction
            if expected_goals >= self.min_over_goals:
                momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                         stats['away_momentum'] == "improving") else 0
                return "Over 2.5", "Moderate", 3, momentum_bonus
            else:
                return "No Bet", "None", 3, 0
        
        # Rule 4: Moderate Confidence Under (Last 5 only)
        rule4_condition = (
            stats['home_last5_gapg'] < self.under_threshold_defense and
            stats['away_last5'] < self.under_threshold_attack
        )
        
        if rule4_condition:
            # VALIDATION: Expected goals must support Under prediction
            if expected_goals <= self.max_under_goals:
                momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                         stats['away_momentum'] == "declining") else 0
                return "Under 2.5", "Moderate", 4, momentum_bonus
            else:
                return "No Bet", "None", 4, 0
        
        # Rule 5: xG-based edge detection
        xg_advantage = (stats['home_xg'] + stats['away_xg']) / 2
        
        if xg_advantage > 1.8 and expected_goals >= 2.5:  # High xG expectation
            return "Over 2.5", "Low", 5, 0
        elif xg_advantage < 1.2 and expected_goals <= 2.5:  # Low xG expectation
            return "Under 2.5", "Low", 5, 0
        
        return "No Bet", "None", 5, 0
    
    def _calculate_poisson_probabilities(self, stats: Dict) -> Tuple[float, float, float, Dict]:
        """Calculate Poisson probabilities with Kalman optimal estimates"""
        
        context = stats['league_context']
        
        # Use Kalman optimal estimates with proper Poisson formula
        lambda_home = (stats['home_attack'] * 
                      (stats['away_defense'] / context['avg_gapg']) * 
                      (1 + context['home_advantage']))
        
        lambda_away = (stats['away_attack'] * 
                      (stats['home_defense'] / context['avg_gapg']))
        
        expected_goals = lambda_home + lambda_away
        
        # Calculate Poisson probabilities
        prob_under = sum(self.poisson_pmf(k, expected_goals) for k in range(3))
        prob_over = 1 - prob_under
        
        return expected_goals, prob_over, prob_under, {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'home_momentum': stats['home_momentum'],
            'away_momentum': stats['away_momentum'],
            'expected_home_goals': lambda_home,
            'expected_away_goals': lambda_away
        }
    
    def _bayesian_probability(self, expected_goals: float) -> Tuple[float, float]:
        """Bayesian probability calculation"""
        # Likelihood of data given Over
        like_over = self.normal_pdf(expected_goals,
                                  self.likelihoods['over']['last10_mean'],
                                  self.likelihoods['over']['last10_std'])
        
        # Likelihood of data given Under
        like_under = self.normal_pdf(expected_goals,
                                   self.likelihoods['under']['last10_mean'],
                                   self.likelihoods['under']['last10_std'])
        
        # Bayesian update
        evidence = like_over * self.prior_over + like_under * self.prior_under
        
        if evidence == 0:
            prob_over = 0.5
        else:
            prob_over = (like_over * self.prior_over) / evidence
        
        return prob_over, 1 - prob_over
    
    def calculate_kelly_stake(self, probability: float, odds: float, confidence: str, 
                             bankroll: float = None) -> Dict:
        """Calculate stake using integrated Kelly calculator"""
        if bankroll is None:
            bankroll = self.bankroll
        
        # Map confidence to Kelly fraction adjustment
        confidence_multipliers = {
            "High": 1.0,
            "Moderate": 0.7,
            "Low": 0.4,
            "None": 0.0
        }
        
        confidence_mult = confidence_multipliers.get(confidence, 0.5)
        
        # Check if we should bet
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
        
        # Calculate stake using Kelly calculator
        staking_result = self.kelly.calculate_stake(
            probability=probability,
            odds=odds,
            bankroll=bankroll,
            max_percent=self.max_stake_pct
        )
        
        # Add edge calculation if not already present
        if 'edge_percent' not in staking_result:
            implied_prob = 1 / odds if odds > 0 else 0.0
            edge = probability - implied_prob
            staking_result['edge_percent'] = edge * 100
            
            # Determine value rating
            if edge > 0.1:
                staking_result['value_rating'] = "Excellent"
            elif edge > 0.05:
                staking_result['value_rating'] = "Good"
            elif edge > 0.02:
                staking_result['value_rating'] = "Fair"
            else:
                staking_result['value_rating'] = "Poor"
            
            staking_result['implied_probability'] = implied_prob
            staking_result['true_probability'] = probability
        
        return staking_result
    
    def predict_with_staking(self, home_stats: dict, away_stats: dict,
                           market_odds: dict, league: str = "default", 
                           bankroll: float = None) -> Dict:
        """
        Complete prediction pipeline with validation
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        # Extract odds from market_odds dict
        over_odds = market_odds.get('over_25', 1.85)
        under_odds = market_odds.get('under_25', 1.95)
        
        # Prepare stats with Kalman optimal fusion
        stats = self._prepare_stats_for_prediction(home_stats, away_stats, league)
        
        # Calculate Poisson probabilities FIRST
        expected_goals, prob_over, prob_under, poisson_details = self._calculate_poisson_probabilities(stats)
        
        # Apply rules WITH VALIDATION against expected goals
        prediction, confidence, rule_number, confidence_boost = self._apply_five_rules_with_validation(
            stats, expected_goals
        )
        
        # If No Bet, return early
        if prediction == "No Bet":
            explanation = self._generate_no_bet_explanation(
                stats, expected_goals, rule_number, prediction
            )
            return {
                'prediction': 'NO BET',
                'confidence': 'None',
                'probability': max(prob_over, prob_under),
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
                'market_odds': over_odds if prob_over > prob_under else under_odds,
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
        
        # Calculate Bayesian probabilities
        bayesian_over, bayesian_under = self._bayesian_probability(expected_goals)
        
        # Combine Poisson and Bayesian (weighted average)
        combined_over = 0.6 * prob_over + 0.4 * bayesian_over
        combined_under = 0.6 * prob_under + 0.4 * bayesian_under
        
        # Determine final probability
        if prediction == "Over 2.5":
            base_probability = combined_over
            market_odd = over_odds
        else:  # Under 2.5
            base_probability = combined_under
            market_odd = under_odds
        
        # Apply confidence boost
        final_probability = min(0.95, base_probability + confidence_boost)
        
        # Calculate stake
        staking_info = self.calculate_kelly_stake(
            probability=final_probability,
            odds=market_odd,
            confidence=confidence,
            bankroll=bankroll
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            prediction, confidence, rule_number, stats, 
            final_probability, staking_info['edge_percent'],
            expected_goals
        )
        
        # Return complete prediction
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
    
    def _generate_explanation(self, prediction: str, confidence: str, rule_number: int, 
                            stats: Dict, probability: float, edge: float,
                            expected_goals: float) -> str:
        """Generate detailed explanation for the prediction"""
        
        template_vars = {
            'prediction': prediction,
            'total_goals': expected_goals,
            'last5_contribution': (stats['home_last5'] + stats['away_last5']) / 2,
            'last10_contribution': (stats['home_last10'] + stats['away_last10']) / 2,
            'edge': edge,
            'probability': probability,
            'confidence': 1.0 / ((stats.get('kalman_home_var', 0.1) + stats.get('kalman_away_var', 0.1)) / 2 + 0.01),
            'variance': (stats.get('kalman_home_var', 0.1) + stats.get('kalman_away_var', 0.1)) / 2,
            'home_defense': stats['home_defense'],
            'away_attack': stats['away_attack'],
            'home_last5': stats['home_last5'],
            'away_last5': stats['away_last5'],
            'home_momentum': stats['home_momentum'],
            'away_momentum': stats['away_momentum'],
            'home_last5_gapg': stats['home_last5_gapg'],
            'xg_avg': (stats['home_xg'] + stats['away_xg']) / 2
        }
        
        template = self.explanation_templates.get(rule_number)
        if template:
            return template.format(**template_vars)
        
        return f"{confidence} confidence {prediction} based on comprehensive statistical analysis."
    
    def _generate_no_bet_explanation(self, stats: Dict, expected_goals: float, 
                                   rule_number: int, predicted_rule: str) -> str:
        """Explanation for no bet scenarios"""
        
        if rule_number in [1, 2, 3, 4]:  # Rule triggered but validation failed
            if predicted_rule == "Over 2.5":
                required = f">={self.min_over_goals} goals"
            else:
                required = f"<={self.max_under_goals} goals"
            
            return self.explanation_templates['no_bet'].format(
                prediction=predicted_rule,
                total_goals=expected_goals,
                required_condition=required
            )
        
        return (
            f"No clear statistical edge. Kalman optimal estimates: "
            f"Home: {stats['kalman_home']:.2f}, Away: {stats['kalman_away']:.2f}. "
            f"Expected goals: {expected_goals:.2f}. Bayesian probability inconclusive."
        )
