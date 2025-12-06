import numpy as np
import math
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

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
    Complete v5.0 Football Predictor with:
    - Kalman Filter for optimal Last5/Last10 fusion
    - Bayesian probability calculation
    - Hybrid decision system
    - Kelly staking with confidence weighting
    """
    
    def __init__(self, bankroll: float = 1000.0, min_confidence: float = 0.60):
        self.bankroll = bankroll
        self.min_confidence = min_confidence
        
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
        
        # Betting parameters
        self.kelly_fraction = 0.25
        self.max_stake_pct = 0.05
        
        # Explanation templates
        self._init_explanation_templates()
        
    def _init_explanation_templates(self):
        """Initialize explanation templates"""
        self.explanation_templates = {
            1: (
                "High Confidence {prediction}: Kalman optimal estimate shows {total_goals:.2f} total goals "
                "with strong momentum. Last5: {last5_contribution:.2f}, Last10: {last10_contribution:.2f}. "
                "Edge: {edge:.1f}%."
            ),
            2: (
                "Moderate Confidence {prediction}: Bayesian probability {probability:.1%} with "
                "Kalman confidence {confidence:.2f}. Expected goals: {total_goals:.2f}. "
                "Edge: {edge:.1f}%."
            ),
            3: (
                "Low Confidence {prediction}: Statistical edge detected but weak signal. "
                "Expected goals: {total_goals:.2f}. Kalman variance: {variance:.3f}. "
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
    
    def _prepare_stats_for_prediction(self, home_stats: TeamStats, away_stats: TeamStats, 
                                    league: str = "default") -> Dict:
        """Prepare statistics with Kalman optimal fusion"""
        context = self.league_context.get(league, self.league_context['default'])
        
        # Kalman optimal fusion of Last5 and Last10
        prior_mean = context['avg_gpg']
        prior_var = self.kalman_params['initial_variance']
        
        # Process noise
        prior_var += self.kalman_params['process_variance']
        
        # Home team: Fusion of home stats
        home_last10 = home_stats.last10_home_gpg
        home_last5 = home_stats.last5_home_gpg
        
        # Away team: Fusion of away stats
        away_last10 = away_stats.last10_away_gpg
        away_last5 = away_stats.last5_away_gpg
        
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
        home_xg_hybrid = 0.6 * home_final_adj + 0.4 * home_stats.avg_xg_for
        away_xg_hybrid = 0.6 * away_final_adj + 0.4 * away_stats.avg_xg_for
        
        # Final estimates
        home_attack_final = 0.5 * home_final_adj + 0.5 * home_xg_hybrid
        away_attack_final = 0.5 * away_final_adj + 0.5 * away_xg_hybrid
        
        # Defense estimates
        home_defense_final = home_stats.last10_home_gapg * 0.6 + home_stats.avg_xg_against * 0.4
        away_defense_final = away_stats.last10_away_gapg * 0.6 + away_stats.avg_xg_against * 0.4
        
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
            'home_xg': home_stats.avg_xg_for,
            'away_xg': away_stats.avg_xg_for,
            'league_context': context,
            'kalman_home': home_final,
            'kalman_away': away_final,
            'kalman_home_var': home_final_var,
            'kalman_away_var': away_final_var
        }
    
    def _apply_five_rules(self, stats: Dict) -> Tuple[str, str, int, float]:
        """
        Enhanced 5-rule system with Kalman optimal estimates
        """
        # Extract Kalman optimal estimates
        home_attack = stats['home_attack']
        away_attack = stats['away_attack']
        home_defense = stats['home_defense']
        away_defense = stats['away_defense']
        
        # Rule 1: High Confidence Over (Both Kalman estimates > threshold)
        rule1_condition = (
            home_attack > self.over_threshold and
            away_attack > self.over_threshold and
            stats['home_last5'] > self.over_threshold and
            stats['away_last5'] > self.over_threshold
        )
        
        if rule1_condition:
            confidence_boost = 0.05 if (stats['home_momentum'] == "improving" or 
                                       stats['away_momentum'] == "improving") else 0
            return "Over 2.5", "High", 1, confidence_boost
        
        # Rule 2: High Confidence Under (Strong Kalman defense vs weak attack)
        rule2_condition = (
            home_defense < self.under_threshold_defense and
            away_attack < self.under_threshold_attack and
            stats['home_last5_gapg'] < self.under_threshold_defense and
            stats['away_last5'] < self.under_threshold_attack
        )
        
        if rule2_condition:
            confidence_boost = 0.05 if (stats['home_momentum'] == "improving" or 
                                       stats['away_momentum'] == "declining") else 0
            return "Under 2.5", "High", 2, confidence_boost
        
        # Rule 3: Moderate Confidence Over (Kalman estimate > threshold)
        rule3_condition = (
            home_attack > self.over_threshold and
            away_attack > self.over_threshold
        )
        
        if rule3_condition:
            momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                     stats['away_momentum'] == "improving") else 0
            return "Over 2.5", "Moderate", 3, momentum_bonus
        
        # Rule 4: Moderate Confidence Under (Kalman defense strong)
        rule4_condition = (
            home_defense < self.under_threshold_defense and
            away_attack < self.under_threshold_attack
        )
        
        if rule4_condition:
            momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                     stats['away_momentum'] == "declining") else 0
            return "Under 2.5", "Moderate", 4, momentum_bonus
        
        # Rule 5: Bayesian/Kalman edge detection
        total_goals = home_attack + away_attack
        if total_goals > 2.8:  # High expected goals
            return "Over 2.5", "Low", 5, 0
        elif total_goals < 2.2:  # Low expected goals
            return "Under 2.5", "Low", 5, 0
        
        return "No Bet", "None", 5, 0
    
    def _calculate_poisson_probabilities(self, stats: Dict) -> Tuple[float, float, float, Dict]:
        """Calculate Poisson probabilities with Kalman optimal estimates"""
        
        context = stats['league_context']
        
        # Use Kalman optimal estimates
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
            'away_momentum': stats['away_momentum']
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
        """Fractional Kelly staking with confidence weighting"""
        if bankroll is None:
            bankroll = self.bankroll
        
        confidence_map = {"High": 0.8, "Moderate": 0.65, "Low": 0.55, "None": 0.0}
        confidence_value = confidence_map.get(confidence, 0.5)
        
        if confidence_value < self.min_confidence:
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
        
        implied_prob = 1 / odds if odds > 0 else 0.0
        edge = probability - implied_prob
        
        if edge <= 0:
            return {
                'stake_amount': 0.0,
                'stake_percent': 0.0,
                'kelly_fraction': 0.0,
                'expected_value': 0.0,
                'risk_level': 'No Value',
                'edge_percent': edge * 100,
                'value_rating': 'Poor',
                'implied_probability': implied_prob
            }
        
        # Kelly formula
        b = odds - 1
        p = probability
        q = 1 - p
        
        if b <= 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly and confidence weighting
        fractional_kelly = max(0, kelly_fraction) * self.kelly_fraction * confidence_value
        
        # Calculate stake
        stake_amount = fractional_kelly * bankroll
        
        # Cap at reasonable levels
        max_stake = bankroll * self.max_stake_pct
        stake_amount = min(stake_amount, max_stake)
        
        # Calculate expected value
        expected_value = (probability * (stake_amount * (odds - 1))) - ((1 - probability) * stake_amount)
        
        # Determine risk level
        if fractional_kelly > 0.1:
            risk_level = "High"
        elif fractional_kelly > 0.05:
            risk_level = "Medium"
        elif fractional_kelly > 0:
            risk_level = "Low"
        else:
            risk_level = "No Bet"
        
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
            'kelly_fraction': fractional_kelly,
            'expected_value': expected_value,
            'risk_level': risk_level,
            'edge_percent': edge * 100,
            'value_rating': value_rating,
            'implied_probability': implied_prob
        }
    
    def predict_match(self, home_stats: TeamStats, away_stats: TeamStats, 
                     over_odds: float, under_odds: float, 
                     league: str = "default", bankroll: float = None) -> Dict:
        """
        Complete prediction for a single match
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            over_odds: Market odds for Over 2.5
            under_odds: Market odds for Under 2.5
            league: League name for context
            bankroll: Current bankroll (overrides instance bankroll)
        
        Returns:
            Complete prediction dictionary
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        # Prepare stats with Kalman optimal fusion
        stats = self._prepare_stats_for_prediction(home_stats, away_stats, league)
        
        # Apply rules
        prediction, confidence, rule_number, confidence_boost = self._apply_five_rules(stats)
        
        # Calculate Poisson probabilities
        expected_goals, prob_over, prob_under, poisson_details = self._calculate_poisson_probabilities(stats)
        
        # Calculate Bayesian probabilities
        bayesian_over, bayesian_under = self._bayesian_probability(expected_goals)
        
        # Combine Poisson and Bayesian (weighted average)
        combined_over = 0.6 * prob_over + 0.4 * bayesian_over
        combined_under = 0.6 * prob_under + 0.4 * bayesian_under
        
        # Determine final probability
        if prediction == "Over 2.5":
            base_probability = combined_over
            market_odd = over_odds
        elif prediction == "Under 2.5":
            base_probability = combined_under
            market_odd = under_odds
        else:
            base_probability = max(combined_over, combined_under)
            market_odd = 2.0
        
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
                'away_momentum': stats['away_momentum'],
                'kalman_home': stats['kalman_home'],
                'kalman_away': stats['kalman_away']
            }
        }
    
    def _generate_explanation(self, prediction: str, confidence: str, rule_number: int, 
                            stats: Dict, probability: float, edge: float,
                            expected_goals: float) -> str:
        """Generate detailed explanation for the prediction"""
        
        if rule_number == 5 and prediction == "No Bet":
            return (
                f"No clear statistical edge. Kalman optimal estimates: "
                f"Home: {stats['kalman_home']:.2f}, Away: {stats['kalman_away']:.2f}. "
                f"Expected goals: {expected_goals:.2f}. Bayesian probability inconclusive."
            )
        
        # Prepare template variables
        template_vars = {
            'prediction': prediction,
            'total_goals': expected_goals,
            'last5_contribution': (stats['home_last5'] + stats['away_last5']) / 2,
            'last10_contribution': (stats['home_last10'] + stats['away_last10']) / 2,
            'edge': edge,
            'probability': probability,
            'confidence': stats.get('kalman_home_var', 0.1) + stats.get('kalman_away_var', 0.1),
            'variance': (stats.get('kalman_home_var', 0.1) + stats.get('kalman_away_var', 0.1)) / 2
        }
        
        # Select appropriate template based on confidence
        if confidence == "High":
            template_key = 1
        elif confidence == "Moderate":
            template_key = 2
        else:
            template_key = 3
        
        template = self.explanation_templates.get(template_key)
        if template:
            return template.format(**template_vars)
        
        return f"{confidence} confidence {prediction} based on Kalman optimal hybrid analysis."

# ============================================================================
# SIMPLE TEST
# ============================================================================

def test_predictor():
    """Test the predictor"""
    
    # Create sample teams
    team_a = TeamStats(
        team_name="Bournemouth",
        matches_played=14,
        home_wins=4, home_draws=2, home_losses=1,
        home_goals_for=10, home_goals_against=5,
        away_wins=1, away_draws=2, away_losses=4,
        away_goals_for=11, away_goals_against=19,
        home_xg=1.55, away_xg=None,
        avg_xg_for=1.53, avg_xg_against=1.32,
        form_last_5="WDDLW",
        attack_strength=1.15, defense_strength=1.35,
        last5_home_gpg=1.8, last5_home_gapg=0.8,
        last5_away_gpg=1.6, last5_away_gapg=3.0,
        last10_home_gpg=1.6, last10_home_gapg=0.71,
        last10_away_gpg=1.57, last10_away_gapg=2.71
    )
    
    team_b = TeamStats(
        team_name="Everton",
        matches_played=14,
        home_wins=3, home_draws=2, home_losses=2,
        home_goals_for=8, home_goals_against=9,
        away_wins=3, away_draws=1, away_losses=3,
        away_goals_for=7, away_goals_against=8,
        home_xg=1.70, away_xg=0.64,
        avg_xg_for=1.29, avg_xg_against=1.52,
        form_last_5="WDLDL",
        attack_strength=1.15, defense_strength=1.00,
        last5_home_gpg=1.2, last5_home_gapg=1.8,
        last5_away_gpg=0.9, last5_away_gapg=1.0,
        last10_home_gpg=1.14, last10_home_gapg=1.29,
        last10_away_gpg=1.0, last10_away_gapg=1.14
    )
    
    # Initialize predictor
    predictor = CompletePhantomPredictor(bankroll=1000.0)
    
    # Make prediction
    result = predictor.predict_match(
        home_stats=team_a,
        away_stats=team_b,
        over_odds=1.85,
        under_odds=1.95,
        league='premier_league'
    )
    
    # Display results
    print("=" * 60)
    print("COMPLETE PHANTOM PREDICTOR v5.0")
    print("=" * 60)
    print(f"Match: {team_a.team_name} vs {team_b.team_name}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Probability: {result['probability']:.1%}")
    print(f"Expected Goals: {result['expected_goals']:.2f}")
    print(f"\nExplanation: {result['explanation']}")
    
    if result['prediction'] != 'NO BET':
        print(f"\nStaking Info:")
        print(f"  Stake: ${result['staking_info']['stake_amount']:.2f}")
        print(f"  Edge: {result['staking_info']['edge_percent']:.1f}%")
        print(f"  Expected Value: ${result['staking_info']['expected_value']:.2f}")
        print(f"  Risk Level: {result['staking_info']['risk_level']}")

if __name__ == "__main__":
    test_predictor()
