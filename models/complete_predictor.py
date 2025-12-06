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
    Complete v4.3 Football Predictor with:
    - Hybrid last5/last10 system
    - Bayesian shrinkage
    - xG integration (60/40)
    - Form momentum detection
    - League context awareness
    - Kelly staking
    - Edge calculation
    
    Maintains compatibility with original app.py
    """
    
    def __init__(self, bankroll: float = 1000.0, min_confidence: float = 0.50):
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
        
        # Bayesian parameters
        self.bayesian_weight = 0.3
        self.min_games_for_stability = 8
        
        # Prediction thresholds
        self.over_threshold = 1.5
        self.under_threshold_defense = 1.0
        self.under_threshold_attack = 1.5
        
        # xG weight
        self.xg_weight = 0.4
        
        # Betting parameters
        self.kelly_fraction = 0.25
        self.max_stake_pct = 0.05
        
        # Explanation templates
        self._init_explanation_templates()
        
    def _init_explanation_templates(self):
        """Initialize explanation templates"""
        self.explanation_templates = {
            1: "High Confidence {prediction}: Both teams average >1.5 goals per game in adjusted last 10 AND last 5 metrics.",
            2: "High Confidence {prediction}: Strong adjusted defense vs weak adjusted attack in both periods.",
            3: "Moderate Confidence {prediction}: Strong attacking form in last 5 matches.",
            4: "Moderate Confidence {prediction}: Recent defensive strength vs attacking weakness.",
            5: "Low Confidence {prediction}: xG-based prediction with slight statistical edge."
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
    
    def _bayesian_adjust(self, stat_value: float, n_games: int, prior_mean: float) -> float:
        """Bayesian shrinkage toward prior mean"""
        if n_games < self.min_games_for_stability:
            weight = n_games / self.min_games_for_stability
            shrunk_value = weight * stat_value + (1 - weight) * prior_mean
            return shrunk_value
        return stat_value
    
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
        """Prepare statistics for prediction"""
        context = self.league_context.get(league, self.league_context['default'])
        
        stats = {
            # Basic stats
            'home_last5_gpg': home_stats.last5_home_gpg,
            'home_last5_gapg': home_stats.last5_home_gapg,
            'home_last10_gpg': home_stats.last10_home_gpg,
            'home_last10_gapg': home_stats.last10_home_gapg,
            
            'away_last5_gpg': away_stats.last5_away_gpg,
            'away_last5_gapg': away_stats.last5_away_gapg,
            'away_last10_gpg': away_stats.last10_away_gpg,
            'away_last10_gapg': away_stats.last10_away_gapg,
            
            # xG stats
            'home_xg_for': home_stats.avg_xg_for,
            'home_xg_against': home_stats.avg_xg_against,
            'away_xg_for': away_stats.avg_xg_for,
            'away_xg_against': away_stats.avg_xg_against,
            
            # Form
            'home_form': home_stats.form_last_5,
            'away_form': away_stats.form_last_5,
            
            # Attack/defense strength
            'home_attack_strength': home_stats.attack_strength,
            'home_defense_strength': home_stats.defense_strength,
            'away_attack_strength': away_stats.attack_strength,
            'away_defense_strength': away_stats.defense_strength,
        }
        
        # Calculate momentum
        stats['home_momentum'], stats['home_momentum_mult'] = self._calculate_form_momentum(
            stats['home_last5_gpg'], stats['home_last10_gpg']
        )
        stats['away_momentum'], stats['away_momentum_mult'] = self._calculate_form_momentum(
            stats['away_last5_gpg'], stats['away_last10_gpg']
        )
        
        # Bayesian adjustments
        stats['home_last10_gpg_adj'] = self._bayesian_adjust(
            stats['home_last10_gpg'], 10, context['avg_gpg']
        )
        stats['home_last10_gapg_adj'] = self._bayesian_adjust(
            stats['home_last10_gapg'], 10, context['avg_gapg']
        )
        stats['away_last10_gpg_adj'] = self._bayesian_adjust(
            stats['away_last10_gpg'], 10, context['avg_gpg']
        )
        stats['away_last10_gapg_adj'] = self._bayesian_adjust(
            stats['away_last10_gpg'], 10, context['avg_gapg']
        )
        
        # Apply momentum to last 5 stats
        stats['home_last5_gpg_adj'] = stats['home_last5_gpg'] * stats['home_momentum_mult']
        stats['home_last5_gapg_adj'] = stats['home_last5_gapg'] * (1/stats['home_momentum_mult'] 
                                                                  if stats['home_momentum'] == "improving" 
                                                                  else stats['home_momentum_mult'])
        stats['away_last5_gpg_adj'] = stats['away_last5_gpg'] * stats['away_momentum_mult']
        stats['away_last5_gapg_adj'] = stats['away_last5_gapg'] * (1/stats['away_momentum_mult'] 
                                                                  if stats['away_momentum'] == "improving" 
                                                                  else stats['away_momentum_mult'])
        
        # xG hybrid metrics
        stats['home_hybrid_gpg'] = (0.6 * stats['home_last10_gpg_adj'] + 0.4 * stats['home_xg_for'])
        stats['home_hybrid_gapg'] = (0.6 * stats['home_last10_gapg_adj'] + 0.4 * stats['home_xg_against'])
        stats['away_hybrid_gpg'] = (0.6 * stats['away_last10_gpg_adj'] + 0.4 * stats['away_xg_for'])
        stats['away_hybrid_gapg'] = (0.6 * stats['away_last10_gapg_adj'] + 0.4 * stats['away_xg_against'])
        
        # Combined final stats
        stats['home_attack_final'] = (0.4 * stats['home_last10_gpg_adj'] + 
                                      0.4 * stats['home_last5_gpg_adj'] + 
                                      0.2 * stats['home_hybrid_gpg'])
        stats['home_defense_final'] = (0.4 * stats['home_last10_gapg_adj'] + 
                                       0.4 * stats['home_last5_gapg_adj'] + 
                                       0.2 * stats['home_hybrid_gapg'])
        stats['away_attack_final'] = (0.4 * stats['away_last10_gpg_adj'] + 
                                      0.4 * stats['away_last5_gpg_adj'] + 
                                      0.2 * stats['away_hybrid_gpg'])
        stats['away_defense_final'] = (0.4 * stats['away_last10_gapg_adj'] + 
                                       0.4 * stats['away_last5_gapg_adj'] + 
                                       0.2 * stats['away_hybrid_gapg'])
        
        stats['league_context'] = context
        
        return stats
    
    def _apply_five_rules(self, stats: Dict) -> Tuple[str, str, int, float]:
        """
        Enhanced 5-rule system
        """
        
        # Rule 1: High Confidence Over
        rule1_condition = (
            stats['home_last10_gpg_adj'] > self.over_threshold and
            stats['away_last10_gpg_adj'] > self.over_threshold and
            stats['home_last5_gpg_adj'] > self.over_threshold and
            stats['away_last5_gpg_adj'] > self.over_threshold
        )
        
        if rule1_condition:
            confidence_boost = 0.05 if (stats['home_momentum'] == "improving" or 
                                       stats['away_momentum'] == "improving") else 0
            return "Over 2.5", "High", 1, confidence_boost
        
        # Rule 2: High Confidence Under
        rule2_condition = (
            stats['home_last10_gapg_adj'] < self.under_threshold_defense and
            stats['away_last10_gpg_adj'] < self.under_threshold_attack and
            stats['home_last5_gapg_adj'] < self.under_threshold_defense and
            stats['away_last5_gpg_adj'] < self.under_threshold_attack
        )
        
        if rule2_condition:
            confidence_boost = 0.05 if (stats['home_momentum'] == "improving" or 
                                       stats['away_momentum'] == "declining") else 0
            return "Under 2.5", "High", 2, confidence_boost
        
        # Rule 3: Moderate Confidence Over
        rule3_condition = (
            stats['home_last5_gpg_adj'] > self.over_threshold and
            stats['away_last5_gpg_adj'] > self.over_threshold
        )
        
        if rule3_condition:
            momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                     stats['away_momentum'] == "improving") else 0
            return "Over 2.5", "Moderate", 3, momentum_bonus
        
        # Rule 4: Moderate Confidence Under
        rule4_condition = (
            stats['home_last5_gapg_adj'] < self.under_threshold_defense and
            stats['away_last5_gpg_adj'] < self.under_threshold_attack
        )
        
        if rule4_condition:
            momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                     stats['away_momentum'] == "declining") else 0
            return "Under 2.5", "Moderate", 4, momentum_bonus
        
        # Rule 5: Check xG edge
        xg_advantage = (stats['home_hybrid_gpg'] + stats['away_hybrid_gpg']) / 2
        
        if xg_advantage > 1.8:
            return "Over 2.5", "Low", 5, 0
        elif xg_advantage < 1.2:
            return "Under 2.5", "Low", 5, 0
        
        return "No Bet", "None", 5, 0
    
    def _calculate_poisson_probabilities(self, stats: Dict) -> Tuple[float, float, float, Dict]:
        """Calculate Poisson probabilities"""
        
        context = stats['league_context']
        
        lambda_home = (stats['home_attack_final'] * 
                      (stats['away_defense_final'] / max(0.1, context['avg_gapg'])) * 
                      (1 + context['home_advantage']))
        
        lambda_away = (stats['away_attack_final'] * 
                      (stats['home_defense_final'] / max(0.1, context['avg_gapg'])))
        
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
    
    def calculate_kelly_stake(self, probability: float, odds: float, confidence: str, 
                             bankroll: float = None) -> Dict:
        """Fractional Kelly staking"""
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
                'value_rating': 'None'
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
                'value_rating': 'Poor'
            }
        
        # Kelly formula
        b = odds - 1
        p = probability
        q = 1 - p
        
        if b <= 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly
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
            'value_rating': value_rating
        }
    
    # ==================== UNIVERSAL COMPATIBILITY METHODS ====================
    
    def predict_with_staking(self, home_stats: TeamStats, away_stats: TeamStats,
                           over_odds: float, under_odds: float,
                           league: str = "default", bankroll: float = None) -> Dict:
        """
        Universal compatibility method - matches most common app.py signatures
        """
        return self._predict_core(home_stats, away_stats, over_odds, under_odds, league, bankroll)
    
    def predict_match(self, home_stats: TeamStats, away_stats: TeamStats,
                     over_odds: float, under_odds: float,
                     league: str = "default", bankroll: float = None) -> Dict:
        """
        Alternative name for compatibility
        """
        return self._predict_core(home_stats, away_stats, over_odds, under_odds, league, bankroll)
    
    def _predict_core(self, home_stats: TeamStats, away_stats: TeamStats,
                     over_odds: float, under_odds: float,
                     league: str = "default", bankroll: float = None) -> Dict:
        """
        Core prediction logic
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        # Prepare stats
        stats = self._prepare_stats_for_prediction(home_stats, away_stats, league)
        
        # Apply rules
        prediction, confidence, rule_number, confidence_boost = self._apply_five_rules(stats)
        
        # Calculate probabilities
        expected_goals, prob_over, prob_under, poisson_details = self._calculate_poisson_probabilities(stats)
        
        # Determine final probability
        if prediction == "Over 2.5":
            base_probability = prob_over
            market_odd = over_odds
        elif prediction == "Under 2.5":
            base_probability = prob_under
            market_odd = under_odds
        else:
            base_probability = max(prob_over, prob_under)
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
            prediction, confidence, rule_number, stats, final_probability, staking_info['edge_percent']
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
                'home_attack': stats['home_attack_final'],
                'home_defense': stats['home_defense_final'],
                'away_attack': stats['away_attack_final'],
                'away_defense': stats['away_defense_final'],
                'home_momentum': stats['home_momentum'],
                'away_momentum': stats['away_momentum']
            }
        }
    
    def _generate_explanation(self, prediction: str, confidence: str, rule_number: int, 
                            stats: Dict, probability: float, edge: float) -> str:
        """Generate explanation"""
        
        if rule_number == 5 and prediction == "No Bet":
            return "No clear statistical edge. Bayesian adjusted metrics don't meet criteria."
        
        template = self.explanation_templates.get(rule_number)
        if template:
            return template.format(prediction=prediction)
        
        return f"{confidence} confidence {prediction} based on comprehensive statistical analysis."

# ============================================================================
# TEST WITH MULTIPLE SIGNATURES
# ============================================================================

def test_all_signatures():
    """Test all possible method signatures"""
    
    # Create sample teams
    team_a = TeamStats(
        team_name="Team A",
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
        team_name="Team B",
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
    
    predictor = CompletePhantomPredictor(bankroll=1000.0)
    
    # Test Method 1: predict_with_staking (most likely what your app uses)
    try:
        result1 = predictor.predict_with_staking(
            home_stats=team_a,
            away_stats=team_b,
            over_odds=1.85,
            under_odds=1.95,
            league='premier_league',
            bankroll=1000.0
        )
        print("✅ Method 1 (predict_with_staking): SUCCESS")
    except Exception as e:
        print(f"❌ Method 1 (predict_with_staking): {e}")
    
    # Test Method 2: predict_match
    try:
        result2 = predictor.predict_match(
            home_stats=team_a,
            away_stats=team_b,
            over_odds=1.85,
            under_odds=1.95,
            league='premier_league',
            bankroll=1000.0
        )
        print("✅ Method 2 (predict_match): SUCCESS")
    except Exception as e:
        print(f"❌ Method 2 (predict_match): {e}")
    
    return predictor

if __name__ == "__main__":
    predictor = test_all_signatures()
    print("\n✅ All compatibility tests complete!")
