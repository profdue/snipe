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

class KalmanHybridPredictor:
    """
    Mathematically Optimal Hybrid Predictor using:
    1. Kalman Filter for optimal Last5/Last10 fusion
    2. Bayesian inference for probability calculation
    3. Pure Python/Numpy implementations (no scipy dependency)
    """
    
    def __init__(self, bankroll: float = 1000.0, min_confidence: float = 0.60):
        self.bankroll = bankroll
        self.min_confidence = min_confidence
        
        # League-specific parameters
        self.league_params = {
            'premier_league': {'avg_total': 2.8, 'home_adv': 0.15, 'over_freq': 0.48},
            'la_liga': {'avg_total': 2.6, 'home_adv': 0.18, 'over_freq': 0.45},
            'bundesliga': {'avg_total': 3.1, 'home_adv': 0.12, 'over_freq': 0.52},
            'serie_a': {'avg_total': 2.5, 'home_adv': 0.20, 'over_freq': 0.43},
            'ligue_1': {'avg_total': 2.7, 'home_adv': 0.16, 'over_freq': 0.47},
            'default': {'avg_total': 2.7, 'home_adv': 0.16, 'over_freq': 0.47}
        }
        
        # Kalman Filter parameters
        self.kalman_params = {
            'process_variance': 0.08,
            'measurement_variance_last5': 0.25,
            'measurement_variance_last10': 0.12,
            'initial_variance': 0.5
        }
        
        # Bayesian prior parameters
        self.prior_over = 0.47
        self.prior_under = 0.53
        
        # Likelihood parameters
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
        
        # Trend detection
        self.trend_filter = np.array([0.05, 0.1, 0.15, 0.25, 0.45])
        self.momentum_threshold = 0.15
        
        # Betting parameters
        self.kelly_fraction = 0.25
        self.max_stake_pct = 0.05
        
    # ==================== CORE MATHEMATICAL METHODS ====================
    
    def normal_pdf(self, x: float, mean: float, std: float) -> float:
        """Pure Python implementation of normal PDF (replaces scipy.stats.norm.pdf)"""
        if std <= 0:
            return 0.0
        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)
    
    def poisson_pmf(self, k: int, lambd: float) -> float:
        """Pure Python implementation of Poisson PMF"""
        return (lambd ** k * math.exp(-lambd)) / math.factorial(k)
    
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
    
    def kalman_predict_goals(self, home_stats: TeamStats, away_stats: TeamStats, 
                           league: str = "default") -> Dict[str, float]:
        """
        Optimal fusion of Last 5 and Last 10 data using Kalman Filter
        """
        params = self.league_params.get(league, self.league_params['default'])
        
        # Start with league average as prior
        prior_mean = params['avg_total']
        prior_var = self.kalman_params['initial_variance']
        
        # Add process noise
        prior_var += self.kalman_params['process_variance']
        
        # Measurement 1: Last 10 games
        last10_measurement = ((home_stats.last10_home_gpg + away_stats.last10_away_gpg) / 2) * 2
        last10_measurement = max(0.5, min(5.0, last10_measurement))
        
        # Update with Last 10 data
        posterior_mean, posterior_var = self.kalman_update(
            prior_mean, prior_var,
            last10_measurement, self.kalman_params['measurement_variance_last10']
        )
        
        # Measurement 2: Last 5 games
        last5_measurement = ((home_stats.last5_home_gpg + away_stats.last5_away_gpg) / 2) * 2
        last5_measurement = max(0.5, min(5.0, last5_measurement))
        
        # Update with Last 5 data
        final_mean, final_var = self.kalman_update(
            posterior_mean, posterior_var,
            last5_measurement, self.kalman_params['measurement_variance_last5']
        )
        
        # Calculate home/away breakdown
        home_share = 0.5 + params['home_adv'] / 2
        home_goals = final_mean * home_share
        away_goals = final_mean * (1 - home_share)
        
        return {
            'total_goals': final_mean,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'variance': final_var,
            'confidence': 1.0 / (final_var + 0.01),
            'last5_contribution': last5_measurement,
            'last10_contribution': last10_measurement
        }
    
    def bayesian_probability(self, kalman_result: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate P(Over|Data) using Bayes' Theorem with Kalman output
        """
        total_goals = kalman_result['total_goals']
        confidence = kalman_result['confidence']
        
        # Calculate likelihoods using our normal_pdf function
        like_over = self.normal_pdf(total_goals, 
                                  self.likelihoods['over']['last10_mean'],
                                  self.likelihoods['over']['last10_std'] / np.sqrt(confidence))
        
        like_under = self.normal_pdf(total_goals,
                                   self.likelihoods['under']['last10_mean'],
                                   self.likelihoods['under']['last10_std'] / np.sqrt(confidence))
        
        # Apply Bayes' Theorem
        evidence = like_over * self.prior_over + like_under * self.prior_under
        
        if evidence == 0:
            prob_over = 0.5
        else:
            prob_over = (like_over * self.prior_over) / evidence
        
        # Adjust for confidence
        prob_over = self._confidence_adjust(prob_over, confidence)
        
        return prob_over, 1 - prob_over
    
    def _confidence_adjust(self, probability: float, confidence: float) -> float:
        """Adjust probability based on confidence level"""
        if confidence > 1.5:  # High confidence
            if probability > 0.5:
                adjusted = 0.5 + (probability - 0.5) * 1.2
            else:
                adjusted = 0.5 - (0.5 - probability) * 1.2
        elif confidence > 1.0:  # Moderate confidence
            if probability > 0.5:
                adjusted = 0.5 + (probability - 0.5) * 1.1
            else:
                adjusted = 0.5 - (0.5 - probability) * 1.1
        else:  # Low confidence
            adjusted = 0.5 + (probability - 0.5) * 0.8
        
        return max(0.01, min(0.99, adjusted))
    
    def detect_trend_momentum(self, home_stats: TeamStats, away_stats: TeamStats) -> Dict[str, Any]:
        """
        Detect trends using simple moving averages (no scipy dependency)
        """
        # Simple momentum calculation
        home_last5 = home_stats.last5_home_gpg
        home_last10 = home_stats.last10_home_gpg
        away_last5 = away_stats.last5_away_gpg
        away_last10 = away_stats.last10_away_gpg
        
        # Calculate momentum (change from last10 to last5)
        home_momentum = (home_last5 - home_last10) / max(0.1, home_last10)
        away_momentum = (away_last5 - away_last10) / max(0.1, away_last10)
        
        # Determine trend direction
        home_trend = "improving" if home_momentum > self.momentum_threshold else \
                    "declining" if home_momentum < -self.momentum_threshold else "stable"
        
        away_trend = "improving" if away_momentum > self.momentum_threshold else \
                    "declining" if away_momentum < -self.momentum_threshold else "stable"
        
        # Simple consistency measure
        home_consistency = 1.0 / (abs(home_last5 - home_last10) + 0.1)
        away_consistency = 1.0 / (abs(away_last5 - away_last10) + 0.1)
        
        return {
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'home_trend': home_trend,
            'away_trend': away_trend,
            'home_consistency': home_consistency,
            'away_consistency': away_consistency,
            'total_momentum': home_momentum + away_momentum,
            'trend_score': (home_momentum + away_momentum) * 
                          (home_consistency + away_consistency) / 2
        }
    
    # ==================== PREDICTION DECISION SYSTEM ====================
    
    def hybrid_decision_system(self, prob_over: float, prob_under: float,
                             kalman_result: Dict[str, float],
                             trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final prediction decision using hybrid criteria
        """
        total_goals = kalman_result['total_goals']
        confidence = kalman_result['confidence']
        trend_score = trend_data['trend_score']
        
        # Calculate decision score (0-100)
        decision_score = 0
        
        # Component 1: Probability strength (40 points max)
        prob_diff = abs(prob_over - 0.5)
        decision_score += min(40, prob_diff * 80)
        
        # Component 2: Kalman confidence (30 points max)
        decision_score += min(30, (confidence - 0.5) * 60)
        
        # Component 3: Trend momentum (20 points max)
        if trend_score > 0.3:
            decision_score += 20
        elif trend_score > 0.15:
            decision_score += 10
        elif trend_score < -0.3:
            decision_score -= 10
        
        # Component 4: Goal expectation (10 points max)
        if total_goals > 3.0:
            decision_score += 10 if prob_over > 0.5 else -5
        elif total_goals < 2.0:
            decision_score += 10 if prob_under > 0.5 else -5
        
        # Normalize score
        decision_score = max(0, min(100, decision_score))
        
        # Determine prediction
        if prob_over > prob_under:
            prediction = "OVER 2.5"
            base_prob = prob_over
        else:
            prediction = "UNDER 2.5"
            base_prob = prob_under
        
        # Set confidence levels
        if decision_score >= 70:
            confidence_level = "HIGH"
            kelly_mult = 1.0
        elif decision_score >= 50:
            confidence_level = "MODERATE"
            kelly_mult = 0.7
        elif decision_score >= 30:
            confidence_level = "LOW"
            kelly_mult = 0.4
        else:
            confidence_level = "NO BET"
            kelly_mult = 0.0
            
        # Apply trend-based probability adjustment
        if trend_data['total_momentum'] > 0.2 and prediction == "OVER 2.5":
            base_prob = min(0.95, base_prob * 1.1)
        elif trend_data['total_momentum'] < -0.2 and prediction == "UNDER 2.5":
            base_prob = min(0.95, base_prob * 1.1)
        
        return {
            'prediction': prediction,
            'probability': base_prob,
            'confidence': confidence_level,
            'decision_score': decision_score,
            'kelly_multiplier': kelly_mult,
            'expected_goals': total_goals,
            'kalman_confidence': confidence,
            'trend_score': trend_score
        }
    
    # ==================== BETTING & STAKING ====================
    
    def calculate_optimal_stake(self, probability: float, odds: float, 
                              confidence_mult: float, bankroll: float = None) -> Dict[str, float]:
        """
        Calculate optimal stake using fractional Kelly
        """
        if bankroll is None:
            bankroll = self.bankroll
            
        # Calculate edge
        implied_prob = 1.0 / odds if odds > 0 else 0
        edge = probability - implied_prob
        
        if edge <= 0 or probability <= implied_prob:
            return {
                'stake_amount': 0.0,
                'stake_percent': 0.0,
                'kelly_fraction': 0.0,
                'expected_value': 0.0,
                'edge_percent': edge * 100,
                'value_rating': 'NO VALUE',
                'implied_probability': implied_prob,
                'true_probability': probability
            }
        
        # Kelly Criterion
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly and confidence multiplier
        fractional_kelly = max(0, kelly_fraction) * self.kelly_fraction * confidence_mult
        
        # Calculate stake
        stake_amount = fractional_kelly * bankroll
        
        # Apply maximum stake limit
        max_stake = bankroll * self.max_stake_pct
        stake_amount = min(stake_amount, max_stake)
        
        # Calculate expected value
        ev_win = stake_amount * (odds - 1) * p
        ev_loss = stake_amount * q
        expected_value = ev_win - ev_loss
        
        # Determine value rating
        if edge > 0.15:
            value_rating = "EXCELLENT"
        elif edge > 0.10:
            value_rating = "VERY GOOD"
        elif edge > 0.05:
            value_rating = "GOOD"
        elif edge > 0.02:
            value_rating = "FAIR"
        else:
            value_rating = "POOR"
        
        return {
            'stake_amount': stake_amount,
            'stake_percent': stake_amount / bankroll,
            'kelly_fraction': fractional_kelly,
            'expected_value': expected_value,
            'edge_percent': edge * 100,
            'value_rating': value_rating,
            'implied_probability': implied_prob,
            'true_probability': probability
        }
    
    # ==================== MAIN PREDICTION METHOD ====================
    
    def predict_match(self, home_stats: TeamStats, away_stats: TeamStats,
                     over_odds: float, under_odds: float,
                     league: str = "default", bankroll: float = None) -> Dict[str, Any]:
        """
        Complete prediction pipeline
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        # 1. Kalman Filter optimal fusion
        kalman_result = self.kalman_predict_goals(home_stats, away_stats, league)
        
        # 2. Bayesian probability calculation
        prob_over, prob_under = self.bayesian_probability(kalman_result)
        
        # 3. Trend detection
        trend_data = self.detect_trend_momentum(home_stats, away_stats)
        
        # 4. Hybrid decision system
        decision = self.hybrid_decision_system(prob_over, prob_under, kalman_result, trend_data)
        
        # 5. If no bet, return early
        if decision['confidence'] == "NO BET":
            return {
                'prediction': 'NO BET',
                'confidence': 'NO BET',
                'probability': max(prob_over, prob_under),
                'expected_goals': kalman_result['total_goals'],
                'explanation': self._generate_no_bet_explanation(kalman_result, trend_data),
                'staking_info': {'stake_amount': 0.0, 'edge_percent': 0.0},
                'analysis': {
                    'kalman_result': kalman_result,
                    'trend_data': trend_data,
                    'prob_over': prob_over,
                    'prob_under': prob_under
                }
            }
        
        # 6. Determine market odds
        if decision['prediction'] == "OVER 2.5":
            market_odds = over_odds
            probability = prob_over
        else:
            market_odds = under_odds
            probability = prob_under
        
        # 7. Calculate optimal stake
        staking_info = self.calculate_optimal_stake(
            probability=probability,
            odds=market_odds,
            confidence_mult=decision['kelly_multiplier'],
            bankroll=bankroll
        )
        
        # 8. Generate explanation
        explanation = self._generate_explanation(
            decision, kalman_result, trend_data, staking_info
        )
        
        # 9. Return complete prediction
        return {
            'prediction': decision['prediction'],
            'confidence': decision['confidence'],
            'probability': probability,
            'expected_goals': kalman_result['total_goals'],
            'decision_score': decision['decision_score'],
            'explanation': explanation,
            'staking_info': staking_info,
            'market_odds': market_odds,
            'analysis': {
                'kalman_result': kalman_result,
                'trend_data': trend_data,
                'prob_over': prob_over,
                'prob_under': prob_under,
                'home_last5_gpg': home_stats.last5_home_gpg,
                'home_last10_gpg': home_stats.last10_home_gpg,
                'away_last5_gpg': away_stats.last5_away_gpg,
                'away_last10_gpg': away_stats.last10_away_gpg
            }
        }
    
    # ==================== EXPLANATION GENERATION ====================
    
    def _generate_explanation(self, decision: Dict[str, Any], 
                            kalman_result: Dict[str, float],
                            trend_data: Dict[str, Any],
                            staking_info: Dict[str, float]) -> str:
        """Generate detailed explanation"""
        
        prediction = decision['prediction']
        confidence = decision['confidence']
        total_goals = kalman_result['total_goals']
        
        explanation_parts = []
        
        # Kalman analysis
        explanation_parts.append(
            f"Kalman estimate: {total_goals:.2f} total goals "
            f"(Last5: {kalman_result['last5_contribution']:.2f}, "
            f"Last10: {kalman_result['last10_contribution']:.2f})"
        )
        
        # Trend analysis
        explanation_parts.append(
            f"Trend: Home {trend_data['home_trend']} "
            f"(Δ: {trend_data['home_momentum']:.0%}), "
            f"Away {trend_data['away_trend']} "
            f"(Δ: {trend_data['away_momentum']:.0%})"
        )
        
        # Probability and decision score
        explanation_parts.append(
            f"Probability: {decision['probability']:.1%} {prediction} "
            f"(Score: {decision['decision_score']:.1f}/100)"
        )
        
        # Staking rationale
        if staking_info['stake_amount'] > 0:
            explanation_parts.append(
                f"Edge: {staking_info['edge_percent']:.1f}% → "
                f"Stake: ${staking_info['stake_amount']:.2f}"
            )
        
        return " | ".join(explanation_parts)
    
    def _generate_no_bet_explanation(self, kalman_result: Dict[str, float],
                                   trend_data: Dict[str, Any]) -> str:
        """Explanation for no bet scenarios"""
        return (
            f"No clear edge. Expected goals: {kalman_result['total_goals']:.2f} "
            f"(Last5: {kalman_result['last5_contribution']:.2f}, "
            f"Last10: {kalman_result['last10_contribution']:.2f}). "
            f"Insufficient statistical advantage."
        )

# ============================================================================
# SIMPLE TEST FUNCTION
# ============================================================================

def test_predictor():
    """Test the predictor with sample data"""
    
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
    
    # Initialize predictor
    predictor = KalmanHybridPredictor(bankroll=1000.0)
    
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
    print("KALMAN HYBRID PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Match: {team_a.team_name} vs {team_b.team_name}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Probability: {result['probability']:.1%}")
    print(f"Expected Goals: {result['expected_goals']:.2f}")
    
    if result['prediction'] != 'NO BET':
        print(f"\nExplanation: {result['explanation']}")
        print(f"\nStaking Info:")
        print(f"  Stake: ${result['staking_info']['stake_amount']:.2f}")
        print(f"  Edge: {result['staking_info']['edge_percent']:.1f}%")
        print(f"  Value Rating: {result['staking_info']['value_rating']}")

if __name__ == "__main__":
    test_predictor()
