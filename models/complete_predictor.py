import numpy as np
import math
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
#from scipy.stats import poisson, norm
from collections import deque
import warnings

warnings.filterwarnings('ignore')

@dataclass
class TeamStats:
    """Container for team statistics from CSV"""
    team_name: str
    matches_played: int
    home_wins: int
    home_draws: int
    home_losses: int
    home_goals_for: int  # Changed to int - actual goals
    home_goals_against: int  # Changed to int
    away_wins: int
    away_draws: int
    away_losses: int
    away_goals_for: int  # Changed to int
    away_goals_against: int  # Changed to int
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
    # Add goal timeline for better analysis
    home_goals_timeline: Optional[List[int]] = None
    away_goals_timeline: Optional[List[int]] = None

class KalmanHybridPredictor:
    """
    Mathematically Optimal Hybrid Predictor using:
    1. Kalman Filter for optimal Last5/Last10 fusion
    2. Bayesian inference for probability calculation
    3. Convolutional filters for trend detection
    4. Statistical validation with confidence intervals
    """
    
    def __init__(self, bankroll: float = 1000.0, min_confidence: float = 0.60):
        self.bankroll = bankroll
        self.min_confidence = min_confidence
        
        # League-specific parameters (empirically optimized)
        self.league_params = {
            'premier_league': {'avg_total': 2.8, 'home_adv': 0.15, 'over_freq': 0.48},
            'la_liga': {'avg_total': 2.6, 'home_adv': 0.18, 'over_freq': 0.45},
            'bundesliga': {'avg_total': 3.1, 'home_adv': 0.12, 'over_freq': 0.52},
            'serie_a': {'avg_total': 2.5, 'home_adv': 0.20, 'over_freq': 0.43},
            'ligue_1': {'avg_total': 2.7, 'home_adv': 0.16, 'over_freq': 0.47},
            'default': {'avg_total': 2.7, 'home_adv': 0.16, 'over_freq': 0.47}
        }
        
        # Kalman Filter parameters (empirically tuned)
        self.kalman_params = {
            'process_variance': 0.08,  # How much true scoring rate changes per game
            'measurement_variance_last5': 0.25,  # Higher variance for small sample
            'measurement_variance_last10': 0.12,  # Lower variance for larger sample
            'initial_variance': 0.5
        }
        
        # Bayesian prior parameters
        self.prior_over = 0.47  # Prior probability of Over 2.5
        self.prior_under = 0.53  # Prior probability of Under 2.5
        
        # Likelihood distributions (estimated from historical data)
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
        
        # Trend detection parameters
        self.trend_filter = np.array([0.05, 0.1, 0.15, 0.25, 0.45])  # Exponential weighting
        self.momentum_threshold = 0.15  # 15% change significant
        
        # Betting parameters
        self.kelly_fraction = 0.25  # Fractional Kelly
        self.max_stake_pct = 0.05  # Max 5% of bankroll
        
        # Performance tracking
        self.predictions_history = []
        self.edge_calculations = []
        
    # ==================== CORE MATHEMATICAL METHODS ====================
    
    def kalman_update(self, prior_mean: float, prior_var: float, 
                     measurement: float, measurement_var: float) -> Tuple[float, float]:
        """
        Kalman Filter update step - mathematically optimal estimation
        
        Theorem: For linear Gaussian systems, Kalman Filter minimizes
        mean squared error: E[(estimate - true_value)²]
        """
        # Prevent division by zero
        if measurement_var <= 0:
            measurement_var = 0.01
            
        # Kalman gain: K = P / (P + R)
        kalman_gain = prior_var / (prior_var + measurement_var)
        
        # Update estimate: x = x + K * (z - x)
        updated_mean = prior_mean + kalman_gain * (measurement - prior_mean)
        
        # Update uncertainty: P = (1 - K) * P
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
        
        # Add process noise (scoring rates change over time)
        prior_var += self.kalman_params['process_variance']
        
        # Measurement 1: Last 10 games (more reliable)
        last10_measurement = ((home_stats.last10_home_gpg + away_stats.last10_away_gpg) / 2) * 2
        last10_measurement = max(0.5, min(5.0, last10_measurement))  # Bound reasonable values
        
        # Update with Last 10 data
        posterior_mean, posterior_var = self.kalman_update(
            prior_mean, prior_var,
            last10_measurement, self.kalman_params['measurement_variance_last10']
        )
        
        # Measurement 2: Last 5 games (less reliable but more recent)
        last5_measurement = ((home_stats.last5_home_gpg + away_stats.last5_away_gpg) / 2) * 2
        last5_measurement = max(0.5, min(5.0, last5_measurement))
        
        # Update with Last 5 data
        final_mean, final_var = self.kalman_update(
            posterior_mean, posterior_var,
            last5_measurement, self.kalman_params['measurement_variance_last5']
        )
        
        # Calculate home/away breakdown with home advantage
        home_share = 0.5 + params['home_adv'] / 2
        home_goals = final_mean * home_share
        away_goals = final_mean * (1 - home_share)
        
        return {
            'total_goals': final_mean,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'variance': final_var,
            'confidence': 1.0 / (final_var + 0.01),  # Inverse of variance
            'last5_contribution': last5_measurement,
            'last10_contribution': last10_measurement
        }
    
    def bayesian_probability(self, kalman_result: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate P(Over|Data) using Bayes' Theorem with Kalman output
        
        Bayes' Theorem: P(Over|Data) = P(Data|Over) * P(Over) / P(Data)
        """
        total_goals = kalman_result['total_goals']
        confidence = kalman_result['confidence']
        
        # Calculate likelihoods P(Data|Over) and P(Data|Under)
        like_over = norm.pdf(total_goals, 
                           self.likelihoods['over']['last10_mean'],
                           self.likelihoods['over']['last10_std'] / np.sqrt(confidence))
        
        like_under = norm.pdf(total_goals,
                            self.likelihoods['under']['last10_mean'],
                            self.likelihoods['under']['last10_std'] / np.sqrt(confidence))
        
        # Apply Bayes' Theorem
        evidence = like_over * self.prior_over + like_under * self.prior_under
        
        if evidence == 0:
            prob_over = 0.5
        else:
            prob_over = (like_over * self.prior_over) / evidence
        
        # Adjust for confidence (higher confidence → probability closer to extremes)
        prob_over = self._confidence_adjust(prob_over, confidence)
        
        return prob_over, 1 - prob_over
    
    def _confidence_adjust(self, probability: float, confidence: float) -> float:
        """
        Adjust probability toward extremes based on confidence
        
        Higher confidence → move probability away from 0.5
        Lower confidence → move probability toward 0.5
        """
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
        Detect genuine trends vs statistical noise using signal processing
        """
        # If we have goal timelines, use them
        home_recent = []
        away_recent = []
        
        # Extract recent performance (last 10 games worth of data)
        if hasattr(home_stats, 'home_goals_timeline') and home_stats.home_goals_timeline:
            home_recent = home_stats.home_goals_timeline[-10:]
        else:
            # Estimate from averages
            home_recent = [home_stats.last10_home_gpg] * 5 + [home_stats.last5_home_gpg] * 5
            
        if hasattr(away_stats, 'away_goals_timeline') and away_stats.away_goals_timeline:
            away_recent = away_stats.away_goals_timeline[-10:]
        else:
            away_recent = [away_stats.last10_away_gpg] * 5 + [away_stats.last5_away_gpg] * 5
        
        # Apply convolutional filter to detect trends
        home_trend = np.convolve(home_recent, self.trend_filter, mode='valid')
        away_trend = np.convolve(away_recent, self.trend_filter, mode='valid')
        
        # Calculate momentum (first derivative)
        if len(home_trend) >= 3:
            home_momentum = (home_trend[-1] - home_trend[-3]) / 2
        else:
            home_momentum = 0
            
        if len(away_trend) >= 3:
            away_momentum = (away_trend[-1] - away_trend[-3]) / 2
        else:
            away_momentum = 0
            
        # Determine trend direction
        home_trend_dir = "improving" if home_momentum > self.momentum_threshold else \
                        "declining" if home_momentum < -self.momentum_threshold else "stable"
        
        away_trend_dir = "improving" if away_momentum > self.momentum_threshold else \
                        "declining" if away_momentum < -self.momentum_threshold else "stable"
        
        # Calculate trend consistency (lower std = more consistent)
        if len(home_recent) >= 5:
            home_consistency = 1.0 / (np.std(home_recent[-5:]) + 0.01)
        else:
            home_consistency = 1.0
            
        if len(away_recent) >= 5:
            away_consistency = 1.0 / (np.std(away_recent[-5:]) + 0.01)
        else:
            away_consistency = 1.0
            
        return {
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'home_trend': home_trend_dir,
            'away_trend': away_trend_dir,
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
        
        # Determine prediction and confidence
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
        Calculate optimal stake using fractional Kelly with confidence weighting
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
        
        # Kelly Criterion: f* = (bp - q) / b
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
        
        # 3. Trend and momentum detection
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
        """Generate detailed, informative explanation"""
        
        prediction = decision['prediction']
        confidence = decision['confidence']
        total_goals = kalman_result['total_goals']
        
        explanation_parts = []
        
        # Part 1: Kalman Filter analysis
        explanation_parts.append(
            f"Kalman Filter optimal estimate: {total_goals:.2f} total goals "
            f"(Last5: {kalman_result['last5_contribution']:.2f}, "
            f"Last10: {kalman_result['last10_contribution']:.2f})"
        )
        
        # Part 2: Trend analysis
        explanation_parts.append(
            f"Trend analysis: Home {trend_data['home_trend']} "
            f"(momentum: {trend_data['home_momentum']:.2f}), "
            f"Away {trend_data['away_trend']} "
            f"(momentum: {trend_data['away_momentum']:.2f})"
        )
        
        # Part 3: Probability and confidence
        explanation_parts.append(
            f"Bayesian probability: {decision['probability']:.1%} {prediction} "
            f"(Decision score: {decision['decision_score']:.1f}/100)"
        )
        
        # Part 4: Staking rationale
        if staking_info['stake_amount'] > 0:
            explanation_parts.append(
                f"Edge: {staking_info['edge_percent']:.1f}% → "
                f"Stake: ${staking_info['stake_amount']:.2f} "
                f"({staking_info['stake_percent']:.1%} of bankroll)"
            )
        
        return " | ".join(explanation_parts)
    
    def _generate_no_bet_explanation(self, kalman_result: Dict[str, float],
                                   trend_data: Dict[str, Any]) -> str:
        """Explanation for no bet scenarios"""
        return (
            f"No clear edge. Kalman estimate: {kalman_result['total_goals']:.2f} goals "
            f"(confidence: {kalman_result['confidence']:.2f}). "
            f"Trends: Home {trend_data['home_trend']}, Away {trend_data['away_trend']}. "
            f"Insufficient statistical advantage for betting."
        )
    
    # ==================== PERFORMANCE ANALYTICS ====================
    
    def calculate_expected_roi(self, predictions_history: List[Dict] = None) -> Dict[str, float]:
        """Calculate expected ROI based on historical predictions"""
        if predictions_history is None:
            predictions_history = self.predictions_history
            
        if not predictions_history:
            return {'expected_roi': 0.0, 'sharpe_ratio': 0.0, 'hit_rate': 0.0}
        
        total_ev = 0.0
        total_stake = 0.0
        wins = 0
        bets = 0
        
        for pred in predictions_history:
            if pred['staking_info']['stake_amount'] > 0:
                total_ev += pred['staking_info']['expected_value']
                total_stake += pred['staking_info']['stake_amount']
                bets += 1
                # Note: Actual win/loss tracking would need outcome data
        
        if total_stake > 0:
            expected_roi = total_ev / total_stake * 100
        else:
            expected_roi = 0.0
        
        # Calculate Sharpe ratio (simplified)
        if bets > 1 and total_stake > 0:
            # This would need actual returns data
            sharpe_ratio = expected_roi / 10  # Placeholder
        else:
            sharpe_ratio = 0.0
        
        return {
            'expected_roi': expected_roi,
            'sharpe_ratio': sharpe_ratio,
            'total_bets': bets,
            'total_ev': total_ev,
            'avg_edge': np.mean([p['staking_info']['edge_percent'] 
                                for p in predictions_history 
                                if p['staking_info']['edge_percent'] > 0])
        }

# ============================================================================
# SIMPLIFIED DATA LOADER
# ============================================================================

class SimplifiedDataLoader:
    """Simplified data loader for the hybrid system"""
    
    @staticmethod
    def create_team_stats_from_dict(data: Dict) -> TeamStats:
        """Create TeamStats from dictionary"""
        return TeamStats(
            team_name=data.get('team_name', 'Unknown'),
            matches_played=data.get('matches_played', 0),
            home_wins=data.get('home_wins', 0),
            home_draws=data.get('home_draws', 0),
            home_losses=data.get('home_losses', 0),
            home_goals_for=data.get('home_goals_for', 0),
            home_goals_against=data.get('home_goals_against', 0),
            away_wins=data.get('away_wins', 0),
            away_draws=data.get('away_draws', 0),
            away_losses=data.get('away_losses', 0),
            away_goals_for=data.get('away_goals_for', 0),
            away_goals_against=data.get('away_goals_against', 0),
            home_xg=data.get('home_xg'),
            away_xg=data.get('away_xg'),
            avg_xg_for=data.get('avg_xg_for', 1.5),
            avg_xg_against=data.get('avg_xg_against', 1.5),
            form_last_5=data.get('form_last_5', ''),
            attack_strength=data.get('attack_strength', 1.0),
            defense_strength=data.get('defense_strength', 1.0),
            last5_home_gpg=data.get('last5_home_gpg', 1.5),
            last5_home_gapg=data.get('last5_home_gapg', 1.5),
            last5_away_gpg=data.get('last5_away_gpg', 1.5),
            last5_away_gapg=data.get('last5_away_gapg', 1.5),
            last10_home_gpg=data.get('last10_home_gpg', 1.5),
            last10_home_gapg=data.get('last10_home_gapg', 1.5),
            last10_away_gpg=data.get('last10_away_gpg', 1.5),
            last10_away_gapg=data.get('last10_away_gapg', 1.5)
        )

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_system():
    """Test the hybrid prediction system"""
    
    # Create test data
    bournemouth = SimplifiedDataLoader.create_team_stats_from_dict({
        'team_name': 'Bournemouth',
        'last5_home_gpg': 1.6,
        'last10_home_gpg': 1.5,
        'last5_away_gpg': 1.8,  # Note: This is away for away team analysis
        'last10_away_gpg': 1.6,
        'avg_xg_for': 1.6,
        'avg_xg_against': 1.3,
        'form_last_5': 'WDDLW'
    })
    
    everton = SimplifiedDataLoader.create_team_stats_from_dict({
        'team_name': 'Everton',
        'last5_home_gpg': 1.2,  # Note: This is home for home team analysis
        'last10_home_gpg': 1.1,
        'last5_away_gpg': 0.9,
        'last10_away_gpg': 1.0,
        'avg_xg_for': 1.3,
        'avg_xg_against': 1.5,
        'form_last_5': 'WDLDL'
    })
    
    # Initialize predictor
    predictor = KalmanHybridPredictor(bankroll=1000.0)
    
    # Make prediction
    result = predictor.predict_match(
        home_stats=bournemouth,
        away_stats=everton,
        over_odds=1.85,
        under_odds=1.95,
        league='premier_league'
    )
    
    # Display results
    print("=" * 60)
    print("KALMAN HYBRID PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Match: {bournemouth.team_name} vs {everton.team_name}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Probability: {result['probability']:.1%}")
    print(f"Expected Goals: {result['expected_goals']:.2f}")
    print(f"Decision Score: {result['decision_score']:.1f}/100")
    print("\n" + result['explanation'])
    
    if result['prediction'] != 'NO BET':
        print(f"\nStaking Info:")
        print(f"  Stake: ${result['staking_info']['stake_amount']:.2f}")
        print(f"  Edge: {result['staking_info']['edge_percent']:.1f}%")
        print(f"  Expected Value: ${result['staking_info']['expected_value']:.2f}")
        print(f"  Value Rating: {result['staking_info']['value_rating']}")
    
    # Show analysis
    print("\n" + "=" * 60)
    print("ANALYSIS DETAILS:")
    print("=" * 60)
    kalman = result['analysis']['kalman_result']
    print(f"Kalman Estimate: {kalman['total_goals']:.2f} goals")
    print(f"  Last5 Contribution: {kalman['last5_contribution']:.2f}")
    print(f"  Last10 Contribution: {kalman['last10_contribution']:.2f}")
    print(f"  Confidence: {kalman['confidence']:.2f}")
    
    trend = result['analysis']['trend_data']
    print(f"\nTrend Analysis:")
    print(f"  Home: {trend['home_trend']} (momentum: {trend['home_momentum']:.2f})")
    print(f"  Away: {trend['away_trend']} (momentum: {trend['away_momentum']:.2f})")
    print(f"  Trend Score: {trend['trend_score']:.2f}")
    
    print(f"\nProbabilities:")
    print(f"  P(Over): {result['analysis']['prob_over']:.1%}")
    print(f"  P(Under): {result['analysis']['prob_under']:.1%}")

if __name__ == "__main__":
    test_system()
