import numpy as np
import math
from typing import Dict, Tuple, Optional

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
    """
    
    def __init__(self, bankroll: float = 1000.0, min_confidence: float = 0.55):
        self.bankroll = bankroll
        self.min_confidence = min_confidence
        
        # League context (can be overridden per league)
        self.league_context = {
            'premier_league': {'avg_gpg': 1.47, 'avg_gapg': 1.27, 'home_advantage': 0.25},
            'la_liga': {'avg_gpg': 1.32, 'avg_gapg': 1.18, 'home_advantage': 0.28},
            'bundesliga': {'avg_gpg': 1.58, 'avg_gapg': 1.42, 'home_advantage': 0.22},
            'default': {'avg_gpg': 1.5, 'avg_gapg': 1.3, 'home_advantage': 0.25}
        }
        
        # Bayesian parameters
        self.bayesian_weight = 0.3
        self.min_games_for_stability = 8
        
        # Prediction thresholds (adjustable based on league)
        self.over_threshold = 1.5
        self.under_threshold_defense = 1.0
        self.under_threshold_attack = 1.5
        
        # xG weight (60% actual, 40% xG)
        self.xg_weight = 0.4
        
    def poisson_pmf(self, k: int, lambd: float) -> float:
        """Calculate Poisson probability mass function"""
        return (lambd ** k * math.exp(-lambd)) / math.factorial(k)
    
    def _bayesian_adjust(self, stat_value: float, n_games: int, prior_mean: float) -> float:
        """Bayesian shrinkage toward prior mean"""
        if n_games < self.min_games_for_stability:
            weight = n_games / self.min_games_for_stability
            shrunk_value = weight * stat_value + (1 - weight) * prior_mean
            return shrunk_value
        return stat_value
    
    def _calculate_form_momentum(self, last5_value: float, last10_value: float) -> Tuple[str, float]:
        """Detect if team is improving or declining"""
        if last10_value == 0:  # Avoid division by zero
            return "stable", 1.0
            
        ratio = last5_value / last10_value
        
        if ratio > 1.15:  # 15% improvement
            return "improving", 1.1  # Boost by 10%
        elif ratio < 0.85:  # 15% decline
            return "declining", 0.9  # Reduce by 10%
        return "stable", 1.0
    
    def _extract_and_process_stats(self, home_stats: Dict, away_stats: Dict, league: str = "default") -> Dict:
        """Extract and process all relevant statistics with Bayesian adjustments"""
        context = self.league_context.get(league, self.league_context['default'])
        
        stats = {}
        
        # Extract basic stats
        stats['home_last5_gpg'] = home_stats.get('last5_home_gpg', home_stats.get('home_gpg', 0))
        stats['home_last5_gapg'] = home_stats.get('last5_home_gapg', home_stats.get('home_gapg', 0))
        stats['home_last10_gpg'] = home_stats.get('last10_home_gpg', stats['home_last5_gpg'])
        stats['home_last10_gapg'] = home_stats.get('last10_home_gapg', stats['home_last5_gapg'])
        
        stats['away_last5_gpg'] = away_stats.get('last5_away_gpg', away_stats.get('away_gpg', 0))
        stats['away_last5_gapg'] = away_stats.get('last5_away_gapg', away_stats.get('away_gapg', 0))
        stats['away_last10_gpg'] = away_stats.get('last10_away_gpg', stats['away_last5_gpg'])
        stats['away_last10_gapg'] = away_stats.get('last10_away_gapg', stats['away_last5_gapg'])
        
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
            stats['away_last10_gpg'], 10, context['avg_gpg']
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
        home_gpg_last10 = home_stats.get('gpg_last10', 0)
        away_gpg_last10 = away_stats.get('gpg_last10', 0)
        home_gapg_last10 = home_stats.get('gapg_last10', 0)
        away_gapg_last10 = away_stats.get('gapg_last10', 0)
        
        stats['home_hybrid_gpg'] = (1 - self.xg_weight) * home_gpg_last10 + self.xg_weight * home_stats.get('avg_xg_for', home_gpg_last10)
        stats['home_hybrid_gapg'] = (1 - self.xg_weight) * home_gapg_last10 + self.xg_weight * home_stats.get('avg_xg_against', home_gapg_last10)
        stats['away_hybrid_gpg'] = (1 - self.xg_weight) * away_gpg_last10 + self.xg_weight * away_stats.get('avg_xg_for', away_gpg_last10)
        stats['away_hybrid_gapg'] = (1 - self.xg_weight) * away_gapg_last10 + self.xg_weight * away_stats.get('avg_xg_against', away_gapg_last10)
        
        # Combined final stats (40% last10_adj, 40% last5_adj, 20% hybrid)
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
        Enhanced 5-rule system with Bayesian adjustments and momentum
        Returns: (prediction, confidence, rule_number, confidence_boost)
        """
        
        # Rule 1: High Confidence Over (Both periods with momentum)
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
        
        # Rule 2: High Confidence Under (Strong defense vs weak attack)
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
        
        # Rule 3: Moderate Confidence Over (Last 5 only with momentum)
        rule3_condition = (
            stats['home_last5_gpg_adj'] > self.over_threshold and
            stats['away_last5_gpg_adj'] > self.over_threshold
        )
        
        if rule3_condition:
            momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                     stats['away_momentum'] == "improving") else 0
            return "Over 2.5", "Moderate", 3, momentum_bonus
        
        # Rule 4: Moderate Confidence Under (Last 5 only with momentum)
        rule4_condition = (
            stats['home_last5_gapg_adj'] < self.under_threshold_defense and
            stats['away_last5_gpg_adj'] < self.under_threshold_attack
        )
        
        if rule4_condition:
            momentum_bonus = 0.03 if (stats['home_momentum'] == "improving" or 
                                     stats['away_momentum'] == "declining") else 0
            return "Under 2.5", "Moderate", 4, momentum_bonus
        
        # Rule 5: Check xG edge before giving up
        xg_advantage = (stats['home_hybrid_gpg'] + stats['away_hybrid_gpg']) / 2
        
        if xg_advantage > 1.8:  # High xG expectation
            return "Over 2.5", "Low", 5, 0
        elif xg_advantage < 1.2:  # Low xG expectation
            return "Under 2.5", "Low", 5, 0
        
        return "No Bet", "None", 5, 0
    
    def _calculate_poisson_probabilities(self, stats: Dict) -> Tuple[float, float, float, Dict]:
        """Enhanced Poisson calculation with all adjustments"""
        
        context = stats['league_context']
        
        # Calculate expected goals with home advantage
        lambda_home = (stats['home_attack_final'] * 
                      (stats['away_defense_final'] / context['avg_gapg']) * 
                      (1 + context['home_advantage']))
        
        lambda_away = (stats['away_attack_final'] * 
                      (stats['home_defense_final'] / context['avg_gapg']))
        
        expected_goals = lambda_home + lambda_away
        
        # Calculate Poisson probabilities
        total_lambda = lambda_home + lambda_away
        
        # Probability of 0, 1, or 2 goals (Under 2.5)
        prob_under = sum(self.poisson_pmf(k, total_lambda) for k in range(3))
        prob_over = 1 - prob_under
        
        return expected_goals, prob_over, prob_under, {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'home_momentum': stats['home_momentum'],
            'away_momentum': stats['away_momentum'],
            'expected_home_goals': lambda_home,
            'expected_away_goals': lambda_away
        }
    
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
                'edge_percent': 0.0
            }
        
        implied_prob = 1 / odds
        edge = probability - implied_prob
        
        if edge <= 0:
            return {
                'stake_amount': 0.0,
                'stake_percent': 0.0,
                'kelly_fraction': 0.0,
                'expected_value': 0.0,
                'risk_level': 'No Value',
                'edge_percent': edge * 100
            }
        
        # Kelly formula
        b = odds - 1
        p = probability
        q = 1 - p
        
        if b <= 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly (25%) and confidence weighting
        kelly_fraction = max(0, kelly_fraction)  # Ensure non-negative
        fractional_kelly = kelly_fraction * 0.25 * confidence_value
        
        # Calculate stake
        stake_amount = fractional_kelly * bankroll
        
        # Cap at reasonable levels (max 5% of bankroll)
        max_stake = bankroll * 0.05
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
        
        return {
            'stake_amount': stake_amount,
            'stake_percent': stake_amount / bankroll,
            'kelly_fraction': fractional_kelly,
            'expected_value': expected_value,
            'risk_level': risk_level,
            'edge_percent': edge * 100,
            'implied_probability': implied_prob,
            'value_rating': 'Excellent' if edge > 0.1 else 'Good' if edge > 0.05 else 'Fair' if edge > 0.02 else 'Poor'
        }
    
    def predict_with_staking(self, home_stats: Dict, away_stats: Dict, 
                            market_odds: Dict, league: str = "default", 
                            bankroll: float = None) -> Dict:
        """
        Complete prediction with staking recommendations
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            market_odds: Dictionary with 'over_25' and/or 'under_25' odds
            league: League name for context
            bankroll: Current bankroll (overrides instance bankroll)
        
        Returns:
            Complete prediction dictionary with staking info
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        # Process stats
        stats = self._extract_and_process_stats(home_stats, away_stats, league)
        
        # Apply rules
        prediction, confidence, rule_number, confidence_boost = self._apply_five_rules(stats)
        
        # Calculate Poisson probabilities
        expected_goals, prob_over, prob_under, poisson_details = self._calculate_poisson_probabilities(stats)
        
        # Determine final probability
        if prediction == "Over 2.5":
            base_probability = prob_over
        elif prediction == "Under 2.5":
            base_probability = prob_under
        else:
            base_probability = 0.5
        
        # Apply confidence boost
        final_probability = min(0.95, base_probability + confidence_boost)
        
        # Get appropriate market odds
        if prediction == "Over 2.5":
            market_odd = market_odds.get('over_25', 1.85)
        elif prediction == "Under 2.5":
            market_odd = market_odds.get('under_25', 1.95)
        else:
            market_odd = 2.0  # Default if no bet
        
        # Calculate stake
        staking_info = self.calculate_kelly_stake(
            probability=final_probability,
            odds=market_odd,
            confidence=confidence,
            bankroll=bankroll
        )
        
        # Generate detailed explanation
        explanation = self._generate_explanation(
            prediction, confidence, rule_number, stats, 
            final_probability, staking_info['edge_percent']
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
                'away_momentum': stats['away_momentum'],
                'xg_advantage': (stats['home_hybrid_gpg'] + stats['away_hybrid_gpg']) / 2
            }
        }
    
    def _generate_explanation(self, prediction: str, confidence: str, rule_number: int, 
                            stats: Dict, probability: float, edge: float) -> str:
        """Generate detailed explanation for the prediction"""
        
        explanations = {
            1: f"High Confidence {prediction}: Both teams average >1.5 goals per game in adjusted last 10 AND last 5 metrics. "
                f"Home: {stats['home_last10_gpg_adj']:.2f} → {stats['home_last5_gpg_adj']:.2f} GPG ({stats['home_momentum']}). "
                f"Away: {stats['away_last10_gpg_adj']:.2f} → {stats['away_last5_gpg_adj']:.2f} GPG ({stats['away_momentum']}). "
                f"xG hybrid: {(stats['home_hybrid_gpg'] + stats['away_hybrid_gpg'])/2:.2f}. Edge: {edge:.1f}%.",
            
            2: f"High Confidence {prediction}: Strong adjusted defense (<1.0 GApg) vs weak adjusted attack (<1.5 GPG) in both periods. "
                f"Home defense: {stats['home_last10_gapg_adj']:.2f} → {stats['home_last5_gapg_adj']:.2f} GApg ({stats['home_momentum']}). "
                f"Away attack: {stats['away_last10_gpg_adj']:.2f} → {stats['away_last5_gpg_adj']:.2f} GPG ({stats['away_momentum']}). "
                f"Expected goals: {stats['home_attack_final']:.2f} vs {stats['away_attack_final']:.2f}. Edge: {edge:.1f}%.",
            
            3: f"Moderate Confidence {prediction}: Strong attacking form in last 5 matches (>1.5 GPG). "
                f"Home: {stats['home_last5_gpg_adj']:.2f} GPG ({stats['home_momentum']}). "
                f"Away: {stats['away_last5_gpg_adj']:.2f} GPG ({stats['away_momentum']}). "
                f"Poisson probability: {probability:.1%}. Edge: {edge:.1f}%.",
            
            4: f"Moderate Confidence {prediction}: Recent defensive strength vs attacking weakness. "
                f"Home defense (last5): {stats['home_last5_gapg_adj']:.2f} GApg. "
                f"Away attack (last5): {stats['away_last5_gpg_adj']:.2f} GPG. "
                f"Momentum: Home {stats['home_momentum']}, Away {stats['away_momentum']}. Edge: {edge:.1f}%.",
            
            5: f"Low Confidence {prediction}: xG-based prediction with slight statistical edge. "
                f"xG advantage: {(stats['home_hybrid_gpg'] + stats['away_hybrid_gpg'])/2:.2f}. "
                f"Expected total goals: {stats['home_attack_final'] + stats['away_attack_final']:.2f}. "
                f"Poisson probability: {probability:.1%}. Edge: {edge:.1f}%."
        }
        
        if rule_number == 5 and prediction == "No Bet":
            return (f"No clear statistical edge. Bayesian adjusted metrics don't meet criteria. "
                   f"Home: {stats['home_attack_final']:.2f} attack, {stats['home_defense_final']:.2f} defense. "
                   f"Away: {stats['away_attack_final']:.2f} attack, {stats['away_defense_final']:.2f} defense. "
                   f"xG hybrid: {(stats['home_hybrid_gpg'] + stats['away_hybrid_gpg'])/2:.2f}.")
        
        return explanations.get(rule_number, f"{confidence} confidence {prediction} based on comprehensive statistical analysis.")
    
    def batch_predict(self, matches: list, market_odds_list: list, league: str = "default") -> list:
        """Predict multiple matches at once"""
        predictions = []
        
        for i, match in enumerate(matches):
            home_stats, away_stats = match
            market_odds = market_odds_list[i] if i < len(market_odds_list) else {'over_25': 1.85, 'under_25': 1.95}
            
            prediction = self.predict_with_staking(
                home_stats, away_stats, market_odds, league
            )
            predictions.append(prediction)
        
        return predictions
    
    def analyze_bankroll_growth(self, predictions: list, initial_bankroll: float = 1000.0) -> Dict:
        """Simulate bankroll growth based on predictions"""
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        bet_history = []
        
        for pred in predictions:
            if pred['prediction'] != "No Bet" and pred['staking_info']['stake_amount'] > 0:
                stake = pred['staking_info']['stake_amount']
                odds = pred['market_odds']
                
                # Simulate outcome (for testing, you'd use actual results)
                # Here we use the probability to determine outcome
                import random
                if random.random() < pred['probability']:
                    # Win
                    profit = stake * (odds - 1)
                    bankroll += profit
                    outcome = "WIN"
                else:
                    # Lose
                    bankroll -= stake
                    outcome = "LOSE"
                
                bet_history.append({
                    'prediction': pred['prediction'],
                    'stake': stake,
                    'odds': odds,
                    'probability': pred['probability'],
                    'outcome': outcome,
                    'profit': stake * (odds - 1) if outcome == "WIN" else -stake
                })
                bankroll_history.append(bankroll)
        
        return {
            'final_bankroll': bankroll,
            'total_bets': len(bet_history),
            'total_profit': bankroll - initial_bankroll,
            'roi': ((bankroll - initial_bankroll) / initial_bankroll) * 100,
            'bankroll_history': bankroll_history,
            'bet_history': bet_history
        }
