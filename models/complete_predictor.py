import numpy as np
import math
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class TeamStats:
    """Container for team statistics from CSV"""
    team_name: str
    matches_played: int
    home_wins: int
    home_draws: int
    home_losses: int
    home_goals_for: float
    home_goals_against: float
    away_wins: int
    away_draws: int
    away_losses: int
    away_goals_for: float
    away_goals_against: float
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
    """
    
    def __init__(self, bankroll: float = 1000.0, min_confidence: float = 0.50):
        self.bankroll = bankroll
        self.min_confidence = min_confidence
        
        # League context (can be overridden per league)
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
        
        # Prediction thresholds (adjustable based on league)
        self.over_threshold = 1.5  # Goals per game
        self.under_threshold_defense = 1.0  # Goals against per game
        self.under_threshold_attack = 1.5  # Goals per game
        
        # xG weight (60% actual, 40% xG)
        self.xg_weight = 0.4
        
        # Template-based explanations
        self._init_explanation_templates()
        
    def _init_explanation_templates(self):
        """Initialize explanation templates for cleaner code"""
        self.explanation_templates = {
            1: (
                "High Confidence {prediction}: Both teams average >1.5 goals per game in adjusted "
                "last 10 AND last 5 metrics. Home: {home_last10_gpg_adj:.2f} → {home_last5_gpg_adj:.2f} GPG "
                "({home_momentum}). Away: {away_last10_gpg_adj:.2f} → {away_last5_gpg_adj:.2f} GPG ({away_momentum}). "
                "xG hybrid: {xg_avg:.2f}. Edge: {edge:.1f}%."
            ),
            2: (
                "High Confidence {prediction}: Strong adjusted defense (<1.0 GApg) vs weak adjusted attack "
                "(<1.5 GPG) in both periods. Home defense: {home_last10_gapg_adj:.2f} → {home_last5_gapg_adj:.2f} GApg "
                "({home_momentum}). Away attack: {away_last10_gpg_adj:.2f} → {away_last5_gpg_adj:.2f} GPG "
                "({away_momentum}). Expected goals: {home_attack_final:.2f} vs {away_attack_final:.2f}. "
                "Edge: {edge:.1f}%."
            ),
            3: (
                "Moderate Confidence {prediction}: Strong attacking form in last 5 matches (>1.5 GPG). "
                "Home: {home_last5_gpg_adj:.2f} GPG ({home_momentum}). "
                "Away: {away_last5_gpg_adj:.2f} GPG ({away_momentum}). "
                "Poisson probability: {probability:.1%}. Edge: {edge:.1f}%."
            ),
            4: (
                "Moderate Confidence {prediction}: Recent defensive strength vs attacking weakness. "
                "Home defense (last5): {home_last5_gapg_adj:.2f} GApg. "
                "Away attack (last5): {away_last5_gpg_adj:.2f} GPG. "
                "Momentum: Home {home_momentum}, Away {away_momentum}. Edge: {edge:.1f}%."
            ),
            5: (
                "Low Confidence {prediction}: xG-based prediction with slight statistical edge. "
                "xG advantage: {xg_avg:.2f}. "
                "Expected total goals: {total_goals:.2f}. "
                "Poisson probability: {probability:.1%}. Edge: {edge:.1f}%."
            )
        }
    
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
    
    def _prepare_stats_for_prediction(self, home_stats: TeamStats, away_stats: TeamStats, league: str = "default") -> Dict:
        """Prepare statistics for prediction from TeamStats objects"""
        context = self.league_context.get(league, self.league_context['default'])
        
        # Extract and prepare stats in the format expected by the predictor
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
            stats['away_last10_gapg'], 10, context['avg_gapg']
        )
        
        # Apply momentum to last 5 stats
        stats['home_last5_gpg_adj'] = stats['home_last5_gpg'] * stats['home_momentum_mult']
        stats['home_last5_gapg_adj'] = stats['home_last5_gapg'] * (1/stats['home_momentum_mult'] 
                                                                  if stats['home_momentum'] == "improving" 
                                                                  else stats['home_momentum_mult'])
        stats['away_last5_gpg_adj'] = stats['away_last5_gpg'] * stats['away_momentum_mult']
        stats['away_last5_gapg_adj'] = stats['away_last5_gpg'] * (1/stats['away_momentum_mult'] 
                                                                  if stats['away_momentum'] == "improving" 
                                                                  else stats['away_momentum_mult'])
        
        # xG hybrid metrics (60% actual goals, 40% xG)
        stats['home_hybrid_gpg'] = (0.6 * stats['home_last10_gpg_adj'] + 0.4 * stats['home_xg_for'])
        stats['home_hybrid_gapg'] = (0.6 * stats['home_last10_gapg_adj'] + 0.4 * stats['home_xg_against'])
        stats['away_hybrid_gpg'] = (0.6 * stats['away_last10_gpg_adj'] + 0.4 * stats['away_xg_for'])
        stats['away_hybrid_gapg'] = (0.6 * stats['away_last10_gapg_adj'] + 0.4 * stats['away_xg_against'])
        
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
        
        # Prepare stats
        stats = self._prepare_stats_for_prediction(home_stats, away_stats, league)
        
        # Apply rules
        prediction, confidence, rule_number, confidence_boost = self._apply_five_rules(stats)
        
        # Calculate Poisson probabilities
        expected_goals, prob_over, prob_under, poisson_details = self._calculate_poisson_probabilities(stats)
        
        # Determine final probability
        if prediction == "Over 2.5":
            base_probability = prob_over
            market_odd = over_odds
        elif prediction == "Under 2.5":
            base_probability = prob_under
            market_odd = under_odds
        else:
            base_probability = 0.5
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
        
        if rule_number == 5 and prediction == "No Bet":
            return (
                f"No clear statistical edge. Bayesian adjusted metrics don't meet criteria. "
                f"Home: {stats['home_attack_final']:.2f} attack, {stats['home_defense_final']:.2f} defense. "
                f"Away: {stats['away_attack_final']:.2f} attack, {stats['away_defense_final']:.2f} defense. "
                f"xG hybrid: {(stats['home_hybrid_gpg'] + stats['away_hybrid_gpg'])/2:.2f}."
            )
        
        # Prepare template variables
        template_vars = {
            'prediction': prediction,
            'home_last10_gpg_adj': stats['home_last10_gpg_adj'],
            'home_last5_gpg_adj': stats['home_last5_gpg_adj'],
            'home_momentum': stats['home_momentum'],
            'away_last10_gpg_adj': stats['away_last10_gpg_adj'],
            'away_last5_gpg_adj': stats['away_last5_gpg_adj'],
            'away_momentum': stats['away_momentum'],
            'xg_avg': (stats['home_hybrid_gpg'] + stats['away_hybrid_gpg']) / 2,
            'edge': edge,
            'home_last10_gapg_adj': stats['home_last10_gapg_adj'],
            'home_last5_gapg_adj': stats['home_last5_gapg_adj'],
            'home_attack_final': stats['home_attack_final'],
            'away_attack_final': stats['away_attack_final'],
            'probability': probability,
            'total_goals': stats['home_attack_final'] + stats['away_attack_final']
        }
        
        template = self.explanation_templates.get(rule_number)
        if template:
            return template.format(**template_vars)
        
        return f"{confidence} confidence {prediction} based on comprehensive statistical analysis."

# ============================================================================
# DATA LOADER AND PROCESSOR
# ============================================================================

class FootballDataLoader:
    """Loads and processes football data from CSV files"""
    
    @staticmethod
    def load_teams_from_csv(csv_path: str) -> Dict[str, TeamStats]:
        """Load team statistics from CSV file"""
        df = pd.read_csv(csv_path)
        teams = {}
        
        for _, row in df.iterrows():
            stats = TeamStats(
                team_name=row['team_name'],
                matches_played=int(row['matches_played']),
                home_wins=int(row['home_wins']),
                home_draws=int(row['home_draws']),
                home_losses=int(row['home_losses']),
                home_goals_for=float(row['home_goals_for']),
                home_goals_against=float(row['home_goals_against']),
                away_wins=int(row['away_wins']),
                away_draws=int(row['away_draws']),
                away_losses=int(row['away_losses']),
                away_goals_for=float(row['away_goals_for']),
                away_goals_against=float(row['away_goals_against']),
                home_xg=float(row['home_xg']) if pd.notna(row.get('home_xg', np.nan)) else None,
                away_xg=float(row['away_xg']) if pd.notna(row.get('away_xg', np.nan)) else None,
                avg_xg_for=float(row['avg_xg_for']),
                avg_xg_against=float(row['avg_xg_against']),
                form_last_5=row['form_last_5'],
                attack_strength=float(row['attack_strength']),
                defense_strength=float(row['defense_strength']),
                last5_home_gpg=float(row['last5_home_gpg']),
                last5_home_gapg=float(row['last5_home_gapg']),
                last5_away_gpg=float(row['last5_away_gpg']),
                last5_away_gapg=float(row['last5_away_gapg']),
                last10_home_gpg=float(row['last10_home_gpg']),
                last10_home_gapg=float(row['last10_home_gapg']),
                last10_away_gpg=float(row['last10_away_gpg']),
                last10_away_gapg=float(row['last10_away_gapg'])
            )
            teams[row['team_name']] = stats
        
        return teams

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Example usage"""
    # Initialize predictor
    predictor = CompletePhantomPredictor(bankroll=1000.0, min_confidence=0.50)
    
    # Example: Load data
    # data_loader = FootballDataLoader()
    # premier_league_teams = data_loader.load_teams_from_csv("premier_league_teams.csv")
    
    # Example stats (simulated)
    bournemouth_stats = TeamStats(
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
        last5_home_gpg=1.4, last5_home_gapg=0.8,
        last5_away_gpg=1.6, last5_away_gapg=3.0,
        last10_home_gpg=1.43, last10_home_gapg=0.71,
        last10_away_gpg=1.57, last10_away_gapg=2.71
    )
    
    everton_stats = TeamStats(
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
        last5_away_gpg=0.8, last5_away_gapg=1.0,
        last10_home_gpg=1.14, last10_home_gapg=1.29,
        last10_away_gpg=1.0, last10_away_gapg=1.14
    )
    
    # Make prediction
    prediction = predictor.predict_match(
        home_stats=bournemouth_stats,
        away_stats=everton_stats,
        over_odds=1.85,
        under_odds=1.95,
        league="premier_league",
        bankroll=1000.0
    )
    
    # Print results
    print(f"Prediction: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Probability: {prediction['probability']:.1%}")
    print(f"Expected Goals: {prediction['expected_goals']:.2f}")
    print(f"\nExplanation: {prediction['explanation']}")
    print(f"\nStaking Info:")
    print(f"  Stake: ${prediction['staking_info']['stake_amount']:.2f}")
    print(f"  Edge: {prediction['staking_info']['edge_percent']:.1f}%")
    print(f"  Expected Value: ${prediction['staking_info']['expected_value']:.2f}")
    print(f"  Risk Level: {prediction['staking_info']['risk_level']}")

if __name__ == "__main__":
    main()
