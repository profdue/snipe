import numpy as np
import math
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass
from enum import Enum

class BetType(Enum):
    """Bet types supported by the system"""
    OVER_25 = "Over 2.5"
    UNDER_25 = "Under 2.5"
    BTTS_YES = "BTTS Yes"
    BTTS_NO = "BTTS No"
    HOME_WIN = "Home Win"
    AWAY_WIN = "Away Win"
    DRAW = "Draw"
    HOME_DOUBLE_CHANCE = "Home or Draw"
    AWAY_DOUBLE_CHANCE = "Away or Draw"
    NO_BET = "No Bet"

class TeamStyle(Enum):
    """Playing style classifications"""
    POSSESSION = "Possession"
    COUNTER = "Counter"
    HIGH_PRESS = "High Press"
    LOW_BLOCK = "Low Block"
    BALANCED = "Balanced"

@dataclass
class EnhancedTeamStats:
    """Container for enhanced team statistics from CSV"""
    # Identity Metrics
    team_name: str
    matches_played: int
    possession_avg: float
    shots_per_game: float
    shots_on_target_pg: float
    conversion_rate: float  # As decimal (0.12 for 12%)
    xg_for_avg: float
    xg_against_avg: float
    
    # Home/Away Split
    home_wins: int
    home_draws: int
    home_losses: int
    away_wins: int
    away_draws: int
    away_losses: int
    home_goals_for: int
    home_goals_against: int
    away_goals_for: int
    away_goals_against: int
    
    # Defense Patterns
    clean_sheet_pct: float  # As decimal
    clean_sheet_pct_home: float  # As decimal
    clean_sheet_pct_away: float  # As decimal
    failed_to_score_pct: float  # As decimal
    failed_to_score_pct_home: float  # As decimal
    failed_to_score_pct_away: float  # As decimal
    
    # Transition Patterns
    btts_pct: float  # As decimal
    btts_pct_home: float  # As decimal
    btts_pct_away: float  # As decimal
    over25_pct: float  # As decimal
    over25_pct_home: float  # As decimal
    over25_pct_away: float  # As decimal
    
    # Recent Form
    last5_form: str
    last5_wins: int
    last5_draws: int
    last5_losses: int
    last5_goals_for: int
    last5_goals_against: int
    
    def __post_init__(self):
        """Handle percentage strings and other data cleaning"""
        # Convert percentage strings to proper decimal values
        percentage_fields = [
            'possession_avg', 'conversion_rate', 
            'clean_sheet_pct', 'clean_sheet_pct_home', 'clean_sheet_pct_away',
            'failed_to_score_pct', 'failed_to_score_pct_home', 'failed_to_score_pct_away',
            'btts_pct', 'btts_pct_home', 'btts_pct_away',
            'over25_pct', 'over25_pct_home', 'over25_pct_away'
        ]
        
        for field in percentage_fields:
            value = getattr(self, field)
            if isinstance(value, str):
                # Remove % sign and convert to proper decimal
                try:
                    cleaned = value.replace('%', '').strip()
                    if cleaned:
                        # Convert percentage to decimal (e.g., "6%" -> 0.06)
                        setattr(self, field, float(cleaned) / 100.0)
                    else:
                        setattr(self, field, 0.0)
                except (ValueError, AttributeError):
                    setattr(self, field, 0.0)
            elif isinstance(value, (int, float)):
                # If it's already a number, check if it's > 1 (likely a percentage)
                if value > 1.0 and field not in ['possession_avg']:  # possession can be > 100 in some systems
                    # Assume it's a percentage (e.g., 6.0 means 6%)
                    setattr(self, field, value / 100.0)
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                setattr(self, field, 0.0)
        
        # Handle possession separately (should be between 0-1)
        if self.possession_avg > 1.0:
            self.possession_avg = self.possession_avg / 100.0
        
        # Handle form string
        if not self.last5_form or (isinstance(self.last5_form, float) and np.isnan(self.last5_form)):
            self.last5_form = ""
        
        # Ensure numeric fields are proper types
        int_fields = [
            'matches_played', 'home_wins', 'home_draws', 'home_losses',
            'away_wins', 'away_draws', 'away_losses', 'home_goals_for',
            'home_goals_against', 'away_goals_for', 'away_goals_against',
            'last5_wins', 'last5_draws', 'last5_losses', 'last5_goals_for',
            'last5_goals_against'
        ]
        
        for field in int_fields:
            value = getattr(self, field)
            if isinstance(value, str):
                try:
                    setattr(self, field, int(float(value)))
                except:
                    setattr(self, field, 0)
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                setattr(self, field, 0)
        
        # Ensure float fields are proper types
        float_fields = [
            'shots_per_game', 'shots_on_target_pg', 'xg_for_avg', 'xg_against_avg'
        ]
        
        for field in float_fields:
            value = getattr(self, field)
            if isinstance(value, str):
                try:
                    setattr(self, field, float(value))
                except:
                    setattr(self, field, 0.0)
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                setattr(self, field, 0.0)
    
    @property
    def points_per_game(self) -> float:
        total_points = (self.home_wins + self.away_wins) * 3 + (self.home_draws + self.away_draws)
        return total_points / self.matches_played if self.matches_played > 0 else 0
    
    @property
    def style(self) -> TeamStyle:
        """Determine team playing style based on metrics"""
        # Convert possession to percentage for classification
        possession_pct = self.possession_avg * 100 if self.possession_avg <= 1.0 else self.possession_avg
        
        if possession_pct >= 55:
            return TeamStyle.POSSESSION
        elif possession_pct <= 45:
            return TeamStyle.COUNTER
        elif self.shots_per_game >= 14:
            return TeamStyle.HIGH_PRESS
        elif self.shots_per_game <= 9:
            return TeamStyle.LOW_BLOCK
        else:
            return TeamStyle.BALANCED
    
    @property
    def attack_efficiency(self) -> float:
        """Goals per shot on target"""
        if self.shots_on_target_pg > 0:
            total_goals = self.home_goals_for + self.away_goals_for
            total_shots_on_target = self.matches_played * self.shots_on_target_pg
            return total_goals / total_shots_on_target if total_shots_on_target > 0 else 0.0
        return 0.0
    
    @property
    def defensive_efficiency(self) -> float:
        """Simplified defensive efficiency metric"""
        goals_conceded = self.home_goals_against + self.away_goals_against
        if goals_conceded > 0:
            return self.matches_played / goals_conceded
        return 10.0  # Arbitrary high value for perfect defense


class EdgeFinderPredictor:
    """
    v1.1 Football Predictor based on "3 Things" Framework:
    1. Team Identity (What they ARE)
    2. Defense (What they STOP)
    3. Transition (How they CHANGE)
    """
    
    def __init__(self, bankroll: float = 1000.0, min_edge: float = 0.03, 
                 max_correlation_exposure: float = 0.10):
        self.bankroll = bankroll
        self.min_edge = min_edge  # Minimum edge percentage (3%)
        self.max_correlation_exposure = max_correlation_exposure
        
        # League context with possession benchmarks
        self.league_context = {
            'premier_league': {
                'avg_gpg': 2.7,  # Fixed: Real Premier League average
                'avg_xg_for': 1.4,  # Average xG per team
                'avg_xg_against': 1.4,  # Average xG conceded
                'home_advantage': 0.15,
                'avg_possession': 0.50,  # As decimal
                'avg_conversion': 0.11  # As decimal
            },
            'la_liga': {
                'avg_gpg': 2.6,
                'avg_xg_for': 1.35,
                'avg_xg_against': 1.35,
                'home_advantage': 0.18,
                'avg_possession': 0.52,
                'avg_conversion': 0.105
            },
            'bundesliga': {
                'avg_gpg': 3.1,
                'avg_xg_for': 1.55,
                'avg_xg_against': 1.55,
                'home_advantage': 0.12,
                'avg_possession': 0.51,
                'avg_conversion': 0.125
            },
            'serie_a': {
                'avg_gpg': 2.5,
                'avg_xg_for': 1.25,
                'avg_xg_against': 1.25,
                'home_advantage': 0.20,
                'avg_possession': 0.49,
                'avg_conversion': 0.095
            },
            'ligue_1': {
                'avg_gpg': 2.7,
                'avg_xg_for': 1.35,
                'avg_xg_against': 1.35,
                'home_advantage': 0.16,
                'avg_possession': 0.50,
                'avg_conversion': 0.108
            },
            'default': {
                'avg_gpg': 2.7,
                'avg_xg_for': 1.4,
                'avg_xg_against': 1.4,
                'home_advantage': 0.16,
                'avg_possession': 0.50,
                'avg_conversion': 0.11
            }
        }
        
        # Style matchup adjustments
        self.style_adjustments = {
            (TeamStyle.POSSESSION, TeamStyle.LOW_BLOCK): {
                'home_goals_mult': 0.9,  # Reduced effect
                'away_goals_mult': 1.0,
                'under_bias': 0.05,      # Reduced bias
                'draw_bias': 0.05
            },
            (TeamStyle.COUNTER, TeamStyle.HIGH_PRESS): {
                'home_goals_mult': 1.1,
                'away_goals_mult': 1.0,
                'over_bias': 0.05,
                'btts_bias': 0.05
            },
            (TeamStyle.HIGH_PRESS, TeamStyle.HIGH_PRESS): {
                'home_goals_mult': 1.05,
                'away_goals_mult': 1.05,
                'over_bias': 0.10,
                'btts_bias': 0.08
            },
        }
        
        # Default style adjustment
        self.default_style_adjustment = {
            'home_goals_mult': 1.0,
            'away_goals_mult': 1.0,
            'over_bias': 0.0,
            'under_bias': 0.0,
            'btts_bias': 0.0,
            'draw_bias': 0.0
        }
        
        # Betting parameters
        self.max_stake_pct = 0.03  # Maximum 3% per bet
        self.min_confidence_for_stake = 0.55  # Minimum probability for betting
        
        # Initialize explanation system
        self._init_explanation_templates()
        
        # Track correlated bets
        self.correlated_bets = []
    
    def _init_explanation_templates(self):
        """Initialize explanation templates for different insights"""
        self.explanation_templates = {
            'style_matchup': "🎯 {style_clash}: {home_style} ({home_possession:.0f}%) vs {away_style} ({away_possession:.0f}%). {insight}",
            'efficiency_mismatch': "⚡ Efficiency Mismatch: {team1} ({conv1:.1f}% conversion) vs {team2} ({conv2:.1f}% conversion). {insight}",
            'defensive_pattern': "🛡️ Defensive Pattern: {team} has {clean_sheets:.1f}% clean sheets, fails to score {failed:.1f}%. {insight}",
            'transition_trend': "📈 Transition Trend: {team} BTTS {btts:.1f}%, Over 2.5 {over25:.1f}%. {insight}",
            'form_momentum': "📊 Form Momentum: {team} last 5: {form}. {insight}",
            'value_finding': "💰 Value Found: Model: {model_prob:.1%} vs Market: {market_prob:.1%} = {edge:.1f}% edge.",
            'no_edge': "❌ No Edge: Model: {model_prob:.1%} vs Market: {market_prob:.1%} = {edge:.1f}% edge."
        }
    
    def analyze_team_identity(self, home_stats: EnhancedTeamStats, 
                            away_stats: EnhancedTeamStats) -> Dict:
        """Analyze the Team Identity dimension"""
        # Convert to percentages for display
        home_possession_pct = home_stats.possession_avg * 100
        away_possession_pct = away_stats.possession_avg * 100
        home_conversion_pct = home_stats.conversion_rate * 100
        away_conversion_pct = away_stats.conversion_rate * 100
        
        analysis = {
            'style_clash': None,
            'possession_difference': home_possession_pct - away_possession_pct,
            'conversion_comparison': home_conversion_pct - away_conversion_pct,
            'shots_difference': home_stats.shots_per_game - away_stats.shots_per_game,
            'efficiency_ratio': home_stats.conversion_rate / away_stats.conversion_rate 
                                if away_stats.conversion_rate > 0 else 1.0,
            'insights': []
        }
        
        # Determine style clash
        home_style = home_stats.style
        away_style = away_stats.style
        analysis['style_clash'] = f"{home_style.value} vs {away_style.value}"
        
        # Generate insights
        if analysis['possession_difference'] > 10:
            analysis['insights'].append(
                f"Home team dominates possession ({home_possession_pct:.1f}% vs {away_possession_pct:.1f}%)"
            )
        elif analysis['possession_difference'] < -10:
            analysis['insights'].append(
                f"Away team dominates possession ({away_possession_pct:.1f}% vs {home_possession_pct:.1f}%)"
            )
        
        if analysis['conversion_comparison'] > 3:
            analysis['insights'].append(
                f"Home team more efficient ({home_conversion_pct:.1f}% vs {away_conversion_pct:.1f}% conversion)"
            )
        elif analysis['conversion_comparison'] < -3:
            analysis['insights'].append(
                f"Away team more efficient ({away_conversion_pct:.1f}% vs {home_conversion_pct:.1f}% conversion)"
            )
        
        return analysis
    
    def analyze_defense_patterns(self, home_stats: EnhancedTeamStats,
                               away_stats: EnhancedTeamStats) -> Dict:
        """Analyze the Defense dimension"""
        # Convert to percentages for display
        home_cs_pct = home_stats.clean_sheet_pct_home * 100
        away_cs_pct = away_stats.clean_sheet_pct_away * 100
        home_failed_pct = home_stats.failed_to_score_pct_home * 100
        away_failed_pct = away_stats.failed_to_score_pct_away * 100
        
        analysis = {
            'home_clean_sheet_strength': home_cs_pct,
            'away_clean_sheet_strength': away_cs_pct,
            'home_scoring_reliability': 100 - home_failed_pct,
            'away_scoring_reliability': 100 - away_failed_pct,
            'defensive_mismatch': None,
            'insights': []
        }
        
        # Determine defensive mismatch
        if home_cs_pct > 30 and away_failed_pct > 30:
            analysis['defensive_mismatch'] = "Strong home defense vs weak away attack"
            analysis['insights'].append("High clean sheet potential for home team")
        elif away_cs_pct > 30 and home_failed_pct > 30:
            analysis['defensive_mismatch'] = "Strong away defense vs weak home attack"
            analysis['insights'].append("High clean sheet potential for away team")
        
        # Additional insights
        if home_cs_pct < 15:
            analysis['insights'].append(f"Home team rarely keeps clean sheets ({home_cs_pct:.1f}%)")
        
        if away_cs_pct < 15:
            analysis['insights'].append(f"Away team rarely keeps clean sheets ({away_cs_pct:.1f}%)")
        
        return analysis
    
    def analyze_transition_trends(self, home_stats: EnhancedTeamStats,
                                away_stats: EnhancedTeamStats) -> Dict:
        """Analyze the Transition dimension"""
        # Convert to percentages for display
        home_btts_pct = home_stats.btts_pct_home * 100
        away_btts_pct = away_stats.btts_pct_away * 100
        home_over25_pct = home_stats.over25_pct_home * 100
        away_over25_pct = away_stats.over25_pct_away * 100
        
        analysis = {
            'combined_btts': (home_btts_pct + away_btts_pct) / 2,
            'combined_over25': (home_over25_pct + away_over25_pct) / 2,
            'home_form_momentum': self._calculate_form_momentum(home_stats),
            'away_form_momentum': self._calculate_form_momentum(away_stats),
            'momentum_difference': None,
            'insights': []
        }
        
        # Calculate momentum difference
        home_ppg_last5 = home_stats.last5_wins * 3 + home_stats.last5_draws
        away_ppg_last5 = away_stats.last5_wins * 3 + away_stats.last5_draws
        analysis['momentum_difference'] = home_ppg_last5 - away_ppg_last5
        
        # Generate insights
        if analysis['combined_btts'] > 65:
            analysis['insights'].append(f"High BTTS probability ({analysis['combined_btts']:.1f}%)")
        
        if analysis['combined_over25'] > 65:
            analysis['insights'].append(f"High Over 2.5 probability ({analysis['combined_over25']:.1f}%)")
        
        if analysis['home_form_momentum'] == 'improving':
            analysis['insights'].append("Home team in improving form")
        elif analysis['home_form_momentum'] == 'declining':
            analysis['insights'].append("Home team in declining form")
        
        if analysis['away_form_momentum'] == 'improving':
            analysis['insights'].append("Away team in improving form")
        elif analysis['away_form_momentum'] == 'declining':
            analysis['insights'].append("Away team in declining form")
        
        return analysis
    
    def _calculate_form_momentum(self, stats: EnhancedTeamStats) -> str:
        """Calculate if team is improving or declining"""
        # Simplified momentum calculation
        last5_ppg = (stats.last5_wins * 3 + stats.last5_draws) / 5 if 5 > 0 else 0
        season_ppg = stats.points_per_game
        
        if season_ppg == 0:
            return "stable"
        
        if last5_ppg > season_ppg * 1.2:
            return "improving"
        elif last5_ppg < season_ppg * 0.8:
            return "declining"
        else:
            return "stable"
    
    def calculate_goal_expectations(self, home_stats: EnhancedTeamStats,
                                  away_stats: EnhancedTeamStats,
                                  league: str = "default") -> Dict:
        """Calculate goal expectations using the "3 Things" framework"""
        context = self.league_context.get(league, self.league_context['default'])
        
        # Get style adjustment
        style_key = (home_stats.style, away_stats.style)
        style_adj = self.style_adjustments.get(style_key, self.default_style_adjustment)
        
        # Base goal expectations from xG (most reliable predictor)
        lambda_home = home_stats.xg_for_avg
        lambda_away = away_stats.xg_for_avg
        
        # Adjust for opponent defense quality
        lambda_home *= (away_stats.xg_against_avg / context['avg_xg_against'])
        lambda_away *= (home_stats.xg_against_avg / context['avg_xg_against'])
        
        # Apply home advantage
        lambda_home *= (1 + context['home_advantage'])
        
        # Apply conversion efficiency
        home_conv_adj = home_stats.conversion_rate / context['avg_conversion'] if context['avg_conversion'] > 0 else 1.0
        away_conv_adj = away_stats.conversion_rate / context['avg_conversion'] if context['avg_conversion'] > 0 else 1.0
        
        lambda_home *= home_conv_adj
        lambda_away *= away_conv_adj
        
        # Apply style adjustments (reduced impact)
        lambda_home *= style_adj['home_goals_mult']
        lambda_away *= style_adj['away_goals_mult']
        
        # Ensure reasonable bounds
        lambda_home = max(0.2, min(4.0, lambda_home))
        lambda_away = max(0.2, min(4.0, lambda_away))
        
        total_goals = lambda_home + lambda_away
        
        # Calculate probabilities with CORRECT Poisson
        prob_over25 = self._poisson_over25_correct(lambda_home, lambda_away)
        prob_under25 = 1 - prob_over25
        prob_btts = self._poisson_btts(lambda_home, lambda_away)
        prob_no_btts = 1 - prob_btts
        
        # Calculate win/draw probabilities using proper Poisson
        home_win_prob, draw_prob, away_win_prob = self._poisson_match_probabilities(lambda_home, lambda_away)
        
        # Apply small biases from style adjustments
        if style_adj.get('over_bias', 0) > 0:
            prob_over25 += style_adj['over_bias']
            prob_under25 = 1 - prob_over25
        elif style_adj.get('under_bias', 0) > 0:
            prob_under25 += style_adj['under_bias']
            prob_over25 = 1 - prob_under25
        
        if style_adj.get('draw_bias', 0) > 0:
            draw_prob += style_adj['draw_bias']
            # Re-normalize
            total = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'total_goals': total_goals,
            'probabilities': {
                'over25': prob_over25,
                'under25': prob_under25,
                'btts_yes': prob_btts,
                'btts_no': prob_no_btts,
                'home_win': home_win_prob,
                'away_win': away_win_prob,
                'draw': draw_prob,
                'home_or_draw': home_win_prob + draw_prob,
                'away_or_draw': away_win_prob + draw_prob
            },
            'style_adjustment': style_adj,
            'efficiency_adjustments': {
                'home_conversion': home_conv_adj,
                'away_conversion': away_conv_adj
            }
        }
    
    def _poisson_over25_correct(self, lambda_home: float, lambda_away: float) -> float:
        """Correct probability of Over 2.5 goals using Poisson distribution"""
        try:
            total_lambda = lambda_home + lambda_away
            # Probability of 0, 1, or 2 goals
            prob_0 = math.exp(-total_lambda)
            prob_1 = total_lambda * math.exp(-total_lambda)
            prob_2 = (total_lambda ** 2) * math.exp(-total_lambda) / 2
            prob_under25 = prob_0 + prob_1 + prob_2
            return 1 - prob_under25  # Over 2.5
        except:
            return 0.5
    
    def _poisson_btts(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate probability of Both Teams to Score"""
        try:
            prob_home_score = 1 - math.exp(-lambda_home)
            prob_away_score = 1 - math.exp(-lambda_away)
            return prob_home_score * prob_away_score
        except:
            return 0.5
    
    def _poisson_match_probabilities(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Calculate proper win/draw probabilities using Poisson distribution"""
        try:
            max_goals = 8  # Reasonable cutoff
            home_win_prob = 0.0
            draw_prob = 0.0
            away_win_prob = 0.0
            
            for i in range(max_goals + 1):  # Home goals
                prob_home_i = math.exp(-lambda_home) * (lambda_home ** i) / math.factorial(i)
                
                for j in range(max_goals + 1):  # Away goals
                    prob_away_j = math.exp(-lambda_away) * (lambda_away ** j) / math.factorial(j)
                    
                    joint_prob = prob_home_i * prob_away_j
                    
                    if i > j:
                        home_win_prob += joint_prob
                    elif i == j:
                        draw_prob += joint_prob
                    else:
                        away_win_prob += joint_prob
            
            # Normalize (probabilities should sum to 1)
            total = home_win_prob + draw_prob + away_win_prob
            if total > 0:
                home_win_prob /= total
                draw_prob /= total
                away_win_prob /= total
            
            return home_win_prob, draw_prob, away_win_prob
            
        except Exception as e:
            # Fallback to simplified method
            print(f"Poisson calculation error: {e}")
            return self._simple_win_probabilities(lambda_home, lambda_away)
    
    def _simple_win_probabilities(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Simplified win probability calculation as fallback"""
        try:
            if lambda_home <= 0 or lambda_away <= 0:
                return 0.33, 0.34, 0.33
            
            # Basic ratio-based approach
            home_strength = lambda_home / (lambda_home + lambda_away)
            draw_prob = 0.25  # Default draw probability
            home_win_prob = home_strength * (1 - draw_prob)
            away_win_prob = (1 - home_strength) * (1 - draw_prob)
            
            return home_win_prob, draw_prob, away_win_prob
        except:
            return 0.33, 0.34, 0.33
    
    def detect_value_bets(self, model_probs: Dict, market_odds: Dict) -> List[Dict]:
        """Detect value bets across all markets"""
        value_bets = []
        
        # Map of our probabilities to market odds
        bet_mappings = [
            ('over25', 'over_25', BetType.OVER_25),
            ('under25', 'under_25', BetType.UNDER_25),
            ('btts_yes', 'btts_yes', BetType.BTTS_YES),
            ('btts_no', 'btts_no', BetType.BTTS_NO),
            ('home_win', 'home_win', BetType.HOME_WIN),
            ('away_win', 'away_win', BetType.AWAY_WIN),
            ('draw', 'draw', BetType.DRAW),
            ('home_or_draw', 'home_draw', BetType.HOME_DOUBLE_CHANCE),
            ('away_or_draw', 'away_draw', BetType.AWAY_DOUBLE_CHANCE)
        ]
        
        for model_key, market_key, bet_type in bet_mappings:
            if model_key in model_probs and market_key in market_odds:
                model_prob = model_probs[model_key]
                market_odd = market_odds[market_key]
                
                if market_odd > 0:
                    implied_prob = 1 / market_odd
                    edge = model_prob - implied_prob
                    
                    if edge >= self.min_edge and model_prob >= self.min_confidence_for_stake:
                        value_bets.append({
                            'bet_type': bet_type,
                            'model_probability': model_prob,
                            'market_odds': market_odd,
                            'implied_probability': implied_prob,
                            'edge_percent': edge * 100,
                            'market_key': market_key
                        })
        
        # Sort by edge (highest first)
        value_bets.sort(key=lambda x: x['edge_percent'], reverse=True)
        return value_bets
    
    def calculate_correlation_factor(self, bet1: Dict, bet2: Dict) -> float:
        """Calculate correlation factor between two bets"""
        # Highly correlated bets (reduce stake)
        high_correlation_pairs = [
            (BetType.OVER_25, BetType.BTTS_YES),
            (BetType.UNDER_25, BetType.BTTS_NO),
            (BetType.HOME_WIN, BetType.BTTS_NO),
        ]
        
        # Anti-correlated bets (increase diversification)
        anti_correlation_pairs = [
            (BetType.OVER_25, BetType.UNDER_25),
            (BetType.BTTS_YES, BetType.BTTS_NO),
            (BetType.HOME_WIN, BetType.AWAY_WIN),
        ]
        
        bet_pair = (bet1['bet_type'], bet2['bet_type'])
        
        if bet_pair in high_correlation_pairs or (bet2['bet_type'], bet1['bet_type']) in high_correlation_pairs:
            return 0.5  # Reduce stake by 50%
        elif bet_pair in anti_correlation_pairs or (bet2['bet_type'], bet1['bet_type']) in anti_correlation_pairs:
            return 1.5  # Increase diversification
        else:
            return 1.0  # Normal correlation
    
    def calculate_kelly_stake(self, probability: float, odds: float, bankroll: float,
                            correlation_factor: float = 1.0, max_percent: float = 0.03) -> Dict:
        """Calculate Kelly stake with correlation adjustment"""
        try:
            q = 1 - probability
            b = odds - 1
            
            if b <= 0 or probability <= 0:
                kelly_fraction = 0
            else:
                kelly_fraction = (b * probability - q) / b
            
            # Apply fractional Kelly (conservative)
            kelly_fraction *= 0.25  # 25% Kelly
            
            # Apply correlation adjustment
            kelly_fraction *= correlation_factor
            
            # Ensure non-negative
            kelly_fraction = max(0, kelly_fraction)
            
            # Calculate stake
            stake_amount = bankroll * kelly_fraction
            
            # Apply maximum stake limit
            max_stake = bankroll * max_percent
            stake_amount = min(stake_amount, max_stake)
            
            # Ensure minimum sensible stake
            if stake_amount < bankroll * 0.005:  # Less than 0.5%
                stake_amount = 0
            
            # Calculate expected value
            expected_value = (probability * (stake_amount * (odds - 1))) - (q * stake_amount)
            
            # Calculate edge
            implied_prob = 1 / odds if odds > 0 else 0
            edge = probability - implied_prob
            
            # Determine value rating
            if edge > 0.10:
                value_rating = "⭐⭐⭐ Excellent"
            elif edge > 0.05:
                value_rating = "⭐⭐ Good"
            elif edge > 0.03:
                value_rating = "⭐ Fair"
            else:
                value_rating = "Poor"
            
            # Determine risk level
            if kelly_fraction > 0.05:
                risk_level = "High"
            elif kelly_fraction > 0.02:
                risk_level = "Medium"
            elif kelly_fraction > 0:
                risk_level = "Low"
            else:
                risk_level = "No Bet"
            
            return {
                'stake_amount': stake_amount,
                'stake_percent': stake_amount / bankroll if bankroll > 0 else 0,
                'kelly_fraction': kelly_fraction,
                'expected_value': expected_value,
                'risk_level': risk_level,
                'edge_percent': edge * 100,
                'value_rating': value_rating,
                'implied_probability': implied_prob,
                'true_probability': probability,
                'max_stake_limit': max_stake,
                'correlation_factor': correlation_factor
            }
        except Exception as e:
            # Return safe defaults if calculation fails
            return {
                'stake_amount': 0,
                'stake_percent': 0,
                'kelly_fraction': 0,
                'expected_value': 0,
                'risk_level': "No Bet",
                'edge_percent': 0,
                'value_rating': "Poor",
                'implied_probability': 0,
                'true_probability': probability,
                'max_stake_limit': bankroll * max_percent,
                'correlation_factor': correlation_factor
            }
    
    def predict_match(self, home_stats: EnhancedTeamStats, away_stats: EnhancedTeamStats,
                     market_odds: Dict, league: str = "default", 
                     bankroll: float = None) -> Dict:
        """
        Main prediction method using the "3 Things" framework
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        # Reset correlated bets
        self.correlated_bets = []
        
        # Analyze all three dimensions
        identity_analysis = self.analyze_team_identity(home_stats, away_stats)
        defense_analysis = self.analyze_defense_patterns(home_stats, away_stats)
        transition_analysis = self.analyze_transition_trends(home_stats, away_stats)
        
        # Calculate goal expectations and probabilities
        goal_expectations = self.calculate_goal_expectations(home_stats, away_stats, league)
        
        # Detect value bets
        value_bets = self.detect_value_bets(goal_expectations['probabilities'], market_odds)
        
        # Calculate stakes for each value bet
        final_bets = []
        total_exposure = 0
        
        for i, bet in enumerate(value_bets):
            # Calculate correlation factor with previous bets
            correlation_factor = 1.0
            if i > 0:
                for prev_bet in final_bets:
                    corr = self.calculate_correlation_factor(bet, prev_bet['bet_details'])
                    correlation_factor = min(correlation_factor, corr)  # Use minimum
            
            # Calculate stake
            staking_result = self.calculate_kelly_stake(
                probability=bet['model_probability'],
                odds=bet['market_odds'],
                bankroll=bankroll,
                correlation_factor=correlation_factor,
                max_percent=self.max_stake_pct
            )
            
            # Only include if stake > 0
            if staking_result['stake_amount'] > 0:
                # Check total exposure
                potential_exposure = total_exposure + staking_result['stake_percent']
                if potential_exposure <= self.max_correlation_exposure:
                    final_bet = {
                        'bet_type': bet['bet_type'].value,
                        'market_odds': bet['market_odds'],
                        'model_probability': bet['model_probability'],
                        'edge_percent': bet['edge_percent'],
                        'staking': staking_result,
                        'explanation': self._generate_bet_explanation(
                            bet, home_stats, away_stats, identity_analysis,
                            defense_analysis, transition_analysis
                        ),
                        'bet_details': bet
                    }
                    final_bets.append(final_bet)
                    total_exposure += staking_result['stake_percent']
                    self.correlated_bets.append(final_bet)
        
        # Generate overall match insights
        match_insights = self._generate_match_insights(
            home_stats, away_stats, identity_analysis,
            defense_analysis, transition_analysis, goal_expectations
        )
        
        # Prepare final result
        result = {
            'match_analysis': {
                'identity': identity_analysis,
                'defense': defense_analysis,
                'transition': transition_analysis,
                'goal_expectations': goal_expectations,
                'match_insights': match_insights
            },
            'value_bets': final_bets,
            'total_exposure_percent': total_exposure,
            'total_expected_value': sum(b['staking']['expected_value'] for b in final_bets),
            'expected_bankroll_growth': (sum(b['staking']['expected_value'] for b in final_bets) / bankroll * 100 
                                        if bankroll > 0 else 0),
            'market_odds_used': market_odds
        }
        
        # Add no-bet message if applicable
        if not final_bets:
            result['recommendation'] = f"NO BET - No value opportunities meeting minimum {self.min_edge*100:.1f}% edge criteria"
            result['reason'] = f"Minimum edge required: {self.min_edge*100:.1f}%"
        else:
            result['recommendation'] = f"Found {len(final_bets)} value bet(s) with {total_exposure*100:.1f}% exposure"
        
        return result
    
    def _generate_bet_explanation(self, bet: Dict, home_stats: EnhancedTeamStats,
                                away_stats: EnhancedTeamStats, identity: Dict,
                                defense: Dict, transition: Dict) -> str:
        """Generate explanation for a specific bet"""
        explanations = []
        
        # Add style matchup insight
        if identity.get('style_clash'):
            explanations.append(
                self.explanation_templates['style_matchup'].format(
                    style_clash=identity['style_clash'],
                    home_style=home_stats.style.value,
                    home_possession=home_stats.possession_avg * 100,
                    away_style=away_stats.style.value,
                    away_possession=away_stats.possession_avg * 100,
                    insight=identity['insights'][0] if identity['insights'] else "Neutral matchup"
                )
            )
        
        # Add efficiency insight
        if abs(identity.get('conversion_comparison', 0)) > 3:
            explanations.append(
                self.explanation_templates['efficiency_mismatch'].format(
                    team1=home_stats.team_name,
                    conv1=home_stats.conversion_rate * 100,
                    team2=away_stats.team_name,
                    conv2=away_stats.conversion_rate * 100,
                    insight="Favors more efficient team" if identity['conversion_comparison'] > 0 else "Favors less wasteful team"
                )
            )
        
        # Add value finding
        explanations.append(
            self.explanation_templates['value_finding'].format(
                model_prob=bet['model_probability'],
                market_prob=1/bet['market_odds'],
                edge=bet['edge_percent']
            )
        )
        
        return " | ".join(explanations)
    
    def _generate_match_insights(self, home_stats: EnhancedTeamStats,
                               away_stats: EnhancedTeamStats, identity: Dict,
                               defense: Dict, transition: Dict,
                               goal_expectations: Dict) -> List[str]:
        """Generate overall match insights"""
        insights = []
        
        # Goal expectation insight
        total_goals = goal_expectations['total_goals']
        if total_goals > 3.0:
            insights.append(f"High-scoring affair expected ({total_goals:.2f} total goals)")
        elif total_goals < 2.0:
            insights.append(f"Low-scoring affair expected ({total_goals:.2f} total goals)")
        
        # Style insight
        if identity.get('possession_difference', 0) > 10:
            insights.append(f"{home_stats.team_name} likely to dominate possession")
        elif identity.get('possession_difference', 0) < -10:
            insights.append(f"{away_stats.team_name} likely to dominate possession")
        
        # Defensive insight
        if defense.get('defensive_mismatch'):
            insights.append(defense['defensive_mismatch'])
        
        # Form insight
        if transition.get('home_form_momentum') == 'improving':
            insights.append(f"{home_stats.team_name} in improving form")
        if transition.get('away_form_momentum') == 'improving':
            insights.append(f"{away_stats.team_name} in improving form")
        
        return insights
