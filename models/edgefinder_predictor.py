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

class ConfidenceLevel(Enum):
    """Confidence scoring levels"""
    HIGH = "⭐⭐⭐ High"
    MEDIUM = "⭐⭐ Medium"
    LOW = "⭐ Low"
    VERY_LOW = "Very Low"

@dataclass
class EnhancedTeamStats:
    """Container for enhanced team statistics from CSV"""
    # Identity Metrics
    team_name: str
    matches_played: int
    possession_avg: float  # As decimal
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
                try:
                    cleaned = value.replace('%', '').strip()
                    if cleaned:
                        setattr(self, field, float(cleaned) / 100.0)
                    else:
                        setattr(self, field, 0.0)
                except (ValueError, AttributeError):
                    setattr(self, field, 0.0)
            elif isinstance(value, (int, float)):
                if value > 1.0 and field not in ['possession_avg']:
                    setattr(self, field, value / 100.0)
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                setattr(self, field, 0.0)
        
        # Handle possession separately
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
    def goals_against_avg(self) -> float:
        """Average goals conceded per game"""
        total_goals_against = self.home_goals_against + self.away_goals_against
        return total_goals_against / self.matches_played if self.matches_played > 0 else 0.0
    
    @property
    def home_attack_strength(self) -> float:
        """Home attack strength relative to league average"""
        home_games = self.home_wins + self.home_draws + self.home_losses
        if home_games > 0:
            return self.home_goals_for / home_games
        return self.xg_for_avg
    
    @property
    def away_attack_strength(self) -> float:
        """Away attack strength relative to league average"""
        away_games = self.away_wins + self.away_draws + self.away_losses
        if away_games > 0:
            return self.away_goals_for / away_games
        return self.xg_for_avg * 0.9  # Away teams typically score 10% less


class EdgeFinderPredictor:
    """
    v1.2 Football Predictor based on "3 Things" Framework:
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
                'avg_gpg': 2.7,
                'avg_xg_for': 1.4,
                'avg_xg_against': 1.4,
                'home_advantage': 0.15,  # +15% boost for home teams
                'avg_possession': 0.50,
                'avg_conversion': 0.11
            },
            'default': {
                'avg_gpg': 2.7,
                'avg_xg_for': 1.4,
                'avg_xg_against': 1.4,
                'home_advantage': 0.15,
                'avg_possession': 0.50,
                'avg_conversion': 0.11
            }
        }
        
        # Style matchup adjustments (ALIGNED WITH LOGIC)
        self.style_adjustments = {
            (TeamStyle.POSSESSION, TeamStyle.LOW_BLOCK): {
                'home_goals_mult': 0.75,  # -25% for possession vs low block
                'away_goals_mult': 1.0,
                'under_bias': 0.10,
                'draw_bias': 0.08,
                'explanation': "Possession teams struggle against organized low blocks"
            },
            (TeamStyle.COUNTER, TeamStyle.HIGH_PRESS): {
                'home_goals_mult': 1.25,  # +25% for counter vs high press
                'away_goals_mult': 1.0,
                'over_bias': 0.10,
                'btts_bias': 0.08,
                'explanation': "Counter-attacking teams thrive against high defensive lines"
            },
            (TeamStyle.POSSESSION, TeamStyle.POSSESSION): {
                'home_goals_mult': 1.0,
                'away_goals_mult': 1.0,
                'draw_bias': 0.05,
                'explanation': "Possession vs possession leads to controlled, lower-scoring games"
            },
            (TeamStyle.HIGH_PRESS, TeamStyle.HIGH_PRESS): {
                'home_goals_mult': 1.15,
                'away_goals_mult': 1.15,
                'over_bias': 0.15,
                'btts_bias': 0.10,
                'explanation': "High press vs high press leads to end-to-end action"
            },
        }
        
        # Default style adjustment
        self.default_style_adjustment = {
            'home_goals_mult': 1.0,
            'away_goals_mult': 1.0,
            'over_bias': 0.0,
            'under_bias': 0.0,
            'btts_bias': 0.0,
            'draw_bias': 0.0,
            'explanation': "Neutral style matchup"
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
            'identity': "🎯 **Team Identity**: {home_team} ({home_style}, {home_possession:.0f}% possession, {home_conversion:.1f}% conversion) vs {away_team} ({away_style}, {away_possession:.0f}% possession, {away_conversion:.1f}% conversion). {insight}",
            'defense': "🛡️ **Defensive Pattern**: {team} keeps clean sheets {clean_sheets:.1f}% of time, fails to score {failed:.1f}%. {insight}",
            'transition': "📈 **Transition Trend**: {team} BTTS {btts:.1f}%, Over 2.5 {over25:.1f}%, recent form: {form}. {insight}",
            'style_matchup': "⚔️ **Style Clash**: {home_style} vs {away_style}. {explanation}",
            'efficiency_edge': "⚡ **Efficiency Edge**: {team1} ({conv1:.1f}% conversion) vs {team2} ({conv2:.1f}% conversion). {team} more clinical.",
            'form_momentum': "📊 **Form Momentum**: {team} {momentum} form ({form_record}). {insight}",
            'venue_effect': "🏟️ **Venue Effect**: {home_team} home xG: {home_xg:.2f}, {away_team} away xG: {away_xg:.2f}. {insight}",
            'value_finding': "💰 **Value Found**: Model: {model_prob:.1%} vs Market: {market_prob:.1%} = {edge:.1f}% edge.",
            'confidence': "🎯 **Confidence**: {score}/10 - {level} ({reason})"
        }
    
    def analyze_team_identity(self, home_stats: EnhancedTeamStats, 
                            away_stats: EnhancedTeamStats) -> Dict:
        """Analyze the Team Identity dimension"""
        # Convert to percentages for display
        home_possession_pct = home_stats.possession_avg * 100
        away_possession_pct = away_stats.possession_avg * 100
        home_conversion_pct = home_stats.conversion_rate * 100
        away_conversion_pct = away_stats.conversion_rate * 100
        
        # Calculate efficiency edge
        conversion_diff = home_conversion_pct - away_conversion_pct
        possession_diff = home_possession_pct - away_possession_pct
        
        analysis = {
            'style_clash': None,
            'possession_difference': possession_diff,
            'conversion_comparison': conversion_diff,
            'shots_difference': home_stats.shots_per_game - away_stats.shots_per_game,
            'efficiency_ratio': home_stats.conversion_rate / away_stats.conversion_rate 
                                if away_stats.conversion_rate > 0 else 1.0,
            'insights': [],
            'confidence_factors': []
        }
        
        # Determine style clash
        home_style = home_stats.style
        away_style = away_stats.style
        analysis['style_clash'] = f"{home_style.value} vs {away_style.value}"
        
        # Style matchup insights
        style_key = (home_style, away_style)
        style_adj = self.style_adjustments.get(style_key, self.default_style_adjustment)
        if 'explanation' in style_adj:
            analysis['insights'].append(style_adj['explanation'])
        
        # Generate identity insights
        if abs(possession_diff) > 10:
            if possession_diff > 10:
                analysis['insights'].append(
                    f"{home_stats.team_name} dominates possession ({home_possession_pct:.1f}% vs {away_possession_pct:.1f}%)"
                )
                analysis['confidence_factors'].append(('possession_dominance', 1))
            else:
                analysis['insights'].append(
                    f"{away_stats.team_name} dominates possession ({away_possession_pct:.1f}% vs {home_possession_pct:.1f}%)"
                )
                analysis['confidence_factors'].append(('possession_dominance', 1))
        
        if abs(conversion_diff) > 3:
            if conversion_diff > 3:
                analysis['insights'].append(
                    f"{home_stats.team_name} more efficient ({home_conversion_pct:.1f}% vs {away_conversion_pct:.1f}% conversion)"
                )
                analysis['confidence_factors'].append(('efficiency_edge', 2))
            else:
                analysis['insights'].append(
                    f"{away_stats.team_name} more efficient ({away_conversion_pct:.1f}% vs {home_conversion_pct:.1f}% conversion)"
                )
                analysis['confidence_factors'].append(('efficiency_edge', 2))
        
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
            'insights': [],
            'confidence_factors': []
        }
        
        # Determine defensive mismatch (ALIGNED WITH LOGIC)
        if home_cs_pct > 40 and away_failed_pct > 40:
            analysis['defensive_mismatch'] = "Strong home defense vs weak away attack"
            analysis['insights'].append(f"{home_stats.team_name} strong home defense ({home_cs_pct:.1f}% clean sheets) vs {away_stats.team_name} poor away attack ({away_failed_pct:.1f}% failed to score)")
            analysis['confidence_factors'].append(('defensive_mismatch', 3))
        elif away_cs_pct > 40 and home_failed_pct > 40:
            analysis['defensive_mismatch'] = "Strong away defense vs weak home attack"
            analysis['insights'].append(f"{away_stats.team_name} strong away defense ({away_cs_pct:.1f}% clean sheets) vs {home_stats.team_name} poor home attack ({home_failed_pct:.1f}% failed to score)")
            analysis['confidence_factors'].append(('defensive_mismatch', 3))
        
        # Additional defensive insights
        if home_cs_pct < 15:
            analysis['insights'].append(f"{home_stats.team_name} rarely keeps clean sheets at home ({home_cs_pct:.1f}%)")
            analysis['confidence_factors'].append(('defensive_weakness', 1))
        
        if away_cs_pct < 15:
            analysis['insights'].append(f"{away_stats.team_name} rarely keeps clean sheets away ({away_cs_pct:.1f}%)")
            analysis['confidence_factors'].append(('defensive_weakness', 1))
        
        # Scoring reliability insights
        if home_failed_pct > 40:
            analysis['insights'].append(f"{home_stats.team_name} frequently fails to score at home ({home_failed_pct:.1f}%)")
        
        if away_failed_pct > 40:
            analysis['insights'].append(f"{away_stats.team_name} frequently fails to score away ({away_failed_pct:.1f}%)")
        
        return analysis
    
    def analyze_transition_trends(self, home_stats: EnhancedTeamStats,
                                away_stats: EnhancedTeamStats) -> Dict:
        """Analyze the Transition dimension"""
        # Convert to percentages for display
        home_btts_pct = home_stats.btts_pct_home * 100
        away_btts_pct = away_stats.btts_pct_away * 100
        home_over25_pct = home_stats.over25_pct_home * 100
        away_over25_pct = away_stats.over25_pct_away * 100
        
        # Calculate form momentum
        home_momentum = self._calculate_form_momentum(home_stats)
        away_momentum = self._calculate_form_momentum(away_stats)
        
        analysis = {
            'combined_btts': (home_btts_pct + away_btts_pct) / 2,
            'combined_over25': (home_over25_pct + away_over25_pct) / 2,
            'home_form_momentum': home_momentum,
            'away_form_momentum': away_momentum,
            'momentum_difference': None,
            'insights': [],
            'confidence_factors': []
        }
        
        # Calculate momentum difference
        home_ppg_last5 = home_stats.last5_wins * 3 + home_stats.last5_draws
        away_ppg_last5 = away_stats.last5_wins * 3 + away_stats.last5_draws
        analysis['momentum_difference'] = home_ppg_last5 - away_ppg_last5
        
        # Generate transition insights
        if analysis['combined_btts'] > 65:
            analysis['insights'].append(f"High BTTS probability ({analysis['combined_btts']:.1f}% historical)")
            analysis['confidence_factors'].append(('high_btts', 2))
        elif analysis['combined_btts'] < 35:
            analysis['insights'].append(f"Low BTTS probability ({analysis['combined_btts']:.1f}% historical)")
            analysis['confidence_factors'].append(('low_btts', 2))
        
        if analysis['combined_over25'] > 65:
            analysis['insights'].append(f"High Over 2.5 probability ({analysis['combined_over25']:.1f}% historical)")
            analysis['confidence_factors'].append(('high_over25', 2))
        elif analysis['combined_over25'] < 35:
            analysis['insights'].append(f"Low Over 2.5 probability ({analysis['combined_over25']:.1f}% historical)")
            analysis['confidence_factors'].append(('low_over25', 2))
        
        # Form momentum insights (ALIGNED WITH LOGIC: 40% weight to recent form)
        if home_momentum == 'improving':
            analysis['insights'].append(f"{home_stats.team_name} in IMPROVING form ({home_stats.last5_form})")
            analysis['confidence_factors'].append(('improving_form', 2))
        elif home_momentum == 'declining':
            analysis['insights'].append(f"{home_stats.team_name} in DECLINING form ({home_stats.last5_form})")
            analysis['confidence_factors'].append(('declining_form', 2))
        
        if away_momentum == 'improving':
            analysis['insights'].append(f"{away_stats.team_name} in IMPROVING form ({away_stats.last5_form})")
            analysis['confidence_factors'].append(('improving_form', 2))
        elif away_momentum == 'declining':
            analysis['insights'].append(f"{away_stats.team_name} in DECLINING form ({away_stats.last5_form})")
            analysis['confidence_factors'].append(('declining_form', 2))
        
        return analysis
    
    def _calculate_form_momentum(self, stats: EnhancedTeamStats) -> str:
        """Calculate if team is improving or declining (40% weight to recent form)"""
        last5_ppg = (stats.last5_wins * 3 + stats.last5_draws) / 5 if 5 > 0 else 0
        season_ppg = stats.points_per_game
        
        if season_ppg == 0:
            return "stable"
        
        # 40% weight to recent form vs 60% to season average
        weighted_ppg = (last5_ppg * 0.4) + (season_ppg * 0.6)
        
        if weighted_ppg > season_ppg * 1.15:  # 15% improvement
            return "improving"
        elif weighted_ppg < season_ppg * 0.85:  # 15% decline
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
        
        # BASE AVERAGE: Use venue-specific attack strength (ALIGNED WITH LOGIC)
        home_base_attack = home_stats.home_attack_strength
        away_base_attack = away_stats.away_attack_strength
        
        # EFFICIENCY ADJUSTMENT: Team conversion vs opponent defense
        home_efficiency = home_stats.conversion_rate / context['avg_conversion'] if context['avg_conversion'] > 0 else 1.0
        away_efficiency = away_stats.conversion_rate / context['avg_conversion'] if context['avg_conversion'] > 0 else 1.0
        
        # OPPONENT DEFENSE ADJUSTMENT: Adjust for opponent's defensive quality
        home_vs_away_defense = away_stats.xg_against_avg / context['avg_xg_against'] if context['avg_xg_against'] > 0 else 1.0
        away_vs_home_defense = home_stats.xg_against_avg / context['avg_xg_against'] if context['avg_xg_against'] > 0 else 1.0
        
        # STYLE ADJUSTMENT: Apply style matchup multipliers
        home_style_mult = style_adj['home_goals_mult']
        away_style_mult = style_adj['away_goals_mult']
        
        # VENUE EFFECTS: Apply home advantage (ALIGNED WITH LOGIC: +0.3 for home, -0.2 for away)
        home_venue_boost = 1 + context['home_advantage']  # +15% for Premier League
        away_venue_penalty = 0.85  # Away teams typically score 15% less
        
        # FORM MOMENTUM: Apply form adjustments (40% weight to recent form)
        home_form_adj = self._calculate_form_adjustment(home_stats)
        away_form_adj = self._calculate_form_adjustment(away_stats)
        
        # FINAL GOAL EXPECTATION CALCULATION (ALIGNED WITH LOGIC FORMULA)
        lambda_home = (home_base_attack * 
                      home_efficiency * 
                      home_vs_away_defense * 
                      home_style_mult * 
                      home_venue_boost * 
                      home_form_adj)
        
        lambda_away = (away_base_attack * 
                      away_efficiency * 
                      away_vs_home_defense * 
                      away_style_mult * 
                      away_venue_penalty * 
                      away_form_adj)
        
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
        
        # Apply style biases
        if style_adj.get('over_bias', 0) > 0:
            prob_over25 = min(0.95, prob_over25 + style_adj['over_bias'])
            prob_under25 = 1 - prob_over25
        elif style_adj.get('under_bias', 0) > 0:
            prob_under25 = min(0.95, prob_under25 + style_adj['under_bias'])
            prob_over25 = 1 - prob_under25
        
        if style_adj.get('draw_bias', 0) > 0:
            draw_prob = min(0.5, draw_prob + style_adj['draw_bias'])
            # Re-normalize
            total = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        if style_adj.get('btts_bias', 0) > 0:
            prob_btts = min(0.95, prob_btts + style_adj['btts_bias'])
            prob_no_btts = 1 - prob_btts
        
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
            'adjustment_factors': {
                'home_efficiency': home_efficiency,
                'away_efficiency': away_efficiency,
                'home_venue_boost': home_venue_boost,
                'away_venue_penalty': away_venue_penalty,
                'home_form_adj': home_form_adj,
                'away_form_adj': away_form_adj
            }
        }
    
    def _calculate_form_adjustment(self, stats: EnhancedTeamStats) -> float:
        """Calculate form adjustment factor (40% weight to recent form)"""
        last5_ppg = (stats.last5_wins * 3 + stats.last5_draws) / 5 if 5 > 0 else 0
        season_ppg = stats.points_per_game
        
        if season_ppg == 0:
            return 1.0
        
        # 40% weight to recent form vs 60% to season average
        weighted_ppg = (last5_ppg * 0.4) + (season_ppg * 0.6)
        
        # Convert to adjustment factor
        if weighted_ppg > season_ppg * 1.15:
            return 1.15  # +15% for improving form
        elif weighted_ppg < season_ppg * 0.85:
            return 0.85  # -15% for declining form
        else:
            return 1.0
    
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
            max_goals = 8
            home_win_prob = 0.0
            draw_prob = 0.0
            away_win_prob = 0.0
            
            for i in range(max_goals + 1):
                prob_home_i = math.exp(-lambda_home) * (lambda_home ** i) / math.factorial(i)
                
                for j in range(max_goals + 1):
                    prob_away_j = math.exp(-lambda_away) * (lambda_away ** j) / math.factorial(j)
                    
                    joint_prob = prob_home_i * prob_away_j
                    
                    if i > j:
                        home_win_prob += joint_prob
                    elif i == j:
                        draw_prob += joint_prob
                    else:
                        away_win_prob += joint_prob
            
            # Normalize
            total = home_win_prob + draw_prob + away_win_prob
            if total > 0:
                home_win_prob /= total
                draw_prob /= total
                away_win_prob /= total
            
            return home_win_prob, draw_prob, away_win_prob
            
        except Exception as e:
            return self._simple_win_probabilities(lambda_home, lambda_away)
    
    def _simple_win_probabilities(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Simplified win probability calculation as fallback"""
        try:
            if lambda_home <= 0 or lambda_away <= 0:
                return 0.33, 0.34, 0.33
            
            home_strength = lambda_home / (lambda_home + lambda_away)
            draw_prob = 0.25
            home_win_prob = home_strength * (1 - draw_prob)
            away_win_prob = (1 - home_strength) * (1 - draw_prob)
            
            return home_win_prob, draw_prob, away_win_prob
        except:
            return 0.33, 0.34, 0.33
    
    def calculate_confidence_score(self, analysis: Dict, goal_expectations: Dict) -> Dict:
        """Calculate confidence score (1-10) based on analysis factors"""
        confidence_factors = []
        total_score = 5  # Start at neutral 5/10
        
        # Collect all confidence factors from analyses
        for dimension in ['identity', 'defense', 'transition']:
            if dimension in analysis:
                factors = analysis[dimension].get('confidence_factors', [])
                confidence_factors.extend(factors)
        
        # Apply confidence factors
        for factor, weight in confidence_factors:
            total_score += weight
            if total_score > 10:
                total_score = 10
            elif total_score < 1:
                total_score = 1
        
        # Adjust based on data quality
        total_matches = analysis.get('home_matches', 0) + analysis.get('away_matches', 0)
        if total_matches < 10:
            total_score -= 2  # Penalize for small sample
        elif total_matches > 20:
            total_score += 1  # Reward for large sample
        
        # Ensure bounds
        total_score = max(1, min(10, total_score))
        
        # Determine confidence level
        if total_score >= 8:
            level = ConfidenceLevel.HIGH
            reason = "Strong statistical signals, clear style matchup, consistent data"
        elif total_score >= 6:
            level = ConfidenceLevel.MEDIUM
            reason = "Moderate statistical signals, some conflicting data"
        elif total_score >= 4:
            level = ConfidenceLevel.LOW
            reason = "Weak statistical signals, inconsistent data"
        else:
            level = ConfidenceLevel.VERY_LOW
            reason = "Very weak signals, unreliable data"
        
        return {
            'score': total_score,
            'level': level,
            'reason': reason,
            'factors': confidence_factors
        }
    
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
                        # Determine value rating (ALIGNED WITH LOGIC)
                        if edge > 0.05:
                            value_rating = "⭐⭐⭐ Golden Nugget"
                        elif edge > 0.03:
                            value_rating = "⭐⭐ Value Bet"
                        elif edge > 0.01:
                            value_rating = "⭐ Consider"
                        else:
                            value_rating = "Poor"
                        
                        value_bets.append({
                            'bet_type': bet_type,
                            'model_probability': model_prob,
                            'market_odds': market_odd,
                            'implied_probability': implied_prob,
                            'edge_percent': edge * 100,
                            'market_key': market_key,
                            'value_rating': value_rating
                        })
        
        # Sort by edge (highest first)
        value_bets.sort(key=lambda x: x['edge_percent'], reverse=True)
        return value_bets
    
    def calculate_correlation_factor(self, bet1: Dict, bet2: Dict) -> float:
        """Calculate correlation factor between two bets"""
        high_correlation_pairs = [
            (BetType.OVER_25, BetType.BTTS_YES),
            (BetType.UNDER_25, BetType.BTTS_NO),
            (BetType.HOME_WIN, BetType.BTTS_NO),
        ]
        
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
            return 1.0
    
    def calculate_kelly_stake(self, probability: float, odds: float, bankroll: float,
                            correlation_factor: float = 1.0, max_percent: float = 0.03,
                            confidence_score: int = 5) -> Dict:
        """Calculate Kelly stake with correlation and confidence adjustment"""
        try:
            q = 1 - probability
            b = odds - 1
            
            if b <= 0 or probability <= 0:
                kelly_fraction = 0
            else:
                kelly_fraction = (b * probability - q) / b
            
            # Apply fractional Kelly (ALIGNED WITH LOGIC: K factor)
            kelly_fraction *= 0.25  # Conservative K=0.25
            
            # Apply correlation adjustment
            kelly_fraction *= correlation_factor
            
            # Apply confidence adjustment
            confidence_mult = confidence_score / 10.0  # 0.5 for score=5, 1.0 for score=10
            kelly_fraction *= confidence_mult
            
            # Ensure non-negative
            kelly_fraction = max(0, kelly_fraction)
            
            # Calculate stake (ALIGNED WITH LOGIC: Stake = Bankroll × (Edge / (Odds-1)) × K)
            stake_amount = bankroll * kelly_fraction
            
            # Apply maximum stake limit (ALIGNED WITH LOGIC: Max 3% per bet)
            max_stake = bankroll * max_percent
            stake_amount = min(stake_amount, max_stake)
            
            # Minimum sensible stake
            if stake_amount < bankroll * 0.005:
                stake_amount = 0
            
            # Calculate expected value
            expected_value = (probability * (stake_amount * (odds - 1))) - (q * stake_amount)
            
            # Calculate edge
            implied_prob = 1 / odds if odds > 0 else 0
            edge = probability - implied_prob
            
            # Determine risk level based on kelly fraction
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
                'implied_probability': implied_prob,
                'true_probability': probability,
                'max_stake_limit': max_stake,
                'correlation_factor': correlation_factor,
                'confidence_multiplier': confidence_mult
            }
        except Exception as e:
            return {
                'stake_amount': 0,
                'stake_percent': 0,
                'kelly_fraction': 0,
                'expected_value': 0,
                'risk_level': "No Bet",
                'edge_percent': 0,
                'implied_probability': 0,
                'true_probability': probability,
                'max_stake_limit': bankroll * max_percent,
                'correlation_factor': correlation_factor,
                'confidence_multiplier': 1.0
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
        
        # Calculate confidence score
        combined_analysis = {
            'identity': identity_analysis,
            'defense': defense_analysis,
            'transition': transition_analysis,
            'home_matches': home_stats.matches_played,
            'away_matches': away_stats.matches_played
        }
        confidence = self.calculate_confidence_score(combined_analysis, goal_expectations)
        
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
                    correlation_factor = min(correlation_factor, corr)
            
            # Calculate stake with confidence adjustment
            staking_result = self.calculate_kelly_stake(
                probability=bet['model_probability'],
                odds=bet['market_odds'],
                bankroll=bankroll,
                correlation_factor=correlation_factor,
                max_percent=self.max_stake_pct,
                confidence_score=confidence['score']
            )
            
            # Only include if stake > 0
            if staking_result['stake_amount'] > 0:
                # Check total exposure (ALIGNED WITH LOGIC: Max 5% on correlated bets)
                potential_exposure = total_exposure + staking_result['stake_percent']
                if potential_exposure <= self.max_correlation_exposure:
                    final_bet = {
                        'bet_type': bet['bet_type'].value,
                        'market_odds': bet['market_odds'],
                        'model_probability': bet['model_probability'],
                        'edge_percent': bet['edge_percent'],
                        'value_rating': bet['value_rating'],
                        'staking': staking_result,
                        'explanation': self._generate_bet_explanation(
                            bet, home_stats, away_stats, identity_analysis,
                            defense_analysis, transition_analysis, goal_expectations
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
                'confidence': confidence,
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
                                defense: Dict, transition: Dict,
                                goal_expectations: Dict) -> str:
        """Generate explanation for a specific bet"""
        explanations = []
        
        # Add style matchup insight
        if identity.get('style_clash'):
            style_key = (home_stats.style, away_stats.style)
            style_adj = self.style_adjustments.get(style_key, self.default_style_adjustment)
            
            explanations.append(
                self.explanation_templates['style_matchup'].format(
                    home_style=home_stats.style.value,
                    away_style=away_stats.style.value,
                    explanation=style_adj.get('explanation', 'Neutral matchup')
                )
            )
        
        # Add efficiency insight
        if abs(identity.get('conversion_comparison', 0)) > 3:
            if identity['conversion_comparison'] > 0:
                efficient_team = home_stats.team_name
            else:
                efficient_team = away_stats.team_name
            
            explanations.append(
                self.explanation_templates['efficiency_edge'].format(
                    team1=home_stats.team_name,
                    conv1=home_stats.conversion_rate * 100,
                    team2=away_stats.team_name,
                    conv2=away_stats.conversion_rate * 100,
                    team=efficient_team
                )
            )
        
        # Add form momentum insight
        if transition.get('home_form_momentum') in ['improving', 'declining']:
            explanations.append(
                self.explanation_templates['form_momentum'].format(
                    team=home_stats.team_name,
                    momentum=transition['home_form_momentum'],
                    form_record=home_stats.last5_form,
                    insight="Recent form heavily weighted (40%) in calculations"
                )
            )
        
        if transition.get('away_form_momentum') in ['improving', 'declining']:
            explanations.append(
                self.explanation_templates['form_momentum'].format(
                    team=away_stats.team_name,
                    momentum=transition['away_form_momentum'],
                    form_record=away_stats.last5_form,
                    insight="Recent form heavily weighted (40%) in calculations"
                )
            )
        
        # Add value finding with confidence
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
            insights.append(f"Over 2.5 probability: {goal_expectations['probabilities']['over25']:.1%}")
        elif total_goals < 2.0:
            insights.append(f"Low-scoring affair expected ({total_goals:.2f} total goals)")
            insights.append(f"Under 2.5 probability: {goal_expectations['probabilities']['under25']:.1%}")
        
        # Style insight
        if identity.get('possession_difference', 0) > 10:
            insights.append(f"{home_stats.team_name} likely to dominate possession")
        elif identity.get('possession_difference', 0) < -10:
            insights.append(f"{away_stats.team_name} likely to dominate possession")
        
        # Defensive insight
        if defense.get('defensive_mismatch'):
            insights.append(defense['defensive_mismatch'])
        
        # BTTS insight
        btts_prob = goal_expectations['probabilities']['btts_yes']
        if btts_prob > 0.65:
            insights.append(f"High BTTS probability ({btts_prob:.1%})")
        elif btts_prob < 0.35:
            insights.append(f"Low BTTS probability ({btts_prob:.1%})")
        
        # Form insight
        if transition.get('home_form_momentum') == 'improving':
            insights.append(f"{home_stats.team_name} in improving form (recent form weighted 40%)")
        elif transition.get('home_form_momentum') == 'declining':
            insights.append(f"{home_stats.team_name} in declining form (recent form weighted 40%)")
        
        if transition.get('away_form_momentum') == 'improving':
            insights.append(f"{away_stats.team_name} in improving form (recent form weighted 40%)")
        elif transition.get('away_form_momentum') == 'declining':
            insights.append(f"{away_stats.team_name} in declining form (recent form weighted 40%)")
        
        return insights
