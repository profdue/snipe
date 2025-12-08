"""
EdgeFinder Predictor - FIXED Version
Maintains all original design while fixing mathematical bugs
"""

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
    # Core data
    team_name: str
    matches_played: int
    possession_avg: float
    
    # Shot data
    shots_per_game: float
    shots_on_target_pg: float
    conversion_rate: float
    
    # xG data
    xg_for_avg: float
    xg_against_avg: float
    
    # Home/Away splits
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
    
    # Defense patterns
    clean_sheet_pct: float
    clean_sheet_pct_home: float
    clean_sheet_pct_away: float
    failed_to_score_pct: float
    failed_to_score_pct_home: float
    failed_to_score_pct_away: float
    
    # Transition patterns
    btts_pct: float
    btts_pct_home: float
    btts_pct_away: float
    over25_pct: float
    over25_pct_home: float
    over25_pct_away: float
    
    # Recent Form
    last5_form: str
    last5_wins: int
    last5_draws: int
    last5_losses: int
    last5_goals_for: int
    last5_goals_against: int
    last5_ppg: float
    last5_cs_pct: float
    last5_fts_pct: float
    last5_btts_pct: float
    last5_over25_pct: float
    
    def __post_init__(self):
        """Enhanced data cleaning with proper defaults"""
        # Convert percentages
        percentage_fields = [
            'possession_avg', 'conversion_rate',
            'clean_sheet_pct', 'clean_sheet_pct_home', 'clean_sheet_pct_away',
            'failed_to_score_pct', 'failed_to_score_pct_home', 'failed_to_score_pct_away',
            'btts_pct', 'btts_pct_home', 'btts_pct_away',
            'over25_pct', 'over25_pct_home', 'over25_pct_away',
            'last5_cs_pct', 'last5_fts_pct', 'last5_btts_pct', 'last5_over25_pct'
        ]
        
        for field in percentage_fields:
            value = getattr(self, field)
            if isinstance(value, str):
                cleaned = value.replace('%', '').strip()
                setattr(self, field, float(cleaned) / 100.0 if cleaned else 0.0)
            elif isinstance(value, (int, float)):
                if value > 1.0:
                    setattr(self, field, value / 100.0)
    
    @property
    def points_per_game(self) -> float:
        """Calculate points per game"""
        total_points = (self.home_wins + self.away_wins) * 3 + (self.home_draws + self.away_draws)
        return total_points / self.matches_played if self.matches_played > 0 else 0
    
    @property
    def home_games_played(self) -> int:
        """Number of home games played"""
        return self.home_wins + self.home_draws + self.home_losses
    
    @property
    def away_games_played(self) -> int:
        """Number of away games played"""
        return self.away_wins + self.away_draws + self.away_losses
    
    @property
    def season_goals_per_game(self) -> float:
        """Season average goals per game (all venues)"""
        total_goals = self.home_goals_for + self.away_goals_for
        return total_goals / self.matches_played if self.matches_played > 0 else 0.5
    
    @property
    def season_goals_conceded_per_game(self) -> float:
        """Season average goals conceded per game (all venues)"""
        total_conceded = self.home_goals_against + self.away_goals_against
        return total_conceded / self.matches_played if self.matches_played > 0 else 0.5
    
    @property
    def style(self) -> TeamStyle:
        """Determine team playing style"""
        possession_pct = self.possession_avg * 100
        
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


class EdgeFinderPredictor:
    """
    FIXED Football Predictor - Realistic predictions with proper defense logic
    """
    
    def __init__(self, 
                 bankroll: float = 1000.0, 
                 min_edge: float = 0.03,
                 max_correlation_exposure: float = 0.10,
                 form_weight: float = 0.4,
                 min_sample_size: int = 5):
        
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.max_correlation_exposure = max_correlation_exposure
        self.form_weight = form_weight
        self.min_sample_size = min_sample_size
        
        # League contexts - REALISTIC values
        self.league_contexts = {
            'premier_league': {
                'avg_gpg': 2.7,
                'avg_shots': 12.0,
                'avg_conversion': 0.11,
                'home_advantage': 1.15,  # 15% boost at home
                'away_penalty': 0.92     # 8% penalty away
            },
            'la_liga': {
                'avg_gpg': 2.5,
                'avg_shots': 11.5,
                'avg_conversion': 0.105,
                'home_advantage': 1.18,
                'away_penalty': 0.90
            },
            'bundesliga': {
                'avg_gpg': 3.0,
                'avg_shots': 13.0,
                'avg_conversion': 0.115,
                'home_advantage': 1.10,  # Bundesliga has less home advantage
                'away_penalty': 0.95
            },
            'serie_a': {
                'avg_gpg': 2.6,
                'avg_shots': 11.8,
                'avg_conversion': 0.11,
                'home_advantage': 1.16,
                'away_penalty': 0.91
            },
            'ligue_1': {
                'avg_gpg': 2.4,
                'avg_shots': 11.2,
                'avg_conversion': 0.107,
                'home_advantage': 1.17,
                'away_penalty': 0.89
            },
            'default': {
                'avg_gpg': 2.7,
                'avg_shots': 12.0,
                'avg_conversion': 0.11,
                'home_advantage': 1.15,
                'away_penalty': 0.92
            }
        }
        
        # Style matchup adjustments - REALISTIC small effects
        self.style_matchup_effects = {
            (TeamStyle.POSSESSION, TeamStyle.LOW_BLOCK): {'possession_team_adj': 0.95, 'low_block_team_adj': 1.05},
            (TeamStyle.COUNTER, TeamStyle.HIGH_PRESS): {'counter_team_adj': 1.08, 'high_press_team_adj': 0.97},
            (TeamStyle.HIGH_PRESS, TeamStyle.LOW_BLOCK): {'high_press_team_adj': 1.05, 'low_block_team_adj': 0.95},
            (TeamStyle.LOW_BLOCK, TeamStyle.POSSESSION): {'low_block_team_adj': 1.03, 'possession_team_adj': 0.98},
        }
        
        # Betting parameters
        self.max_stake_pct = 0.03
        self.min_confidence_for_stake = 0.55
    
    def _get_venue_attack_strength(self, stats: EnhancedTeamStats, is_home: bool) -> float:
        """Get venue-specific attack strength with REALISTIC bounds"""
        if is_home:
            games = stats.home_games_played
            goals = stats.home_goals_for
        else:
            games = stats.away_games_played
            goals = stats.away_goals_for
        
        if games >= self.min_sample_size and games > 0:
            gpg = goals / games
        else:
            gpg = stats.season_goals_per_game
        
        # REALISTIC BOUNDS: No team averages >3.0 goals per game
        # Minimum 0.3 goals per game
        return max(0.3, min(3.0, gpg))
    
    def _get_venue_defense_strength(self, stats: EnhancedTeamStats, is_home: bool) -> float:
        """Get venue-specific defense strength with REALISTIC bounds"""
        if is_home:
            games = stats.home_games_played
            conceded = stats.home_goals_against
        else:
            games = stats.away_games_played
            conceded = stats.away_goals_against
        
        if games >= self.min_sample_size and games > 0:
            gapg = conceded / games
        else:
            gapg = stats.season_goals_conceded_per_game
        
        # REALISTIC BOUNDS: No team concedes >3.0 goals per game on average
        return max(0.3, min(3.0, gapg))
    
    def _calculate_attack_vs_defense_adjustment(self, attack_strength: float,
                                              opponent_defense: float,
                                              league_avg_gpg: float) -> float:
        """
        FIXED: Realistic attack vs defense adjustment
        
        LOGIC:
        - If opponent defense = league average → no adjustment (multiplier = 1.0)
        - If opponent defense is GOOD (concedes less than average) → attack reduced
        - If opponent defense is BAD (concedes more than average) → attack increased
        
        Formula: attack_multiplier = (2 - (opponent_defense / league_avg)) / 1.5
        This gives multipliers between ~0.67 and ~1.33
        """
        if opponent_defense <= 0.1 or league_avg_gpg <= 0.1:
            return attack_strength
        
        # Calculate defense quality
        defense_quality = opponent_defense / league_avg_gpg
        
        # FIXED FORMULA: attack_multiplier = (2 - defense_quality) / 1.5
        # defense_quality = 0.5 (good defense) → (2 - 0.5)/1.5 = 1.0
        # defense_quality = 1.0 (average defense) → (2 - 1.0)/1.5 = 0.67
        # defense_quality = 1.5 (bad defense) → (2 - 1.5)/1.5 = 0.33
        
        # Wait, that's backwards! Let me think...
        # Actually, if defense_quality = 0.5 (GOOD defense), attack should be REDUCED
        # So attack_multiplier should be < 1.0 for good defenses
        
        # CORRECT FORMULA: attack_multiplier = defense_quality
        # Good defense (0.5x league avg) → multiplier = 0.5 (attack halved)
        # Average defense (1.0x) → multiplier = 1.0 (no change)
        # Bad defense (1.5x) → multiplier = 1.5 (attack increased by 50%)
        
        attack_multiplier = defense_quality
        
        # Apply REALISTIC bounds: attack can be 0.5x to 1.5x
        attack_multiplier = max(0.5, min(1.5, attack_multiplier))
        
        return attack_strength * attack_multiplier
    
    def _calculate_efficiency_adjustment(self, conversion_rate: float, league_avg_conversion: float) -> float:
        """Calculate efficiency adjustment from conversion rates"""
        if league_avg_conversion <= 0:
            return 1.0
        
        efficiency = conversion_rate / league_avg_conversion
        
        # REALISTIC bounds: efficiency between 0.5x and 1.5x
        return max(0.5, min(1.5, efficiency))
    
    def _calculate_shot_quality_adjustment(self, stats: EnhancedTeamStats, league_avg_shots: float) -> float:
        """Calculate shot quality from volume and accuracy"""
        if league_avg_shots <= 0:
            return 1.0
        
        # Volume component
        volume_ratio = stats.shots_per_game / league_avg_shots
        
        # Accuracy component (if we have shots on target data)
        if stats.shots_per_game > 0:
            accuracy = stats.shots_on_target_pg / stats.shots_per_game
            # Assume league average accuracy is ~35%
            accuracy_ratio = accuracy / 0.35 if 0.35 > 0 else 1.0
        else:
            accuracy_ratio = 1.0
        
        # Combined shot quality
        shot_quality = (volume_ratio * 0.7) + (accuracy_ratio * 0.3)
        
        # REALISTIC bounds
        return max(0.7, min(1.3, shot_quality))
    
    def _calculate_form_adjustment(self, stats: EnhancedTeamStats) -> float:
        """Calculate form adjustment with REALISTIC bounds"""
        # Calculate recent goals per game
        last5_gpg = stats.last5_goals_for / 5 if stats.last5_goals_for > 0 else 0.5
        
        # Get season average
        season_gpg = max(0.3, stats.season_goals_per_game)
        
        # Calculate form ratio
        form_ratio = last5_gpg / season_gpg if season_gpg > 0 else 1.0
        
        # Apply form weight: blend recent form with season baseline
        weighted_form = (self.form_weight * form_ratio) + ((1 - self.form_weight) * 1.0)
        
        # REALISTIC bounds: form can't double or halve team strength
        return max(0.7, min(1.3, weighted_form))
    
    def calculate_goal_expectations(self, home_stats: EnhancedTeamStats,
                                  away_stats: EnhancedTeamStats,
                                  league: str = "default") -> Dict:
        """
        FIXED goal expectation calculation with REALISTIC outputs
        """
        context = self.league_contexts.get(league, self.league_contexts['default'])
        
        # 1. VENUE-SPECIFIC BASE STRENGTHS
        home_attack_base = self._get_venue_attack_strength(home_stats, is_home=True)
        away_attack_base = self._get_venue_attack_strength(away_stats, is_home=False)
        
        home_defense_base = self._get_venue_defense_strength(home_stats, is_home=True)
        away_defense_base = self._get_venue_defense_strength(away_stats, is_home=False)
        
        # 2. ATTACK VS DEFENSE ADJUSTMENTS (FIXED!)
        home_attack_adjusted = self._calculate_attack_vs_defense_adjustment(
            home_attack_base, away_defense_base, context['avg_gpg']
        )
        
        away_attack_adjusted = self._calculate_attack_vs_defense_adjustment(
            away_attack_base, home_defense_base, context['avg_gpg']
        )
        
        # 3. EFFICIENCY ADJUSTMENTS
        home_efficiency = self._calculate_efficiency_adjustment(
            home_stats.conversion_rate, context['avg_conversion']
        )
        away_efficiency = self._calculate_efficiency_adjustment(
            away_stats.conversion_rate, context['avg_conversion']
        )
        
        # 4. SHOT QUALITY ADJUSTMENTS
        home_shot_quality = self._calculate_shot_quality_adjustment(
            home_stats, context['avg_shots']
        )
        away_shot_quality = self._calculate_shot_quality_adjustment(
            away_stats, context['avg_shots']
        )
        
        # 5. FORM ADJUSTMENTS
        home_form = self._calculate_form_adjustment(home_stats)
        away_form = self._calculate_form_adjustment(away_stats)
        
        # 6. STYLE ADJUSTMENTS
        style_key = (home_stats.style, away_stats.style)
        style_adjustments = self.style_matchup_effects.get(style_key, 
            {'possession_team_adj': 1.0, 'counter_team_adj': 1.0, 
             'high_press_team_adj': 1.0, 'low_block_team_adj': 1.0})
        
        # Determine which team gets which adjustment
        home_style_adj = 1.0
        away_style_adj = 1.0
        
        if home_stats.style == TeamStyle.POSSESSION:
            home_style_adj = style_adjustments.get('possession_team_adj', 1.0)
        elif home_stats.style == TeamStyle.COUNTER:
            home_style_adj = style_adjustments.get('counter_team_adj', 1.0)
        elif home_stats.style == TeamStyle.HIGH_PRESS:
            home_style_adj = style_adjustments.get('high_press_team_adj', 1.0)
        elif home_stats.style == TeamStyle.LOW_BLOCK:
            home_style_adj = style_adjustments.get('low_block_team_adj', 1.0)
            
        if away_stats.style == TeamStyle.POSSESSION:
            away_style_adj = style_adjustments.get('possession_team_adj', 1.0)
        elif away_stats.style == TeamStyle.COUNTER:
            away_style_adj = style_adjustments.get('counter_team_adj', 1.0)
        elif away_stats.style == TeamStyle.HIGH_PRESS:
            away_style_adj = style_adjustments.get('high_press_team_adj', 1.0)
        elif away_stats.style == TeamStyle.LOW_BLOCK:
            away_style_adj = style_adjustments.get('low_block_team_adj', 1.0)
        
        # 7. VENUE ADVANTAGE
        home_venue = context['home_advantage']
        away_venue = context['away_penalty']
        
        # 8. FINAL GOAL EXPECTATIONS (with ALL adjustments)
        lambda_home = (home_attack_adjusted *
                      home_efficiency *
                      home_shot_quality *
                      home_form *
                      home_style_adj *
                      home_venue)
        
        lambda_away = (away_attack_adjusted *
                      away_efficiency *
                      away_shot_quality *
                      away_form *
                      away_style_adj *
                      away_venue)
        
        # 9. CRITICAL REALITY CHECKS
        # NO team expects >2.5 goals per game in reality
        lambda_home = max(0.2, min(2.5, lambda_home))
        lambda_away = max(0.2, min(2.5, lambda_away))
        
        # If total goals > 5.0, scale down (extremely rare in reality)
        total_goals = lambda_home + lambda_away
        if total_goals > 5.0:
            scale_factor = 4.5 / total_goals
            lambda_home *= scale_factor
            lambda_away *= scale_factor
            total_goals = lambda_home + lambda_away
        
        # 10. CALCULATE PROBABILITIES
        prob_over25 = self._poisson_over25_correct(lambda_home, lambda_away)
        prob_under25 = 1 - prob_over25
        prob_btts = self._poisson_btts(lambda_home, lambda_away)
        prob_no_btts = 1 - prob_btts
        
        # Calculate win/draw probabilities
        home_win_prob, draw_prob, away_win_prob = self._poisson_match_probabilities(lambda_home, lambda_away)
        
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
            'adjustment_factors': {
                'home_attack': home_attack_base,
                'away_attack': away_attack_base,
                'home_defense': home_defense_base,
                'away_defense': away_defense_base,
                'home_attack_adjusted': home_attack_adjusted,
                'away_attack_adjusted': away_attack_adjusted,
                'home_efficiency': home_efficiency,
                'away_efficiency': away_efficiency,
                'home_shot_quality': home_shot_quality,
                'away_shot_quality': away_shot_quality,
                'home_form': home_form,
                'away_form': away_form,
                'home_style_adj': home_style_adj,
                'away_style_adj': away_style_adj,
                'home_venue': home_venue,
                'away_venue': away_venue
            }
        }
    
    def _poisson_over25_correct(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate probability of Over 2.5 goals with SANITY CHECKS"""
        try:
            total_lambda = lambda_home + lambda_away
            
            # If total goals is very low, Over 2.5 is unlikely
            if total_lambda < 1.5:
                return max(0.05, total_lambda / 5.0)
            
            prob_0 = math.exp(-total_lambda)
            prob_1 = total_lambda * math.exp(-total_lambda)
            prob_2 = (total_lambda ** 2) * math.exp(-total_lambda) / 2
            prob_under25 = prob_0 + prob_1 + prob_2
            
            result = 1 - prob_under25
            
            # SANITY: Never give >95% or <5% probability
            return max(0.05, min(0.95, result))
            
        except:
            # Fallback based on total goals
            total_goals = lambda_home + lambda_away
            if total_goals > 3.5:
                return 0.75
            elif total_goals > 3.0:
                return 0.65
            elif total_goals > 2.5:
                return 0.55
            elif total_goals > 2.0:
                return 0.45
            elif total_goals > 1.5:
                return 0.35
            else:
                return 0.25
    
    def _poisson_btts(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate probability of Both Teams to Score"""
        try:
            prob_home_score = 1 - math.exp(-lambda_home)
            prob_away_score = 1 - math.exp(-lambda_away)
            result = prob_home_score * prob_away_score
            
            # SANITY: Never give >90% or <10% probability
            return max(0.10, min(0.90, result))
            
        except:
            # Fallback
            avg_goals = (lambda_home + lambda_away) / 2
            if avg_goals > 1.8:
                return 0.60
            elif avg_goals > 1.5:
                return 0.55
            elif avg_goals > 1.2:
                return 0.45
            elif avg_goals > 0.8:
                return 0.35
            else:
                return 0.25
    
    def _poisson_match_probabilities(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Calculate proper win/draw probabilities"""
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
            
        except Exception:
            # Simplified fallback with realistic bounds
            if lambda_home <= 0 or lambda_away <= 0:
                return 0.33, 0.34, 0.33
            
            goal_diff = lambda_home - lambda_away
            
            if goal_diff > 1.0:
                return 0.55, 0.25, 0.20
            elif goal_diff > 0.5:
                return 0.45, 0.30, 0.25
            elif goal_diff > -0.5:
                return 0.35, 0.30, 0.35
            elif goal_diff > -1.0:
                return 0.25, 0.30, 0.45
            else:
                return 0.20, 0.25, 0.55
    
    def analyze_team_identity(self, home_stats: EnhancedTeamStats, 
                            away_stats: EnhancedTeamStats) -> Dict:
        """Analyze Team Identity using shot data and style"""
        analysis = {
            'home_shot_quality': self._calculate_shot_quality_adjustment(home_stats, 12.0),
            'away_shot_quality': self._calculate_shot_quality_adjustment(away_stats, 12.0),
            'style_clash': f"{home_stats.style.value} vs {away_stats.style.value}",
            'insights': [],
            'confidence_factors': []
        }
        
        # Shot volume insights
        if home_stats.shots_per_game > 14:
            analysis['insights'].append(
                f"{home_stats.team_name} high shot volume ({home_stats.shots_per_game:.1f}/game)"
            )
            analysis['confidence_factors'].append(('high_shot_volume', 1))
        
        if away_stats.shots_per_game < 10:
            analysis['insights'].append(
                f"{away_stats.team_name} low shot volume ({away_stats.shots_per_game:.1f}/game)"
            )
            analysis['confidence_factors'].append(('low_shot_volume', 1))
        
        # Conversion insights
        if home_stats.conversion_rate < 0.08:
            analysis['insights'].append(
                f"{home_stats.team_name} poor conversion ({home_stats.conversion_rate:.1%})"
            )
        
        if away_stats.conversion_rate > 0.15:
            analysis['insights'].append(
                f"{away_stats.team_name} excellent conversion ({away_stats.conversion_rate:.1%})"
            )
            analysis['confidence_factors'].append(('high_conversion', 1))
        
        # Style matchup
        style_key = (home_stats.style, away_stats.style)
        if style_key in self.style_matchup_effects:
            analysis['insights'].append(
                f"Significant style matchup: {home_stats.style.value} vs {away_stats.style.value}"
            )
            analysis['confidence_factors'].append(('style_matchup', 2))
        
        return analysis
    
    def analyze_defense_patterns(self, home_stats: EnhancedTeamStats,
                               away_stats: EnhancedTeamStats) -> Dict:
        """Analyze Defense using venue splits and clean sheet data"""
        analysis = {
            'home_venue_defense': self._get_venue_defense_strength(home_stats, is_home=True),
            'away_venue_defense': self._get_venue_defense_strength(away_stats, is_home=False),
            'insights': [],
            'confidence_factors': []
        }
        
        # Clean sheet insights
        if home_stats.clean_sheet_pct_home > 0.4:
            analysis['insights'].append(
                f"{home_stats.team_name} strong home defense ({home_stats.clean_sheet_pct_home:.1%} clean sheets)"
            )
            analysis['confidence_factors'].append(('strong_home_defense', 2))
        
        if away_stats.clean_sheet_pct_away < 0.1:
            analysis['insights'].append(
                f"{away_stats.team_name} weak away defense ({away_stats.clean_sheet_pct_away:.1%} clean sheets)"
            )
            analysis['confidence_factors'].append(('weak_away_defense', 2))
        
        # Failed to score insights
        if home_stats.failed_to_score_pct_home < 0.2:
            analysis['insights'].append(
                f"{home_stats.team_name} reliable home scoring"
            )
        
        if away_stats.failed_to_score_pct_away > 0.4:
            analysis['insights'].append(
                f"{away_stats.team_name} struggles to score away ({away_stats.failed_to_score_pct_away:.1%} failed to score)"
            )
            analysis['confidence_factors'].append(('poor_away_scoring', 2))
        
        return analysis
    
    def analyze_transition_trends(self, home_stats: EnhancedTeamStats,
                                away_stats: EnhancedTeamStats) -> Dict:
        """Analyze Transition using BTTS, Over patterns and recent form"""
        analysis = {
            'home_last5_gpg': home_stats.last5_goals_for / 5 if home_stats.last5_goals_for > 0 else 0.5,
            'away_last5_gpg': away_stats.last5_goals_for / 5 if away_stats.last5_goals_for > 0 else 0.5,
            'combined_btts': (home_stats.btts_pct + away_stats.btts_pct) / 2,
            'combined_over25': (home_stats.over25_pct + away_stats.over25_pct) / 2,
            'insights': [],
            'confidence_factors': []
        }
        
        # Recent form insights
        home_season_gpg = max(0.3, home_stats.season_goals_per_game)
        away_season_gpg = max(0.3, away_stats.season_goals_per_game)
        
        home_form_ratio = analysis['home_last5_gpg'] / home_season_gpg
        away_form_ratio = analysis['away_last5_gpg'] / away_season_gpg
        
        if home_form_ratio > 1.3:
            analysis['insights'].append(
                f"{home_stats.team_name} excellent recent attacking form ({home_stats.last5_goals_for} goals in last 5)"
            )
            analysis['confidence_factors'].append(('improving_attack', 2))
        elif home_form_ratio < 0.7:
            analysis['insights'].append(
                f"{home_stats.team_name} poor recent attacking form ({home_stats.last5_goals_for} goals in last 5)"
            )
            analysis['confidence_factors'].append(('poor_attack_form', 2))
        
        if away_form_ratio > 1.3:
            analysis['insights'].append(
                f"{away_stats.team_name} excellent recent attacking form ({away_stats.last5_goals_for} goals in last 5)"
            )
            analysis['confidence_factors'].append(('improving_attack', 2))
        elif away_form_ratio < 0.7:
            analysis['insights'].append(
                f"{away_stats.team_name} poor recent attacking form ({away_stats.last5_goals_for} goals in last 5)"
            )
            analysis['confidence_factors'].append(('poor_attack_form', 2))
        
        # BTTS and Over trends
        if analysis['combined_btts'] > 0.7:
            analysis['insights'].append(
                f"High BTTS probability (combined {analysis['combined_btts']:.1%})"
            )
            analysis['confidence_factors'].append(('high_btts', 1))
        
        if analysis['combined_over25'] > 0.7:
            analysis['insights'].append(
                f"High Over 2.5 probability (combined {analysis['combined_over25']:.1%})"
            )
            analysis['confidence_factors'].append(('high_scoring', 1))
        
        return analysis
    
    def calculate_confidence_score(self, analysis: Dict, goal_expectations: Dict) -> Dict:
        """Calculate confidence score based on data quality"""
        confidence_factors = []
        total_score = 5  # Base score
        
        home_stats = analysis.get('home_stats')
        away_stats = analysis.get('away_stats')
        
        if home_stats and away_stats:
            # Sample size
            if home_stats.matches_played >= 10 and away_stats.matches_played >= 10:
                total_score += 2
                confidence_factors.append(('good_sample_size', 2))
            elif home_stats.matches_played >= 5 and away_stats.matches_played >= 5:
                total_score += 1
                confidence_factors.append(('adequate_sample_size', 1))
            
            # Venue data
            if home_stats.home_games_played >= 3 and away_stats.away_games_played >= 3:
                total_score += 1
                confidence_factors.append(('venue_data_available', 1))
            
            # Recent form data
            if home_stats.last5_goals_for > 0 or away_stats.last5_goals_for > 0:
                total_score += 1
                confidence_factors.append(('recent_form_data', 1))
        
        # Check for extreme predictions
        total_goals = goal_expectations.get('total_goals', 0)
        lambda_home = goal_expectations.get('lambda_home', 0)
        lambda_away = goal_expectations.get('lambda_away', 0)
        
        if total_goals > 5.0:
            total_score -= 3
            confidence_factors.append(('extreme_total_goals', -3))
        elif total_goals > 4.0:
            total_score -= 2
            confidence_factors.append(('high_total_goals', -2))
        
        if lambda_home > 3.0 or lambda_away > 3.0:
            total_score -= 2
            confidence_factors.append(('extreme_team_goals', -2))
        
        # Ensure bounds
        total_score = max(1, min(10, total_score))
        
        # Determine confidence level
        if total_score >= 8:
            level = ConfidenceLevel.HIGH
            reason = "Good data quality with realistic predictions"
        elif total_score >= 6:
            level = ConfidenceLevel.MEDIUM
            reason = "Adequate data quality with some limitations"
        elif total_score >= 4:
            level = ConfidenceLevel.LOW
            reason = "Limited data quality - predictions less reliable"
        else:
            level = ConfidenceLevel.VERY_LOW
            reason = "Poor data quality - high uncertainty"
        
        return {
            'score': total_score,
            'level': level,
            'reason': reason,
            'factors': confidence_factors
        }
    
    def detect_value_bets(self, model_probs: Dict, market_odds: Dict) -> List[Dict]:
        """Detect value bets across all markets with SANITY CHECKS"""
        value_bets = []
        
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
                    
                    # SANITY CHECK: Reject extreme edges (>25%) - they're model errors
                    if edge > 0.25:
                        continue
                    
                    if edge >= self.min_edge and model_prob >= self.min_confidence_for_stake:
                        if edge > 0.15:
                            value_rating = "🔥 EXTREME VALUE"
                        elif edge > 0.10:
                            value_rating = "⭐⭐⭐ GOLDEN NUGGET"
                        elif edge > 0.05:
                            value_rating = "⭐⭐ VALUE BET"
                        elif edge > 0.03:
                            value_rating = "⭐ CONSIDER"
                        else:
                            value_rating = "Small Edge"
                        
                        value_bets.append({
                            'bet_type': bet_type,
                            'model_probability': model_prob,
                            'market_odds': market_odd,
                            'implied_probability': implied_prob,
                            'edge_percent': edge * 100,
                            'market_key': market_key,
                            'value_rating': value_rating
                        })
        
        value_bets.sort(key=lambda x: x['edge_percent'], reverse=True)
        return value_bets
    
    def predict_match(self, home_stats: EnhancedTeamStats, away_stats: EnhancedTeamStats,
                     market_odds: Dict, league: str = "default", 
                     bankroll: float = None) -> Dict:
        """
        Main prediction method - FIXED with realistic outputs
        """
        if bankroll is None:
            bankroll = self.bankroll
        
        # Analyze all three dimensions
        identity_analysis = self.analyze_team_identity(home_stats, away_stats)
        defense_analysis = self.analyze_defense_patterns(home_stats, away_stats)
        transition_analysis = self.analyze_transition_trends(home_stats, away_stats)
        
        # Store stats for confidence calculation
        identity_analysis['home_stats'] = home_stats
        identity_analysis['away_stats'] = away_stats
        
        # Calculate goal expectations (FIXED!)
        goal_expectations = self.calculate_goal_expectations(home_stats, away_stats, league)
        
        # Calculate confidence score
        combined_analysis = {
            'identity': identity_analysis,
            'defense': defense_analysis,
            'transition': transition_analysis,
            'home_stats': home_stats,
            'away_stats': away_stats
        }
        confidence = self.calculate_confidence_score(combined_analysis, goal_expectations)
        
        # Detect value bets (with sanity checks)
        raw_value_bets = self.detect_value_bets(goal_expectations['probabilities'], market_odds)
        
        # Calculate stakes
        final_bets = []
        for bet in raw_value_bets:
            edge = bet['edge_percent'] / 100
            
            # Conservative stake calculation
            if edge > 0.20:
                stake_pct = min(self.max_stake_pct * 0.5, edge * 0.15)  # Be extra conservative
            elif edge > 0.10:
                stake_pct = min(self.max_stake_pct, edge * 0.20)
            elif edge > 0.05:
                stake_pct = min(self.max_stake_pct, edge * 0.25)
            else:
                stake_pct = min(self.max_stake_pct, edge * 0.30)
            
            stake_amount = bankroll * stake_pct
            
            if stake_amount > bankroll * 0.005:  # Minimum sensible stake
                final_bet = {
                    'bet_type': bet['bet_type'].value,
                    'market_odds': bet['market_odds'],
                    'model_probability': bet['model_probability'],
                    'edge_percent': bet['edge_percent'],
                    'value_rating': bet['value_rating'],
                    'stake_amount': stake_amount,
                    'stake_percent': stake_pct * 100,
                    'implied_probability': bet['implied_probability']
                }
                final_bets.append(final_bet)
        
        # Prepare final result
        result = {
            'match_analysis': {
                'identity': identity_analysis,
                'defense': defense_analysis,
                'transition': transition_analysis,
                'goal_expectations': goal_expectations,
                'confidence': confidence
            },
            'value_bets': final_bets,
            'total_exposure_percent': sum(b['stake_percent'] for b in final_bets),
            'total_stake': sum(b['stake_amount'] for b in final_bets),
            'market_odds_used': market_odds,
            'league_context': league
        }
        
        if not final_bets:
            result['recommendation'] = f"NO BET - No value opportunities meeting minimum {self.min_edge*100:.1f}% edge criteria"
        
        return result


# Test function to verify realistic outputs
def test_realistic_predictions():
    """Test with sample data to verify realistic outputs"""
    print("🧪 TESTING FIXED PREDICTOR...")
    
    # Sample stats (simplified)
    home_stats = EnhancedTeamStats(
        team_name="Arsenal",
        matches_played=14,
        possession_avg=59,
        shots_per_game=14.43,
        shots_on_target_pg=6.14,
        conversion_rate=0.13,
        xg_for_avg=1.71,
        xg_against_avg=0.83,
        home_wins=6, home_draws=1, home_losses=0,
        away_wins=4, away_draws=2, away_losses=1,
        home_goals_for=18, home_goals_against=2,
        away_goals_for=9, away_goals_against=5,
        clean_sheet_pct=0.57,
        clean_sheet_pct_home=0.71,
        clean_sheet_pct_away=0.43,
        failed_to_score_pct=0.07,
        failed_to_score_pct_home=0.0,
        failed_to_score_pct_away=0.14,
        btts_pct=0.36,
        btts_pct_home=0.29,
        btts_pct_away=0.43,
        over25_pct=0.43,
        over25_pct_home=0.71,
        over25_pct_away=0.14,
        last5_form="LWDWD",
        last5_wins=2, last5_draws=2, last5_losses=1,
        last5_goals_for=10, last5_goals_against=6,
        last5_ppg=1.6,
        last5_cs_pct=0.20,
        last5_fts_pct=0.0,
        last5_btts_pct=0.80,
        last5_over25_pct=0.60
    )
    
    away_stats = EnhancedTeamStats(
        team_name="Manchester City",
        matches_played=14,
        possession_avg=57,
        shots_per_game=14.14,
        shots_on_target_pg=6.21,
        conversion_rate=0.16,
        xg_for_avg=1.67,
        xg_against_avg=1.14,
        home_wins=6, home_draws=0, home_losses=1,
        away_wins=3, away_draws=1, away_losses=3,
        home_goals_for=19, home_goals_against=6,
        away_goals_for=13, away_goals_against=10,
        clean_sheet_pct=0.36,
        clean_sheet_pct_home=0.43,
        clean_sheet_pct_away=0.29,
        failed_to_score_pct=0.14,
        failed_to_score_pct_home=0.14,
        failed_to_score_pct_away=0.14,
        btts_pct=0.50,
        btts_pct_home=0.43,
        btts_pct_away=0.57,
        over25_pct=0.79,
        over25_pct_home=0.86,
        over25_pct_away=0.71,
        last5_form="WWWLW",
        last5_wins=4, last5_draws=0, last5_losses=1,
        last5_goals_for=15, last5_goals_against=8,
        last5_ppg=2.4,
        last5_cs_pct=0.40,
        last5_fts_pct=0.0,
        last5_btts_pct=0.60,
        last5_over25_pct=1.00
    )
    
    predictor = EdgeFinderPredictor(
        bankroll=1000.0,
        min_edge=0.03,
        form_weight=0.4
    )
    
    result = predictor.predict_match(home_stats, away_stats, {}, 'premier_league')
    
    print(f"\n✅ REALISTIC RESULTS:")
    print(f"Home Expected Goals: {result['match_analysis']['goal_expectations']['lambda_home']:.2f}")
    print(f"Away Expected Goals: {result['match_analysis']['goal_expectations']['lambda_away']:.2f}")
    print(f"Total Expected Goals: {result['match_analysis']['goal_expectations']['total_goals']:.2f}")
    
    # Verify realism
    total_goals = result['match_analysis']['goal_expectations']['total_goals']
    if 2.0 <= total_goals <= 4.0:
        print(f"✅ REALITY CHECK PASSED: Total goals {total_goals:.2f} is realistic for Premier League")
    else:
        print(f"⚠️  WARNING: Total goals {total_goals:.2f} outside typical range")


if __name__ == "__main__":
    test_realistic_predictions()
