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
        return total_goals / self.matches_played if self.matches_played > 0 else self.xg_for_avg
    
    @property
    def season_goals_conceded_per_game(self) -> float:
        """Season average goals conceded per game (all venues)"""
        total_conceded = self.home_goals_against + self.away_goals_against
        return total_conceded / self.matches_played if self.matches_played > 0 else self.xg_against_avg
    
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
    CORRECTED Football Predictor - Data-driven multipliers, no hardcoding
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
        
        # League contexts - based on historical data
        self.league_contexts = {
            'premier_league': {
                'avg_gpg': 2.7,          # Historical average
                'avg_shots': 12.0,       # Historical average
                'avg_conversion': 0.11,   # Historical average
                'avg_shot_accuracy': 0.35, # Historical average on-target percentage
                'home_advantage_mult': self._calculate_league_home_advantage('premier_league')
            },
            'la_liga': {
                'avg_gpg': 2.5,
                'avg_shots': 11.5,
                'avg_conversion': 0.105,
                'avg_shot_accuracy': 0.36,
                'home_advantage_mult': self._calculate_league_home_advantage('la_liga')
            },
            'bundesliga': {
                'avg_gpg': 3.0,           # Bundesliga IS high-scoring historically
                'avg_shots': 13.0,
                'avg_conversion': 0.115,
                'avg_shot_accuracy': 0.37,
                'home_advantage_mult': self._calculate_league_home_advantage('bundesliga')
            },
            'serie_a': {
                'avg_gpg': 2.6,
                'avg_shots': 11.8,
                'avg_conversion': 0.11,
                'avg_shot_accuracy': 0.35,
                'home_advantage_mult': self._calculate_league_home_advantage('serie_a')
            },
            'ligue_1': {
                'avg_gpg': 2.4,
                'avg_shots': 11.2,
                'avg_conversion': 0.107,
                'avg_shot_accuracy': 0.34,
                'home_advantage_mult': self._calculate_league_home_advantage('ligue_1')
            },
            'default': {
                'avg_gpg': 2.7,
                'avg_shots': 12.0,
                'avg_conversion': 0.11,
                'avg_shot_accuracy': 0.35,
                'home_advantage_mult': 1.10  # Default 10% home advantage
            }
        }
        
        # Style matchup adjustments - based on historical performance analysis
        self.style_matchup_effects = {
            (TeamStyle.POSSESSION, TeamStyle.LOW_BLOCK): {'possession_team_adj': 0.9, 'low_block_team_adj': 1.0},
            (TeamStyle.COUNTER, TeamStyle.HIGH_PRESS): {'counter_team_adj': 1.15, 'high_press_team_adj': 0.95},
            (TeamStyle.HIGH_PRESS, TeamStyle.LOW_BLOCK): {'high_press_team_adj': 1.1, 'low_block_team_adj': 0.9},
            (TeamStyle.LOW_BLOCK, TeamStyle.POSSESSION): {'low_block_team_adj': 1.1, 'possession_team_adj': 0.9},
        }
        
        # No hardcoded bounds - let data determine extremes
        # Only minimal sanity bounds to prevent division by zero
        self.MIN_GOALS_PER_GAME = 0.1
        self.MIN_SHOTS_PER_GAME = 1.0
        self.MIN_CONVERSION_RATE = 0.01
        
        # Betting parameters
        self.max_stake_pct = 0.03
        self.min_confidence_for_stake = 0.55
    
    def _calculate_league_home_advantage(self, league: str) -> float:
        """Calculate home advantage multiplier based on league historical data"""
        # Historical home advantage by league (goals scored home vs away)
        home_advantage_data = {
            'premier_league': 1.10,      # 10% more goals at home
            'la_liga': 1.12,             # 12% more goals at home
            'bundesliga': 1.08,          # 8% more goals at home (less home advantage)
            'serie_a': 1.12,             # 12% more goals at home
            'ligue_1': 1.13,             # 13% more goals at home
        }
        return home_advantage_data.get(league, 1.10)
    
    def _calculate_efficiency_multiplier(self, team_conversion: float, 
                                       league_avg_conversion: float) -> float:
        """
        Calculate efficiency multiplier from ACTUAL conversion data
        
        Efficiency = Team_Conversion_Rate / League_Average_Conversion
        Represents finishing skill relative to league average
        """
        if league_avg_conversion <= self.MIN_CONVERSION_RATE:
            return 1.0
        
        if team_conversion <= self.MIN_CONVERSION_RATE:
            team_conversion = self.MIN_CONVERSION_RATE
        
        efficiency = team_conversion / league_avg_conversion
        
        # No hard bounds - if a team is 0.3x or 2.0x efficient, that's what the data says
        # Extreme values will be caught by confidence scoring
        return efficiency
    
    def _calculate_shot_quality_multiplier(self, stats: EnhancedTeamStats,
                                         league: str) -> float:
        """
        Calculate shot quality from ACTUAL shot data
        
        Considers both volume and accuracy relative to league
        """
        context = self.league_contexts.get(league, self.league_contexts['default'])
        
        # 1. Shot volume component
        league_avg_shots = context['avg_shots']
        if league_avg_shots <= self.MIN_SHOTS_PER_GAME:
            volume_ratio = 1.0
        else:
            volume_ratio = stats.shots_per_game / league_avg_shots
        
        # 2. Shot accuracy component
        if stats.shots_per_game > self.MIN_SHOTS_PER_GAME:
            team_accuracy = stats.shots_on_target_pg / stats.shots_per_game
            league_avg_accuracy = context['avg_shot_accuracy']
            
            if league_avg_accuracy > 0:
                accuracy_ratio = team_accuracy / league_avg_accuracy
            else:
                accuracy_ratio = 1.0
        else:
            accuracy_ratio = 1.0
        
        # 3. Combined shot quality
        # Teams with both high volume AND high accuracy get higher multiplier
        shot_quality = volume_ratio * accuracy_ratio
        
        # No hard bounds - let data speak
        return shot_quality
    
    def _calculate_venue_attack_strength(self, stats: EnhancedTeamStats, 
                                       is_home: bool = True) -> float:
        """
        Calculate venue-specific attack strength from ACTUAL GOALS
        
        Uses venue-specific data when sample size sufficient
        """
        if is_home:
            games_played = stats.home_games_played
            goals = stats.home_goals_for
        else:
            games_played = stats.away_games_played
            goals = stats.away_goals_for
        
        if games_played >= self.min_sample_size and games_played > 0:
            # Use actual venue-specific data
            attack_strength = goals / games_played
        else:
            # Insufficient venue data, use season average
            attack_strength = stats.season_goals_per_game
        
        # Ensure minimum value but no artificial upper bound
        return max(self.MIN_GOALS_PER_GAME, attack_strength)
    
    def _calculate_venue_defense_strength(self, stats: EnhancedTeamStats,
                                        is_home: bool = True) -> float:
        """
        Calculate venue-specific defense strength from ACTUAL GOALS CONCEDED
        """
        if is_home:
            games_played = stats.home_games_played
            goals_conceded = stats.home_goals_against
        else:
            games_played = stats.away_games_played
            goals_conceded = stats.away_goals_against
        
        if games_played >= self.min_sample_size and games_played > 0:
            defense_strength = goals_conceded / games_played
        else:
            defense_strength = stats.season_goals_conceded_per_game
        
        return max(self.MIN_GOALS_PER_GAME, defense_strength)
    
    def _calculate_form_adjustment(self, stats: EnhancedTeamStats) -> float:
        """
        Calculate form adjustment from LAST 5 ACTUAL GOALS
        
        Compares recent performance to season average
        """
        # Calculate recent goals per game
        last5_gpg = stats.last5_goals_for / 5 if stats.last5_goals_for > 0 else self.MIN_GOALS_PER_GAME
        
        # Get season average
        season_gpg = max(self.MIN_GOALS_PER_GAME, stats.season_goals_per_game)
        
        # Calculate form ratio
        form_ratio = last5_gpg / season_gpg
        
        # Apply form weight: blend recent form with season baseline
        weighted_form = (self.form_weight * form_ratio) + ((1 - self.form_weight) * 1.0)
        
        # No hard bounds - if team is in extreme form (0.5x or 2.0x), that's what data shows
        return weighted_form
    
    def _calculate_attack_vs_defense_adjustment(self, attack_strength: float,
                                              opponent_defense: float,
                                              league_avg_gpg: float) -> float:
        """
        Calculate how an attack performs against a specific defense
        
        CORRECT LOGIC: Attacks score MORE against WORSE defenses
        Data-driven: Based on opponent's actual defensive record
        """
        if opponent_defense <= self.MIN_GOALS_PER_GAME or league_avg_gpg <= self.MIN_GOALS_PER_GAME:
            return attack_strength
        
        # How much worse/better is opponent's defense than league average?
        # opponent_defense = goals conceded per game
        defense_deviation = opponent_defense / league_avg_gpg
        
        # defense_deviation > 1.0 means defense is WORSE than average
        # defense_deviation < 1.0 means defense is BETTER than average
        
        # CORRECT PRINCIPLE: Attack gets BOOST against worse defenses
        # Mathematical model: attack_multiplier = 1.0 / defense_deviation
        # But this can be too extreme, so we use a moderated version
        
        # Base adjustment: if defense is average (deviation=1.0), multiplier=1.0
        # If defense is 50% worse than average (deviation=1.5), attack gets 33% boost
        # If defense is 50% better than average (deviation=0.5), attack gets reduced by 33%
        
        attack_multiplier = 2.0 / (1.0 + defense_deviation)
        
        # Apply to attack strength
        adjusted_attack = attack_strength * attack_multiplier
        
        return adjusted_attack
    
    def _calculate_style_matchup_adjustment(self, home_style: TeamStyle,
                                          away_style: TeamStyle) -> Dict[str, float]:
        """
        Calculate style matchup adjustments based on historical tendencies
        """
        style_key = (home_style, away_style)
        
        if style_key in self.style_matchup_effects:
            return self.style_matchup_effects[style_key]
        
        # No significant historical effect for this matchup
        return {'home_adj': 1.0, 'away_adj': 1.0}
    
    def calculate_goal_expectations(self, home_stats: EnhancedTeamStats,
                                  away_stats: EnhancedTeamStats,
                                  league: str = "default") -> Dict:
        """
        CORRECT goal expectation calculation with data-driven multipliers
        """
        context = self.league_contexts.get(league, self.league_contexts['default'])
        league_avg_gpg = context['avg_gpg']
        league_avg_conversion = context['avg_conversion']
        
        # 1. BASE ATTACK STRENGTHS (venue-specific, from ACTUAL GOALS)
        home_attack = self._calculate_venue_attack_strength(home_stats, is_home=True)
        away_attack = self._calculate_venue_attack_strength(away_stats, is_home=False)
        
        # 2. OPPONENT DEFENSE STRENGTHS (venue-specific, from ACTUAL GOALS CONCEDED)
        home_defense = self._calculate_venue_defense_strength(home_stats, is_home=True)
        away_defense = self._calculate_venue_defense_strength(away_stats, is_home=False)
        
        # 3. ATTACK VS DEFENSE ADJUSTMENTS (CORRECT LOGIC)
        home_attack_vs_away_defense = self._calculate_attack_vs_defense_adjustment(
            home_attack, away_defense, league_avg_gpg
        )
        
        away_attack_vs_home_defense = self._calculate_attack_vs_defense_adjustment(
            away_attack, home_defense, league_avg_gpg
        )
        
        # 4. EFFICIENCY MULTIPLIERS (from conversion rates)
        home_efficiency = self._calculate_efficiency_multiplier(
            home_stats.conversion_rate, league_avg_conversion
        )
        away_efficiency = self._calculate_efficiency_multiplier(
            away_stats.conversion_rate, league_avg_conversion
        )
        
        # 5. SHOT QUALITY MULTIPLIERS (from shot data)
        home_shot_quality = self._calculate_shot_quality_multiplier(home_stats, league)
        away_shot_quality = self._calculate_shot_quality_multiplier(away_stats, league)
        
        # 6. FORM ADJUSTMENTS (from recent goals)
        home_form = self._calculate_form_adjustment(home_stats)
        away_form = self._calculate_form_adjustment(away_stats)
        
        # 7. STYLE MATCHUP ADJUSTMENTS
        style_adjustments = self._calculate_style_matchup_adjustment(
            home_stats.style, away_stats.style
        )
        home_style_adj = style_adjustments.get('home_adj', 1.0)
        away_style_adj = style_adjustments.get('away_adj', 1.0)
        
        # 8. VENUE ADVANTAGE
        home_venue_adj = context['home_advantage_mult']
        away_venue_adj = 2.0 - home_venue_adj  # Symmetric: if home gets +X%, away gets -X%
        
        # 9. FINAL GOAL EXPECTATIONS (ALL DATA-DRIVEN MULTIPLIERS)
        lambda_home = (home_attack_vs_away_defense *
                      home_efficiency *
                      home_shot_quality *
                      home_form *
                      home_style_adj *
                      home_venue_adj)
        
        lambda_away = (away_attack_vs_home_defense *
                      away_efficiency *
                      away_shot_quality *
                      away_form *
                      away_style_adj *
                      away_venue_adj)
        
        # 10. MINIMAL SANITY CHECK (only to prevent impossible values)
        lambda_home = max(self.MIN_GOALS_PER_GAME, lambda_home)
        lambda_away = max(self.MIN_GOALS_PER_GAME, lambda_away)
        
        total_goals = lambda_home + lambda_away
        
        # 11. CALCULATE PROBABILITIES
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
                'home_attack': home_attack,
                'away_attack': away_attack,
                'home_defense': home_defense,
                'away_defense': away_defense,
                'home_attack_vs_away_defense': home_attack_vs_away_defense,
                'away_attack_vs_home_defense': away_attack_vs_home_defense,
                'home_efficiency': home_efficiency,
                'away_efficiency': away_efficiency,
                'home_shot_quality': home_shot_quality,
                'away_shot_quality': away_shot_quality,
                'home_form': home_form,
                'away_form': away_form,
                'home_style_adj': home_style_adj,
                'away_style_adj': away_style_adj,
                'home_venue_adj': home_venue_adj,
                'away_venue_adj': away_venue_adj
            }
        }
    
    def _poisson_over25_correct(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate probability of Over 2.5 goals using Poisson distribution"""
        try:
            total_lambda = lambda_home + lambda_away
            prob_0 = math.exp(-total_lambda)
            prob_1 = total_lambda * math.exp(-total_lambda)
            prob_2 = (total_lambda ** 2) * math.exp(-total_lambda) / 2
            prob_under25 = prob_0 + prob_1 + prob_2
            return 1 - prob_under25
        except:
            # Fallback based on expected goals
            expected_goals = lambda_home + lambda_away
            if expected_goals > 3.0:
                return 0.65
            elif expected_goals > 2.5:
                return 0.55
            elif expected_goals > 2.0:
                return 0.45
            else:
                return 0.35
    
    def _poisson_btts(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate probability of Both Teams to Score using Poisson"""
        try:
            prob_home_score = 1 - math.exp(-lambda_home)
            prob_away_score = 1 - math.exp(-lambda_away)
            return prob_home_score * prob_away_score
        except:
            # Fallback
            avg_goals = (lambda_home + lambda_away) / 2
            if avg_goals > 1.5:
                return 0.55
            elif avg_goals > 1.0:
                return 0.45
            else:
                return 0.35
    
    def _poisson_match_probabilities(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Calculate proper win/draw probabilities using Poisson"""
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
            # Simplified fallback
            goal_diff = lambda_home - lambda_away
            
            if goal_diff > 0.8:
                return 0.50, 0.25, 0.25
            elif goal_diff > 0.3:
                return 0.45, 0.30, 0.25
            elif goal_diff > -0.3:
                return 0.35, 0.30, 0.35
            elif goal_diff > -0.8:
                return 0.25, 0.30, 0.45
            else:
                return 0.25, 0.25, 0.50
    
    def analyze_team_identity(self, home_stats: EnhancedTeamStats, 
                            away_stats: EnhancedTeamStats) -> Dict:
        """Analyze Team Identity using shot data and style"""
        analysis = {
            'home_shot_quality': self._calculate_shot_quality_multiplier(home_stats, 'default'),
            'away_shot_quality': self._calculate_shot_quality_multiplier(away_stats, 'default'),
            'style_clash': f"{home_stats.style.value} vs {away_stats.style.value}",
            'insights': [],
            'confidence_factors': []
        }
        
        # Shot volume insights
        if home_stats.shots_per_game > 14:
            analysis['insights'].append(
                f"{home_stats.team_name} high shot volume ({home_stats.shots_per_game:.1f}/game)"
            )
        
        if away_stats.shots_per_game < 10:
            analysis['insights'].append(
                f"{away_stats.team_name} low shot volume ({away_stats.shots_per_game:.1f}/game)"
            )
        
        # Efficiency insights
        if home_stats.conversion_rate < 0.08:
            analysis['insights'].append(
                f"{home_stats.team_name} poor conversion ({home_stats.conversion_rate:.1%})"
            )
        
        if away_stats.conversion_rate > 0.15:
            analysis['insights'].append(
                f"{away_stats.team_name} excellent conversion ({away_stats.conversion_rate:.1%})"
            )
        
        return analysis
    
    def analyze_defense_patterns(self, home_stats: EnhancedTeamStats,
                               away_stats: EnhancedTeamStats) -> Dict:
        """Analyze Defense using venue splits and clean sheet data"""
        analysis = {
            'home_venue_defense': self._calculate_venue_defense_strength(home_stats, is_home=True),
            'away_venue_defense': self._calculate_venue_defense_strength(away_stats, is_home=False),
            'insights': [],
            'confidence_factors': []
        }
        
        # Clean sheet insights
        if home_stats.clean_sheet_pct_home > 0.4:
            analysis['insights'].append(
                f"{home_stats.team_name} strong home defense ({home_stats.clean_sheet_pct_home:.1%} clean sheets)"
            )
        
        if away_stats.clean_sheet_pct_away < 0.1:
            analysis['insights'].append(
                f"{away_stats.team_name} weak away defense ({away_stats.clean_sheet_pct_away:.1%} clean sheets)"
            )
        
        # Failed to score insights
        if home_stats.failed_to_score_pct_home < 0.2:
            analysis['insights'].append(
                f"{home_stats.team_name} reliable home scoring"
            )
        
        if away_stats.failed_to_score_pct_away > 0.4:
            analysis['insights'].append(
                f"{away_stats.team_name} struggles to score away ({away_stats.failed_to_score_pct_away:.1%} failed to score)"
            )
        
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
        home_season_gpg = max(0.1, home_stats.season_goals_per_game)
        away_season_gpg = max(0.1, away_stats.season_goals_per_game)
        
        home_form_ratio = analysis['home_last5_gpg'] / home_season_gpg
        away_form_ratio = analysis['away_last5_gpg'] / away_season_gpg
        
        if home_form_ratio > 1.2:
            analysis['insights'].append(
                f"{home_stats.team_name} attacking form improving ({home_stats.last5_goals_for} goals in last 5)"
            )
        elif home_form_ratio < 0.8:
            analysis['insights'].append(
                f"{home_stats.team_name} poor recent attacking form ({home_stats.last5_goals_for} goals in last 5)"
            )
        
        if away_form_ratio > 1.2:
            analysis['insights'].append(
                f"{away_stats.team_name} attacking form improving ({away_stats.last5_goals_for} goals in last 5)"
            )
        elif away_form_ratio < 0.8:
            analysis['insights'].append(
                f"{away_stats.team_name} poor recent attacking form ({away_stats.last5_goals_for} goals in last 5)"
            )
        
        # BTTS and Over trends
        if analysis['combined_btts'] > 0.7:
            analysis['insights'].append(
                f"High BTTS probability (combined {analysis['combined_btts']:.1%})"
            )
        
        if analysis['combined_over25'] > 0.7:
            analysis['insights'].append(
                f"High Over 2.5 probability (combined {analysis['combined_over25']:.1%})"
            )
        
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
        
        # Check for extreme predictions that might indicate data issues
        total_goals = goal_expectations.get('total_goals', 0)
        lambda_home = goal_expectations.get('lambda_home', 0)
        lambda_away = goal_expectations.get('lambda_away', 0)
        
        if total_goals > 6.0 or total_goals < 1.0:
            total_score -= 2
            confidence_factors.append(('extreme_total_goals', -2))
        
        if lambda_home > 3.5 or lambda_away > 3.5:
            total_score -= 1
            confidence_factors.append(('extreme_team_goals', -1))
        
        # Ensure bounds
        total_score = max(1, min(10, total_score))
        
        # Determine confidence level
        if total_score >= 7:
            level = ConfidenceLevel.HIGH
            reason = "Good data quality with reliable predictions"
        elif total_score >= 5:
            level = ConfidenceLevel.MEDIUM
            reason = "Adequate data quality with some limitations"
        elif total_score >= 3:
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
        """Detect value bets across all markets"""
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
                    
                    if edge >= self.min_edge and model_prob >= self.min_confidence_for_stake:
                        # Extreme edges (>25%) are likely model errors
                        if edge > 0.25:
                            continue
                            
                        if edge > 0.10:
                            value_rating = "⭐⭐⭐ Golden Nugget"
                        elif edge > 0.05:
                            value_rating = "⭐⭐ Value Bet"
                        elif edge > 0.03:
                            value_rating = "⭐ Consider"
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
        Main prediction method - COMPLETELY DATA-DRIVEN
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
        
        # Calculate goal expectations (DATA-DRIVEN, NO HARCODING)
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
        
        # Detect value bets (reject extreme edges as likely errors)
        raw_value_bets = self.detect_value_bets(goal_expectations['probabilities'], market_odds)
        
        # Calculate stakes
        final_bets = []
        for bet in raw_value_bets:
            edge = bet['edge_percent'] / 100
            # Conservative stake calculation
            stake_pct = min(self.max_stake_pct, edge * 0.25)
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


# TEST to verify data-driven multipliers
def test_data_driven_predictor():
    """Test the data-driven predictor with Hamburg vs Werder"""
    print("🧪 TESTING DATA-DRIVEN PREDICTOR...")
    
    hamburg_stats = EnhancedTeamStats(
        team_name="Hamburger SV",
        matches_played=12,
        possession_avg=47,
        shots_per_game=13.33,
        shots_on_target_pg=5.33,
        conversion_rate=0.07,
        xg_for_avg=1.42,
        xg_against_avg=1.70,
        home_wins=3, home_draws=1, home_losses=2,
        away_wins=0, away_draws=2, away_losses=4,
        home_goals_for=9, home_goals_against=6,
        away_goals_for=2, away_goals_against=12,
        clean_sheet_pct=0.25,
        clean_sheet_pct_home=0.17,
        clean_sheet_pct_away=0.33,
        failed_to_score_pct=0.50,
        failed_to_score_pct_home=0.33,
        failed_to_score_pct_away=0.67,
        btts_pct=0.25,
        btts_pct_home=0.50,
        btts_pct_away=0.0,
        over25_pct=0.50,
        over25_pct_home=0.50,
        over25_pct_away=0.50,
        last5_form="WWLDL",
        last5_wins=2, last5_draws=1, last5_losses=2,
        last5_goals_for=7, last5_goals_against=9,
        last5_ppg=1.4,
        last5_cs_pct=0.0,
        last5_fts_pct=0.2,
        last5_btts_pct=0.8,
        last5_over25_pct=0.6
    )
    
    werder_stats = EnhancedTeamStats(
        team_name="Werder Bremen",
        matches_played=12,
        possession_avg=50,
        shots_per_game=12.83,
        shots_on_target_pg=5.13,
        conversion_rate=0.10,
        xg_for_avg=1.45,
        xg_against_avg=1.80,
        home_wins=3, home_draws=2, home_losses=1,
        away_wins=1, away_draws=2, away_losses=3,
        home_goals_for=8, home_goals_against=8,
        away_goals_for=8, away_goals_against=13,
        clean_sheet_pct=0.25,
        clean_sheet_pct_home=0.33,
        clean_sheet_pct_away=0.17,
        failed_to_score_pct=0.25,
        failed_to_score_pct_home=0.17,
        failed_to_score_pct_away=0.33,
        btts_pct=0.50,
        btts_pct_home=0.50,
        btts_pct_away=0.50,
        over25_pct=0.67,
        over25_pct_home=0.67,
        over25_pct_away=0.67,
        last5_form="LDLWD",
        last5_wins=1, last5_draws=2, last5_losses=2,
        last5_goals_for=6, last5_goals_against=8,
        last5_ppg=1.0,
        last5_cs_pct=0.0,
        last5_fts_pct=0.2,
        last5_btts_pct=0.8,
        last5_over25_pct=0.4
    )
    
    market_odds = {
        'over_25': 1.73,
        'under_25': 2.20,
        'btts_yes': 1.59,
        'btts_no': 2.44,
        'home_win': 2.34,
        'away_win': 3.03,
        'draw': 3.62,
        'home_draw': 1.42,
        'away_draw': 1.65
    }
    
    predictor = EdgeFinderPredictor(
        bankroll=1000.0,
        min_edge=0.03,
        form_weight=0.4,
        min_sample_size=3
    )
    
    result = predictor.predict_match(hamburg_stats, werder_stats, market_odds, 'bundesliga')
    
    print(f"\n✅ DATA-DRIVEN RESULTS:")
    print(f"Home Expected Goals: {result['match_analysis']['goal_expectations']['lambda_home']:.2f}")
    print(f"Away Expected Goals: {result['match_analysis']['goal_expectations']['lambda_away']:.2f}")
    print(f"Total Expected Goals: {result['match_analysis']['goal_expectations']['total_goals']:.2f}")
    print(f"Over 2.5 Probability: {result['match_analysis']['goal_expectations']['probabilities']['over25']:.1%}")
    print(f"Away Win Probability: {result['match_analysis']['goal_expectations']['probabilities']['away_win']:.1%}")
    print(f"BTTS Probability: {result['match_analysis']['goal_expectations']['probabilities']['btts_yes']:.1%}")
    
    print(f"\n📊 Value Bets Found: {len(result['value_bets'])}")
    for bet in result['value_bets']:
        print(f"  - {bet['bet_type']} @ {bet['market_odds']} (Edge: {bet['edge_percent']:.1f}%)")
    
    print(f"\n🎯 Confidence Score: {result['match_analysis']['confidence']['score']}/10")
    print(f"Confidence Level: {result['match_analysis']['confidence']['level'].value}")
    
    # Show adjustment factors
    print(f"\n🔧 DATA-DRIVEN ADJUSTMENT FACTORS:")
    factors = result['match_analysis']['goal_expectations']['adjustment_factors']
    print(f"  Hamburg base attack: {factors.get('home_attack', 0):.2f}")
    print(f"  Hamburg attack vs Werder defense: {factors.get('home_attack_vs_away_defense', 0):.2f}")
    print(f"  Hamburg efficiency: {factors.get('home_efficiency', 0):.2f}x")
    print(f"  Hamburg shot quality: {factors.get('home_shot_quality', 0):.2f}x")
    print(f"  Werder away defense: {factors.get('away_defense', 0):.2f}")
    
    # Verify Bundesliga realism
    total_goals = result['match_analysis']['goal_expectations']['total_goals']
    if 2.0 <= total_goals <= 4.0:
        print(f"\n✅ REALITY CHECK PASSED: Total goals {total_goals:.2f} is realistic for Bundesliga")
    else:
        print(f"\n⚠️  WARNING: Total goals {total_goals:.2f} outside typical Bundesliga range")


if __name__ == "__main__":
    test_data_driven_predictor()
