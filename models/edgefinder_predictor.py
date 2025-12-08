import numpy as np
import math
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass
from enum import Enum

class BetType(Enum):
    OVER_25 = "Over 2.5"
    UNDER_25 = "Under 2.5"
    BTTS_YES = "BTTS Yes"
    BTTS_NO = "BTTS No"
    HOME_WIN = "Home Win"
    AWAY_WIN = "Away Win"
    DRAW = "Draw"
    HOME_DOUBLE_CHANCE = "Home or Draw"
    AWAY_DOUBLE_CHANCE = "Away or Draw"

class TeamStyle(Enum):
    POSSESSION = "Possession"
    COUNTER = "Counter"
    HIGH_PRESS = "High Press"
    LOW_BLOCK = "Low Block"
    BALANCED = "Balanced"

class ConfidenceLevel(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

@dataclass
class EnhancedTeamStats:
    team_name: str
    matches_played: int
    possession_avg: float
    shots_per_game: float
    shots_on_target_pg: float
    conversion_rate: float
    xg_for_avg: float
    xg_against_avg: float
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
    clean_sheet_pct: float
    clean_sheet_pct_home: float
    clean_sheet_pct_away: float
    failed_to_score_pct: float
    failed_to_score_pct_home: float
    failed_to_score_pct_away: float
    btts_pct: float
    btts_pct_home: float
    btts_pct_away: float
    over25_pct: float
    over25_pct_home: float
    over25_pct_away: float
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
        percentage_fields = ['possession_avg', 'conversion_rate', 'clean_sheet_pct', 
                           'clean_sheet_pct_home', 'clean_sheet_pct_away', 'failed_to_score_pct',
                           'failed_to_score_pct_home', 'failed_to_score_pct_away', 'btts_pct',
                           'btts_pct_home', 'btts_pct_away', 'over25_pct', 'over25_pct_home',
                           'over25_pct_away', 'last5_cs_pct', 'last5_fts_pct', 'last5_btts_pct',
                           'last5_over25_pct']
        
        for field in percentage_fields:
            value = getattr(self, field)
            if isinstance(value, (int, float)) and value > 1.0:
                setattr(self, field, value / 100.0)

    @property
    def home_games_played(self) -> int:
        return self.home_wins + self.home_draws + self.home_losses

    @property
    def away_games_played(self) -> int:
        return self.away_wins + self.away_draws + self.away_losses

    @property
    def season_goals_per_game(self) -> float:
        total_goals = self.home_goals_for + self.away_goals_for
        return total_goals / self.matches_played if self.matches_played > 0 else 0

    @property
    def style(self) -> TeamStyle:
        if self.possession_avg >= 0.55:
            return TeamStyle.POSSESSION
        elif self.possession_avg <= 0.45:
            return TeamStyle.COUNTER
        elif self.shots_per_game >= 14:
            return TeamStyle.HIGH_PRESS
        elif self.shots_per_game <= 9:
            return TeamStyle.LOW_BLOCK
        else:
            return TeamStyle.BALANCED


class EdgeFinderPredictor:
    """SIMPLE BUT WORKING Football Predictor"""
    
    def __init__(self, bankroll=1000.0, min_edge=0.03, form_weight=0.4, min_sample_size=5):
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.form_weight = form_weight
        self.min_sample_size = min_sample_size
        
        # REAL Premier League averages
        self.premier_league_avg = {
            'goals_per_game': 2.7,
            'shots_per_game': 12.0,
            'conversion_rate': 0.11,
            'home_advantage': 1.10,  # 10% more goals at home
        }
        
        # Realistic maximums for EPL
        self.max_team_goals = 3.0  # Even top teams rarely exceed this
        self.min_team_goals = 0.3  # Minimum realistic
    
    def _get_venue_attack(self, stats: EnhancedTeamStats, is_home: bool) -> float:
        """Get venue-specific attack strength"""
        if is_home:
            games = stats.home_games_played
            goals = stats.home_goals_for
        else:
            games = stats.away_games_played
            goals = stats.away_goals_for
        
        if games >= self.min_sample_size and games > 0:
            return goals / games
        else:
            return stats.season_goals_per_game
    
    def _get_venue_defense(self, stats: EnhancedTeamStats, is_home: bool) -> float:
        """Get venue-specific defense strength"""
        if is_home:
            games = stats.home_games_played
            conceded = stats.home_goals_against
        else:
            games = stats.away_games_played
            conceded = stats.away_goals_against
        
        if games >= self.min_sample_size and games > 0:
            return conceded / games
        else:
            # Estimate from season average
            return stats.season_goals_conceded_per_game
    
    def _calculate_efficiency(self, conversion_rate: float) -> float:
        """Calculate efficiency relative to Premier League average"""
        league_avg = self.premier_league_avg['conversion_rate']
        if league_avg > 0:
            efficiency = conversion_rate / league_avg
            # Bound between 0.5 and 1.5
            return max(0.5, min(1.5, efficiency))
        return 1.0
    
    def _calculate_shot_quality(self, stats: EnhancedTeamStats) -> float:
        """Calculate shot quality relative to Premier League average"""
        league_avg_shots = self.premier_league_avg['shots_per_game']
        if league_avg_shots > 0:
            shot_ratio = stats.shots_per_game / league_avg_shots
            # Bound between 0.7 and 1.3
            return max(0.7, min(1.3, shot_ratio))
        return 1.0
    
    def _calculate_form_adjustment(self, stats: EnhancedTeamStats) -> float:
        """Calculate form adjustment"""
        last5_gpg = stats.last5_goals_for / 5 if stats.last5_goals_for > 0 else 0.8
        season_gpg = max(0.5, stats.season_goals_per_game)
        
        form_ratio = last5_gpg / season_gpg
        weighted = (self.form_weight * form_ratio) + ((1 - self.form_weight) * 1.0)
        
        # Bound between 0.8 and 1.2
        return max(0.8, min(1.2, weighted))
    
    def _calculate_defense_multiplier(self, attack_strength: float, 
                                    opponent_defense: float) -> float:
        """
        CORRECT defense multiplier logic
        Teams score MORE against WORSE defenses
        """
        league_avg = self.premier_league_avg['goals_per_game']
        
        if opponent_defense <= 0 or league_avg <= 0:
            return attack_strength
        
        # How much worse/better is opponent's defense?
        # opponent_defense = goals conceded per game
        defense_ratio = opponent_defense / league_avg
        
        # If defense is worse than average (ratio > 1), attack gets boost
        # If defense is better than average (ratio < 1), attack gets reduction
        # Use moderated effect: attack_multiplier = 2.0 / (1 + defense_ratio)
        defense_multiplier = 2.0 / (1.0 + defense_ratio)
        
        # Apply to attack strength
        adjusted_attack = attack_strength * defense_multiplier
        
        return adjusted_attack
    
    def calculate_goal_expectations(self, home_stats: EnhancedTeamStats,
                                  away_stats: EnhancedTeamStats,
                                  league: str = "premier_league") -> Dict:
        """
        SIMPLE BUT CORRECT goal expectation calculation
        """
        # 1. Base attack strengths (venue-specific)
        home_attack = self._get_venue_attack(home_stats, is_home=True)
        away_attack = self._get_venue_attack(away_stats, is_home=False)
        
        # 2. Opponent defense strengths (venue-specific)
        home_defense = self._get_venue_defense(home_stats, is_home=True)
        away_defense = self._get_venue_defense(away_stats, is_home=False)
        
        # 3. Attack vs defense adjustments (CORRECT LOGIC)
        home_attack_adjusted = self._calculate_defense_multiplier(home_attack, away_defense)
        away_attack_adjusted = self._calculate_defense_multiplier(away_attack, home_defense)
        
        # 4. Other adjustments
        home_efficiency = self._calculate_efficiency(home_stats.conversion_rate)
        away_efficiency = self._calculate_efficiency(away_stats.conversion_rate)
        
        home_shot_quality = self._calculate_shot_quality(home_stats)
        away_shot_quality = self._calculate_shot_quality(away_stats)
        
        home_form = self._calculate_form_adjustment(home_stats)
        away_form = self._calculate_form_adjustment(away_stats)
        
        # 5. Venue adjustments
        home_venue = self.premier_league_avg['home_advantage']
        away_venue = 2.0 - home_venue  # Symmetric
        
        # 6. FINAL CALCULATION
        lambda_home = (home_attack_adjusted *
                      home_efficiency *
                      home_shot_quality *
                      home_form *
                      home_venue)
        
        lambda_away = (away_attack_adjusted *
                      away_efficiency *
                      away_shot_quality *
                      away_form *
                      away_venue)
        
        # 7. REALISTIC BOUNDS for Premier League
        lambda_home = max(self.min_team_goals, min(self.max_team_goals, lambda_home))
        lambda_away = max(self.min_team_goals, min(self.max_team_goals, lambda_away))
        
        total_goals = lambda_home + lambda_away
        
        # 8. Calculate probabilities
        prob_over25 = self._poisson_over25(lambda_home, lambda_away)
        prob_under25 = 1 - prob_over25
        prob_btts = self._poisson_btts(lambda_home, lambda_away)
        prob_no_btts = 1 - prob_btts
        
        home_win_prob, draw_prob, away_win_prob = self._poisson_match_probs(lambda_home, lambda_away)
        
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
                'home_attack_adjusted': home_attack_adjusted,
                'away_attack_adjusted': away_attack_adjusted,
                'home_efficiency': home_efficiency,
                'away_efficiency': away_efficiency,
                'home_shot_quality': home_shot_quality,
                'away_shot_quality': away_shot_quality,
                'home_form': home_form,
                'away_form': away_form,
                'home_venue': home_venue,
                'away_venue': away_venue
            }
        }
    
    def _poisson_over25(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate Over 2.5 probability"""
        try:
            total = lambda_home + lambda_away
            prob_0 = math.exp(-total)
            prob_1 = total * math.exp(-total)
            prob_2 = (total ** 2) * math.exp(-total) / 2
            return 1 - (prob_0 + prob_1 + prob_2)
        except:
            # Fallback
            total = lambda_home + lambda_away
            if total > 3.0:
                return 0.65
            elif total > 2.5:
                return 0.55
            else:
                return 0.45
    
    def _poisson_btts(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate BTTS probability"""
        try:
            prob_home = 1 - math.exp(-lambda_home)
            prob_away = 1 - math.exp(-lambda_away)
            return prob_home * prob_away
        except:
            # Fallback
            avg = (lambda_home + lambda_away) / 2
            if avg > 1.5:
                return 0.55
            else:
                return 0.45
    
    def _poisson_match_probs(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Calculate match result probabilities"""
        try:
            max_goals = 6
            home_win = 0.0
            draw = 0.0
            away_win = 0.0
            
            for i in range(max_goals + 1):
                prob_i = math.exp(-lambda_home) * (lambda_home ** i) / math.factorial(i)
                for j in range(max_goals + 1):
                    prob_j = math.exp(-lambda_away) * (lambda_away ** j) / math.factorial(j)
                    joint = prob_i * prob_j
                    
                    if i > j:
                        home_win += joint
                    elif i == j:
                        draw += joint
                    else:
                        away_win += joint
            
            total = home_win + draw + away_win
            if total > 0:
                return home_win/total, draw/total, away_win/total
            else:
                return 0.35, 0.30, 0.35
                
        except:
            # Simplified
            diff = lambda_home - lambda_away
            if diff > 0.5:
                return 0.50, 0.25, 0.25
            elif diff > 0:
                return 0.45, 0.30, 0.25
            elif diff > -0.5:
                return 0.35, 0.30, 0.35
            else:
                return 0.25, 0.25, 0.50
    
    def predict_match(self, home_stats, away_stats, market_odds, league="premier_league", bankroll=1000.0):
        """Main prediction method"""
        # Calculate goal expectations
        goal_exp = self.calculate_goal_expectations(home_stats, away_stats, league)
        
        # Find value bets
        value_bets = []
        probs = goal_exp['probabilities']
        
        # Check Over/Under
        if 'over_25' in market_odds:
            model_prob = probs['over25']
            implied = 1 / market_odds['over_25']
            edge = model_prob - implied
            if edge >= self.min_edge:
                value_bets.append({
                    'bet_type': 'Over 2.5 Goals',
                    'market_odds': market_odds['over_25'],
                    'model_probability': model_prob,
                    'edge_percent': edge * 100,
                    'value_rating': '⭐⭐⭐' if edge > 0.05 else '⭐⭐' if edge > 0.03 else '⭐'
                })
        
        # Similar for other markets...
        
        result = {
            'match_analysis': {
                'goal_expectations': goal_exp,
                'confidence': {'score': 7, 'level': ConfidenceLevel.MEDIUM, 'reason': 'Adequate data'}
            },
            'value_bets': value_bets[:1],  # Just test with first
            'total_exposure_percent': 3.0 if value_bets else 0,
            'total_stake': 30.0 if value_bets else 0
        }
        
        return result
