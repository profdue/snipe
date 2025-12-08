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
    Football Predictor - Data-driven multipliers
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
        
        # League contexts
        self.league_contexts = {
            'premier_league': {
                'avg_gpg': 2.7,
                'avg_shots': 12.0,
                'avg_conversion': 0.11,
                'avg_shot_accuracy': 0.35,
                'home_advantage_mult': 1.10
            },
            'la_liga': {
                'avg_gpg': 2.5,
                'avg_shots': 11.5,
                'avg_conversion': 0.105,
                'avg_shot_accuracy': 0.36,
                'home_advantage_mult': 1.12
            },
            'bundesliga': {
                'avg_gpg': 3.0,
                'avg_shots': 13.0,
                'avg_conversion': 0.115,
                'avg_shot_accuracy': 0.37,
                'home_advantage_mult': 1.08
            },
            'serie_a': {
                'avg_gpg': 2.6,
                'avg_shots': 11.8,
                'avg_conversion': 0.11,
                'avg_shot_accuracy': 0.35,
                'home_advantage_mult': 1.12
            },
            'ligue_1': {
                'avg_gpg': 2.4,
                'avg_shots': 11.2,
                'avg_conversion': 0.107,
                'avg_shot_accuracy': 0.34,
                'home_advantage_mult': 1.13
            },
            'default': {
                'avg_gpg': 2.7,
                'avg_shots': 12.0,
                'avg_conversion': 0.11,
                'avg_shot_accuracy': 0.35,
                'home_advantage_mult': 1.10
            }
        }
        
        # Style matchup adjustments
        self.style_matchup_effects = {
            (TeamStyle.POSSESSION, TeamStyle.LOW_BLOCK): {'home_adj': 0.9, 'away_adj': 1.0},
            (TeamStyle.COUNTER, TeamStyle.HIGH_PRESS): {'home_adj': 1.15, 'away_adj': 0.95},
            (TeamStyle.HIGH_PRESS, TeamStyle.LOW_BLOCK): {'home_adj': 1.1, 'away_adj': 0.9},
            (TeamStyle.LOW_BLOCK, TeamStyle.POSSESSION): {'home_adj': 1.1, 'away_adj': 0.9},
        }
        
        # Betting parameters
        self.max_stake_pct = 0.03
        self.min_confidence_for_stake = 0.55
        
    def _calculate_attack_vs_defense_adjustment(self, attack_strength: float,
                                              opponent_defense: float,
                                              league_avg_gpg: float) -> float:
        """
        Calculate how an attack performs against a specific defense
        """
        if opponent_defense <= 0.1 or league_avg_gpg <= 0.1:
            return attack_strength
        
        # How much worse/better is opponent's defense than league average?
        defense_quality = opponent_defense / league_avg_gpg
        
        # Attack multiplier: if defense is worse than average, attack gets boost
        attack_multiplier = 2.0 / (1.0 + defense_quality)
        
        # Apply realistic bounds
        attack_multiplier = max(0.5, min(1.8, attack_multiplier))
        
        return attack_strength * attack_multiplier
    
    def calculate_goal_expectations(self, home_stats: EnhancedTeamStats,
                                  away_stats: EnhancedTeamStats,
                                  league: str = "default") -> Dict:
        """
        Main goal expectation calculation
        """
        context = self.league_contexts.get(league, self.league_contexts['default'])
        
        # 1. Base attack/defense strengths
        home_attack_base = max(0.1, home_stats.season_goals_per_game)
        home_defense_base = max(0.1, home_stats.season_goals_conceded_per_game)
        away_attack_base = max(0.1, away_stats.season_goals_per_game)
        away_defense_base = max(0.1, away_stats.season_goals_conceded_per_game)
        
        # 2. Venue adjustments
        home_attack_venue = self._calculate_venue_strength(home_stats, is_home=True, is_attack=True)
        away_attack_venue = self._calculate_venue_strength(away_stats, is_home=False, is_attack=True)
        home_defense_venue = self._calculate_venue_strength(home_stats, is_home=True, is_attack=False)
        away_defense_venue = self._calculate_venue_strength(away_stats, is_home=False, is_attack=False)
        
        # 3. Attack vs defense adjustments
        home_attack_adjusted = self._calculate_attack_vs_defense_adjustment(
            home_attack_venue, away_defense_venue, context['avg_gpg']
        )
        away_attack_adjusted = self._calculate_attack_vs_defense_adjustment(
            away_attack_venue, home_defense_venue, context['avg_gpg']
        )
        
        # 4. Efficiency adjustments
        home_efficiency = self._calculate_efficiency_adjustment(home_stats.conversion_rate, context['avg_conversion'])
        away_efficiency = self._calculate_efficiency_adjustment(away_stats.conversion_rate, context['avg_conversion'])
        
        # 5. Shot quality adjustments
        home_shot_quality = self._calculate_shot_quality_adjustment(home_stats, context['avg_shots'], context['avg_shot_accuracy'])
        away_shot_quality = self._calculate_shot_quality_adjustment(away_stats, context['avg_shots'], context['avg_shot_accuracy'])
        
        # 6. Form adjustments
        home_form = self._calculate_form_adjustment(home_stats)
        away_form = self._calculate_form_adjustment(away_stats)
        
        # 7. Style adjustments
        style_adjustments = self._calculate_style_adjustment(home_stats.style, away_stats.style)
        
        # 8. Venue advantage
        home_venue = context['home_advantage_mult']
        away_venue = 2.0 - home_venue
        
        # 9. Final calculations
        lambda_home = (home_attack_adjusted * 
                      home_efficiency * 
                      home_shot_quality * 
                      home_form * 
                      style_adjustments['home_adj'] * 
                      home_venue)
        
        lambda_away = (away_attack_adjusted * 
                      away_efficiency * 
                      away_shot_quality * 
                      away_form * 
                      style_adjustments['away_adj'] * 
                      away_venue)
        
        # Apply realistic bounds
        lambda_home = max(0.2, min(4.0, lambda_home))
        lambda_away = max(0.2, min(4.0, lambda_away))
        
        # Calculate probabilities
        return self._calculate_probabilities(lambda_home, lambda_away)
    
    def _calculate_venue_strength(self, stats: EnhancedTeamStats, is_home: bool, is_attack: bool) -> float:
        """Calculate venue-specific strength"""
        if is_home:
            games = stats.home_games_played
            if is_attack:
                goals = stats.home_goals_for
            else:
                goals = stats.home_goals_against
        else:
            games = stats.away_games_played
            if is_attack:
                goals = stats.away_goals_for
            else:
                goals = stats.away_goals_against
        
        if games >= self.min_sample_size:
            return max(0.1, goals / games)
        elif is_attack:
            return max(0.1, stats.season_goals_per_game)
        else:
            return max(0.1, stats.season_goals_conceded_per_game)
    
    def _calculate_efficiency_adjustment(self, team_conversion: float, league_avg: float) -> float:
        """Calculate efficiency adjustment"""
        if league_avg <= 0:
            return 1.0
        
        efficiency = team_conversion / league_avg
        return max(0.7, min(1.3, efficiency))
    
    def _calculate_shot_quality_adjustment(self, stats: EnhancedTeamStats, 
                                         league_avg_shots: float, 
                                         league_avg_accuracy: float) -> float:
        """Calculate shot quality adjustment"""
        # Volume component
        volume_ratio = stats.shots_per_game / league_avg_shots if league_avg_shots > 0 else 1.0
        
        # Accuracy component
        if stats.shots_per_game > 0:
            team_accuracy = stats.shots_on_target_pg / stats.shots_per_game
            accuracy_ratio = team_accuracy / league_avg_accuracy if league_avg_accuracy > 0 else 1.0
        else:
            accuracy_ratio = 1.0
        
        # Combined
        shot_quality = volume_ratio * accuracy_ratio
        return max(0.7, min(1.5, shot_quality))
    
    def _calculate_form_adjustment(self, stats: EnhancedTeamStats) -> float:
        """Calculate form adjustment"""
        recent_gpg = stats.last5_goals_for / 5 if stats.last5_goals_for > 0 else 0.5
        season_gpg = max(0.1, stats.season_goals_per_game)
        
        form_ratio = recent_gpg / season_gpg
        weighted_form = (self.form_weight * form_ratio) + ((1 - self.form_weight) * 1.0)
        
        return max(0.7, min(1.5, weighted_form))
    
    def _calculate_style_adjustment(self, home_style: TeamStyle, away_style: TeamStyle) -> Dict:
        """Calculate style adjustments"""
        key = (home_style, away_style)
        if key in self.style_matchup_effects:
            return self.style_matchup_effects[key]
        return {'home_adj': 1.0, 'away_adj': 1.0}
    
    def _calculate_probabilities(self, lambda_home: float, lambda_away: float) -> Dict:
        """Calculate all probabilities from goal expectations"""
        # Over/Under 2.5
        prob_over25 = self._poisson_over25(lambda_home, lambda_away)
        prob_under25 = 1 - prob_over25
        
        # BTTS
        prob_btts = self._poisson_btts(lambda_home, lambda_away)
        prob_no_btts = 1 - prob_btts
        
        # Match result
        home_win_prob, draw_prob, away_win_prob = self._poisson_match_result(lambda_home, lambda_away)
        
        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'total_goals': lambda_home + lambda_away,
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
            }
        }
    
    def _poisson_over25(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate Over 2.5 probability"""
        try:
            total_lambda = lambda_home + lambda_away
            prob_0 = math.exp(-total_lambda)
            prob_1 = total_lambda * math.exp(-total_lambda)
            prob_2 = (total_lambda ** 2) * math.exp(-total_lambda) / 2
            return 1 - (prob_0 + prob_1 + prob_2)
        except:
            # Fallback
            total_goals = lambda_home + lambda_away
            if total_goals > 3.0:
                return 0.65
            elif total_goals > 2.5:
                return 0.55
            elif total_goals > 2.0:
                return 0.45
            else:
                return 0.35
    
    def _poisson_btts(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate BTTS probability"""
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
    
    def _poisson_match_result(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Calculate match result probabilities"""
        try:
            max_goals = 8
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
            
            # Normalize
            total = home_win + draw + away_win
            if total > 0:
                home_win /= total
                draw /= total
                away_win /= total
            
            return home_win, draw, away_win
        except:
            # Simplified fallback
            diff = lambda_home - lambda_away
            
            if diff > 0.8:
                return 0.50, 0.25, 0.25
            elif diff > 0.3:
                return 0.45, 0.30, 0.25
            elif diff > -0.3:
                return 0.35, 0.30, 0.35
            elif diff > -0.8:
                return 0.25, 0.30, 0.45
            else:
                return 0.25, 0.25, 0.50
    
    def find_value_bets(self, model_probs: Dict, market_odds: Dict) -> List[Dict]:
        """Find value bets based on edge"""
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
                        # Calculate stake
                        edge_pct = edge * 100
                        stake_pct = min(self.max_stake_pct, edge * 0.5)
                        stake_amount = self.bankroll * stake_pct
                        
                        value_rating = "⭐"
                        if edge_pct > 10:
                            value_rating = "⭐⭐⭐"
                        elif edge_pct > 5:
                            value_rating = "⭐⭐"
                        
                        value_bets.append({
                            'bet_type': bet_type.value,
                            'market_odds': market_odd,
                            'model_probability': model_prob,
                            'implied_probability': implied_prob,
                            'edge_percent': edge_pct,
                            'stake_amount': stake_amount,
                            'stake_percent': stake_pct * 100,
                            'value_rating': value_rating
                        })
        
        return sorted(value_bets, key=lambda x: x['edge_percent'], reverse=True)
    
    def predict(self, home_stats: EnhancedTeamStats, away_stats: EnhancedTeamStats,
               market_odds: Dict, league: str = "default") -> Dict:
        """
        Main prediction method
        """
        # Calculate goal expectations
        goal_expectations = self.calculate_goal_expectations(home_stats, away_stats, league)
        
        # Find value bets
        value_bets = self.find_value_bets(goal_expectations['probabilities'], market_odds)
        
        # Calculate totals
        total_stake = sum(b['stake_amount'] for b in value_bets)
        total_exposure = sum(b['stake_percent'] for b in value_bets)
        
        # Basic confidence score
        confidence_score = self._calculate_confidence_score(home_stats, away_stats, goal_expectations)
        
        return {
            'goal_expectations': goal_expectations,
            'value_bets': value_bets,
            'total_stake': total_stake,
            'total_exposure': total_exposure,
            'confidence_score': confidence_score,
            'league': league
        }
    
    def _calculate_confidence_score(self, home_stats: EnhancedTeamStats,
                                  away_stats: EnhancedTeamStats,
                                  goal_expectations: Dict) -> int:
        """Calculate confidence score (1-10)"""
        score = 5  # Base score
        
        # Sample size
        if home_stats.matches_played >= 10 and away_stats.matches_played >= 10:
            score += 2
        elif home_stats.matches_played >= 5 and away_stats.matches_played >= 5:
            score += 1
        
        # Venue data
        if home_stats.home_games_played >= 3 and away_stats.away_games_played >= 3:
            score += 1
        
        # Check for unrealistic predictions
        total_goals = goal_expectations['total_goals']
        if total_goals > 5.0 or total_goals < 1.0:
            score -= 2
        
        return max(1, min(10, score))
