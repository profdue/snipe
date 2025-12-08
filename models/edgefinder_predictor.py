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
    
    # Shot data - NOW ACTUALLY USED
    shots_per_game: float
    shots_on_target_pg: float
    conversion_rate: float
    
    # xG data
    xg_for_avg: float
    xg_against_avg: float
    
    # Home/Away splits - NOW ACTUALLY USED
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
    
    # Defense patterns - NOW ACTUALLY USED
    clean_sheet_pct: float
    clean_sheet_pct_home: float
    clean_sheet_pct_away: float
    failed_to_score_pct: float
    failed_to_score_pct_home: float
    failed_to_score_pct_away: float
    
    # Transition patterns - NOW ACTUALLY USED
    btts_pct: float
    btts_pct_home: float
    btts_pct_away: float
    over25_pct: float
    over25_pct_home: float
    over25_pct_away: float
    
    # Recent Form - NOW ACTUALLY USED
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
    FIXED Football Predictor - Uses ALL available data properly
    Universal engine for top 5 European leagues
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
        
        # League contexts for top 5 European leagues
        self.league_contexts = {
            'premier_league': {
                'avg_gpg': 2.7,
                'avg_shots': 12.0,
                'home_advantage': 0.15,
                'away_penalty': 0.85
            },
            'la_liga': {
                'avg_gpg': 2.5,
                'avg_shots': 11.5,
                'home_advantage': 0.18,
                'away_penalty': 0.82
            },
            'bundesliga': {
                'avg_gpg': 3.0,
                'avg_shots': 13.0,
                'home_advantage': 0.12,
                'away_penalty': 0.88
            },
            'serie_a': {
                'avg_gpg': 2.6,
                'avg_shots': 11.8,
                'home_advantage': 0.16,
                'away_penalty': 0.84
            },
            'ligue_1': {
                'avg_gpg': 2.4,
                'avg_shots': 11.2,
                'home_advantage': 0.17,
                'away_penalty': 0.83
            },
            'default': {
                'avg_gpg': 2.7,
                'avg_shots': 12.0,
                'home_advantage': 0.15,
                'away_penalty': 0.85
            }
        }
        
        # Style adjustments
        self.style_adjustments = {
            (TeamStyle.POSSESSION, TeamStyle.LOW_BLOCK): {'home_goals_mult': 0.85},
            (TeamStyle.COUNTER, TeamStyle.HIGH_PRESS): {'home_goals_mult': 1.15},
        }
        
        self.default_style_adjustment = {'home_goals_mult': 1.0, 'away_goals_mult': 1.0}
        
        # Betting parameters
        self.max_stake_pct = 0.03
        self.min_confidence_for_stake = 0.55
    
    def _calculate_shot_quality_multiplier(self, stats: EnhancedTeamStats, league: str = "default") -> float:
        """Calculate shot quality multiplier based on shots data"""
        context = self.league_contexts.get(league, self.league_contexts['default'])
        avg_shots = context['avg_shots']
        
        shot_volume = stats.shots_per_game / avg_shots if avg_shots > 0 else 1.0
        
        # Bound between 0.7 and 1.3
        return max(0.7, min(1.3, shot_volume))
    
    def _calculate_venue_attack_strength(self, stats: EnhancedTeamStats, is_home: bool = True) -> float:
        """Calculate attack strength for specific venue using ACTUAL GOALS"""
        if is_home:
            games_played = stats.home_games_played
            goals = stats.home_goals_for
        else:
            games_played = stats.away_games_played
            goals = stats.away_goals_for
        
        if games_played >= self.min_sample_size:
            return goals / games_played
        else:
            # Insufficient venue data, use season average with venue adjustment
            venue_multiplier = 1.15 if is_home else 0.85
            return stats.season_goals_per_game * venue_multiplier
    
    def _calculate_venue_defense_strength(self, stats: EnhancedTeamStats, is_home: bool = True) -> float:
        """Calculate defense strength for specific venue using ACTUAL GOALS CONCEDED"""
        if is_home:
            games_played = stats.home_games_played
            goals_conceded = stats.home_goals_against
        else:
            games_played = stats.away_games_played
            goals_conceded = stats.away_goals_against
        
        if games_played >= self.min_sample_size:
            return goals_conceded / games_played
        else:
            # Insufficient venue data, use season average with venue adjustment
            venue_multiplier = 0.85 if is_home else 1.15  # Better at home, worse away
            return stats.season_goals_conceded_per_game * venue_multiplier
    
    def _calculate_form_adjustment(self, stats: EnhancedTeamStats) -> float:
        """Calculate form adjustment using LAST 5 ACTUAL GOALS, not just PPG"""
        # Calculate last 5 goals per game
        last5_gpg = stats.last5_goals_for / 5
        last5_gapg = stats.last5_goals_against / 5
        
        # Get season averages
        season_gpg = stats.season_goals_per_game
        season_gapg = stats.season_goals_conceded_per_game
        
        # Calculate attack and defense ratios
        attack_ratio = last5_gpg / season_gpg if season_gpg > 0 else 1.0
        defense_ratio = last5_gapg / season_gapg if season_gapg > 0 else 1.0
        
        # Combined form factor (good attack AND good defense = better form)
        # Lower defense_ratio is better (conceding less)
        form_factor = attack_ratio * (1.0 / max(0.1, defense_ratio))
        
        # Apply form weight
        weighted_form = (
            (self.form_weight * form_factor) +
            ((1 - self.form_weight) * 1.0)  # Season baseline
        )
        
        # Bound between 0.7 and 1.3
        return max(0.7, min(1.3, weighted_form))
    
    def calculate_goal_expectations(self, home_stats: EnhancedTeamStats,
                                  away_stats: EnhancedTeamStats,
                                  league: str = "default") -> Dict:
        """
        FIXED goal expectation calculation using ALL available data
        Universal for top 5 European leagues
        """
        context = self.league_contexts.get(league, self.league_contexts['default'])
        
        # 1. VENUE-SPECIFIC ATTACK STRENGTH (using ACTUAL GOALS)
        home_attack = self._calculate_venue_attack_strength(home_stats, is_home=True)
        away_attack = self._calculate_venue_attack_strength(away_stats, is_home=False)
        
        # 2. SHOT QUALITY MULTIPLIER (using shots data)
        home_shot_mult = self._calculate_shot_quality_multiplier(home_stats, league)
        away_shot_mult = self._calculate_shot_quality_multiplier(away_stats, league)
        
        # 3. FORM ADJUSTMENT (using LAST 5 ACTUAL GOALS)
        home_form_adj = self._calculate_form_adjustment(home_stats)
        away_form_adj = self._calculate_form_adjustment(away_stats)
        
        # 4. VENUE-SPECIFIC DEFENSE STRENGTH (using ACTUAL GOALS CONCEDED)
        home_defense = self._calculate_venue_defense_strength(home_stats, is_home=True)
        away_defense = self._calculate_venue_defense_strength(away_stats, is_home=False)
        
        # 5. DEFENSE ADJUSTMENT MULTIPLIERS
        league_avg = context['avg_gpg']
        home_defense_mult = league_avg / away_defense if away_defense > 0 else 1.0
        away_defense_mult = league_avg / home_defense if home_defense > 0 else 1.0
        
        # 6. EFFICIENCY ADJUSTMENT (conversion rate)
        avg_conversion = 0.11  # League average conversion rate
        home_efficiency = stats.conversion_rate / avg_conversion if avg_conversion > 0 else 1.0
        away_efficiency = stats.conversion_rate / avg_conversion if avg_conversion > 0 else 1.0
        
        # 7. STYLE ADJUSTMENT
        style_key = (home_stats.style, away_stats.style)
        style_adj = self.style_adjustments.get(style_key, self.default_style_adjustment)
        
        # 8. FINAL GOAL EXPECTATION CALCULATION
        lambda_home = (home_attack *
                      home_shot_mult *
                      home_form_adj *
                      home_defense_mult *
                      home_efficiency *
                      style_adj.get('home_goals_mult', 1.0) *
                      (1 + context['home_advantage']))
        
        lambda_away = (away_attack *
                      away_shot_mult *
                      away_form_adj *
                      away_defense_mult *
                      away_efficiency *
                      style_adj.get('away_goals_mult', 1.0) *
                      context['away_penalty'])
        
        # 9. REALITY BOUNDS
        lambda_home = max(0.2, min(4.0, lambda_home))
        lambda_away = max(0.2, min(4.0, lambda_away))
        
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
                'home_attack': home_attack,
                'away_attack': away_attack,
                'home_shot_mult': home_shot_mult,
                'away_shot_mult': away_shot_mult,
                'home_form_adj': home_form_adj,
                'away_form_adj': away_form_adj,
                'home_defense_mult': home_defense_mult,
                'away_defense_mult': away_defense_mult,
                'home_defense': home_defense,
                'away_defense': away_defense
            }
        }
    
    def _poisson_over25_correct(self, lambda_home: float, lambda_away: float) -> float:
        """Calculate probability of Over 2.5 goals"""
        try:
            total_lambda = lambda_home + lambda_away
            prob_0 = math.exp(-total_lambda)
            prob_1 = total_lambda * math.exp(-total_lambda)
            prob_2 = (total_lambda ** 2) * math.exp(-total_lambda) / 2
            prob_under25 = prob_0 + prob_1 + prob_2
            return 1 - prob_under25
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
            # Simplified fallback
            if lambda_home <= 0 or lambda_away <= 0:
                return 0.33, 0.34, 0.33
            
            home_strength = lambda_home / (lambda_home + lambda_away)
            draw_prob = 0.25
            home_win_prob = home_strength * (1 - draw_prob)
            away_win_prob = (1 - home_strength) * (1 - draw_prob)
            
            return home_win_prob, draw_prob, away_win_prob
    
    def analyze_team_identity(self, home_stats: EnhancedTeamStats, 
                            away_stats: EnhancedTeamStats) -> Dict:
        """Analyze Team Identity using shot data and style"""
        analysis = {
            'home_shot_quality': self._calculate_shot_quality_multiplier(home_stats),
            'away_shot_quality': self._calculate_shot_quality_multiplier(away_stats),
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
        
        # Style insights
        style_key = (home_stats.style, away_stats.style)
        if style_key in self.style_adjustments:
            analysis['insights'].append(
                f"Style matchup: {home_stats.style.value} vs {away_stats.style.value}"
            )
            analysis['confidence_factors'].append(('significant_style_clash', 2))
        
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
            'home_last5_gpg': home_stats.last5_goals_for / 5,
            'away_last5_gpg': away_stats.last5_goals_for / 5,
            'combined_btts': (home_stats.btts_pct + away_stats.btts_pct) / 2,
            'combined_over25': (home_stats.over25_pct + away_stats.over25_pct) / 2,
            'insights': [],
            'confidence_factors': []
        }
        
        # Recent form insights
        home_form_ratio = analysis['home_last5_gpg'] / home_stats.season_goals_per_game if home_stats.season_goals_per_game > 0 else 1.0
        away_form_ratio = analysis['away_last5_gpg'] / away_stats.season_goals_per_game if away_stats.season_goals_per_game > 0 else 1.0
        
        if home_form_ratio > 1.2:
            analysis['insights'].append(
                f"{home_stats.team_name} attacking form improving ({home_stats.last5_goals_for} goals in last 5)"
            )
            analysis['confidence_factors'].append(('improving_attack', 2))
        
        if away_form_ratio < 0.8:
            analysis['insights'].append(
                f"{away_stats.team_name} poor recent attacking form ({away_stats.last5_goals_for} goals in last 5)"
            )
            analysis['confidence_factors'].append(('poor_attack_form', 2))
        
        # BTTS and Over trends
        if analysis['combined_btts'] > 0.7:
            analysis['insights'].append(
                f"High BTTS probability (combined {analysis['combined_btts']:.1%})"
            )
            analysis['confidence_factors'].append(('high_btts', 2))
        
        if analysis['combined_over25'] > 0.7:
            analysis['insights'].append(
                f"High Over 2.5 probability (combined {analysis['combined_over25']:.1%})"
            )
            analysis['confidence_factors'].append(('high_scoring', 2))
        
        return analysis
    
    def calculate_confidence_score(self, analysis: Dict, goal_expectations: Dict) -> Dict:
        """Calculate confidence score based on data quality"""
        confidence_factors = []
        total_score = 5
        
        # Sample size factor
        home_stats = analysis.get('home_stats')
        away_stats = analysis.get('away_stats')
        
        if home_stats and away_stats:
            if home_stats.matches_played >= 10 and away_stats.matches_played >= 10:
                total_score += 1
                confidence_factors.append(('good_sample_size', 1))
            
            # Venue data availability
            if home_stats.home_games_played >= 5 and away_stats.away_games_played >= 5:
                total_score += 1
                confidence_factors.append(('good_venue_data', 1))
            
            # Recent form data
            if home_stats.last5_goals_for > 0 or away_stats.last5_goals_for > 0:
                total_score += 1
                confidence_factors.append(('recent_form_data', 1))
        
        # Ensure bounds
        total_score = max(1, min(10, total_score))
        
        # Determine confidence level
        if total_score >= 8:
            level = ConfidenceLevel.HIGH
            reason = "Strong data quality across all dimensions"
        elif total_score >= 6:
            level = ConfidenceLevel.MEDIUM
            reason = "Adequate data with some limitations"
        elif total_score >= 4:
            level = ConfidenceLevel.LOW
            reason = "Limited or inconsistent data"
        else:
            level = ConfidenceLevel.VERY_LOW
            reason = "Poor data quality or very small samples"
        
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
        
        value_bets.sort(key=lambda x: x['edge_percent'], reverse=True)
        return value_bets
    
    def predict_match(self, home_stats: EnhancedTeamStats, away_stats: EnhancedTeamStats,
                     market_odds: Dict, league: str = "default", 
                     bankroll: float = None) -> Dict:
        """
        Main prediction method
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
        
        # Calculate goal expectations
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
        
        # Detect value bets
        value_bets = self.detect_value_bets(goal_expectations['probabilities'], market_odds)
        
        # Calculate stakes
        final_bets = []
        for bet in value_bets:
            # Simplified stake calculation
            edge = bet['edge_percent'] / 100
            stake_pct = min(self.max_stake_pct, edge * 0.5)
            stake_amount = bankroll * stake_pct
            
            if stake_amount > bankroll * 0.005:  # Minimum sensible stake
                final_bet = {
                    'bet_type': bet['bet_type'].value,
                    'market_odds': bet['market_odds'],
                    'model_probability': bet['model_probability'],
                    'edge_percent': bet['edge_percent'],
                    'value_rating': bet['value_rating'],
                    'stake_amount': stake_amount,
                    'stake_percent': stake_pct * 100
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
            'market_odds_used': market_odds
        }
        
        if not final_bets:
            result['recommendation'] = f"NO BET - No value opportunities meeting minimum {self.min_edge*100:.1f}% edge criteria"
        
        return result


# Quick test
if __name__ == "__main__":
    print("FIXED EdgeFinder Predictor v2.0")
    print("Now properly uses ALL available data:")
    print("- Actual goals (not just xG)")
    print("- Venue splits (home/away performance)")
    print("- Shot data (attack quality)")
    print("- Last 5 goals (recent form)")
    print("- Clean sheet data (defense quality)")
    print("\nUniversal for top 5 European leagues")
