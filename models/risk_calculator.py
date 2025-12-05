"""
Risk Management and Staking Calculator
Uses fractional Kelly criterion with bankroll management
"""

import numpy as np

class RiskCalculator:
    def __init__(self, bankroll=100.0, kelly_fraction=0.25, min_confidence=0.55):
        """
        Initialize risk calculator
        
        Args:
            bankroll: Total bankroll in units
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
            min_confidence: Minimum confidence threshold for bets
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_confidence = min_confidence
        
    def calculate_kelly_stake(self, probability, odds):
        """
        Calculate Kelly criterion stake
        
        Args:
            probability: True probability of outcome
            odds: Decimal odds offered
            
        Returns:
            Fraction of bankroll to stake (0 to 1)
        """
        if odds <= 1.0:
            return 0.0
        
        # Kelly formula: f* = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - p
        b = odds - 1
        p = probability
        q = 1 - p
        
        if b * p - q <= 0:
            return 0.0
        
        kelly_stake = (b * p - q) / b
        
        # Apply fractional Kelly
        fractional_stake = kelly_stake * self.kelly_fraction
        
        # Ensure stake is between 0 and reasonable maximum
        return max(0.0, min(fractional_stake, 0.1))  # Max 10% of bankroll
    
    def calculate_stake_units(self, probability, odds, confidence):
        """
        Calculate stake in units with confidence adjustment
        
        Args:
            probability: True probability
            odds: Decimal odds
            confidence: Confidence score (0 to 1)
            
        Returns:
            Dictionary with stake details
        """
        if confidence < self.min_confidence:
            return {
                'stake_units': 0.0,
                'stake_percent': 0.0,
                'recommended': False,
                'edge': 0.0,
                'risk_level': 'NO_BET'
            }
        
        # Calculate Kelly stake
        kelly_percent = self.calculate_kelly_stake(probability, odds)
        
        if kelly_percent <= 0:
            return {
                'stake_units': 0.0,
                'stake_percent': 0.0,
                'recommended': False,
                'edge': 0.0,
                'risk_level': 'NO_BET'
            }
        
        # Adjust stake by confidence
        confidence_adjusted_percent = kelly_percent * confidence
        
        # Calculate units
        stake_units = confidence_adjusted_percent * self.bankroll
        
        # Determine risk level
        if stake_units >= 2.0:
            risk_level = 'HIGH'
            risk_emoji = '🔥'
        elif stake_units >= 1.0:
            risk_level = 'MEDIUM'
            risk_emoji = '📈'
        elif stake_units >= 0.5:
            risk_level = 'LOW'
            risk_emoji = '📊'
        else:
            risk_level = 'VERY_LOW'
            risk_emoji = '📉'
        
        # Calculate edge
        implied_prob = 1 / odds
        edge = probability - implied_prob
        
        return {
            'stake_units': round(stake_units, 2),
            'stake_percent': round(confidence_adjusted_percent * 100, 1),
            'recommended': True,
            'edge': round(edge * 100, 1),  # As percentage
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'expected_value': round(stake_units * (probability * odds - 1), 2)
        }
    
    def categorize_bet(self, probability, stake_result):
        """
        Categorize bet by strength
        
        Returns:
            Category string with emoji
        """
        if not stake_result['recommended']:
            return "AVOID ❌"
        
        if probability >= 0.70:
            return "STRONG 🔥"
        elif probability >= 0.60:
            return "MODERATE ⚡"
        elif probability >= 0.55:
            return "LIGHT 📊"
        else:
            return "CAUTION ⚠️"
    
    def get_betting_recommendations(self, predictions, market_odds):
        """
        Generate betting recommendations for all markets
        
        Args:
            predictions: Dictionary from prediction_engine
            market_odds: Dictionary with market odds
            
        Returns:
            List of betting recommendations
        """
        recommendations = []
        
        # Check each market
        markets = [
            ('home_win', 'Match Winner: Home', market_odds.get('home', 2.0)),
            ('away_win', 'Match Winner: Away', market_odds.get('away', 4.0)),
            ('draw', 'Match Winner: Draw', market_odds.get('draw', 3.5)),
            ('over_25', 'Total Goals: Over 2.5', market_odds.get('over_25', 2.0)),
            ('under_25', 'Total Goals: Under 2.5', market_odds.get('under_25', 1.8)),
            ('btts_yes', 'BTTS: Yes', market_odds.get('btts', 1.8)),
            ('btts_no', 'BTTS: No', market_odds.get('btts_no', 2.0))
        ]
        
        for market_key, market_name, odds in markets:
            probability = predictions['probabilities'].get(market_key, 0)
            confidence = min(probability * 1.2, 0.95)  # Simple confidence calculation
            
            stake_result = self.calculate_stake_units(probability, odds, confidence)
            
            if stake_result['recommended'] and stake_result['stake_units'] > 0:
                category = self.categorize_bet(probability, stake_result)
                
                recommendations.append({
                    'market': market_name,
                    'probability': round(probability * 100, 1),
                    'odds': odds,
                    'stake_units': stake_result['stake_units'],
                    'edge_percent': stake_result['edge'],
                    'risk_level': stake_result['risk_level'],
                    'risk_emoji': stake_result['risk_emoji'],
                    'category': category,
                    'expected_value': stake_result['expected_value']
                })
        
        # Sort by expected value
        recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return recommendations