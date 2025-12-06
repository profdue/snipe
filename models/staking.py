"""
Kelly Criterion Implementation for Optimal Bet Sizing
Mathematically optimal staking based on edge and bankroll management
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StakeResult:
    """Result of Kelly stake calculation"""
    stake_amount: float
    stake_percent: float
    kelly_fraction: float
    expected_value: float
    risk_level: str
    edge_percent: float
    value_rating: str
    implied_probability: float
    true_probability: float
    max_stake_limit: float
    confidence_multiplier: float = 1.0

class KellyCriterion:
    """
    Kelly Criterion calculator for optimal bet sizing
    
    The Kelly Criterion maximizes long-term bankroll growth by determining
    the optimal stake based on edge and probability.
    
    Formula: f* = (bp - q) / b
    Where:
      f* = fraction of bankroll to bet
      b = decimal odds - 1
      p = win probability
      q = loss probability (1 - p)
    """
    
    def __init__(self, fraction: float = 0.5, max_stake_percent: float = 0.05):
        """
        Initialize Kelly Criterion calculator
        
        Args:
            fraction: Fraction of full Kelly to use (0.5 = half Kelly)
            max_stake_percent: Maximum stake as percentage of bankroll (default 5%)
        """
        if not 0 < fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        if not 0 < max_stake_percent <= 1:
            raise ValueError("Max stake percent must be between 0 and 1")
            
        self.fraction = fraction
        self.max_stake_percent = max_stake_percent
        
        # Risk level thresholds
        self.risk_thresholds = {
            'high': 0.1,    # >10% Kelly fraction = High risk
            'medium': 0.05, # 5-10% Kelly fraction = Medium risk
            'low': 0.01     # 1-5% Kelly fraction = Low risk
        }
        
        # Value rating thresholds based on edge
        self.value_thresholds = {
            'excellent': 0.10,  # >10% edge
            'very_good': 0.08,  # 8-10% edge
            'good': 0.05,       # 5-8% edge
            'fair': 0.02,       # 2-5% edge
            'poor': 0.0         # 0-2% edge
        }
    
    def calculate_edge(self, probability: float, odds: float) -> float:
        """
        Calculate betting edge
        
        Args:
            probability: True win probability (0-1)
            odds: Decimal odds
            
        Returns:
            Edge as decimal (positive = value bet)
        """
        if odds <= 0:
            return 0.0
            
        implied_probability = 1.0 / odds
        edge = probability - implied_probability
        return edge
    
    def calculate_full_kelly(self, probability: float, odds: float) -> float:
        """
        Calculate full Kelly fraction
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds
            
        Returns:
            Full Kelly fraction (0-1)
        """
        if odds <= 1 or probability <= 0 or probability >= 1:
            return 0.0
            
        b = odds - 1
        p = probability
        q = 1 - p
        
        # Kelly formula: f* = (bp - q) / b
        full_kelly = (b * p - q) / b
        
        # Ensure non-negative
        return max(0.0, full_kelly)
    
    def calculate_stake(self, 
                       probability: float, 
                       odds: float, 
                       bankroll: float,
                       confidence_multiplier: float = 1.0,
                       min_probability: float = 0.0) -> StakeResult:
        """
        Calculate optimal stake using fractional Kelly
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds
            bankroll: Current bankroll
            confidence_multiplier: Adjust stake based on confidence (0-1)
            min_probability: Minimum probability to consider betting
            
        Returns:
            StakeResult object with all stake information
        """
        # Validate inputs
        if probability <= min_probability or probability >= 1:
            return self._create_no_bet_result(probability, odds, "Probability out of range")
        
        if odds <= 1:
            return self._create_no_bet_result(probability, odds, "Invalid odds")
        
        if bankroll <= 0:
            return self._create_no_bet_result(probability, odds, "Invalid bankroll")
        
        # Calculate edge
        edge = self.calculate_edge(probability, odds)
        implied_probability = 1.0 / odds
        
        if edge <= 0:
            return self._create_no_bet_result(probability, odds, "No edge", edge)
        
        # Calculate full Kelly
        full_kelly = self.calculate_full_kelly(probability, odds)
        
        if full_kelly <= 0:
            return self._create_no_bet_result(probability, odds, "Negative Kelly", edge)
        
        # Apply fractional Kelly and confidence multiplier
        kelly_fraction = full_kelly * self.fraction * confidence_multiplier
        
        # Ensure non-negative and reasonable
        kelly_fraction = max(0.0, min(kelly_fraction, 0.5))  # Cap at 50% Kelly
        
        # Calculate stake amount
        stake_amount = bankroll * kelly_fraction
        
        # Apply maximum stake limit
        max_stake = bankroll * self.max_stake_percent
        stake_amount = min(stake_amount, max_stake)
        
        # Calculate stake percentage
        stake_percent = stake_amount / bankroll
        
        # Calculate expected value
        expected_value = self._calculate_expected_value(probability, odds, stake_amount)
        
        # Determine risk level
        risk_level = self._determine_risk_level(kelly_fraction)
        
        # Determine value rating
        value_rating = self._determine_value_rating(edge)
        
        return StakeResult(
            stake_amount=stake_amount,
            stake_percent=stake_percent,
            kelly_fraction=kelly_fraction,
            expected_value=expected_value,
            risk_level=risk_level,
            edge_percent=edge * 100,
            value_rating=value_rating,
            implied_probability=implied_probability,
            true_probability=probability,
            max_stake_limit=max_stake,
            confidence_multiplier=confidence_multiplier
        )
    
    def calculate_confidence_stake(self,
                                 probability: float,
                                 odds: float,
                                 bankroll: float,
                                 confidence_level: str = "medium",
                                 min_probability: float = 0.0) -> StakeResult:
        """
        Calculate stake with confidence level adjustment
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds
            bankroll: Current bankroll
            confidence_level: "high", "medium", or "low"
            min_probability: Minimum probability to consider betting
            
        Returns:
            StakeResult object with confidence-adjusted stake
        """
        confidence_multipliers = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4
        }
        
        confidence_multiplier = confidence_multipliers.get(confidence_level.lower(), 0.5)
        
        return self.calculate_stake(
            probability=probability,
            odds=odds,
            bankroll=bankroll,
            confidence_multiplier=confidence_multiplier,
            min_probability=min_probability
        )
    
    def _calculate_expected_value(self, probability: float, odds: float, stake: float) -> float:
        """Calculate expected value of a bet"""
        win_amount = stake * (odds - 1)
        loss_amount = stake
        
        ev_win = probability * win_amount
        ev_loss = (1 - probability) * loss_amount
        
        return ev_win - ev_loss
    
    def _determine_risk_level(self, kelly_fraction: float) -> str:
        """Determine risk level based on Kelly fraction"""
        if kelly_fraction > self.risk_thresholds['high']:
            return "High"
        elif kelly_fraction > self.risk_thresholds['medium']:
            return "Medium"
        elif kelly_fraction > self.risk_thresholds['low']:
            return "Low"
        else:
            return "Very Low"
    
    def _determine_value_rating(self, edge: float) -> str:
        """Determine value rating based on edge"""
        if edge > self.value_thresholds['excellent']:
            return "Excellent"
        elif edge > self.value_thresholds['very_good']:
            return "Very Good"
        elif edge > self.value_thresholds['good']:
            return "Good"
        elif edge > self.value_thresholds['fair']:
            return "Fair"
        else:
            return "Poor"
    
    def _create_no_bet_result(self, 
                            probability: float, 
                            odds: float, 
                            reason: str,
                            edge: float = 0.0) -> StakeResult:
        """Create result for no-bet scenarios"""
        implied_probability = 1.0 / odds if odds > 0 else 0.0
        
        return StakeResult(
            stake_amount=0.0,
            stake_percent=0.0,
            kelly_fraction=0.0,
            expected_value=0.0,
            risk_level="No Bet",
            edge_percent=edge * 100,
            value_rating="None",
            implied_probability=implied_probability,
            true_probability=probability,
            max_stake_limit=0.0,
            confidence_multiplier=0.0
        )
    
    def calculate_bankroll_growth(self, 
                                probability: float, 
                                odds: float, 
                                stake_percent: float,
                                num_bets: int = 100) -> Dict[str, float]:
        """
        Simulate bankroll growth over multiple bets
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds
            stake_percent: Stake as percentage of bankroll
            num_bets: Number of bets to simulate
            
        Returns:
            Dictionary with growth statistics
        """
        if stake_percent <= 0 or stake_percent > 1:
            return {"error": "Invalid stake percentage"}
        
        # Calculate growth per bet
        win_multiplier = 1 + stake_percent * (odds - 1)
        loss_multiplier = 1 - stake_percent
        
        # Expected growth per bet
        expected_growth_per_bet = (probability * np.log(win_multiplier) + 
                                  (1 - probability) * np.log(loss_multiplier))
        
        # Compound growth over multiple bets
        expected_total_growth = np.exp(expected_growth_per_bet * num_bets)
        
        # Calculate probability of ruin
        if loss_multiplier <= 0:
            ruin_probability = 1.0
        else:
            ruin_probability = ((loss_multiplier / win_multiplier) ** 
                              (np.log(loss_multiplier) / np.log(win_multiplier / loss_multiplier)))
        
        return {
            'expected_growth_per_bet': expected_growth_per_bet,
            'expected_total_growth': expected_total_growth,
            'ruin_probability': min(ruin_probability, 1.0),
            'optimal_stake_percent': self.calculate_full_kelly(probability, odds),
            'edge': self.calculate_edge(probability, odds)
        }
    
    def get_recommended_fraction(self, 
                               probability: float, 
                               odds: float,
                               risk_tolerance: str = "medium") -> float:
        """
        Get recommended Kelly fraction based on risk tolerance
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds
            risk_tolerance: "conservative", "moderate", or "aggressive"
            
        Returns:
            Recommended Kelly fraction
        """
        full_kelly = self.calculate_full_kelly(probability, odds)
        
        if full_kelly <= 0:
            return 0.0
        
        risk_factors = {
            'conservative': 0.25,  # Quarter Kelly
            'moderate': 0.5,       # Half Kelly
            'aggressive': 0.75     # Three-quarters Kelly
        }
        
        risk_factor = risk_factors.get(risk_tolerance.lower(), 0.5)
        return full_kelly * risk_factor


# Factory function for easy creation
def create_kelly_calculator(fraction: float = 0.5, 
                          max_stake_percent: float = 0.05) -> KellyCriterion:
    """Factory function to create KellyCriterion instance"""
    return KellyCriterion(fraction=fraction, max_stake_percent=max_stake_percent)


# Example usage
if __name__ == "__main__":
    # Create calculator with half Kelly and 5% max stake
    kelly = KellyCriterion(fraction=0.5, max_stake_percent=0.05)
    
    # Example calculation
    result = kelly.calculate_stake(
        probability=0.65,  # 65% win probability
        odds=2.0,          # 2.0 decimal odds (even money)
        bankroll=1000      # $1000 bankroll
    )
    
    print(f"Stake Amount: ${result.stake_amount:.2f}")
    print(f"Stake Percentage: {result.stake_percent:.1%}")
    print(f"Expected Value: ${result.expected_value:.2f}")
    print(f"Edge: {result.edge_percent:.1f}%")
    print(f"Risk Level: {result.risk_level}")
    print(f"Value Rating: {result.value_rating}")
