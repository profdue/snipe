class KellyCriterion:
    def __init__(self, fraction=0.5):
        """
        Initialize Kelly Criterion calculator
        
        Args:
            fraction: Fraction of full Kelly to use (0.5 = half Kelly)
        """
        self.fraction = fraction
        
    def calculate_stake(self, probability, odds, bankroll, max_percent=0.05):
        """
        Calculate optimal stake using Kelly Criterion
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds
            bankroll: Current bankroll
            max_percent: Maximum stake as percentage of bankroll
        
        Returns:
            Dictionary with stake information
        """
        # Full Kelly calculation
        q = 1 - probability
        b = odds - 1
        
        if b <= 0 or probability <= 0:
            kelly_fraction = 0
        else:
            kelly_fraction = (probability * b - q) / b
        
        # Apply fractional Kelly
        kelly_fraction *= self.fraction
        
        # Ensure non-negative
        kelly_fraction = max(0, kelly_fraction)
        
        # Calculate stake
        stake_amount = bankroll * kelly_fraction
        
        # Apply maximum stake limit
        max_stake = bankroll * max_percent
        stake_amount = min(stake_amount, max_stake)
        
        # Calculate expected value
        expected_value = (probability * (stake_amount * (odds - 1))) - (q * stake_amount)
        
        # Determine risk level
        if kelly_fraction > 0.1:
            risk_level = "High"
        elif kelly_fraction > 0.05:
            risk_level = "Medium"
        elif kelly_fraction > 0:
            risk_level = "Low"
        else:
            risk_level = "No Bet"
        
        return {
            'stake_amount': stake_amount,
            'stake_percent': stake_amount / bankroll,
            'kelly_fraction': kelly_fraction,
            'expected_value': expected_value,
            'risk_level': risk_level,
            'max_stake_limit': max_stake
        }
