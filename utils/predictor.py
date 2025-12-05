import json
import pandas as pd

class OverUnderPredictor:
    def __init__(self, teams_data):
        self.teams = teams_data
    
    def predict(self, home_team, away_team, home_xg, away_xg):
        """Main prediction function"""
        
        # Get team stats
        home_stats = self.teams.get(home_team, {})
        away_stats = self.teams.get(away_team, {})
        
        if not home_stats or not away_stats:
            return {"prediction": "NO_DATA", "confidence": "LOW", "reason": "Missing team data"}
        
        # Extract stats
        home_gpg_10 = home_stats.get("home", {}).get("last10", {}).get("gpg", 0)
        home_ga_10 = home_stats.get("home", {}).get("last10", {}).get("ga_pg", 0)
        home_gpg_5 = home_stats.get("home", {}).get("last5", {}).get("gpg", 0)
        home_ga_5 = home_stats.get("home", {}).get("last5", {}).get("ga_pg", 0)
        
        away_gpg_10 = away_stats.get("away", {}).get("last10", {}).get("gpg", 0)
        away_ga_10 = away_stats.get("away", {}).get("last10", {}).get("ga_pg", 0)
        away_gpg_5 = away_stats.get("away", {}).get("last5", {}).get("gpg", 0)
        away_ga_5 = away_stats.get("away", {}).get("last5", {}).get("ga_pg", 0)
        
        # RULE 1: HIGH CONFIDENCE OVER 2.5
        if (home_gpg_10 > 1.5 and away_gpg_10 > 1.5 and
            home_gpg_5 > 1.5 and away_gpg_5 > 1.5):
            return {
                "prediction": "OVER 2.5",
                "confidence": "HIGH",
                "reason": f"Both teams high-scoring in last 10 & 5 games (Home: {home_gpg_5}/{home_gpg_10}, Away: {away_gpg_5}/{away_gpg_10} GPG)"
            }
        
        # RULE 2: HIGH CONFIDENCE UNDER 2.5
        if (home_ga_10 < 1.0 and away_gpg_10 < 1.5 and
            home_ga_5 < 1.0 and away_gpg_5 < 1.5):
            return {
                "prediction": "UNDER 2.5",
                "confidence": "HIGH",
                "reason": f"Home strong defense & away weak attack in last 10 & 5 games (Home GA: {home_ga_5}/{home_ga_10}, Away GPG: {away_gpg_5}/{away_gpg_10})"
            }
        
        if (away_ga_10 < 1.0 and home_gpg_10 < 1.5 and
            away_ga_5 < 1.0 and home_gpg_5 < 1.5):
            return {
                "prediction": "UNDER 2.5",
                "confidence": "HIGH",
                "reason": f"Away strong defense & home weak attack in last 10 & 5 games (Away GA: {away_ga_5}/{away_ga_10}, Home GPG: {home_gpg_5}/{home_gpg_10})"
            }
        
        # RULE 3: MODERATE CONFIDENCE OVER 2.5 (Last 5 only)
        if (home_gpg_5 > 1.5 and away_gpg_5 > 1.5):
            return {
                "prediction": "OVER 2.5",
                "confidence": "MODERATE",
                "reason": f"Both teams high-scoring in last 5 games only (Home: {home_gpg_5}, Away: {away_gpg_5} GPG)"
            }
        
        # RULE 4: MODERATE CONFIDENCE UNDER 2.5 (Last 5 only)
        if (home_ga_5 < 1.0 and away_gpg_5 < 1.5):
            return {
                "prediction": "UNDER 2.5",
                "confidence": "MODERATE",
                "reason": f"Home strong defense & away weak attack in last 5 games only (Home GA: {home_ga_5}, Away GPG: {away_gpg_5})"
            }
        
        if (away_ga_5 < 1.0 and home_gpg_5 < 1.5):
            return {
                "prediction": "UNDER 2.5",
                "confidence": "MODERATE",
                "reason": f"Away strong defense & home weak attack in last 5 games only (Away GA: {away_ga_5}, Home GPG: {home_gpg_5})"
            }
        
        # RULE 5: NO BET
        return {
            "prediction": "NO BET",
            "confidence": "LOW",
            "reason": "No clear statistical edge for Over/Under"
        }
