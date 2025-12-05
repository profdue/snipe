import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from models.predictor import OverUnderPredictor
from models.staking import KellyCriterion

# Page configuration
st.set_page_config(
    page_title="SNIPE: Football Over/Under Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
    }
    .confidence-high {
        background-color: #C8E6C9;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
    }
    .confidence-moderate {
        background-color: #FFF3CD;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #FFC107;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

class FootballPredictorApp:
    def __init__(self):
        self.predictor = OverUnderPredictor()
        self.staking = KellyCriterion()
        self.leagues = self.load_leagues()
        
    def load_leagues(self):
        """Load available leagues from directory structure"""
        leagues = {}
        leagues_dir = "leagues"
        
        if os.path.exists(leagues_dir):
            for league_name in os.listdir(leagues_dir):
                league_path = os.path.join(leagues_dir, league_name)
                if os.path.isdir(league_path):
                    teams_file = os.path.join(league_path, "teams.csv")
                    matches_file = os.path.join(league_path, "matches.csv")
                    
                    if os.path.exists(teams_file):
                        try:
                            teams_df = pd.read_csv(teams_file)
                            leagues[league_name] = {
                                'teams': teams_df,
                                'matches': pd.read_csv(matches_file) if os.path.exists(matches_file) else None,
                                'name': league_name.replace('_', ' ').title()
                            }
                        except Exception as e:
                            st.warning(f"Could not load {league_name}: {e}")
        return leagues
    
    def calculate_team_stats(self, team_data, last_n=10):
        """Calculate team statistics for last N games"""
        stats = {}
        
        # Calculate goals per game (last 10)
        total_matches = team_data['matches_played']
        home_games = team_data['home_wins'] + team_data['home_draws'] + team_data['home_losses']
        away_games = team_data['away_wins'] + team_data['away_draws'] + team_data['away_losses']
        
        total_goals_for = team_data['home_goals_for'] + team_data['away_goals_for']
        total_goals_against = team_data['home_goals_against'] + team_data['away_goals_against']
        
        stats['gpg_last10'] = total_goals_for / total_matches if total_matches > 0 else 0
        stats['gapg_last10'] = total_goals_against / total_matches if total_matches > 0 else 0
        
        # Calculate form-based stats (last 5)
        form = team_data.get('form_last_5', '')
        form_points = {'W': 3, 'D': 1, 'L': 0}
        recent_form = sum(form_points.get(char, 0) for char in form[-5:]) / 5 if len(form) >= 5 else 0
        
        # Estimate goals from xG
        xg_for = team_data.get('avg_xg_for', stats['gpg_last10'])
        xg_against = team_data.get('avg_xg_against', stats['gapg_last10'])
        
        # Weighted average (60% actual, 40% xG)
        stats['gpg_hybrid'] = 0.6 * stats['gpg_last10'] + 0.4 * xg_for
        stats['gapg_hybrid'] = 0.6 * stats['gapg_last10'] + 0.4 * xg_against
        
        # Attack/defense strength
        stats['attack_strength'] = team_data.get('attack_strength', 1.0)
        stats['defense_strength'] = team_data.get('defense_strength', 1.0)
        
        # Calculate home/away specific stats
        stats['home_gpg'] = team_data['home_goals_for'] / home_games if home_games > 0 else 0
        stats['away_gpg'] = team_data['away_goals_for'] / away_games if away_games > 0 else 0
        stats['home_gapg'] = team_data['home_goals_against'] / home_games if home_games > 0 else 0
        stats['away_gapg'] = team_data['away_goals_against'] / away_games if away_games > 0 else 0
        
        return stats
    
    def run(self):
        # Header
        st.markdown('<h1 class="main-header">⚽ SNIPE: Football Over/Under Predictor</h1>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # League selection
            if self.leagues:
                selected_league = st.selectbox(
                    "Select League",
                    list(self.leagues.keys()),
                    format_func=lambda x: self.leagues[x]['name']
                )
                
                league_data = self.leagues[selected_league]
                teams_df = league_data['teams']
                
                # Team selection
                team_options = teams_df['team_name'].tolist()
                home_team = st.selectbox("Home Team", team_options)
                away_team = st.selectbox("Away Team", [t for t in team_options if t != home_team])
                
                # Betting parameters
                st.subheader("💰 Betting Parameters")
                bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=100000, value=1000, step=100)
                min_confidence = st.slider("Minimum Confidence (%)", 0, 100, 70, 5)
                max_stake_percent = st.slider("Max Stake (% of Bankroll)", 1, 10, 5, 1)
                
                # Additional filters
                st.subheader("🔍 Filters")
                show_all = st.checkbox("Show All Predictions", value=False)
                only_high_confidence = st.checkbox("Only High Confidence Bets", value=True)
                
                # Calculate button
                calculate_btn = st.button("🚀 Calculate Predictions", type="primary", use_container_width=True)
            else:
                st.error("No league data found! Please check the leagues directory.")
                return
        
        # Main content area
        if 'calculate_btn' in locals() and calculate_btn:
            # Get team data
            home_data = teams_df[teams_df['team_name'] == home_team].iloc[0]
            away_data = teams_df[teams_df['team_name'] == away_team].iloc[0]
            
            # Calculate stats
            home_stats = self.calculate_team_stats(home_data)
            away_stats = self.calculate_team_stats(away_data)
            
            # Make prediction
            prediction = self.predictor.predict_over_under(home_stats, away_stats, is_home=True)
            
            # Display prediction
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="text-align: center; margin-bottom: 20px;">🎯 Match Prediction</h3>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div style="text-align: center;">
                            <h4>{home_team}</h4>
                            <p>Home</p>
                        </div>
                        <h2 style="margin: 0 20px;">VS</h2>
                        <div style="text-align: center;">
                            <h4>{away_team}</h4>
                            <p>Away</p>
                        </div>
                    </div>
                    <hr>
                    <div style="text-align: center;">
                        <h2 style="color: {'#4CAF50' if prediction['prediction'] == 'Over 2.5' else '#FF9800'};">
                            {prediction['prediction']}
                        </h2>
                        <p><strong>Confidence:</strong> {prediction['confidence']}</p>
                        <p><strong>Probability:</strong> {prediction['probability']:.1%}</p>
                        <p><strong>Expected Goals:</strong> {prediction['expected_goals']:.2f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Staking recommendation
            if prediction['probability'] > (min_confidence / 100):
                odds = st.number_input("Enter Odds", min_value=1.1, max_value=10.0, value=1.85, step=0.05)
                
                stake_info = self.staking.calculate_stake(
                    probability=prediction['probability'],
                    odds=odds,
                    bankroll=bankroll,
                    max_percent=max_stake_percent / 100
                )
                
                st.markdown(f"""
                <div class="{'confidence-high' if prediction['confidence'] == 'High' else 'confidence-moderate'}">
                    <h4>💰 Staking Recommendation</h4>
                    <p><strong>Recommended Stake:</strong> ${stake_info['stake_amount']:.2f} ({stake_info['stake_percent']:.1%} of bankroll)</p>
                    <p><strong>Expected Value:</strong> ${stake_info['expected_value']:.2f}</p>
                    <p><strong>Risk Level:</strong> {stake_info['risk_level']}</p>
                    <p><strong>Kelly Fraction:</strong> {stake_info['kelly_fraction']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Team statistics
            st.subheader("📊 Team Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {home_team} (Home)")
                stats_data = {
                    'Metric': ['Goals/Game', 'Goals Against/Game', 'xG/Game', 'xGA/Game', 
                              'Attack Strength', 'Defense Strength', 'Recent Form'],
                    'Value': [
                        f"{home_stats['gpg_last10']:.2f}",
                        f"{home_stats['gapg_last10']:.2f}",
                        f"{home_data.get('avg_xg_for', home_stats['gpg_last10']):.2f}",
                        f"{home_data.get('avg_xg_against', home_stats['gapg_last10']):.2f}",
                        f"{home_stats['attack_strength']:.2f}",
                        f"{home_stats['defense_strength']:.2f}",
                        home_data.get('form_last_5', 'N/A')
                    ]
                }
                st.table(pd.DataFrame(stats_data))
            
            with col2:
                st.markdown(f"### {away_team} (Away)")
                stats_data = {
                    'Metric': ['Goals/Game', 'Goals Against/Game', 'xG/Game', 'xGA/Game',
                              'Attack Strength', 'Defense Strength', 'Recent Form'],
                    'Value': [
                        f"{away_stats['gpg_last10']:.2f}",
                        f"{away_stats['gapg_last10']:.2f}",
                        f"{away_data.get('avg_xg_for', away_stats['gpg_last10']):.2f}",
                        f"{away_data.get('avg_xg_against', away_stats['gapg_last10']):.2f}",
                        f"{away_stats['attack_strength']:.2f}",
                        f"{away_stats['defense_strength']:.2f}",
                        away_data.get('form_last_5', 'N/A')
                    ]
                }
                st.table(pd.DataFrame(stats_data))
            
            # Prediction logic explanation
            st.subheader("🔍 Prediction Logic")
            st.info(prediction['explanation'])
            
            # Historical predictions (if matches data available)
            if league_data['matches'] is not None:
                st.subheader("📈 Recent Predictions")
                # Filter recent matches for these teams
                matches_df = league_data['matches']
                recent_matches = matches_df[
                    (matches_df['home_team'] == home_team) | 
                    (matches_df['away_team'] == away_team)
                ].head(10)
                
                if not recent_matches.empty:
                    st.dataframe(recent_matches)
        
        else:
            # Welcome screen
            st.markdown("""
            ## 🎯 Welcome to SNIPE Predictor
            
            This app uses advanced statistical models to predict Over/Under 2.5 goals in football matches.
            
            ### 📊 How It Works
            
            1. **Select a league** from the sidebar
            2. **Choose home and away teams**
            3. **Configure your betting parameters**
            4. **Get predictions** with confidence levels and staking recommendations
            
            ### 🏆 Available Leagues
            
            The system currently supports:
            """)
            
            # Display available leagues
            cols = st.columns(min(3, len(self.leagues)))
            for idx, (league_key, league_info) in enumerate(self.leagues.items()):
                with cols[idx % 3]:
                    st.info(f"**{league_info['name']}**\n\n{len(league_info['teams'])} teams loaded")
            
            st.markdown("""
            ### 📈 Prediction Rules
            
            #### High Confidence Bets:
            1. **Over 2.5**: Both teams >1.5 GPG (Last 10 & 5 games)
            2. **Under 2.5**: Defense <1.0 GA PG vs Attack <1.5 GPG (Last 10 & 5)
            
            #### Moderate Confidence Bets:
            3. **Over 2.5**: Both teams >1.5 GPG (Last 5 games only)
            4. **Under 2.5**: Defense <1.0 GA PG vs Attack <1.5 GPG (Last 5 only)
            
            #### No Bet:
            5. No clear statistical edge
            """)

if __name__ == "__main__":
    app = FootballPredictorApp()
    app.run()
