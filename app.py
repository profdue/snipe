"""
PHANTOM PREDICTOR v4.3 - Main Streamlit Application
Statistically Validated Football Prediction Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from models.prediction_engine import PhantomPredictor
from models.risk_calculator import RiskCalculator

# Page configuration
st.set_page_config(
    page_title="PHANTOM PREDICTOR v4.3",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #CCCCCC;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
        margin-bottom: 1rem;
    }
    
    .betting-card {
        background-color: #1E1E24;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #444;
        margin-bottom: 1rem;
    }
    
    .strong-bet {
        border-left: 4px solid #00D26A;
    }
    
    .moderate-bet {
        border-left: 4px solid #FFB74D;
    }
    
    .light-bet {
        border-left: 4px solid #64B5F6;
    }
    
    .stMetric {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'risk_calculator' not in st.session_state:
    st.session_state.risk_calculator = None
if 'current_league' not in st.session_state:
    st.session_state.current_league = None
if 'team_data' not in st.session_state:
    st.session_state.team_data = None

def initialize_app():
    """Initialize the application with data"""
    data_loader = DataLoader()
    
    # Get available leagues
    available_leagues = data_loader.get_available_leagues()
    
    if not available_leagues:
        st.error("No league data found. Please add league data to the data/leagues folder.")
        return None, None, None
    
    return data_loader, available_leagues

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🔥 PHANTOM PREDICTOR v4.3</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Statistically Validated • Form-First Logic • xG Integration • Risk-Aware Staking</p>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        # Bankroll Management
        st.subheader("💰 Bankroll Management")
        bankroll = st.number_input(
            "Bankroll (units):",
            min_value=10.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Your total betting bankroll"
        )
        
        # Risk Parameters
        st.subheader("🎯 Risk Parameters")
        kelly_fraction = st.slider(
            "Kelly Fraction:",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Fraction of full Kelly criterion to use (0.25 = quarter Kelly)"
        )
        
        min_confidence = st.slider(
            "Minimum Confidence (%):",
            min_value=30,
            max_value=80,
            value=55,
            step=5,
            help="Minimum confidence threshold for placing bets"
        ) / 100
        
        # League Selection
        st.subheader("🏆 League Selection")
        
        data_loader, available_leagues = initialize_app()
        
        if available_leagues:
            selected_league = st.selectbox(
                "Select League:",
                available_leagues,
                index=0 if 'Premier League' in available_leagues else 0
            )
            
            st.info(f"**Current League:** {selected_league}")
            
            # Load league data
            try:
                teams_df = data_loader.load_league_teams(selected_league)
                upcoming_df = data_loader.load_upcoming_matches(selected_league)
                
                st.success(f"📊 {len(teams_df)} teams loaded")
                
                if not upcoming_df.empty:
                    st.success(f"📅 {len(upcoming_df)} upcoming matches")
                
            except Exception as e:
                st.error(f"Error loading league data: {str(e)}")
                teams_df = pd.DataFrame()
                upcoming_df = pd.DataFrame()
        else:
            selected_league = None
            teams_df = pd.DataFrame()
            upcoming_df = pd.DataFrame()
        
        # Features Section
        st.markdown("---")
        st.subheader("🎯 v4.3 FEATURES")
        st.markdown("""
        **Statistical Rigor:**
        • Neutral baseline xG
        • Pure Poisson probabilities
        • Bayesian shrinkage
        • xG integration (60/40)
        
        **Risk-Aware:**
        • Fractional Kelly staking
        • Edge-based decisions
        • Bankroll management
        • Confidence-based stakes
        """)
    
    # Main Content Area
    if selected_league and not teams_df.empty:
        # Initialize predictor and risk calculator
        predictor = PhantomPredictor(teams_df)
        risk_calculator = RiskCalculator(
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_confidence=min_confidence
        )
        
        st.session_state.predictor = predictor
        st.session_state.risk_calculator = risk_calculator
        st.session_state.current_league = selected_league
        st.session_state.team_data = teams_df
        
        # Match Selection Section
        st.header("🎯 SELECT MATCH")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 HOME TEAM")
            home_teams = sorted(teams_df['team_name'].unique())
            home_team = st.selectbox("Select Home Team:", home_teams, key="home_select")
            
            if home_team:
                home_stats = data_loader.get_team_details(selected_league, home_team)
                
                if home_stats:
                    # Home Team Stats
                    st.markdown("### 🏠 HOME STATS")
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Matches Played", home_stats['matches_played'])
                    
                    with stats_col2:
                        home_record = f"{home_stats['home_wins']}-{home_stats['home_draws']}-{home_stats['home_losses']}"
                        st.metric("Home Record", home_record)
                    
                    with stats_col3:
                        st.metric("Points", home_stats.get('home_points', 'N/A'))
                    
                    # Goals Section
                    st.markdown("### ⚽ GOALS")
                    
                    goals_col1, goals_col2 = st.columns(2)
                    
                    with goals_col1:
                        st.metric(
                            "Goals For",
                            home_stats['home_goals_for'],
                            f"{home_stats['home_gpg']:.2f} per game"
                        )
                    
                    with goals_col2:
                        st.metric(
                            "Goals Against",
                            home_stats['home_goals_against'],
                            f"{home_stats['home_gapg']:.2f} per game"
                        )
                    
                    # Advanced Statistics (collapsible)
                    with st.expander("📈 ADVANCED HOME STATISTICS"):
                        adv_col1, adv_col2, adv_col3 = st.columns(3)
                        
                        with adv_col1:
                            st.metric("Win Rate", f"{home_stats['home_win_rate']*100:.1f}%")
                        
                        with adv_col2:
                            st.metric("xG For", f"{home_stats['avg_xg_for']:.2f}")
                        
                        with adv_col3:
                            st.metric("xG Against", f"{home_stats['avg_xg_against']:.2f}")
        
        with col2:
            st.subheader("✈️ AWAY TEAM")
            away_teams = sorted([t for t in teams_df['team_name'].unique() if t != home_team])
            away_team = st.selectbox("Select Away Team:", away_teams, key="away_select")
            
            if away_team:
                away_stats = data_loader.get_team_details(selected_league, away_team)
                
                if away_stats:
                    # Away Team Stats
                    st.markdown("### ✈️ AWAY STATS")
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Matches Played", away_stats['matches_played'])
                    
                    with stats_col2:
                        away_record = f"{away_stats['away_wins']}-{away_stats['away_draws']}-{away_stats['away_losses']}"
                        st.metric("Away Record", away_record)
                    
                    with stats_col3:
                        st.metric("Points", away_stats.get('away_points', 'N/A'))
                    
                    # Goals Section
                    st.markdown("### ⚽ GOALS")
                    
                    goals_col1, goals_col2 = st.columns(2)
                    
                    with goals_col1:
                        st.metric(
                            "Goals For",
                            away_stats['away_goals_for'],
                            f"{away_stats['away_gpg']:.2f} per game"
                        )
                    
                    with goals_col2:
                        st.metric(
                            "Goals Against",
                            away_stats['away_goals_against'],
                            f"{away_stats['away_gapg']:.2f} per game"
                        )
                    
                    # Advanced Statistics (collapsible)
                    with st.expander("📈 ADVANCED AWAY STATISTICS"):
                        adv_col1, adv_col2, adv_col3 = st.columns(3)
                        
                        with adv_col1:
                            st.metric("Win Rate", f"{away_stats['away_win_rate']*100:.1f}%")
                        
                        with adv_col2:
                            st.metric("xG For", f"{away_stats['avg_xg_for']:.2f}")
                        
                        with adv_col3:
                            st.metric("xG Against", f"{away_stats['avg_xg_against']:.2f}")
        
        # Analysis Button
        if home_team and away_team and home_team != away_team:
            st.markdown("---")
            
            if st.button("🚀 RUN PHANTOM ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("🧠 Running statistical analysis..."):
                    # Get xG data if available
                    home_xg = None
                    away_xg = None
                    market_odds = {}
                    
                    if not upcoming_df.empty:
                        match_data = upcoming_df[
                            (upcoming_df['home_team'] == home_team) & 
                            (upcoming_df['away_team'] == away_team)
                        ]
                        
                        if not match_data.empty:
                            match_row = match_data.iloc[0]
                            home_xg = match_row['home_xg']
                            away_xg = match_row['away_xg']
                            
                            market_odds = {
                                'home': match_row.get('market_home_odds', 2.0),
                                'draw': match_row.get('market_draw_odds', 3.5),
                                'away': match_row.get('market_away_odds', 4.0),
                                'over_25': match_row.get('market_over_25_odds', 2.0),
                                'under_25': 1.8,  # Default if not provided
                                'btts': match_row.get('market_btts_odds', 1.8),
                                'btts_no': 2.0  # Default if not provided
                            }
                    
                    # Run prediction
                    prediction = predictor.predict_match(home_team, away_team, home_xg, away_xg)
                    
                    # Display Analysis Results
                    st.header("📊 ANALYSIS RESULTS")
                    
                    # Form Scores
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 🎯 FORM SCORES")
                        st.metric(
                            "Home Form Score",
                            f"{prediction['form_scores']['home']:.2f}",
                            f"{prediction['form_scores']['home']*100:.0f}%"
                        )
                        st.metric(
                            "Away Form Score", 
                            f"{prediction['form_scores']['away']:.2f}",
                            f"{prediction['form_scores']['away']*100:.0f}%"
                        )
                        
                        form_adv = prediction['form_advantage']
                        adv_label = "📈 Home advantage" if form_adv > 0 else "📉 Away advantage"
                        st.metric("Form Advantage", f"{form_adv:+.2f}", adv_label)
                    
                    # Attack & Defense Strengths
                    with col2:
                        st.markdown("### ⚽ ATTACK & DEFENSE STRENGTHS")
                        
                        # Get team strengths
                        home_attack, home_defense = predictor.calculate_team_strengths(home_team, is_home=True)
                        away_attack, away_defense = predictor.calculate_team_strengths(away_team, is_home=False)
                        
                        st.metric("Home Attack", f"{home_attack:.2f}", "1.0 = league average")
                        st.metric("Away Attack", f"{away_attack:.2f}", "1.0 = league average")
                        st.metric("Home Defense", f"{home_defense:.2f}", "Lower = better defense")
                        st.metric("Away Defense", f"{away_defense:.2f}", "Lower = better defense")
                    
                    # Expected Goals
                    with col3:
                        st.markdown("### 🎯 EXPECTED GOALS")
                        
                        st.metric("Home xG", f"{prediction['expected_goals']['home']:.2f}")
                        st.metric("Away xG", f"{prediction['expected_goals']['away']:.2f}")
                        st.metric("Total xG", f"{prediction['total_expected_goals']:.2f}")
                        
                        # Compare to league average
                        league_avg = predictor.league_stats['avg_home_goals'] + predictor.league_stats['avg_away_goals']
                        vs_avg = prediction['total_expected_goals'] - league_avg
                        st.metric("vs League Avg", f"{vs_avg:+.2f}")
                    
                    # Betting Recommendations
                    st.markdown("---")
                    st.header("🔥 BOLD PREDICTIONS")
                    
                    # Get betting recommendations
                    recommendations = risk_calculator.get_betting_recommendations(prediction, market_odds)
                    
                    if recommendations:
                        # Categorize bets
                        strong_bets = [r for r in recommendations if 'STRONG' in r['category']]
                        moderate_bets = [r for r in recommendations if 'MODERATE' in r['category']]
                        light_bets = [r for r in recommendations if 'LIGHT' in r['category']]
                        
                        # Display betting cards
                        if strong_bets:
                            st.subheader(f"🔥 STRONG PLAYS ({len(strong_bets)})")
                            for bet in strong_bets:
                                with st.container():
                                    st.markdown(f"""
                                    <div class="betting-card strong-bet">
                                        <h4>{bet['market']}</h4>
                                        <p>Confidence: <strong>{bet['probability']}%</strong> | Odds: {bet['odds']}</p>
                                        <p>Stake: <strong>{bet['stake_units']} units</strong> | Risk: {bet['risk_level']} {bet['risk_emoji']}</p>
                                        <p>Edge: +{bet['edge_percent']}% | Expected Value: {bet['expected_value']} units</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        if moderate_bets:
                            st.subheader(f"⚡ MODERATE PLAYS ({len(moderate_bets)})")
                            for bet in moderate_bets:
                                with st.container():
                                    st.markdown(f"""
                                    <div class="betting-card moderate-bet">
                                        <h4>{bet['market']}</h4>
                                        <p>Confidence: <strong>{bet['probability']}%</strong> | Odds: {bet['odds']}</p>
                                        <p>Stake: <strong>{bet['stake_units']} units</strong> | Risk: {bet['risk_level']} {bet['risk_emoji']}</p>
                                        <p>Edge: +{bet['edge_percent']}% | Expected Value: {bet['expected_value']} units</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        if light_bets:
                            st.subheader(f"📊 LIGHT PLAYS ({len(light_bets)})")
                            for bet in light_bets:
                                with st.container():
                                    st.markdown(f"""
                                    <div class="betting-card light-bet">
                                        <h4>{bet['market']}</h4>
                                        <p>Confidence: <strong>{bet['probability']}%</strong> | Odds: {bet['odds']}</p>
                                        <p>Stake: <strong>{bet['stake_units']} units</strong> | Risk: {bet['risk_level']} {bet['risk_emoji']}</p>
                                        <p>Edge: +{bet['edge_percent']}% | Expected Value: {bet['expected_value']} units</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Summary
                        total_units = sum([b['stake_units'] for b in recommendations])
                        st.info(f"""
                        **📋 SUMMARY:** 🔥 {len(strong_bets)} STRONG betting opportunities, 
                        ⚡ {len(moderate_bets)} MODERATE, 📊 {len(light_bets)} LIGHT 
                        ({total_units:.2f} total units)
                        """)
                    else:
                        st.warning("⚠️ No betting recommendations meeting confidence threshold")
                    
                    # Expected Scoreline
                    st.markdown("---")
                    st.header("📈 EXPECTED SCORELINE")
                    
                    most_likely = prediction['likely_scorelines'][0]
                    score = f"{most_likely[0][0]} - {most_likely[0][1]}"
                    probability = f"{most_likely[1]*100:.1f}%"
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Most Likely Scoreline", score, probability)
                    
                    with col2:
                        st.write(f"**Based on xG:** {prediction['expected_goals']['home']:.2f} - {prediction['expected_goals']['away']:.2f}")
                        
                        if len(prediction['likely_scorelines']) > 1:
                            st.write("**Top 3 Most Likely Scorelines:**")
                            for i, (scoreline, prob) in enumerate(prediction['likely_scorelines'][:3], 1):
                                st.write(f"{i}. {scoreline[0]}-{scoreline[1]} ({prob*100:.1f}%)")
                    
                    # Visualization
                    st.markdown("---")
                    st.header("📊 PROBABILITY DISTRIBUTION")
                    
                    # Create probability distribution chart
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Match Outcome Probabilities', 'Goals Distribution'),
                        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
                    )
                    
                    # Pie chart for match outcomes
                    outcome_labels = ['Home Win', 'Draw', 'Away Win']
                    outcome_values = [
                        prediction['probabilities']['home_win'],
                        prediction['probabilities']['draw'],
                        prediction['probabilities']['away_win']
                    ]
                    
                    fig.add_trace(
                        go.Pie(
                            labels=outcome_labels,
                            values=outcome_values,
                            hole=0.4,
                            marker=dict(colors=['#FF4B4B', '#FFB74D', '#64B5F6'])
                        ),
                        row=1, col=1
                    )
                    
                    # Bar chart for goal probabilities
                    max_goals = 5
                    goal_probs_home = [poisson.pmf(i, prediction['expected_goals']['home']) for i in range(max_goals)]
                    goal_probs_away = [poisson.pmf(i, prediction['expected_goals']['away']) for i in range(max_goals)]
                    
                    fig.add_trace(
                        go.Bar(
                            x=list(range(max_goals)),
                            y=goal_probs_home,
                            name='Home Goals',
                            marker_color='#FF4B4B'
                        ),
                        row=1, col=2
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=list(range(max_goals)),
                            y=goal_probs_away,
                            name='Away Goals',
                            marker_color='#64B5F6'
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Upcoming Matches Section
        if not upcoming_df.empty:
            st.markdown("---")
            st.header("📅 UPCOMING MATCHES")
            
            # Display upcoming matches
            display_cols = ['date', 'home_team', 'away_team', 'home_xg', 'away_xg']
            display_df = upcoming_df[display_cols].copy()
            display_df.columns = ['Date', 'Home', 'Away', 'Home xG', 'Away xG']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
    
    else:
        st.warning("Please select a league and ensure team data is available.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    **⚡ PHANTOM PREDICTOR v4.3** • League: {selected_league or 'None'} • Statistically Validated • xG Integration
    """)

if __name__ == "__main__":
    main()