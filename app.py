import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from models.edgefinder_predictor import EdgeFinderPredictor, EnhancedTeamStats, ConfidenceLevel


class EdgeFinderFootballApp:
    def __init__(self):
        # Don't initialize predictor here - it needs sidebar values
        self.predictor = None
        self.leagues = {}
        self.market_templates = self.load_market_templates()
        self.league_contexts = {
            'premier_league': {'avg_gpg': 2.7},
            'la_liga': {'avg_gpg': 2.5},
            'bundesliga': {'avg_gpg': 3.0},
            'serie_a': {'avg_gpg': 2.6},
            'ligue_1': {'avg_gpg': 2.4},
            'default': {'avg_gpg': 2.7}
        }
        
    def load_leagues(self) -> Dict:
        """Load available leagues from nested directory structure"""
        leagues = {}
        leagues_dir = "leagues"
        
        if os.path.exists(leagues_dir):
            for league_dir in os.listdir(leagues_dir):
                league_path = os.path.join(leagues_dir, league_dir)
                if os.path.isdir(league_path):
                    for filename in os.listdir(league_path):
                        if filename.lower().endswith('.csv'):
                            filepath = os.path.join(league_path, filename)
                            
                            try:
                                df = pd.read_csv(filepath)
                                
                                # Standardize column names
                                df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
                                
                                # Get league name from directory
                                league_name = league_dir
                                display_name = league_dir.replace('_', ' ').title()
                                
                                # Check if team_name column exists
                                if 'team_name' in df.columns:
                                    team_names = df['team_name'].tolist()
                                else:
                                    team_cols = [col for col in df.columns if 'team' in col.lower() or 'name' in col.lower()]
                                    if team_cols:
                                        team_names = df[team_cols[0]].tolist()
                                    else:
                                        st.warning(f"No team name column found in {filepath}")
                                        continue
                                
                                leagues[league_name] = {
                                    'teams': df,
                                    'name': display_name,
                                    'team_names': team_names
                                }
                                
                                break
                                
                            except Exception as e:
                                st.warning(f"Could not load {filepath}: {str(e)}")
        
        if not leagues:
            st.error(f"No league data found in {leagues_dir} directory.")
            st.info(f"Expected structure: {leagues_dir}/league_name/teams.csv")
            
        return leagues
    
    def load_market_templates(self) -> Dict:
        """Load pre-defined market templates"""
        return {
            'balanced': {
                'home_win': 2.50,
                'draw': 3.40,
                'away_win': 2.80,
                'home_draw': 1.50,
                'away_draw': 1.55,
                'over_25': 1.85,
                'under_25': 2.00,
                'btts_yes': 1.75,
                'btts_no': 2.05
            },
            'favorite_home': {
                'home_win': 1.80,
                'draw': 3.75,
                'away_win': 4.50,
                'home_draw': 1.25,
                'away_draw': 2.10,
                'over_25': 1.90,
                'under_25': 1.95,
                'btts_yes': 1.65,
                'btts_no': 2.25
            },
            'favorite_away': {
                'home_win': 4.00,
                'draw': 3.60,
                'away_win': 1.90,
                'home_draw': 1.80,
                'away_draw': 1.30,
                'over_25': 1.85,
                'under_25': 2.00,
                'btts_yes': 1.70,
                'btts_no': 2.15
            }
        }
    
    def create_team_stats(self, team_data: pd.Series) -> EnhancedTeamStats:
        """Create EnhancedTeamStats object from CSV data with proper cleaning"""
        data_dict = {}
        
        # Define all expected fields
        expected_fields = {
            'team_name': str, 'matches_played': int, 'possession_avg': float,
            'shots_per_game': float, 'shots_on_target_pg': float, 'conversion_rate': float,
            'xg_for_avg': float, 'xg_against_avg': float,
            
            'home_wins': int, 'home_draws': int, 'home_losses': int,
            'away_wins': int, 'away_draws': int, 'away_losses': int,
            'home_goals_for': int, 'home_goals_against': int,
            'away_goals_for': int, 'away_goals_against': int,
            
            'clean_sheet_pct': float, 'clean_sheet_pct_home': float, 'clean_sheet_pct_away': float,
            'failed_to_score_pct': float, 'failed_to_score_pct_home': float, 'failed_to_score_pct_away': float,
            
            'btts_pct': float, 'btts_pct_home': float, 'btts_pct_away': float,
            'over25_pct': float, 'over25_pct_home': float, 'over25_pct_away': float,
            
            'last5_form': str, 'last5_wins': int, 'last5_draws': int, 'last5_losses': int,
            'last5_goals_for': int, 'last5_goals_against': int,
            'last5_ppg': float, 'last5_cs_pct': float, 'last5_fts_pct': float,
            'last5_btts_pct': float, 'last5_over25_pct': float
        }
        
        for field, field_type in expected_fields.items():
            if field in team_data:
                value = team_data[field]
                
                if pd.isna(value):
                    if field_type == str:
                        data_dict[field] = ""
                    elif field_type == int:
                        data_dict[field] = 0
                    else:
                        data_dict[field] = 0.0
                    continue
                
                if isinstance(value, str):
                    value = value.strip()
                
                if field_type == str:
                    data_dict[field] = str(value)
                    
                elif field_type == int:
                    try:
                        if isinstance(value, float):
                            data_dict[field] = int(value)
                        else:
                            data_dict[field] = int(float(str(value)))
                    except:
                        data_dict[field] = 0
                        
                elif field_type == float:
                    try:
                        float_value = float(str(value))
                        data_dict[field] = float_value
                    except:
                        data_dict[field] = 0.0
            else:
                if field_type == str:
                    data_dict[field] = ""
                elif field_type == int:
                    data_dict[field] = 0
                else:
                    data_dict[field] = 0.0
        
        return EnhancedTeamStats(**data_dict)
    
    def get_team_style_icon(self, style: str) -> str:
        """Get emoji icon for team style"""
        icons = {
            'Possession': '🔵',
            'Counter': '⚡',
            'High Press': '🔥',
            'Low Block': '🛡️',
            'Balanced': '⚖️'
        }
        return icons.get(style, '❓')
    
    def format_edge_percentage(self, edge: float) -> Tuple[str, str]:
        """Format edge percentage with color coding"""
        if edge >= 20:
            return f"🔥 EXTREME +{edge:.1f}%", "warning"
        elif edge >= 10:
            return f"⭐⭐⭐ +{edge:.1f}%", "success"
        elif edge >= 5:
            return f"⭐⭐ +{edge:.1f}%", "warning"
        elif edge >= 3:
            return f"⭐ +{edge:.1f}%", "info"
        elif edge > 0:
            return f"+{edge:.1f}%", "secondary"
        elif edge == 0:
            return f"{edge:.1f}%", "secondary"
        else:
            return f"{edge:.1f}%", "error"
    
    def get_confidence_color(self, score: int) -> str:
        """Get color for confidence score"""
        if score >= 8:
            return "#4CAF50"  # Green
        elif score >= 6:
            return "#FF9800"  # Orange
        elif score >= 4:
            return "#FF5722"  # Red-Orange
        else:
            return "#F44336"  # Red
    
    def run(self):
        # Set page config
        st.set_page_config(
            page_title="⚽ EdgeFinder Pro: Football Value Betting System",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load leagues
        self.leagues = self.load_leagues()
        
        if not self.leagues:
            st.error("No league data found. Please check your data directory.")
            return
        
        # Custom CSS
        st.markdown("""
            <style>
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .value-bet-card {
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                border-radius: 10px;
                padding: 1.5rem;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .extreme-edge-card {
                background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
                border-radius: 10px;
                padding: 1.5rem;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border: 2px solid #FF9800;
            }
            .confidence-high {
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                display: inline-block;
                font-weight: bold;
            }
            .confidence-medium {
                background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                display: inline-block;
                font-weight: bold;
            }
            .confidence-low {
                background: linear-gradient(135deg, #FF5722 0%, #D84315 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                display: inline-block;
                font-weight: bold;
            }
            .team-stat-box {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                border-left: 3px solid #4CAF50;
            }
            .logic-framework {
                background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                border-radius: 10px;
                padding: 1.5rem;
                color: white;
                margin: 1rem 0;
            }
            .market-category {
                background-color: #f0f2f6;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                margin: 0.5rem 0;
                font-weight: bold;
            }
            .warning-box {
                background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
                border-radius: 10px;
                padding: 1rem;
                color: white;
                margin: 1rem 0;
                border: 2px solid #FF5722;
            }
            .reality-check {
                background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                border-radius: 10px;
                padding: 1rem;
                color: white;
                margin: 1rem 0;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5rem;">⚽ EdgeFinder Pro v2.1</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">FIXED Football Value Betting System</p>
            <p style="font-size: 1rem; opacity: 0.8;">Now with realistic predictions and proper defense calculations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state
        if 'form_weight' not in st.session_state:
            st.session_state.form_weight = 0.4
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Match Configuration")
            
            # League selection
            selected_league = st.selectbox(
                "Select League",
                list(self.leagues.keys()),
                format_func=lambda x: self.leagues[x]['name']
            )
            
            league_data = self.leagues[selected_league]
            teams_df = league_data['teams']
            team_names = league_data['team_names']
            
            st.write(f"**{len(team_names)} teams loaded**")
            
            # Team selection
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("🏠 Home Team", team_names)
            with col2:
                away_options = [t for t in team_names if t != home_team]
                away_team = st.selectbox("✈️ Away Team", away_options)
            
            # Predictor configuration
            st.subheader("⚡ Predictor Configuration")
            
            with st.expander("Advanced Settings"):
                form_weight = st.slider(
                    "Form Weight %", 
                    20, 60, int(st.session_state.form_weight * 100), 5,
                    key="form_weight_slider"
                ) / 100.0
                
                min_sample_size = st.slider(
                    "Minimum Sample Size", 
                    3, 10, 5, 1,
                    help="Minimum games needed to use venue-specific data"
                )
            
            # Market odds
            st.subheader("💰 Market Odds")
            
            template = st.selectbox(
                "Market Template",
                ["Custom", "Balanced", "Favorite Home", "Favorite Away"]
            )
            
            if template == "Custom":
                # Match Result (1X2)
                st.markdown('<div class="market-category">🏆 Match Result (1X2)</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    home_win = st.number_input("Home Win", value=2.50, min_value=1.01, max_value=20.0, step=0.05, key="home_win_custom")
                with col2:
                    draw = st.number_input("Draw", value=3.40, min_value=1.01, max_value=20.0, step=0.05, key="draw_custom")
                with col3:
                    away_win = st.number_input("Away Win", value=2.80, min_value=1.01, max_value=20.0, step=0.05, key="away_win_custom")
                
                # Double Chance
                st.markdown('<div class="market-category">🔀 Double Chance</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    home_draw = st.number_input("Home or Draw", value=1.50, min_value=1.01, max_value=5.0, step=0.05, key="home_draw_custom")
                with col2:
                    away_draw = st.number_input("Away or Draw", value=1.55, min_value=1.01, max_value=5.0, step=0.05, key="away_draw_custom")
                
                # Goals Markets
                st.markdown('<div class="market-category">⚽ Goals Markets</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    over_odds = st.number_input("Over 2.5 Goals", value=1.85, min_value=1.01, max_value=10.0, step=0.05, key="over_custom")
                with col2:
                    under_odds = st.number_input("Under 2.5 Goals", value=2.00, min_value=1.01, max_value=10.0, step=0.05, key="under_custom")
                
                # BTTS Markets
                st.markdown('<div class="market-category">🎯 Both Teams to Score</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    btts_yes = st.number_input("BTTS - Yes", value=1.75, min_value=1.01, max_value=10.0, step=0.05, key="btts_yes_custom")
                with col2:
                    btts_no = st.number_input("BTTS - No", value=2.05, min_value=1.01, max_value=10.0, step=0.05, key="btts_no_custom")
                
            else:
                template_odds = self.market_templates[template.lower().replace(' ', '_')]
                home_win = template_odds['home_win']
                away_win = template_odds['away_win']
                draw = template_odds['draw']
                home_draw = template_odds['home_draw']
                away_draw = template_odds['away_draw']
                over_odds = template_odds['over_25']
                under_odds = template_odds['under_25']
                btts_yes = template_odds['btts_yes']
                btts_no = template_odds['btts_no']
            
            market_odds = {
                'over_25': over_odds,
                'under_25': under_odds,
                'btts_yes': btts_yes,
                'btts_no': btts_no,
                'home_win': home_win,
                'away_win': away_win,
                'draw': draw,
                'home_draw': home_draw,
                'away_draw': away_draw
            }
            
            # Bankroll settings
            st.subheader("🏦 Bankroll Management")
            
            bankroll = st.number_input("Bankroll ($)", value=1000.0, min_value=100.0, max_value=100000.0, step=100.0)
            min_edge = st.slider("Minimum Edge %", 1.0, 10.0, 3.0, 0.5) / 100
            max_exposure = st.slider("Max Exposure %", 5.0, 20.0, 10.0, 1.0) / 100
            
            # Logic framework display
            with st.expander("📚 Enhanced Logic Framework"):
                st.markdown(f"""
                ### FIXED "3 THINGS" ANALYTICAL FRAMEWORK v2.1
                
                **CORRECTED Defense Multiplier Logic:**
                
                1. **TEAM IDENTITY** (What they ARE)
                - Shot volume & quality (shots_per_game, shots_on_target_pg)
                - Conversion efficiency (conversion_rate) → Now properly reduces goals
                - Playing style classification
                
                2. **DEFENSE** (What they STOP) - **FIXED**  
                - Defense quality = league_avg / team_defense
                - Attack multiplier = 1.0 / defense_quality (NOT direct multiplication!)
                - Proper bounds: 0.4x to 1.8x realistic range
                
                3. **TRANSITION** (How they CHANGE)
                - Last 5 ACTUAL GOALS (not just PPG)
                - Form ratio bounded: 0.7x to 1.5x
                - Realistic form adjustments
                
                **FIXES APPLIED:**
                ✅ Defense multipliers now properly inverted  
                ✅ Efficiency adjustments visible in outputs  
                ✅ Realistic bounds on all multipliers  
                ✅ Reality checks for improbable predictions  
                
                **Value Detection**: Edge = P_model - P_market
                - 🔥 **Extreme Edge**: >20% (verify carefully)
                - ⭐⭐⭐ **Golden Nugget**: Edge > 10% with high confidence
                - ⭐⭐ **Value Bet**: Edge > 5% with moderate confidence  
                - ⭐ **Consider**: Edge > 3% or situational value
                """)
            
            # Analyze button
            analyze_btn = st.button("🔍 Analyze Match", type="primary", use_container_width=True)
        
        # Main content
        if analyze_btn:
            # Initialize predictor ONLY when analyze is clicked
            try:
                self.predictor = EdgeFinderPredictor(
                    bankroll=bankroll,
                    min_edge=min_edge,
                    max_correlation_exposure=max_exposure,
                    form_weight=form_weight,
                    min_sample_size=min_sample_size
                )
                
                # Get team data
                home_data = teams_df[teams_df['team_name'] == home_team].iloc[0]
                away_data = teams_df[teams_df['team_name'] == away_team].iloc[0]
                
                # Create stats objects
                home_stats = self.create_team_stats(home_data)
                away_stats = self.create_team_stats(away_data)
                
                # Display team info
                st.info(f"**Analyzing:** {home_team} vs {away_team} in {self.leagues[selected_league]['name']}")
                
                # Run prediction
                with st.spinner("Analyzing match using FIXED '3 Things' framework..."):
                    result = self.predictor.predict_match(
                        home_stats=home_stats,
                        away_stats=away_stats,
                        market_odds=market_odds,
                        league=selected_league,
                        bankroll=bankroll
                    )
                
                # Display results
                self.display_results(result, home_stats, away_stats, selected_league)
                
            except Exception as e:
                st.error(f"Error analyzing match: {str(e)}")
                st.exception(e)
        else:
            self.display_welcome_screen()
    
    def display_results(self, result, home_stats, away_stats, league_key):
        """Display analysis results - UPDATED for fixed predictor"""
        
        # Match header
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); 
                  border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0; font-size: 2rem;">{home_stats.team_name} 🆚 {away_stats.team_name}</h2>
            <p style="font-size: 1.2rem; opacity: 0.9;">Fixed Analysis Results - v2.1</p>
            <p style="font-size: 0.9rem; opacity: 0.8;">With corrected defense calculations and realistic bounds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            expected_goals = result['match_analysis']['goal_expectations']['total_goals']
            st.metric("Expected Goals", f"{expected_goals:.2f}")
        with col2:
            value_bets = len(result['value_bets'])
            st.metric("Value Bets Found", value_bets)
        with col3:
            total_exposure = result['total_exposure_percent']
            st.metric("Total Exposure", f"{total_exposure:.1f}%")
        with col4:
            total_stake = result['total_stake']
            st.metric("Total Stake", f"${total_stake:.2f}")
        
        # Reality check for total goals
        league_avg = self.league_contexts.get(league_key, self.league_contexts['default'])['avg_gpg']
        total_goals = result['match_analysis']['goal_expectations']['total_goals']
        
        if total_goals > league_avg * 1.5:
            st.markdown("""
            <div class="warning-box">
                <h4 style="margin: 0; font-size: 1.2rem;">⚠️ HIGH-SCORING MATCH ALERT</h4>
                <p style="margin: 5px 0; font-size: 0.9rem;">
                    Expected total goals ({total_goals:.2f}) is significantly above {league_key.replace('_', ' ')} average ({league_avg:.1f}).
                    This could indicate genuine high-scoring potential or model limitations.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Value bets
        if result['value_bets']:
            st.markdown("### 🎯 Value Bets Recommended")
            
            # Check for extreme edges
            extreme_bets = [b for b in result['value_bets'] if b['edge_percent'] > 20]
            if extreme_bets:
                st.markdown("""
                <div class="warning-box">
                    <h4 style="margin: 0; font-size: 1.2rem;">🔥 EXTREME EDGE DETECTION</h4>
                    <p style="margin: 5px 0; font-size: 0.9rem;">
                        Some bets show edges >20%. While this could indicate genuine market inefficiency,
                        it's rare in established leagues. Verify data quality and consider smaller stakes.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            for bet in result['value_bets']:
                edge_text, _ = self.format_edge_percentage(bet['edge_percent'])
                value_rating = bet.get('value_rating', '⭐ Consider')
                
                # Determine card class based on edge size
                if bet['edge_percent'] > 20:
                    card_class = "extreme-edge-card"
                    edge_icon = "🔥"
                elif "⭐⭐⭐" in value_rating:
                    card_class = "value-bet-card"
                    edge_icon = "⭐⭐⭐"
                elif "⭐⭐" in value_rating:
                    card_class = "value-bet-card"
                    edge_icon = "⭐⭐"
                else:
                    card_class = "value-bet-card"
                    edge_icon = "⭐"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h3 style="margin: 0; font-size: 1.5rem;">{bet['bet_type']} @ {bet['market_odds']}</h3>
                    <p style="margin: 5px 0; font-size: 1.1rem;">
                        <strong>Value Rating:</strong> {value_rating} &nbsp;•&nbsp;
                        <strong>Edge:</strong> {edge_icon} {edge_text} &nbsp;•&nbsp;
                        <strong>Probability:</strong> {bet['model_probability']:.1%} vs {bet['implied_probability']:.1%} market
                    </p>
                    <p style="margin: 5px 0; font-size: 1rem;">
                        <strong>Stake:</strong> ${bet['stake_amount']:.2f} ({bet['stake_percent']:.1f}% of bankroll)
                    </p>
                    {f'<p style="font-size: 0.9rem; opacity: 0.9; margin-top: 5px;"><strong>⚠️ Extreme edge - verify carefully</strong></p>' if bet['edge_percent'] > 20 else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Show quick details
                with st.expander(f"📊 Quick analysis for {bet['bet_type']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Probabilities:**")
                        st.write(f"Model: {bet['model_probability']:.1%}")
                        st.write(f"Market Implied: {bet['implied_probability']:.1%}")
                        st.write(f"Edge: {bet['edge_percent']:.1f}%")
                        if bet['edge_percent'] > 20:
                            st.warning("Extreme edge - unusual for this league")
                    with col2:
                        st.write("**Bet Details:**")
                        st.write(f"Odds: {bet['market_odds']}")
                        st.write(f"Stake: ${bet['stake_amount']:.2f}")
                        st.write(f"Bankroll %: {bet['stake_percent']:.1f}%")
                        st.write(f"Value Rating: {value_rating}")
        else:
            st.warning(f"""
            ⚠️ **No value bets found**
            
            The model didn't find any betting opportunities meeting the criteria:
            - Minimum edge: {self.predictor.min_edge*100:.1f}%
            - Minimum confidence probability: {self.predictor.min_confidence_for_stake:.1%}
            
            **Possible reasons:**
            1. Market odds accurately reflect true probabilities
            2. Match is too unpredictable
            3. Try adjusting the minimum edge requirement
            """)
        
        # Team analysis with enhanced stats
        st.markdown("### 🔍 Team Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get adjustment factors
            adj_factors = result['match_analysis']['goal_expectations'].get('adjustment_factors', {})
            
            st.markdown(f"#### 🏠 {home_stats.team_name}")
            st.markdown(f"""
            <div class="team-stat-box">
                <p><strong>Style:</strong> {self.get_team_style_icon(home_stats.style.value)} {home_stats.style.value}</p>
                <p><strong>Shots/Game:</strong> {home_stats.shots_per_game:.1f} ({home_stats.shots_on_target_pg:.1f} on target)</p>
                <p><strong>Conversion:</strong> {home_stats.conversion_rate:.1%} (Efficiency: {adj_factors.get('home_efficiency', 1.0):.2f}x)</p>
                <p><strong>Home Attack:</strong> {adj_factors.get('home_attack', 0):.2f} goals/game</p>
                <p><strong>Home Defense:</strong> {adj_factors.get('home_defense', 0):.2f} conceded/game</p>
                <p><strong>Home Clean Sheets:</strong> {home_stats.clean_sheet_pct_home:.1%}</p>
                <p><strong>Recent Form:</strong> {home_stats.last5_form} ({home_stats.last5_goals_for} goals for, {home_stats.last5_goals_against} against)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"#### ✈️ {away_stats.team_name}")
            st.markdown(f"""
            <div class="team-stat-box">
                <p><strong>Style:</strong> {self.get_team_style_icon(away_stats.style.value)} {away_stats.style.value}</p>
                <p><strong>Shots/Game:</strong> {away_stats.shots_per_game:.1f} ({away_stats.shots_on_target_pg:.1f} on target)</p>
                <p><strong>Conversion:</strong> {away_stats.conversion_rate:.1%} (Efficiency: {adj_factors.get('away_efficiency', 1.0):.2f}x)</p>
                <p><strong>Away Attack:</strong> {adj_factors.get('away_attack', 0):.2f} goals/game</p>
                <p><strong>Away Defense:</strong> {adj_factors.get('away_defense', 0):.2f} conceded/game</p>
                <p><strong>Away Clean Sheets:</strong> {away_stats.clean_sheet_pct_away:.1%}</p>
                <p><strong>Recent Form:</strong> {away_stats.last5_form} ({away_stats.last5_goals_for} goals for, {away_stats.last5_goals_against} against)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Three Things Analysis
        analysis = result['match_analysis']
        
        st.markdown("### 📊 Enhanced Three Things Analysis")
        
        tabs = st.tabs(["1. Team Identity", "2. Defense", "3. Transition"])
        
        with tabs[0]:
            identity = analysis['identity']
            if identity.get('insights'):
                for insight in identity['insights']:
                    st.info(f"• {insight}")
            else:
                st.write("No significant identity mismatches detected.")
            
            if identity.get('style_clash'):
                st.write(f"**Style Matchup:** {identity['style_clash']}")
            
            # Show shot quality comparison
            if 'home_shot_quality' in identity and 'away_shot_quality' in identity:
                st.write(f"**Shot Quality Comparison:**")
                st.write(f"- {home_stats.team_name}: {identity['home_shot_quality']:.2f}x league average")
                st.write(f"- {away_stats.team_name}: {identity['away_shot_quality']:.2f}x league average")
        
        with tabs[1]:
            defense = analysis['defense']
            if defense.get('insights'):
                for insight in defense['insights']:
                    st.warning(f"• {insight}")
            else:
                st.write("No significant defensive patterns detected.")
            
            # Show venue defense stats
            if 'home_venue_defense' in defense and 'away_venue_defense' in defense:
                st.write(f"**Venue Defense Strength:**")
                st.write(f"- {home_stats.team_name} home defense: {defense['home_venue_defense']:.2f} goals/game")
                st.write(f"- {away_stats.team_name} away defense: {defense['away_venue_defense']:.2f} goals/game")
        
        with tabs[2]:
            transition = analysis['transition']
            if transition.get('insights'):
                for insight in transition['insights']:
                    st.success(f"• {insight}")
            else:
                st.write("No significant transition trends detected.")
            
            # Show recent form comparison
            if 'home_last5_gpg' in transition and 'away_last5_gpg' in transition:
                st.write(f"**Recent Goal Form (last 5 games):**")
                st.write(f"- {home_stats.team_name}: {transition['home_last5_gpg']:.2f} goals/game")
                st.write(f"- {away_stats.team_name}: {transition['away_last5_gpg']:.2f} goals/game")
            
            # Show pattern probabilities
            if 'combined_btts' in transition and 'combined_over25' in transition:
                st.write(f"**Historical Patterns:**")
                st.write(f"- Combined BTTS probability: {transition['combined_btts']:.1%}")
                st.write(f"- Combined Over 2.5 probability: {transition['combined_over25']:.1%}")
        
        # Goal expectations with adjustments
        goal_exp = analysis['goal_expectations']
        st.markdown("### ⚽ Goal Expectations & Adjustments")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Home Expected Goals", f"{goal_exp['lambda_home']:.2f}")
            st.caption(f"Attack: {goal_exp['adjustment_factors'].get('home_attack', 0):.2f}/game")
        with col2:
            st.metric("Away Expected Goals", f"{goal_exp['lambda_away']:.2f}")
            st.caption(f"Attack: {goal_exp['adjustment_factors'].get('away_attack', 0):.2f}/game")
        with col3:
            st.metric("Total Expected Goals", f"{goal_exp['total_goals']:.2f}")
            st.caption(f"League avg: {league_avg:.1f}/game")
        
        # Show adjustment factors - FIXED LABELS
        with st.expander("🔧 Goal Expectation Adjustments Applied"):
            adj_factors = goal_exp.get('adjustment_factors', {})
            if adj_factors:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{home_stats.team_name} Adjustments:**")
                    st.write(f"- Base Attack: {adj_factors.get('home_attack', 0):.2f} goals/game")
                    st.write(f"- Shot Quality: {adj_factors.get('home_shot_mult', 1.0):.2f}x")
                    st.write(f"- Form: {adj_factors.get('home_form_adj', 1.0):.2f}x")
                    st.write(f"- Efficiency: {adj_factors.get('home_efficiency', 1.0):.2f}x")
                    st.write(f"- Attack vs {away_stats.team_name} Defense: {adj_factors.get('away_defense_mult', 1.0):.2f}x")
                    st.write(f"- Home Advantage: {adj_factors.get('home_advantage', 1.0):.2f}x")
                
                with col2:
                    st.write(f"**{away_stats.team_name} Adjustments:**")
                    st.write(f"- Base Attack: {adj_factors.get('away_attack', 0):.2f} goals/game")
                    st.write(f"- Shot Quality: {adj_factors.get('away_shot_mult', 1.0):.2f}x")
                    st.write(f"- Form: {adj_factors.get('away_form_adj', 1.0):.2f}x")
                    st.write(f"- Efficiency: {adj_factors.get('away_efficiency', 1.0):.2f}x")
                    st.write(f"- Attack vs {home_stats.team_name} Defense: {adj_factors.get('home_defense_mult', 1.0):.2f}x")
                    st.write(f"- Away Penalty: {adj_factors.get('away_penalty', 1.0):.2f}x")
                
                # Add defense quality summary
                st.markdown("---")
                st.write("**Defense Quality Summary:**")
                col1, col2 = st.columns(2)
                with col1:
                    home_def = adj_factors.get('home_defense', league_avg)
                    home_def_quality = league_avg / home_def if home_def > 0 else 1.0
                    st.write(f"{home_stats.team_name} home defense:")
                    st.write(f"- Goals conceded: {home_def:.2f}/game")
                    st.write(f"- Quality vs avg: {home_def_quality:.2f}x")
                    st.write(f"- Effect on opponent: {1.0/home_def_quality:.2f}x scoring")
                
                with col2:
                    away_def = adj_factors.get('away_defense', league_avg)
                    away_def_quality = league_avg / away_def if away_def > 0 else 1.0
                    st.write(f"{away_stats.team_name} away defense:")
                    st.write(f"- Goals conceded: {away_def:.2f}/game")
                    st.write(f"- Quality vs avg: {away_def_quality:.2f}x")
                    st.write(f"- Effect on opponent: {1.0/away_def_quality:.2f}x scoring")
            else:
                st.write("No detailed adjustment factors available.")
        
        # Probabilities
        st.markdown("#### 📊 Probability Breakdown")
        
        prob_cols = st.columns(3)
        with prob_cols[0]:
            st.write("**Over/Under**")
            st.write(f"Over 2.5: {goal_exp['probabilities']['over25']:.1%}")
            st.write(f"Under 2.5: {goal_exp['probabilities']['under25']:.1%}")
        
        with prob_cols[1]:
            st.write("**BTTS**")
            st.write(f"BTTS Yes: {goal_exp['probabilities']['btts_yes']:.1%}")
            st.write(f"BTTS No: {goal_exp['probabilities']['btts_no']:.1%}")
        
        with prob_cols[2]:
            st.write("**Match Result**")
            st.write(f"Home Win: {goal_exp['probabilities']['home_win']:.1%}")
            st.write(f"Draw: {goal_exp['probabilities']['draw']:.1%}")
            st.write(f"Away Win: {goal_exp['probabilities']['away_win']:.1%}")
        
        # Confidence score
        if 'confidence' in analysis:
            confidence = analysis['confidence']
            st.markdown("### 🎯 Confidence Assessment")
            
            confidence_score = confidence['score']
            confidence_color = self.get_confidence_color(confidence_score)
            
            # Add explanation based on score
            explanation = ""
            if confidence_score >= 8:
                explanation = "Strong data quality with realistic predictions"
            elif confidence_score >= 6:
                explanation = "Adequate data but some limitations exist"
            elif confidence_score >= 4:
                explanation = "Limited data quality - predictions less reliable"
            else:
                explanation = "Poor data - predictions highly uncertain"
            
            st.markdown(f"""
            <div style="background-color: {confidence_color}; color: white; padding: 1rem; 
                      border-radius: 10px; text-align: center; margin: 1rem 0;">
                <div style="font-size: 1.5rem; font-weight: bold;">Confidence Score: {confidence_score}/10</div>
                <div style="font-size: 1.2rem;">{confidence['level'].value}</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">{explanation}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">{confidence['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if confidence.get('factors'):
                with st.expander("📊 Confidence Factors"):
                    st.write("**Factors affecting confidence:**")
                    for factor, weight in confidence['factors']:
                        sign = "+" if weight > 0 else ""
                        st.write(f"• {factor.replace('_', ' ').title()}: {sign}{weight}")
            
            # Warning for low confidence
            if confidence_score < 6:
                st.warning("""
                ⚠️ **Moderate/Low Confidence Warning**
                
                Predictions have limited reliability due to:
                1. Data quality issues
                2. Small sample sizes
                3. Inconsistent team performance
                
                Consider these predictions with extra caution.
                """)
    
    def display_welcome_screen(self):
        """Display welcome screen"""
        st.markdown("""
        ## 🎯 Welcome to EdgeFinder Pro v2.1
        
        ### **FIXED Football Value Betting System**
        
        **Critical Bug Fixes Applied:**
        
        1. ✅ **Defense Multiplier Logic Fixed** - No more unrealistic 4.82 expected goals!
        2. ✅ **Efficiency Adjustments Properly Applied** - Conversion rates now matter
        3. ✅ **Realistic Bounds Added** - Predictions stay within plausible ranges
        4. ✅ **Reality Check Warnings** - Alerts for improbable predictions
        5. ✅ **Correct Defense Quality Display** - Labels now show correct meaning
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔵 1. Team Identity
            **What they ARE**
            - Shot volume & quality
            - Conversion efficiency (NOW PROPERLY APPLIED)
            - Playing style
            - League-adjusted metrics
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ 2. Defense - **FIXED**
            **What they STOP**
            - Defense quality = league_avg / goals_conceded
            - Attack multiplier = 1.0 / defense_quality ✓
            - Realistic bounds: 0.4x to 1.8x
            - Venue-specific performance
            """)
        
        with col3:
            st.markdown("""
            ### 📈 3. Transition
            **How they CHANGE**
            - Last 5 actual goals scored/conceded
            - BTTS venue patterns
            - Over/Under venue trends
            - Form momentum with realistic bounds
            """)
        
        st.markdown("""
        ---
        
        ### 🚀 Getting Started
        
        1. **Select a league** from the sidebar
        2. **Choose home and away teams**
        3. **Configure market odds** (use templates or custom)
        4. **Set your bankroll** and edge requirements
        5. **Click "Analyze Match"** to find realistic value bets
        
        ### ⚡ Key Improvements in v2.1
        
        **Critical Bug Fixes:**
        - Defense multipliers now properly inverted
        - Efficiency adjustments visible and applied
        - Realistic bounds prevent improbable outputs
        - Correct labels in adjustment display
        
        **Enhanced User Experience:**
        - Reality check warnings
        - Extreme edge detection (>20%)
        - Improved confidence scoring
        - Clearer adjustment explanations
        
        **Realistic Outputs:**
        - Bundesliga matches: 2.0-3.5 expected goals (not 4.82!)
        - Probabilities match actual league patterns
        - Value bets are genuine, not mathematical artifacts
        
        ### 💰 How We Find Value Now
        
        **CORRECTED Prediction Logic:**
        ```
        Defense_Quality = League_Avg_Goals / Team_Defense
        Attack_Multiplier = 1.0 / Defense_Quality  # FIXED!
        
        Expected_Goals = Base_Attack × Shot_Quality × Form × Attack_Multiplier × Efficiency × Venue
        Edge = P_model - P_market
        ```
        
        **Realistic Value Detection:**
        - 🔥 **Extreme Edge**: >20% (verify carefully - rare in established leagues)
        - ⭐⭐⭐ **Golden Nugget**: Edge > 10% with high confidence
        - ⭐⭐ **Value Bet**: Edge > 5% with moderate confidence  
        - ⭐ **Consider**: Edge > 3% or situational value
        
        ---
        
        ### 🏆 Available Leagues
        """)
        
        for league_key, league_info in self.leagues.items():
            with st.expander(f"{league_info['name']} ({len(league_info['team_names'])} teams)"):
                st.write(f"**Teams available:**")
                teams = league_info['team_names']
                cols = st.columns(3)
                for i, team in enumerate(teams[:12]):
                    cols[i % 3].write(f"• {team}")
                if len(teams) > 12:
                    st.write(f"... and {len(teams) - 12} more")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>⚡ <strong>EdgeFinder Pro v2.1</strong> - Fixed Defense Calculations</p>
            <p>✅ <strong>Now produces realistic predictions for all top 5 European leagues</strong></p>
            <p>⚠️ <strong>Disclaimer:</strong> Sports betting involves risk. This tool provides mathematical probabilities only.</p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = EdgeFinderFootballApp()
    app.run()
