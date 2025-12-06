import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
from models.edgefinder_predictor import EdgeFinderPredictor, EnhancedTeamStats, BetType


class EdgeFinderFootballApp:
    def __init__(self):
        self.predictor = EdgeFinderPredictor(bankroll=1000.0, min_edge=0.03)
        self.leagues = self.load_leagues()
        self.market_templates = self.load_market_templates()
        
    def load_leagues(self) -> Dict:
        """Load available leagues from CSV files"""
        leagues = {}
        leagues_dir = "leagues"
        
        if os.path.exists(leagues_dir):
            for filename in os.listdir(leagues_dir):
                if filename.endswith('.csv'):
                    league_name = filename.replace('.csv', '')
                    filepath = os.path.join(leagues_dir, filename)
                    
                    try:
                        df = pd.read_csv(filepath)
                        
                        # Standardize column names (handle different formats)
                        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
                        
                        leagues[league_name] = {
                            'teams': df,
                            'name': league_name.replace('_', ' ').title(),
                            'team_names': df['team_name'].tolist()
                        }
                        
                    except Exception as e:
                        st.warning(f"Could not load {filename}: {e}")
        
        # If no CSV files found, create a sample league
        if not leagues:
            leagues = self.create_sample_league()
            
        return leagues
    
    def create_sample_league(self) -> Dict:
        """Create a sample league for demonstration"""
        # Sample data (based on our enhanced CSV format)
        sample_data = {
            'team_name': ['Manchester United', 'West Ham', 'Arsenal', 'Manchester City'],
            'matches_played': [14, 14, 14, 14],
            'possession_avg': [53.0, 43.0, 59.0, 57.0],
            'shots_per_game': [15.29, 9.86, 14.43, 14.14],
            'shots_on_target_pg': [6.36, 4.21, 6.14, 6.21],
            'conversion_rate': [10.0, 12.0, 13.0, 16.0],
            'xg_for_avg': [1.69, 1.18, 1.71, 1.67],
            'xg_against_avg': [1.29, 1.77, 0.83, 1.14],
            'home_wins': [4, 2, 6, 6],
            'home_draws': [1, 0, 1, 0],
            'home_losses': [2, 5, 0, 1],
            'away_wins': [2, 1, 4, 3],
            'away_draws': [3, 3, 2, 1],
            'away_losses': [2, 3, 1, 3],
            'home_goals_for': [12, 8, 18, 19],
            'home_goals_against': [8, 17, 2, 6],
            'away_goals_for': [10, 8, 9, 13],
            'away_goals_against': [13, 11, 5, 10],
            'clean_sheet_pct': [7.0, 7.0, 57.0, 36.0],
            'clean_sheet_pct_home': [14.0, 0.0, 71.0, 43.0],
            'clean_sheet_pct_away': [0.0, 14.0, 43.0, 29.0],
            'failed_to_score_pct': [21.0, 36.0, 7.0, 14.0],
            'failed_to_score_pct_home': [29.0, 43.0, 0.0, 14.0],
            'failed_to_score_pct_away': [14.0, 29.0, 14.0, 14.0],
            'btts_pct': [71.0, 57.0, 36.0, 50.0],
            'btts_pct_home': [57.0, 57.0, 29.0, 43.0],
            'btts_pct_away': [86.0, 57.0, 43.0, 57.0],
            'over25_pct': [64.0, 64.0, 43.0, 79.0],
            'over25_pct_home': [71.0, 71.0, 71.0, 86.0],
            'over25_pct_away': [57.0, 57.0, 14.0, 71.0],
            'last5_form': ['DDDWL', 'WDDLW', 'WWWDD', 'WLWDW'],
            'last5_wins': [1, 2, 3, 4],
            'last5_draws': [3, 2, 2, 0],
            'last5_losses': [1, 1, 0, 1],
            'last5_goals_for': [7, 9, 11, 15],
            'last5_goals_against': [7, 8, 4, 9]
        }
        
        df = pd.DataFrame(sample_data)
        
        return {
            'premier_league': {
                'teams': df,
                'name': 'Premier League',
                'team_names': df['team_name'].tolist()
            }
        }
    
    def load_market_templates(self) -> Dict:
        """Load pre-defined market templates for different match types"""
        return {
            'balanced': {
                'over_25': 1.85,
                'under_25': 2.00,
                'btts_yes': 1.70,
                'btts_no': 2.10,
                'home_win': 2.10,
                'away_win': 3.50,
                'draw': 3.40,
                'home_draw': 1.40,
                'away_draw': 1.80
            },
            'favorite_home': {
                'over_25': 1.65,
                'under_25': 2.20,
                'btts_yes': 1.80,
                'btts_no': 1.95,
                'home_win': 1.50,
                'away_win': 6.00,
                'draw': 4.00,
                'home_draw': 1.20,
                'away_draw': 2.50
            },
            'favorite_away': {
                'over_25': 1.70,
                'under_25': 2.10,
                'btts_yes': 1.75,
                'btts_no': 2.00,
                'home_win': 4.00,
                'away_win': 1.80,
                'draw': 3.60,
                'home_draw': 1.80,
                'away_draw': 1.30
            },
            'high_scoring': {
                'over_25': 1.50,
                'under_25': 2.63,
                'btts_yes': 1.57,
                'btts_no': 2.38,
                'home_win': 2.40,
                'away_win': 2.90,
                'draw': 3.50,
                'home_draw': 1.50,
                'away_draw': 1.50
            },
            'low_scoring': {
                'over_25': 2.10,
                'under_25': 1.73,
                'btts_yes': 2.25,
                'btts_no': 1.57,
                'home_win': 2.30,
                'away_win': 3.20,
                'draw': 3.10,
                'home_draw': 1.40,
                'away_draw': 1.50
            }
        }
    
    def create_team_stats(self, team_data: pd.Series) -> EnhancedTeamStats:
        """Create EnhancedTeamStats object from CSV data"""
        # Convert NaN values to appropriate defaults
        data_dict = {}
        for field in EnhancedTeamStats.__dataclass_fields__:
            if field in team_data:
                value = team_data[field]
                # Handle NaN values
                if pd.isna(value):
                    # Set defaults based on field type
                    field_type = EnhancedTeamStats.__dataclass_fields__[field].type
                    if field_type == str:
                        data_dict[field] = ""
                    elif field_type == int:
                        data_dict[field] = 0
                    elif field_type == float:
                        data_dict[field] = 0.0
                else:
                    data_dict[field] = value
            else:
                # Field not in CSV, use default
                field_type = EnhancedTeamStats.__dataclass_fields__[field].type
                if field_type == str:
                    data_dict[field] = ""
                elif field_type == int:
                    data_dict[field] = 0
                elif field_type == float:
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
        if edge >= 10:
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
    
    def run(self):
        st.set_page_config(
            page_title="⚽ EdgeFinder Pro: Football Value Betting System",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
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
            .insight-card {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #4CAF50;
            }
            .style-card {
                background-color: #e3f2fd;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #2196F3;
            }
            .defense-card {
                background-color: #fff3e0;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #FF9800;
            }
            .transition-card {
                background-color: #f3e5f5;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #9C27B0;
            }
            .stake-metric {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border: 1px solid #e0e0e0;
            }
            .team-stat-box {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                border-left: 3px solid #4CAF50;
            }
            .stat-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 15px;
                font-size: 0.85rem;
                font-weight: 600;
                margin: 0.25rem;
            }
            .badge-success { background-color: #4CAF50; color: white; }
            .badge-warning { background-color: #FF9800; color: white; }
            .badge-danger { background-color: #F44336; color: white; }
            .badge-info { background-color: #2196F3; color: white; }
            .badge-secondary { background-color: #9E9E9E; color: white; }
            </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5rem;">⚽ EdgeFinder Pro</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">Cross-League Football Value Betting System</p>
            <p style="font-size: 1rem; opacity: 0.8;">"3 Things" Framework: Identity • Defense • Transition</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Match Configuration")
            
            if not self.leagues:
                st.error("No league data found. Please add CSV files to the 'leagues' directory.")
                return
            
            # League selection
            selected_league = st.selectbox(
                "Select League",
                list(self.leagues.keys()),
                format_func=lambda x: self.leagues[x]['name'],
                help="Choose the league for analysis"
            )
            
            league_data = self.leagues[selected_league]
            teams_df = league_data['teams']
            team_names = league_data['team_names']
            
            # Team selection
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox(
                    "🏠 Home Team",
                    team_names,
                    help="Select the home team"
                )
            with col2:
                # Filter out home team from away options
                away_options = [t for t in team_names if t != home_team]
                away_team = st.selectbox(
                    "✈️ Away Team",
                    away_options,
                    help="Select the away team"
                )
            
            # Market odds configuration
            st.subheader("💰 Market Odds Configuration")
            
            # Market template selection
            template = st.selectbox(
                "Market Template",
                ["Custom", "Balanced", "Favorite Home", "Favorite Away", "High Scoring", "Low Scoring"],
                help="Select a pre-defined market template or set custom odds"
            )
            
            if template == "Custom":
                # Custom odds input
                st.markdown("**Custom Odds Input**")
                
                col1, col2 = st.columns(2)
                with col1:
                    over_odds = st.number_input("Over 2.5", value=1.85, min_value=1.01, max_value=10.0, step=0.05)
                    btts_yes = st.number_input("BTTS Yes", value=1.70, min_value=1.01, max_value=10.0, step=0.05)
                    home_win = st.number_input("Home Win", value=2.10, min_value=1.01, max_value=20.0, step=0.05)
                    home_draw = st.number_input("Home or Draw", value=1.40, min_value=1.01, max_value=5.0, step=0.05)
                
                with col2:
                    under_odds = st.number_input("Under 2.5", value=2.00, min_value=1.01, max_value=10.0, step=0.05)
                    btts_no = st.number_input("BTTS No", value=2.10, min_value=1.01, max_value=10.0, step=0.05)
                    away_win = st.number_input("Away Win", value=3.50, min_value=1.01, max_value=20.0, step=0.05)
                    away_draw = st.number_input("Away or Draw", value=1.80, min_value=1.01, max_value=5.0, step=0.05)
                
                draw = st.number_input("Draw", value=3.40, min_value=1.01, max_value=20.0, step=0.05)
            else:
                # Use template odds
                template_odds = self.market_templates[template.lower().replace(' ', '_')]
                over_odds = template_odds['over_25']
                under_odds = template_odds['under_25']
                btts_yes = template_odds['btts_yes']
                btts_no = template_odds['btts_no']
                home_win = template_odds['home_win']
                away_win = template_odds['away_win']
                draw = template_odds['draw']
                home_draw = template_odds['home_draw']
                away_draw = template_odds['away_draw']
            
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
            
            # Bankroll management
            st.subheader("🏦 Bankroll Management")
            
            bankroll = st.number_input(
                "Bankroll ($)",
                value=1000.0,
                min_value=100.0,
                max_value=100000.0,
                step=100.0,
                help="Your total betting bankroll"
            )
            
            min_edge = st.slider(
                "Minimum Edge %",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Minimum edge required to place a bet"
            ) / 100  # Convert to decimal
            
            max_exposure = st.slider(
                "Max Correlation Exposure %",
                min_value=5.0,
                max_value=20.0,
                value=10.0,
                step=1.0,
                help="Maximum total exposure for correlated bets"
            ) / 100
            
            # Advanced options
            with st.expander("⚡ Advanced Options"):
                use_style_analysis = st.checkbox("Use Style Matchup Analysis", value=True)
                use_efficiency_adjustment = st.checkbox("Use Efficiency Adjustment", value=True)
                use_form_momentum = st.checkbox("Use Form Momentum", value=True)
                show_detailed_stats = st.checkbox("Show Detailed Statistics", value=True)
            
            # Action button
            analyze_btn = st.button(
                "🔍 Analyze Match & Find Value",
                type="primary",
                use_container_width=True,
                help="Run complete analysis using the '3 Things' framework"
            )
        
        # Main content area
        if analyze_btn:
            # Get team data
            home_data = teams_df[teams_df['team_name'] == home_team].iloc[0]
            away_data = teams_df[teams_df['team_name'] == away_team].iloc[0]
            
            # Create stats objects
            home_stats = self.create_team_stats(home_data)
            away_stats = self.create_team_stats(away_data)
            
            # Update predictor settings
            self.predictor.min_edge = min_edge
            self.predictor.max_correlation_exposure = max_exposure
            self.predictor.bankroll = bankroll
            
            # Run prediction
            with st.spinner("🧠 Analyzing match with '3 Things' framework..."):
                result = self.predictor.predict_match(
                    home_stats=home_stats,
                    away_stats=away_stats,
                    market_odds=market_odds,
                    league=selected_league,
                    bankroll=bankroll
                )
            
            # Display results
            self.display_results(result, home_stats, away_stats, market_odds)
        else:
            # Welcome screen
            self.display_welcome_screen()
    
    def display_results(self, result: Dict, home_stats: EnhancedTeamStats, 
                       away_stats: EnhancedTeamStats, market_odds: Dict):
        """Display analysis results"""
        
        # Match header
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); 
                  border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0; font-size: 2rem;">{home_stats.team_name} 🆚 {away_stats.team_name}</h2>
            <p style="font-size: 1.2rem; opacity: 0.9;">EdgeFinder Pro Analysis Results</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📊 Expected Goals",
                f"{result['match_analysis']['goal_expectations']['total_goals']:.2f}",
                help="Total expected goals in the match"
            )
        
        with col2:
            value_bets_count = len(result['value_bets'])
            st.metric(
                "💰 Value Bets Found",
                value_bets_count,
                delta="Good" if value_bets_count > 0 else "No Value",
                delta_color="normal" if value_bets_count > 0 else "off"
            )
        
        with col3:
            total_exposure = result['total_exposure_percent'] * 100
            st.metric(
                "📈 Total Exposure",
                f"{total_exposure:.1f}%",
                help="Total bankroll exposure across all bets"
            )
        
        # Value bets section
        if result['value_bets']:
            st.markdown("### 🎯 Value Bets Recommended")
            
            for i, bet in enumerate(result['value_bets'], 1):
                edge_text, edge_color = self.format_edge_percentage(bet['edge_percent'])
                
                with st.container():
                    st.markdown(f"""
                    <div class="value-bet-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h3 style="margin: 0; font-size: 1.5rem;">{bet['bet_type']} @ {bet['market_odds']}</h3>
                                <p style="margin: 5px 0; font-size: 1.1rem;">
                                    <strong>Edge:</strong> {edge_text} &nbsp;•&nbsp;
                                    <strong>Model Probability:</strong> {bet['model_probability']:.1%} &nbsp;•&nbsp;
                                    <strong>Stake:</strong> ${bet['staking']['stake_amount']:.2f} ({bet['staking']['stake_percent']:.1%})
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 2rem;">💰</span>
                            </div>
                        </div>
                        <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                            <p style="margin: 0; font-size: 0.95rem;">{bet['explanation']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Stake details
                    with st.expander(f"📊 Detailed Stake Analysis for {bet['bet_type']}"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Kelly Fraction",
                                f"{bet['staking']['kelly_fraction']:.3f}",
                                help="Raw Kelly fraction before adjustments"
                            )
                        
                        with col2:
                            st.metric(
                                "Expected Value",
                                f"${bet['staking']['expected_value']:.2f}",
                                help="Mathematical expected value of this bet"
                            )
                        
                        with col3:
                            st.metric(
                                "Risk Level",
                                bet['staking']['risk_level'],
                                help="Risk assessment based on stake size"
                            )
                        
                        with col4:
                            st.metric(
                                "Correlation Factor",
                                f"{bet['staking']['correlation_factor']:.2f}",
                                help="Adjustment for correlation with other bets"
                            )
        else:
            st.warning("""
            ⚠️ **No Value Bets Found**
            
            The system didn't find any betting opportunities meeting your criteria:
            - Minimum edge: {:.1f}%
            - Minimum confidence: {:.0f}%
            - Correlation exposure limit: {:.0f}%
            
            Consider:
            1. Adjusting the minimum edge requirement
            2. Checking different markets
            3. Reviewing if this match has unpredictable factors
            """.format(
                self.predictor.min_edge * 100,
                self.predictor.min_confidence_for_stake * 100,
                self.predictor.max_correlation_exposure * 100
            ))
        
        # "3 Things" Analysis
        st.markdown("### 🔍 Complete '3 Things' Analysis")
        
        analysis = result['match_analysis']
        
        # Team Identity Analysis
        st.markdown("#### 1️⃣ Team Identity (What they ARE)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.display_team_identity(home_stats, "Home")
        
        with col2:
            self.display_team_identity(away_stats, "Away")
        
        # Style matchup insights
        if analysis['identity']['insights']:
            st.markdown("""
            <div class="style-card">
                <h4 style="margin: 0 0 10px 0;">🎯 Style Matchup Analysis</h4>
            """, unsafe_allow_html=True)
            
            for insight in analysis['identity']['insights']:
                st.markdown(f"• {insight}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Defense Analysis
        st.markdown("#### 2️⃣ Defense (What they STOP)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.display_defense_analysis(home_stats, "Home")
        
        with col2:
            self.display_defense_analysis(away_stats, "Away")
        
        # Defense insights
        if analysis['defense']['insights']:
            st.markdown("""
            <div class="defense-card">
                <h4 style="margin: 0 0 10px 0;">🛡️ Defensive Pattern Analysis</h4>
            """, unsafe_allow_html=True)
            
            for insight in analysis['defense']['insights']:
                st.markdown(f"• {insight}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Transition Analysis
        st.markdown("#### 3️⃣ Transition (How they CHANGE)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.display_transition_analysis(home_stats, "Home")
        
        with col2:
            self.display_transition_analysis(away_stats, "Away")
        
        # Transition insights
        if analysis['transition']['insights']:
            st.markdown("""
            <div class="transition-card">
                <h4 style="margin: 0 0 10px 0;">📈 Transition Trend Analysis</h4>
            """, unsafe_allow_html=True)
            
            for insight in analysis['transition']['insights']:
                st.markdown(f"• {insight}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Match insights
        if analysis['match_insights']:
            st.markdown("### 💡 Key Match Insights")
            
            for insight in analysis['match_insights']:
                st.markdown(f"""
                <div class="insight-card">
                    <p style="margin: 0;">{insight}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Goal expectations details
        st.markdown("### ⚽ Goal Expectations Breakdown")
        
        goal_exp = analysis['goal_expectations']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🏠 Home Expected Goals",
                f"{goal_exp['lambda_home']:.2f}",
                help="Expected goals for home team"
            )
        
        with col2:
            st.metric(
                "✈️ Away Expected Goals",
                f"{goal_exp['lambda_away']:.2f}",
                help="Expected goals for away team"
            )
        
        with col3:
            st.metric(
                "📊 Total Expected Goals",
                f"{goal_exp['total_goals']:.2f}",
                help="Total expected goals in match"
            )
        
        # Probability breakdown
        st.markdown("#### 📈 Market Probabilities")
        
        prob_cols = st.columns(3)
        
        with prob_cols[0]:
            st.markdown("**Over/Under 2.5**")
            st.metric("P(Over 2.5)", f"{goal_exp['probabilities']['over25']:.1%}")
            st.metric("P(Under 2.5)", f"{goal_exp['probabilities']['under25']:.1%}")
        
        with prob_cols[1]:
            st.markdown("**Both Teams to Score**")
            st.metric("P(BTTS Yes)", f"{goal_exp['probabilities']['btts_yes']:.1%}")
            st.metric("P(BTTS No)", f"{goal_exp['probabilities']['btts_no']:.1%}")
        
        with prob_cols[2]:
            st.markdown("**Match Result**")
            st.metric("P(Home Win)", f"{goal_exp['probabilities']['home_win']:.1%}")
            st.metric("P(Draw)", f"{goal_exp['probabilities']['draw']:.1%}")
            st.metric("P(Away Win)", f"{goal_exp['probabilities']['away_win']:.1%}")
        
        # Bankroll summary
        if result['value_bets']:
            st.markdown("### 🏦 Bankroll Impact Summary")
            
            total_ev = result['total_expected_value']
            growth_pct = result['expected_bankroll_growth']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Expected Value",
                    f"${total_ev:.2f}",
                    help="Sum of expected values for all bets"
                )
            
            with col2:
                st.metric(
                    "Expected Bankroll Growth",
                    f"{growth_pct:.2f}%",
                    help="Expected percentage increase in bankroll"
                )
            
            with col3:
                st.metric(
                    "Total Correlation Exposure",
                    f"{result['total_exposure_percent']*100:.1f}%",
                    help="Total bankroll exposure across correlated bets"
                )
    
    def display_team_identity(self, stats: EnhancedTeamStats, prefix: str):
        """Display team identity analysis"""
        style_icon = self.get_team_style_icon(stats.style.value)
        
        st.markdown(f"""
        <div class="team-stat-box">
            <h4 style="margin: 0 0 10px 0;">{style_icon} {prefix}: {stats.team_name}</h4>
            <p style="margin: 5px 0;"><strong>Style:</strong> {stats.style.value}</p>
            <p style="margin: 5px 0;"><strong>Possession:</strong> {stats.possession_avg:.1f}%</p>
            <p style="margin: 5px 0;"><strong>Shots/Game:</strong> {stats.shots_per_game:.1f}</p>
            <p style="margin: 5px 0;"><strong>Conversion:</strong> {stats.conversion_rate:.1f}%</p>
            <p style="margin: 5px 0;"><strong>Attack Efficiency:</strong> {stats.attack_efficiency:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_defense_analysis(self, stats: EnhancedTeamStats, prefix: str):
        """Display defense analysis"""
        st.markdown(f"""
        <div class="team-stat-box">
            <h4 style="margin: 0 0 10px 0;">🛡️ {prefix} Defense Analysis</h4>
            <p style="margin: 5px 0;">
                <strong>Clean Sheets:</strong> 
                <span class="stat-badge {'badge-success' if stats.clean_sheet_pct > 30 else 'badge-warning' if stats.clean_sheet_pct > 15 else 'badge-danger'}">
                    {stats.clean_sheet_pct:.1f}%
                </span>
            </p>
            <p style="margin: 5px 0;">
                <strong>Failed to Score:</strong> 
                <span class="stat-badge {'badge-danger' if stats.failed_to_score_pct > 30 else 'badge-warning' if stats.failed_to_score_pct > 20 else 'badge-success'}">
                    {stats.failed_to_score_pct:.1f}%
                </span>
            </p>
            <p style="margin: 5px 0;"><strong>Defensive Efficiency:</strong> {stats.defensive_efficiency:.2f}</p>
            <p style="margin: 5px 0;"><strong>Home CS%:</strong> {stats.clean_sheet_pct_home:.1f}%</p>
            <p style="margin: 5px 0;"><strong>Away CS%:</strong> {stats.clean_sheet_pct_away:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_transition_analysis(self, stats: EnhancedTeamStats, prefix: str):
        """Display transition analysis"""
        momentum = self.predictor._calculate_form_momentum(stats)
        momentum_icon = "📈" if momentum == "improving" else "📊" if momentum == "stable" else "📉"
        
        st.markdown(f"""
        <div class="team-stat-box">
            <h4 style="margin: 0 0 10px 0;">📈 {prefix} Transition Analysis</h4>
            <p style="margin: 5px 0;">
                <strong>Form:</strong> {momentum_icon} {stats.last5_form} ({momentum})
            </p>
            <p style="margin: 5px 0;">
                <strong>BTTS Frequency:</strong> 
                <span class="stat-badge {'badge-success' if stats.btts_pct > 65 else 'badge-warning' if stats.btts_pct > 50 else 'badge-info'}">
                    {stats.btts_pct:.1f}%
                </span>
            </p>
            <p style="margin: 5px 0;">
                <strong>Over 2.5 Frequency:</strong> 
                <span class="stat-badge {'badge-success' if stats.over25_pct > 65 else 'badge-warning' if stats.over25_pct > 50 else 'badge-info'}">
                    {stats.over25_pct:.1f}%
                </span>
            </p>
            <p style="margin: 5px 0;"><strong>Last 5 Record:</strong> {stats.last5_wins}-{stats.last5_draws}-{stats.last5_losses}</p>
            <p style="margin: 5px 0;"><strong>Last 5 Goals:</strong> {stats.last5_goals_for}-{stats.last5_goals_against}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_welcome_screen(self):
        """Display welcome screen"""
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## 🎯 Welcome to EdgeFinder Pro
            
            ### **The "3 Things" Football Value Betting System**
            
            Our revolutionary approach identifies market inefficiencies by analyzing what actually matters in football:
            """)
            
            # Three Things Explanation
            st.markdown("""
            <div style="padding: 2rem; background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); 
                      border-radius: 15px; color: white; margin: 1rem 0;">
                <h3 style="margin: 0 0 1rem 0; color: white;">The "3 Things" Framework</h3>
                
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <div style="text-align: center; flex: 1; padding: 1rem;">
                        <h4 style="color: white; margin: 0.5rem 0;">🔵 1. Team Identity</h4>
                        <p style="margin: 0; opacity: 0.9;">What they ARE<br>Possession, Shots, Conversion Rate</p>
                    </div>
                    
                    <div style="text-align: center; flex: 1; padding: 1rem; border-left: 1px solid rgba(255,255,255,0.2); 
                              border-right: 1px solid rgba(255,255,255,0.2);">
                        <h4 style="color: white; margin: 0.5rem 0;">🛡️ 2. Defense</h4>
                        <p style="margin: 0; opacity: 0.9;">What they STOP<br>Clean Sheets, Failed to Score %</p>
                    </div>
                    
                    <div style="text-align: center; flex: 1; padding: 1rem;">
                        <h4 style="color: white; margin: 0.5rem 0;">📈 3. Transition</h4>
                        <p style="margin: 0; opacity: 0.9;">How they CHANGE<br>BTTS %, Over 2.5 %, Recent Form</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 🎓 How It Works
            
            1. **Analyze Team Identity**
               - Playing style (Possession vs Counter)
               - Shot efficiency and conversion rates
               - Attack and defense efficiency metrics
            
            2. **Evaluate Defense Patterns**
               - Clean sheet probabilities
               - Scoring reliability
               - Home vs away defensive strength
            
            3. **Assess Transition Trends**
               - Game-by-game outcome patterns
               - Form momentum (improving/declining)
               - Market vs reality discrepancies
            
            ### 💰 Value Detection Engine
            
            Our system calculates **true probabilities** using:
            - Style-based goal expectation adjustments
            - Efficiency-weighted metrics
            - Poisson distribution modeling
            - League-context aware adjustments
            
            We only bet when: `P_model - P_market > Minimum Edge`
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Quick Start Guide
            
            1. **Select League** from sidebar
            2. **Choose Teams** (Home & Away)
            3. **Set Market Odds** (use templates or custom)
            4. **Configure Bankroll** settings
            5. **Click Analyze** to find value
            
            ### ⚙️ Recommended Settings
            
            **For Beginners:**
            - Min Edge: 5%
            - Max Exposure: 8%
            - Use market templates
            
            **For Experienced:**
            - Min Edge: 3%
            - Max Exposure: 12%
            - Enable all advanced options
            
            ### 📊 System Performance
            
            Based on our analysis:
            - **Correctly identified** Man Utd vs West Ham value
            - **Predicted** Lille 1-0 Marseille outcome
            - **Spotted** Monaco's defensive issues
            - **Consistent edge** in style mismatches
            """)
            
            # Available leagues
            if self.leagues:
                st.markdown("### 🏆 Available Leagues")
                
                for league_key, league_info in self.leagues.items():
                    with st.expander(f"{league_info['name']}"):
                        st.write(f"**Teams:** {len(league_info['team_names'])}")
                        st.write(f"**Sample Teams:**")
                        st.write(", ".join(league_info['team_names'][:5]) + "...")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
            <p style="margin: 0; color: #666;">
                ⚠️ <strong>Disclaimer:</strong> Sports betting involves risk. This tool provides mathematical probabilities, 
                not guarantees. Only bet what you can afford to lose.
            </p>
            <p style="margin: 0.5rem 0 0 0; color: #666;">
                Built with ❤️ using Streamlit • Based on statistical analysis of top 5 European leagues
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = EdgeFinderFootballApp()
    app.run()
