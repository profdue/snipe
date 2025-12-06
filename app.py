import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from models.edgefinder_predictor import EdgeFinderPredictor, EnhancedTeamStats


class EdgeFinderFootballApp:
    def __init__(self):
        # Don't initialize predictor here - do it after page config
        self.predictor = None
        self.leagues = {}
        self.market_templates = self.load_market_templates()
        
    def load_leagues(self) -> Dict:
        """Load available leagues from nested directory structure"""
        leagues = {}
        leagues_dir = "leagues"
        
        if os.path.exists(leagues_dir):
            for league_dir in os.listdir(leagues_dir):
                league_path = os.path.join(leagues_dir, league_dir)
                if os.path.isdir(league_path):
                    # Look for CSV files in the league directory
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
                                    # Try to find team name column
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
                                
                                break  # Found a CSV, move to next league
                                
                            except Exception as e:
                                st.warning(f"Could not load {filepath}: {str(e)}")
        
        # If no leagues found, show error
        if not leagues:
            st.error(f"No league data found in {leagues_dir} directory.")
            st.info(f"Expected structure: {leagues_dir}/league_name/teams.csv")
            st.info(f"Current structure: {os.listdir(leagues_dir) if os.path.exists(leagues_dir) else 'Directory not found'}")
            
        return leagues
    
    def load_market_templates(self) -> Dict:
        """Load pre-defined market templates"""
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
            }
        }
    
    def create_team_stats(self, team_data: pd.Series) -> EnhancedTeamStats:
        """Create EnhancedTeamStats object from CSV data with proper cleaning"""
        data_dict = {}
        
        # Define all expected fields
        expected_fields = {
            # Identity Metrics
            'team_name': str, 'matches_played': int, 'possession_avg': float,
            'shots_per_game': float, 'shots_on_target_pg': float, 'conversion_rate': float,
            'xg_for_avg': float, 'xg_against_avg': float,
            
            # Home/Away Split
            'home_wins': int, 'home_draws': int, 'home_losses': int,
            'away_wins': int, 'away_draws': int, 'away_losses': int,
            'home_goals_for': int, 'home_goals_against': int,
            'away_goals_for': int, 'away_goals_against': int,
            
            # Defense Patterns
            'clean_sheet_pct': float, 'clean_sheet_pct_home': float, 'clean_sheet_pct_away': float,
            'failed_to_score_pct': float, 'failed_to_score_pct_home': float, 'failed_to_score_pct_away': float,
            
            # Transition Patterns
            'btts_pct': float, 'btts_pct_home': float, 'btts_pct_away': float,
            'over25_pct': float, 'over25_pct_home': float, 'over25_pct_away': float,
            
            # Recent Form
            'last5_form': str, 'last5_wins': int, 'last5_draws': int, 'last5_losses': int,
            'last5_goals_for': int, 'last5_goals_against': int
        }
        
        for field, field_type in expected_fields.items():
            if field in team_data:
                value = team_data[field]
                
                # Special handling for percentage fields
                if field in ['possession_avg', 'conversion_rate', 'clean_sheet_pct', 
                            'clean_sheet_pct_home', 'clean_sheet_pct_away',
                            'failed_to_score_pct', 'failed_to_score_pct_home', 'failed_to_score_pct_away',
                            'btts_pct', 'btts_pct_home', 'btts_pct_away',
                            'over25_pct', 'over25_pct_home', 'over25_pct_away']:
                    
                    if isinstance(value, str):
                        # Remove % and convert
                        try:
                            cleaned = value.replace('%', '').strip()
                            if cleaned:
                                data_dict[field] = float(cleaned)
                            else:
                                data_dict[field] = 0.0
                        except:
                            data_dict[field] = 0.0
                    elif pd.isna(value):
                        data_dict[field] = 0.0
                    else:
                        data_dict[field] = float(value)
                        
                # Handle other fields
                elif pd.isna(value):
                    if field_type == str:
                        data_dict[field] = ""
                    elif field_type == int:
                        data_dict[field] = 0
                    else:
                        data_dict[field] = 0.0
                else:
                    # Convert to correct type
                    try:
                        if field_type == str:
                            data_dict[field] = str(value)
                        elif field_type == int:
                            # Handle floats that should be ints
                            if isinstance(value, float):
                                data_dict[field] = int(value)
                            else:
                                data_dict[field] = int(float(value))
                        else:  # float
                            data_dict[field] = float(value)
                    except:
                        # Default on error
                        if field_type == str:
                            data_dict[field] = ""
                        elif field_type == int:
                            data_dict[field] = 0
                        else:
                            data_dict[field] = 0.0
            else:
                # Field not in CSV
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
        # Set page config FIRST
        st.set_page_config(
            page_title="⚽ EdgeFinder Pro: Football Value Betting System",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Now initialize predictor
        self.predictor = EdgeFinderPredictor(bankroll=1000.0, min_edge=0.03)
        
        # Load leagues
        self.leagues = self.load_leagues()
        
        # If no leagues loaded, show error and return
        if not self.leagues:
            st.error("""
            ## ⚠️ No League Data Found
            
            Please ensure your CSV files are in the correct structure:
            
            ```
            leagues/
            ├── premier_league/
            │   └── teams.csv
            ├── bundesliga/
            │   └── teams.csv
            └── la_liga/
                └── teams.csv
            ```
            
            **Current directory structure:**
            """)
            
            if os.path.exists("leagues"):
                for root, dirs, files in os.walk("leagues"):
                    level = root.replace("leagues", "").count(os.sep)
                    indent = " " * 2 * level
                    st.code(f"{indent}{os.path.basename(root)}/")
                    subindent = " " * 2 * (level + 1)
                    for file in files:
                        st.code(f"{subindent}{file}")
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
            
            # League selection
            selected_league = st.selectbox(
                "Select League",
                list(self.leagues.keys()),
                format_func=lambda x: self.leagues[x]['name']
            )
            
            league_data = self.leagues[selected_league]
            teams_df = league_data['teams']
            team_names = league_data['team_names']
            
            # Show some team stats
            st.write(f"**{len(team_names)} teams loaded**")
            
            # Team selection
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("🏠 Home Team", team_names)
            with col2:
                away_options = [t for t in team_names if t != home_team]
                away_team = st.selectbox("✈️ Away Team", away_options)
            
            # Market odds
            st.subheader("💰 Market Odds")
            
            template = st.selectbox(
                "Market Template",
                ["Custom", "Balanced", "Favorite Home", "Favorite Away"]
            )
            
            if template == "Custom":
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
            
            # Bankroll settings
            st.subheader("🏦 Bankroll Management")
            
            bankroll = st.number_input("Bankroll ($)", value=1000.0, min_value=100.0, max_value=100000.0, step=100.0)
            min_edge = st.slider("Minimum Edge %", 1.0, 10.0, 3.0, 0.5) / 100
            max_exposure = st.slider("Max Correlation Exposure %", 5.0, 20.0, 10.0, 1.0) / 100
            
            # Update predictor settings
            self.predictor.bankroll = bankroll
            self.predictor.min_edge = min_edge
            self.predictor.max_correlation_exposure = max_exposure
            
            # Analyze button
            analyze_btn = st.button("🔍 Analyze Match", type="primary", use_container_width=True)
        
        # Main content
        if analyze_btn:
            # Get team data
            try:
                home_data = teams_df[teams_df['team_name'] == home_team].iloc[0]
                away_data = teams_df[teams_df['team_name'] == away_team].iloc[0]
                
                # Create stats objects
                home_stats = self.create_team_stats(home_data)
                away_stats = self.create_team_stats(away_data)
                
                # Display team info
                st.info(f"**Analyzing:** {home_team} vs {away_team}")
                
                # Run prediction
                with st.spinner("Analyzing match using '3 Things' framework..."):
                    result = self.predictor.predict_match(
                        home_stats=home_stats,
                        away_stats=away_stats,
                        market_odds=market_odds,
                        league=selected_league,
                        bankroll=bankroll
                    )
                
                # Display results
                self.display_results(result, home_stats, away_stats)
                
            except Exception as e:
                st.error(f"Error analyzing match: {str(e)}")
                st.code(f"Traceback: {e}", language="python")
        else:
            self.display_welcome_screen()
    
    def display_results(self, result, home_stats, away_stats):
        """Display analysis results"""
        
        # Match header
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); 
                  border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0; font-size: 2rem;">{home_stats.team_name} 🆚 {away_stats.team_name}</h2>
            <p style="font-size: 1.2rem; opacity: 0.9;">EdgeFinder Pro Analysis Results</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            expected_goals = result['match_analysis']['goal_expectations']['total_goals']
            st.metric("Expected Goals", f"{expected_goals:.2f}")
        with col2:
            value_bets = len(result['value_bets'])
            st.metric("Value Bets Found", value_bets)
        with col3:
            st.metric("Total Exposure", f"{result['total_exposure_percent']*100:.1f}%")
        
        # Value bets
        if result['value_bets']:
            st.markdown("### 🎯 Value Bets Recommended")
            
            for bet in result['value_bets']:
                edge_text, _ = self.format_edge_percentage(bet['edge_percent'])
                
                st.markdown(f"""
                <div class="value-bet-card">
                    <h3 style="margin: 0; font-size: 1.5rem;">{bet['bet_type']} @ {bet['market_odds']}</h3>
                    <p style="margin: 5px 0; font-size: 1.1rem;">
                        <strong>Edge:</strong> {edge_text} &nbsp;•&nbsp;
                        <strong>Probability:</strong> {bet['model_probability']:.1%} &nbsp;•&nbsp;
                        <strong>Stake:</strong> ${bet['staking']['stake_amount']:.2f}
                    </p>
                    <p style="margin: 10px 0 0 0; font-size: 0.95rem; opacity: 0.9;">{bet['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show stake details
                with st.expander(f"📊 Detailed stake analysis for {bet['bet_type']}"):
                    st.write(f"**Kelly Fraction:** {bet['staking']['kelly_fraction']:.3f}")
                    st.write(f"**Expected Value:** ${bet['staking']['expected_value']:.2f}")
                    st.write(f"**Risk Level:** {bet['staking']['risk_level']}")
                    st.write(f"**Value Rating:** {bet['staking']['value_rating']}")
        else:
            st.warning(f"""
            ⚠️ **No value bets found**
            
            The model didn't find any betting opportunities meeting the criteria:
            - Minimum edge: {self.predictor.min_edge*100:.1f}%
            - Minimum confidence: {self.predictor.min_confidence_for_stake*100:.0f}%
            
            **Possible reasons:**
            1. Market odds accurately reflect true probabilities
            2. Match is too unpredictable
            3. Try adjusting the minimum edge requirement
            """)
        
        # Team analysis
        st.markdown("### 🔍 Team Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### 🏠 {home_stats.team_name}")
            st.markdown(f"""
            <div class="team-stat-box">
                <p><strong>Style:</strong> {self.get_team_style_icon(home_stats.style.value)} {home_stats.style.value}</p>
                <p><strong>Possession:</strong> {home_stats.possession_avg:.1f}%</p>
                <p><strong>Conversion:</strong> {home_stats.conversion_rate:.1f}%</p>
                <p><strong>Clean Sheets:</strong> {home_stats.clean_sheet_pct:.1f}%</p>
                <p><strong>Form:</strong> {home_stats.last5_form}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"#### ✈️ {away_stats.team_name}")
            st.markdown(f"""
            <div class="team-stat-box">
                <p><strong>Style:</strong> {self.get_team_style_icon(away_stats.style.value)} {away_stats.style.value}</p>
                <p><strong>Possession:</strong> {away_stats.possession_avg:.1f}%</p>
                <p><strong>Conversion:</strong> {away_stats.conversion_rate:.1f}%</p>
                <p><strong>Clean Sheets:</strong> {away_stats.clean_sheet_pct:.1f}%</p>
                <p><strong>Form:</strong> {away_stats.last5_form}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Match insights
        analysis = result['match_analysis']
        
        # Show identity analysis
        st.markdown("### 🎯 Identity Analysis")
        if analysis['identity']['insights']:
            for insight in analysis['identity']['insights']:
                st.info(f"• {insight}")
        else:
            st.write("No significant style mismatches detected.")
        
        # Show defense analysis
        st.markdown("### 🛡️ Defense Analysis")
        if analysis['defense']['insights']:
            for insight in analysis['defense']['insights']:
                st.warning(f"• {insight}")
        else:
            st.write("No significant defensive patterns detected.")
        
        # Show transition analysis
        st.markdown("### 📈 Transition Analysis")
        if analysis['transition']['insights']:
            for insight in analysis['transition']['insights']:
                st.success(f"• {insight}")
        else:
            st.write("No significant transition trends detected.")
        
        # Show match insights
        if analysis['match_insights']:
            st.markdown("### 💡 Key Match Insights")
            for insight in analysis['match_insights']:
                st.info(f"• {insight}")
        
        # Goal expectations
        goal_exp = analysis['goal_expectations']
        st.markdown("### ⚽ Goal Expectations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Home Expected Goals", f"{goal_exp['lambda_home']:.2f}")
        with col2:
            st.metric("Away Expected Goals", f"{goal_exp['lambda_away']:.2f}")
        
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
    
    def display_welcome_screen(self):
        """Display welcome screen"""
        st.markdown("""
        ## 🎯 Welcome to EdgeFinder Pro
        
        ### **The "3 Things" Football Value Betting System**
        
        Our system analyzes football matches using three key dimensions:
        
        """)
        
        # Three columns for the three things
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔵 1. Team Identity
            **What they ARE**
            - Possession style
            - Shot efficiency
            - Conversion rates
            - Playing style classification
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ 2. Defense
            **What they STOP**
            - Clean sheet probabilities
            - Scoring reliability
            - Defensive efficiency
            - Home/away defensive strength
            """)
        
        with col3:
            st.markdown("""
            ### 📈 3. Transition
            **How they CHANGE**
            - BTTS patterns
            - Over/Under trends
            - Form momentum
            - Game outcome patterns
            """)
        
        st.markdown("""
        ---
        
        ### 🚀 Getting Started
        
        1. **Select a league** from the sidebar
        2. **Choose home and away teams**
        3. **Set market odds** (use templates or custom)
        4. **Configure your bankroll** settings
        5. **Click "Analyze Match"** to find value bets
        
        ### 💰 How We Find Value
        
        Our system calculates **true probabilities** using:
        - Style-based goal expectation adjustments
        - Efficiency-weighted metrics  
        - Poisson distribution modeling
        - League-context aware adjustments
        
        We only recommend bets where: **P_model - P_market > Minimum Edge**
        
        ---
        
        ### 🏆 Available Leagues
        """)
        
        # Show available leagues
        for league_key, league_info in self.leagues.items():
            with st.expander(f"{league_info['name']} ({len(league_info['team_names'])} teams)"):
                st.write(f"**Teams available:**")
                # Show teams in columns
                teams = league_info['team_names']
                cols = st.columns(3)
                for i, team in enumerate(teams[:12]):  # Show first 12 teams
                    cols[i % 3].write(f"• {team}")
                if len(teams) > 12:
                    st.write(f"... and {len(teams) - 12} more")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>⚠️ <strong>Disclaimer:</strong> Sports betting involves risk. This tool provides mathematical probabilities only.</p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = EdgeFinderFootballApp()
    app.run()
