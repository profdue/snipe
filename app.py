import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from models.edgefinder_predictor import EdgeFinderPredictor, EnhancedTeamStats, ConfidenceLevel


class EdgeFinderFootballApp:
    def __init__(self):
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
            'last5_goals_for': int, 'last5_goals_against': int
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
        
        # Now initialize predictor
        self.predictor = EdgeFinderPredictor(bankroll=1000.0, min_edge=0.03)
        
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
            
            # Logic framework display
            with st.expander("📚 Logic Framework"):
                st.markdown("""
                ### THE "3 THINGS" ANALYTICAL FRAMEWORK
                
                1. **TEAM IDENTITY** (What they ARE)
                - Possession style and efficiency
                - Conversion rates matter more than volume
                
                2. **DEFENSE** (What they STOP)
                - Clean sheet frequency reveals stability
                - Failed-to-score percentage shows reliability
                
                3. **TRANSITION** (How they CHANGE)
                - BTTS and Over/Under patterns
                - Recent form weighted 40% vs season 60%
                
                **Value Detection**: Edge = P_model - P_market
                - ⭐⭐⭐ Golden Nugget: Edge > 5%
                - ⭐⭐ Value Bet: Edge > 3%
                - ⭐ Consider: Edge > 1%
                """)
            
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
                st.exception(e)
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
        
        # Quick stats with confidence
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            expected_goals = result['match_analysis']['goal_expectations']['total_goals']
            st.metric("Expected Goals", f"{expected_goals:.2f}")
        with col2:
            value_bets = len(result['value_bets'])
            st.metric("Value Bets Found", value_bets)
        with col3:
            st.metric("Total Exposure", f"{result['total_exposure_percent']*100:.1f}%")
        with col4:
            confidence = result['match_analysis']['confidence']
            confidence_score = confidence['score']
            confidence_color = self.get_confidence_color(confidence_score)
            st.markdown(f"""
            <div style="background-color: {confidence_color}; color: white; padding: 0.5rem; 
                      border-radius: 10px; text-align: center;">
                <div style="font-size: 1.2rem; font-weight: bold;">Confidence</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{confidence_score}/10</div>
                <div style="font-size: 0.8rem;">{confidence['level'].value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Value bets
        if result['value_bets']:
            st.markdown("### 🎯 Value Bets Recommended")
            
            for bet in result['value_bets']:
                edge_text, _ = self.format_edge_percentage(bet['edge_percent'])
                value_rating = bet.get('value_rating', '⭐ Consider')
                
                # Determine card color based on value rating
                if "⭐⭐⭐" in value_rating:
                    card_color = "linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%)"
                elif "⭐⭐" in value_rating:
                    card_color = "linear-gradient(135deg, #FF9800 0%, #F57C00 100%)"
                else:
                    card_color = "linear-gradient(135deg, #2196F3 0%, #1976D2 100%)"
                
                st.markdown(f"""
                <div style="background: {card_color}; border-radius: 10px; padding: 1.5rem; 
                          color: white; margin: 1rem 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="margin: 0; font-size: 1.5rem;">{bet['bet_type']} @ {bet['market_odds']}</h3>
                    <p style="margin: 5px 0; font-size: 1.1rem;">
                        <strong>Value Rating:</strong> {value_rating} &nbsp;•&nbsp;
                        <strong>Edge:</strong> {edge_text} &nbsp;•&nbsp;
                        <strong>Probability:</strong> {bet['model_probability']:.1%} &nbsp;•&nbsp;
                        <strong>Stake:</strong> ${bet['staking']['stake_amount']:.2f}
                    </p>
                    <p style="margin: 10px 0 0 0; font-size: 0.95rem; opacity: 0.9;">{bet['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show stake details
                with st.expander(f"📊 Detailed stake analysis for {bet['bet_type']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Stake Calculation:**")
                        st.write(f"Kelly Fraction: {bet['staking']['kelly_fraction']:.3f}")
                        st.write(f"Correlation Factor: {bet['staking']['correlation_factor']:.2f}")
                        st.write(f"Confidence Multiplier: {bet['staking']['confidence_multiplier']:.2f}")
                        st.write(f"Max Stake Limit: ${bet['staking']['max_stake_limit']:.2f}")
                    with col2:
                        st.write("**Expected Performance:**")
                        st.write(f"Expected Value: ${bet['staking']['expected_value']:.2f}")
                        st.write(f"Risk Level: {bet['staking']['risk_level']}")
                        st.write(f"Model Probability: {bet['model_probability']:.1%}")
                        st.write(f"Market Implied: {bet['staking']['implied_probability']:.1%}")
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
                <p><strong>Possession:</strong> {home_stats.possession_avg:.1%}</p>
                <p><strong>Conversion:</strong> {home_stats.conversion_rate:.1%}</p>
                <p><strong>Clean Sheets:</strong> {home_stats.clean_sheet_pct:.1%}</p>
                <p><strong>Failed to Score:</strong> {home_stats.failed_to_score_pct:.1%}</p>
                <p><strong>Form:</strong> {home_stats.last5_form} ({home_stats.last5_wins}W-{home_stats.last5_draws}D-{home_stats.last5_losses}L)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"#### ✈️ {away_stats.team_name}")
            st.markdown(f"""
            <div class="team-stat-box">
                <p><strong>Style:</strong> {self.get_team_style_icon(away_stats.style.value)} {away_stats.style.value}</p>
                <p><strong>Possession:</strong> {away_stats.possession_avg:.1%}</p>
                <p><strong>Conversion:</strong> {away_stats.conversion_rate:.1%}</p>
                <p><strong>Clean Sheets:</strong> {away_stats.clean_sheet_pct:.1%}</p>
                <p><strong>Failed to Score:</strong> {away_stats.failed_to_score_pct:.1%}</p>
                <p><strong>Form:</strong> {away_stats.last5_form} ({away_stats.last5_wins}W-{away_stats.last5_draws}D-{away_stats.last5_losses}L)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Match insights with confidence factors
        analysis = result['match_analysis']
        confidence = analysis['confidence']
        
        st.markdown("### 🎯 Confidence Analysis")
        
        # Show confidence factors
        if confidence['factors']:
            st.write("**Key Confidence Factors:**")
            for factor, weight in confidence['factors']:
                if weight >= 3:
                    emoji = "🔥"
                elif weight >= 2:
                    emoji = "⚡"
                else:
                    emoji = "📊"
                st.write(f"{emoji} {factor.replace('_', ' ').title()} (+{weight})")
        
        st.write(f"**Overall:** {confidence['reason']}")
        
        # Show analysis by dimension
        st.markdown("### 📊 Three Things Analysis")
        
        tabs = st.tabs(["1. Team Identity", "2. Defense", "3. Transition"])
        
        with tabs[0]:
            identity = analysis['identity']
            if identity['insights']:
                for insight in identity['insights']:
                    st.info(f"• {insight}")
            else:
                st.write("No significant identity mismatches detected.")
            
            # Show style matchup
            if identity.get('style_clash'):
                st.write(f"**Style Matchup:** {identity['style_clash']}")
        
        with tabs[1]:
            defense = analysis['defense']
            if defense['insights']:
                for insight in defense['insights']:
                    st.warning(f"• {insight}")
            else:
                st.write("No significant defensive patterns detected.")
            
            # Show key defensive stats
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{home_stats.team_name} Defense:**")
                st.write(f"- Clean Sheets: {defense['home_clean_sheet_strength']:.1f}%")
                st.write(f"- Scoring Reliability: {defense['home_scoring_reliability']:.1f}%")
            with col2:
                st.write(f"**{away_stats.team_name} Defense:**")
                st.write(f"- Clean Sheets: {defense['away_clean_sheet_strength']:.1f}%")
                st.write(f"- Scoring Reliability: {defense['away_scoring_reliability']:.1f}%")
        
        with tabs[2]:
            transition = analysis['transition']
            if transition['insights']:
                for insight in transition['insights']:
                    st.success(f"• {insight}")
            else:
                st.write("No significant transition trends detected.")
            
            # Show form momentum
            st.write(f"**Form Momentum:**")
            st.write(f"- {home_stats.team_name}: {transition['home_form_momentum'].upper()} ({home_stats.last5_form})")
            st.write(f"- {away_stats.team_name}: {transition['away_form_momentum'].upper()} ({away_stats.last5_form})")
            
            # Show pattern probabilities
            st.write(f"**Historical Patterns:**")
            st.write(f"- Combined BTTS: {transition['combined_btts']:.1f}%")
            st.write(f"- Combined Over 2.5: {transition['combined_over25']:.1f}%")
        
        # Show match insights
        if analysis['match_insights']:
            st.markdown("### 💡 Key Match Insights")
            for insight in analysis['match_insights']:
                st.info(f"• {insight}")
        
        # Goal expectations
        goal_exp = analysis['goal_expectations']
        st.markdown("### ⚽ Goal Expectations")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Home Expected Goals", f"{goal_exp['lambda_home']:.2f}")
            st.caption(f"Season xG: {home_stats.xg_for_avg:.2f}")
        with col2:
            st.metric("Away Expected Goals", f"{goal_exp['lambda_away']:.2f}")
            st.caption(f"Season xG: {away_stats.xg_for_avg:.2f}")
        with col3:
            st.metric("Total Expected Goals", f"{goal_exp['total_goals']:.2f}")
            st.caption(f"League Average: {self.predictor.league_context['premier_league']['avg_gpg']}")
        
        # Show adjustment factors
        with st.expander("🔧 Goal Expectation Adjustments"):
            adj_factors = goal_exp['adjustment_factors']
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{home_stats.team_name} Adjustments:**")
                st.write(f"- Efficiency: {adj_factors['home_efficiency']:.2f}x")
                st.write(f"- Venue Boost: {adj_factors['home_venue_boost']:.2f}x")
                st.write(f"- Form Adjustment: {adj_factors['home_form_adj']:.2f}x")
            with col2:
                st.write(f"**{away_stats.team_name} Adjustments:**")
                st.write(f"- Efficiency: {adj_factors['away_efficiency']:.2f}x")
                st.write(f"- Venue Penalty: {adj_factors['away_venue_penalty']:.2f}x")
                st.write(f"- Form Adjustment: {adj_factors['away_form_adj']:.2f}x")
        
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
            - Form momentum (40% weight)
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
        
        **Prediction Engine Logic:**
        ```
        Expected_Goals = Base_Avg × Style_Adjustment × Efficiency_Adjustment
        Edge = P_model - P_market
        ```
        
        **Value Detection Thresholds:**
        - ⭐⭐⭐ **Golden Nugget**: Edge > 5% with high confidence
        - ⭐⭐ **Value Bet**: Edge > 3% with moderate confidence  
        - ⭐ **Consider**: Edge > 1% or situational value
        
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
            <p>⚠️ <strong>Disclaimer:</strong> Sports betting involves risk. This tool provides mathematical probabilities only.</p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = EdgeFinderFootballApp()
    app.run()
