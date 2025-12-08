import streamlit as st
import pandas as pd
import os
from models.edgefinder_predictor import EdgeFinderPredictor, EnhancedTeamStats

class FootballPredictorApp:
    def __init__(self):
        self.predictor = None
        self.leagues = {}
        self.load_leagues()
        
    def load_leagues(self):
        """Load league data from CSV files"""
        leagues_dir = "leagues"
        
        if os.path.exists(leagues_dir):
            for league_dir in os.listdir(leagues_dir):
                league_path = os.path.join(leagues_dir, league_dir)
                if os.path.isdir(league_path):
                    for filename in os.listdir(league_path):
                        if filename.lower() == 'teams.csv':
                            filepath = os.path.join(league_path, filename)
                            try:
                                df = pd.read_csv(filepath)
                                
                                # Clean column names
                                df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
                                
                                if 'team_name' in df.columns:
                                    self.leagues[league_dir] = {
                                        'name': league_dir.replace('_', ' ').title(),
                                        'teams': df,
                                        'team_names': df['team_name'].tolist()
                                    }
                            except Exception as e:
                                st.warning(f"Could not load {filepath}: {e}")
        
    def create_team_stats(self, team_data: pd.Series) -> EnhancedTeamStats:
        """Create EnhancedTeamStats from DataFrame row"""
        data = {}
        
        # Map all required fields
        field_mapping = {
            'team_name': ('team_name', str),
            'matches_played': ('matches_played', int),
            'possession_avg': ('possession_avg', float),
            'shots_per_game': ('shots_per_game', float),
            'shots_on_target_pg': ('shots_on_target_pg', float),
            'conversion_rate': ('conversion_rate', float),
            'xg_for_avg': ('xg_for_avg', float),
            'xg_against_avg': ('xg_against_avg', float),
            
            # Home/Away records
            'home_wins': ('home_wins', int),
            'home_draws': ('home_draws', int),
            'home_losses': ('home_losses', int),
            'away_wins': ('away_wins', int),
            'away_draws': ('away_draws', int),
            'away_losses': ('away_losses', int),
            
            # Goals
            'home_goals_for': ('home_goals_for', int),
            'home_goals_against': ('home_goals_against', int),
            'away_goals_for': ('away_goals_for', int),
            'away_goals_against': ('away_goals_against', int),
            
            # Defense patterns
            'clean_sheet_pct': ('clean_sheet_pct', float),
            'clean_sheet_pct_home': ('clean_sheet_pct_home', float),
            'clean_sheet_pct_away': ('clean_sheet_pct_away', float),
            'failed_to_score_pct': ('failed_to_score_pct', float),
            'failed_to_score_pct_home': ('failed_to_score_pct_home', float),
            'failed_to_score_pct_away': ('failed_to_score_pct_away', float),
            
            # Transition patterns
            'btts_pct': ('btts_pct', float),
            'btts_pct_home': ('btts_pct_home', float),
            'btts_pct_away': ('btts_pct_away', float),
            'over25_pct': ('over25_pct', float),
            'over25_pct_home': ('over25_pct_home', float),
            'over25_pct_away': ('over25_pct_away', float),
            
            # Recent form
            'last5_form': ('last5_form', str),
            'last5_wins': ('last5_wins', int),
            'last5_draws': ('last5_draws', int),
            'last5_losses': ('last5_losses', int),
            'last5_goals_for': ('last5_goals_for', int),
            'last5_goals_against': ('last5_goals_against', int),
            'last5_ppg': ('last5_ppg', float),
            'last5_cs_pct': ('last5_cs_pct', float),
            'last5_fts_pct': ('last5_fts_pct', float),
            'last5_btts_pct': ('last5_btts_pct', float),
            'last5_over25_pct': ('last5_over25_pct', float)
        }
        
        for model_field, (data_field, field_type) in field_mapping.items():
            if data_field in team_data:
                value = team_data[data_field]
                if pd.isna(value):
                    if field_type == str:
                        data[model_field] = ""
                    elif field_type == int:
                        data[model_field] = 0
                    else:
                        data[model_field] = 0.0
                else:
                    try:
                        if field_type == str:
                            data[model_field] = str(value)
                        elif field_type == int:
                            data[model_field] = int(float(value))
                        else:
                            data[model_field] = float(value)
                    except:
                        if field_type == str:
                            data[model_field] = ""
                        elif field_type == int:
                            data[model_field] = 0
                        else:
                            data[model_field] = 0.0
            else:
                # Default values
                if field_type == str:
                    data[model_field] = ""
                elif field_type == int:
                    data[model_field] = 0
                else:
                    data[model_field] = 0.0
        
        return EnhancedTeamStats(**data)
    
    def run(self):
        st.set_page_config(
            page_title="Football Value Bet Finder",
            page_icon="⚽",
            layout="wide"
        )
        
        st.title("⚽ Football Value Bet Finder")
        
        if not self.leagues:
            st.error("No league data found. Please check the 'leagues' directory.")
            return
        
        # Sidebar
        with st.sidebar:
            st.header("Match Configuration")
            
            # League selection
            selected_league = st.selectbox(
                "Select League",
                list(self.leagues.keys()),
                format_func=lambda x: self.leagues[x]['name']
            )
            
            league_data = self.leagues[selected_league]
            teams_df = league_data['teams']
            team_names = league_data['team_names']
            
            # Team selection
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("Home Team", team_names)
            with col2:
                away_options = [t for t in team_names if t != home_team]
                away_team = st.selectbox("Away Team", away_options)
            
            # Market odds
            st.header("Market Odds")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                home_win = st.number_input("Home Win", value=2.50, min_value=1.01, step=0.05)
                draw = st.number_input("Draw", value=3.40, min_value=1.01, step=0.05)
                away_win = st.number_input("Away Win", value=2.80, min_value=1.01, step=0.05)
            
            with col2:
                over_25 = st.number_input("Over 2.5", value=1.85, min_value=1.01, step=0.05)
                under_25 = st.number_input("Under 2.5", value=2.00, min_value=1.01, step=0.05)
            
            with col3:
                btts_yes = st.number_input("BTTS Yes", value=1.75, min_value=1.01, step=0.05)
                btts_no = st.number_input("BTTS No", value=2.05, min_value=1.01, step=0.05)
            
            market_odds = {
                'over_25': over_25,
                'under_25': under_25,
                'btts_yes': btts_yes,
                'btts_no': btts_no,
                'home_win': home_win,
                'away_win': away_win,
                'draw': draw,
                'home_draw': min(1.0/home_win + 1.0/draw, 1.0) if home_win > 0 and draw > 0 else 1.5,
                'away_draw': min(1.0/away_win + 1.0/draw, 1.0) if away_win > 0 and draw > 0 else 1.55
            }
            
            # Predictor settings
            st.header("Settings")
            bankroll = st.number_input("Bankroll ($)", value=1000.0, min_value=100.0)
            min_edge = st.slider("Minimum Edge %", 1.0, 10.0, 3.0) / 100
            
            analyze_btn = st.button("Analyze Match", type="primary", use_container_width=True)
        
        # Main content
        if analyze_btn:
            # Get team data
            home_data = teams_df[teams_df['team_name'] == home_team].iloc[0]
            away_data = teams_df[teams_df['team_name'] == away_team].iloc[0]
            
            # Create stats objects
            home_stats = self.create_team_stats(home_data)
            away_stats = self.create_team_stats(away_data)
            
            # Initialize predictor
            self.predictor = EdgeFinderPredictor(
                bankroll=bankroll,
                min_edge=min_edge
            )
            
            # Run prediction
            with st.spinner("Analyzing match..."):
                result = self.predictor.predict(
                    home_stats=home_stats,
                    away_stats=away_stats,
                    market_odds=market_odds,
                    league=selected_league
                )
            
            # Display results
            self.display_results(result, home_stats, away_stats)
    
    def display_results(self, result, home_stats, away_stats):
        """Display prediction results"""
        st.header(f"{home_stats.team_name} vs {away_stats.team_name}")
        
        # Goal expectations
        goal_exp = result['goal_expectations']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Home Expected Goals", f"{goal_exp['lambda_home']:.2f}")
        with col2:
            st.metric("Away Expected Goals", f"{goal_exp['lambda_away']:.2f}")
        with col3:
            st.metric("Total Expected Goals", f"{goal_exp['total_goals']:.2f}")
        
        # Probabilities
        st.subheader("Probabilities")
        probs = goal_exp['probabilities']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Match Result**")
            st.write(f"Home: {probs['home_win']:.1%}")
            st.write(f"Draw: {probs['draw']:.1%}")
            st.write(f"Away: {probs['away_win']:.1%}")
        
        with col2:
            st.write("**Goals Markets**")
            st.write(f"Over 2.5: {probs['over25']:.1%}")
            st.write(f"Under 2.5: {probs['under25']:.1%}")
        
        with col3:
            st.write("**Both Teams to Score**")
            st.write(f"BTTS Yes: {probs['btts_yes']:.1%}")
            st.write(f"BTTS No: {probs['btts_no']:.1%}")
        
        # Value bets
        st.subheader("🎯 Value Bets")
        
        if result['value_bets']:
            for bet in result['value_bets']:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{bet['bet_type']}**")
                        st.write(f"Odds: {bet['market_odds']}")
                    
                    with col2:
                        st.write(f"Model: {bet['model_probability']:.1%}")
                        st.write(f"Market: {bet['implied_probability']:.1%}")
                    
                    with col3:
                        edge_color = "green" if bet['edge_percent'] > 0 else "red"
                        st.markdown(f"**Edge: <span style='color:{edge_color}'>{bet['edge_percent']:+.1f}%</span>**", 
                                  unsafe_allow_html=True)
                        st.write(f"Rating: {bet['value_rating']}")
                    
                    with col4:
                        st.write(f"**Stake:**")
                        st.write(f"${bet['stake_amount']:.2f}")
                        st.write(f"({bet['stake_percent']:.1f}%)")
                    
                    st.divider()
            
            st.info(f"**Total Stake:** ${result['total_stake']:.2f} ({result['total_exposure']:.1f}% of bankroll)")
        else:
            st.warning("No value bets found meeting the minimum edge requirement.")
        
        # Confidence
        st.subheader("Confidence Score")
        score = result['confidence_score']
        if score >= 8:
            confidence_text = "⭐⭐⭐ High Confidence"
            color = "green"
        elif score >= 6:
            confidence_text = "⭐⭐ Medium Confidence"
            color = "orange"
        else:
            confidence_text = "⭐ Low Confidence"
            color = "red"
        
        st.markdown(f"**Score:** {score}/10 - <span style='color:{color}'>{confidence_text}</span>", 
                   unsafe_allow_html=True)
        
        # Team styles
        st.subheader("Team Styles")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{home_stats.team_name}**: {home_stats.style.value}")
        with col2:
            st.write(f"**{away_stats.team_name}**: {away_stats.style.value}")

if __name__ == "__main__":
    app = FootballPredictorApp()
    app.run()
