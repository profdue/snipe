import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="SNIPE - Football Predictor",
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("⚽ SNIPE: Football Over/Under Predictor")
st.markdown("### Simple Version - No File Dependencies")

# HARDCODED TEAM DATA (From your last 5 matches)
team_stats = {
    "Arsenal": {
        "home": {"last5": {"gpg": 2.00, "ga_pg": 0.40}},
        "away": {"last5": {"gpg": 1.60, "ga_pg": 0.80}}
    },
    "Manchester City": {
        "home": {"last5": {"gpg": 3.20, "ga_pg": 0.80}},
        "away": {"last5": {"gpg": 1.60, "ga_pg": 1.60}}
    },
    "Aston Villa": {
        "home": {"last5": {"gpg": 2.20, "ga_pg": 0.40}},
        "away": {"last5": {"gpg": 1.80, "ga_pg": 1.60}}
    },
    "Brighton": {
        "home": {"last5": {"gpg": 2.40, "ga_pg": 1.60}},
        "away": {"last5": {"gpg": 1.60, "ga_pg": 1.20}}
    },
    "Newcastle": {
        "home": {"last5": {"gpg": 1.80, "ga_pg": 1.20}},
        "away": {"last5": {"gpg": 1.40, "ga_pg": 1.80}}
    },
    "Tottenham": {
        "home": {"last5": {"gpg": 1.00, "ga_pg": 1.60}},
        "away": {"last5": {"gpg": 2.00, "ga_pg": 1.80}}
    },
    "Crystal Palace": {
        "home": {"last5": {"gpg": 1.60, "ga_pg": 1.20}},
        "away": {"last5": {"gpg": 1.20, "ga_pg": 0.80}}
    },
    "Brentford": {
        "home": {"last5": {"gpg": 2.40, "ga_pg": 1.20}},
        "away": {"last5": {"gpg": 0.80, "ga_pg": 1.80}}
    },
    "Liverpool": {
        "home": {"last5": {"gpg": 1.20, "ga_pg": 1.40}},
        "away": {"last5": {"gpg": 1.20, "ga_pg": 2.00}}
    },
    "Chelsea": {
        "home": {"last5": {"gpg": 1.60, "ga_pg": 1.40}},
        "away": {"last5": {"gpg": 1.60, "ga_pg": 1.00}}
    }
}

# Prediction function
def predict_match(home_team, away_team, home_xg, away_xg):
    """Simple prediction using last 5 games only"""
    
    if home_team not in team_stats or away_team not in team_stats:
        return {"prediction": "NO DATA", "confidence": "LOW", "reason": "Team data missing"}
    
    # Get stats
    home_gpg = team_stats[home_team]["home"]["last5"]["gpg"]
    home_ga = team_stats[home_team]["home"]["last5"]["ga_pg"]
    away_gpg = team_stats[away_team]["away"]["last5"]["gpg"]
    away_ga = team_stats[away_team]["away"]["last5"]["ga_pg"]
    
    # RULE 1: OVER 2.5 (Last 5 only)
    if home_gpg > 1.5 and away_gpg > 1.5:
        return {
            "prediction": "OVER 2.5",
            "confidence": "MODERATE",
            "reason": f"Both teams high-scoring (Home: {home_gpg} GPG, Away: {away_gpg} GPG)"
        }
    
    # RULE 2: UNDER 2.5 (Last 5 only)
    if home_ga < 1.0 and away_gpg < 1.5:
        return {
            "prediction": "UNDER 2.5",
            "confidence": "MODERATE",
            "reason": f"Home strong defense & away weak attack (Home GA: {home_ga}, Away GPG: {away_gpg})"
        }
    
    if away_ga < 1.0 and home_gpg < 1.5:
        return {
            "prediction": "UNDER 2.5",
            "confidence": "MODERATE",
            "reason": f"Away strong defense & home weak attack (Away GA: {away_ga}, Home GPG: {home_gpg})"
        }
    
    return {
        "prediction": "NO BET",
        "confidence": "LOW",
        "reason": "No clear statistical edge"
    }

# Sidebar
st.sidebar.header("⚙️ Prediction Rules")
st.sidebar.markdown("""
**Using Last 5 Games Only:**
- **Over 2.5**: Both teams >1.5 GPG
- **Under 2.5**: Defense <1.0 GA PG vs Attack <1.5 GPG
- **No Bet**: No clear edge
""")

# Main tabs
tab1, tab2 = st.tabs(["📊 Predict Match", "📈 Team Stats"])

with tab1:
    st.header("Predict a Match")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Home Team", sorted(team_stats.keys()), key="home")
        home_xg = st.number_input("Home xG", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    
    with col2:
        away_team = st.selectbox("Away Team", sorted(team_stats.keys()), key="away")
        away_xg = st.number_input("Away xG", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    if st.button("🔮 Get Prediction", type="primary"):
        if home_team == away_team:
            st.error("Please select different teams!")
        else:
            with st.spinner("Analyzing..."):
                result = predict_match(home_team, away_team, home_xg, away_xg)
                
                # Display result
                st.markdown("---")
                if result["prediction"] == "OVER 2.5":
                    st.success(f"🎯 **Prediction**: {result['prediction']}")
                elif result["prediction"] == "UNDER 2.5":
                    st.warning(f"🎯 **Prediction**: {result['prediction']}")
                else:
                    st.info(f"🎯 **Prediction**: {result['prediction']}")
                
                st.metric("Confidence", result["confidence"])
                st.write(f"**Reason**: {result['reason']}")
                
                # Show stats
                st.markdown("**📊 Team Statistics (Last 5 Games):**")
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.write(f"**{home_team} (Home):**")
                    st.write(f"- GPG: {team_stats[home_team]['home']['last5']['gpg']:.2f}")
                    st.write(f"- GA PG: {team_stats[home_team]['home']['last5']['ga_pg']:.2f}")
                
                with stats_col2:
                    st.write(f"**{away_team} (Away):**")
                    st.write(f"- GPG: {team_stats[away_team]['away']['last5']['gpg']:.2f}")
                    st.write(f"- GA PG: {team_stats[away_team]['away']['last5']['ga_pg']:.2f}")

with tab2:
    st.header("Team Statistics")
    
    selected_team = st.selectbox("Select a team", sorted(team_stats.keys()), key="stats")
    
    if selected_team:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Performance (Last 5)")
            st.metric("Goals Per Game", f"{team_stats[selected_team]['home']['last5']['gpg']:.2f}")
            st.metric("Goals Against Per Game", f"{team_stats[selected_team]['home']['last5']['ga_pg']:.2f}")
            
            # Simple rating
            gpg = team_stats[selected_team]['home']['last5']['gpg']
            if gpg > 2.0:
                st.write("🔥 **Attack Rating**: Excellent")
            elif gpg > 1.5:
                st.write("✅ **Attack Rating**: Good")
            elif gpg > 1.0:
                st.write("⚡ **Attack Rating**: Average")
            else:
                st.write("⚠️ **Attack Rating**: Poor")
        
        with col2:
            st.subheader("✈️ Away Performance (Last 5)")
            st.metric("Goals Per Game", f"{team_stats[selected_team]['away']['last5']['gpg']:.2f}")
            st.metric("Goals Against Per Game", f"{team_stats[selected_team]['away']['last5']['ga_pg']:.2f}")
            
            gapg = team_stats[selected_team]['away']['last5']['ga_pg']
            if gapg < 1.0:
                st.write("🛡️ **Defense Rating**: Excellent")
            elif gapg < 1.5:
                st.write("✅ **Defense Rating**: Good")
            elif gapg < 2.0:
                st.write("⚡ **Defense Rating**: Average")
            else:
                st.write("⚠️ **Defense Rating**: Poor")

# Footer
st.markdown("---")
st.markdown("""
**SNIPE v1.0** • Simple Over/Under Predictor • Last 5 Games Analysis
""")
