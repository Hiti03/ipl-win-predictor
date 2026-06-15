import streamlit as st
import pickle
import numpy as np

# 1. Load the model, the encoders, and the saved feature order
model = pickle.load(open('model/rf_model.pkl', 'rb'))
le_bat = pickle.load(open('model/le_bat.pkl', 'rb'))
le_bowl = pickle.load(open('model/le_bowl.pkl', 'rb'))
le_venue = pickle.load(open('model/le_venue.pkl', 'rb'))
feature_order = pickle.load(open('model/features.pkl', 'rb'))

teams = sorted(le_bat.classes_)
venues = sorted(le_venue.classes_)

# 2. Page setup
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏")
st.title("🏏 IPL Win Predictor")
st.write("Predict the batting team's chance of winning the chase.")

# 3. Inputs the user fills in
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting team", teams)
with col2:
    bowling_team = st.selectbox("Bowling team", [t for t in teams if t != batting_team])

venue = st.selectbox("Venue", venues)

col3, col4 = st.columns(2)
with col3:
    target = st.number_input("Target", min_value=1, max_value=300, value=180)
    current_score = st.number_input("Current score", min_value=0, max_value=target, value=80)
with col4:
    overs_completed = st.number_input("Completed overs", min_value=0, max_value=19, value=10)
    balls_this_over = st.number_input("Balls in this over", min_value=0, max_value=5, value=0)

wickets_gone = st.number_input("Wickets gone", min_value=0, max_value=9, value=2)
last_18 = st.number_input("Wickets in the last 3 overs (optional)", min_value=0, max_value=9, value=0)

# 4. When the button is pressed, calculate everything and predict
if st.button("Predict"):
    balls_bowled = overs_completed * 6 + balls_this_over
    overs_bowled = balls_bowled / 6
    balls_remaining = 120 - balls_bowled
    overs_remaining = balls_remaining / 6

    runs_remaining = target - current_score
    wickets_remaining = 10 - wickets_gone

    crr = current_score / overs_bowled if overs_bowled > 0 else 0
    rrr = runs_remaining / overs_remaining if overs_remaining > 0 else 0
    run_rate_diff = crr - rrr

    bat_enc = le_bat.transform([batting_team])[0]
    bowl_enc = le_bowl.transform([bowling_team])[0]
    venue_enc = le_venue.transform([venue])[0]

    # all 13 values, stored by name
    values = {
        'runs_scored': current_score,
        'wickets_remaining': wickets_remaining,
        'overs_bowled': overs_bowled,
        'runs_remaining': runs_remaining,
        'crr': crr,
        'rrr': rrr,
        'target': target,
        'bat_enc': bat_enc,
        'bowl_enc': bowl_enc,
        'venue_enc': venue_enc,
        'balls_remaining': balls_remaining,
        'run_rate_diff': run_rate_diff,
        'last_18_balls_wickets': last_18,
    }

    # put them in the EXACT order features.pkl expects
    X = np.array([[values[f] for f in feature_order]])

    win_prob = model.predict_proba(X)[0][1]

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric(f"{batting_team} win", f"{win_prob*100:.1f}%")
    c2.metric(f"{bowling_team} win", f"{(1 - win_prob)*100:.1f}%")
    st.progress(float(win_prob))