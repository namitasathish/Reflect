
import streamlit as st
import joblib
import pandas as pd
import datetime
import re

# ─── 0. Page config & custom CSS ─────────────────────────────────
st.set_page_config(page_title="Reflect: Mood Journal", layout="wide")
st.markdown("""
<style>
/* Global dark background and text color */
.block-container {
    background-color: #121212 !important;
    color: #e0e0e0 !important;
}
/* Text area styling */
textarea {
    background-color: #1e1e1e !important;
    color: #e0e0e0 !important;
    border: 1px solid #333333 !important;
}
/* Card style for metrics and entries */
.card {
    background: #1e1e1e;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.5);
    color: #e0e0e0;
}
/* Dataframe and table styling */
.dataframe, .stTable {
    background-color: #1e1e1e !important;
    color: #e0e0e0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── 1. Load model artifacts ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/mood_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, le = load_artifacts()

# ─── 2. Helpers ────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

score_map = {
    'joy': 80, 'neutral': 0, 'fear': -60, 'anger': -80,
    'sadness': -70, 'disgust': -60, 'shame': -90, 'guilt': -50
}

def predict(text: str):
    c = clean_text(text)
    vec = vectorizer.transform([c])
    label = model.predict(vec)[0]
    emotion = le.inverse_transform([label])[0]
    score = score_map.get(emotion, 0)
    return emotion, score

# ─── 3. State ──────────────────────────────────────────────────────
if "entries" not in st.session_state:
    st.session_state.entries = []

# ─── 4. Navigation Tabs ────────────────────────────────────────────
tabs = st.tabs([
    f"Journal", 
    f"History", 
    f"Trends"
])
journal_tab, history_tab, trends_tab = tabs

# ─── 5. Journal Tab ────────────────────────────────────────────────
with journal_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Write a New Entry")
    with st.form("entry_form", clear_on_submit=True):
        entry_text = st.text_area("How are you feeling today?", height=150)
        submitted = st.form_submit_button("Save Entry")
    if submitted and entry_text.strip():
        emotion, score = predict(entry_text)
        timestamp = datetime.datetime.now()
        st.session_state.entries.append({
            "timestamp": timestamp,
            "text": entry_text,
            "emotion": emotion,
            "score": score
        })
        col1, col2 = st.columns(2)
        col1.metric("Latest Emotion", emotion.capitalize(), delta="", help="Detected emotion")
        col2.metric("Latest Mood Score", f"{score}", delta="", help="Score from -100 to 100")
        st.success("Entry saved!")
    st.markdown("</div>", unsafe_allow_html=True)

# ─── 6. History Tab ────────────────────────────────────────────────
with history_tab:
    st.subheader("Past Entries")
    if not st.session_state.entries:
        st.info("No entries yet. Go to Journal to write one.")
    else:
        for entry in reversed(st.session_state.entries):
            st.markdown(f"""
                <div class='card'>
                  <strong>{entry['timestamp'].strftime('%Y-%m-%d %H:%M')}</strong> 
                  <span style="float:right; color:#ff6e6e;">{entry['emotion'].capitalize()} ({entry['score']})</span>
                  <p style="margin-top:0.5rem;">{entry['text']}</p>
                </div>
                <br/>
            """, unsafe_allow_html=True)

# ─── 7. Trends Tab ─────────────────────────────────────────────────
with trends_tab:
    st.subheader("Mood Trends (Entry-by-Entry)")
    if not st.session_state.entries:
        st.info("Add entries in Journal first.")
    else:
        # Combine all entries and plot score vs timestamp
        df_entries = pd.DataFrame(st.session_state.entries).sort_values("timestamp")
        df_entries = df_entries.set_index("timestamp")
        st.line_chart(df_entries["score"], height=400, use_container_width=True)
        st.caption("Mood Score for Every Journal Entry Over Time")

# ─── 8. Support Panel ────────────────────────────────────────────── Support Panel ──────────────────────────────────────────────
if len(st.session_state.entries) >= 3:
    last3 = [e["score"] for e in st.session_state.entries[-3:]]
    if all(s < 0 for s in last3):
        with st.expander("🚨 Need Help? Click here for resources", expanded=True):
            st.markdown("""
            - **National Helpline**: 123-456-7890  
            - **Meditation App**: [Headspace](https://www.headspace.com)  
            - **Self-Help Articles**: [American Psychological Association](https://www.psychologytoday.com/us)  
            - **Online Counseling**: [BetterHelp](https://www.betterhelp.com)
            """)
