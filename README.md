# Reflect: Mood Journal
Reflect is a personal mood journal web app built using Streamlit that helps users track, understand, and reflect on their emotional health over time. 
By analyzing journal entries with a pre trained ML model, the app identifies emotional tone, assigns a mood score, and visualizes trends creating a personal space for emotional self awareness and wellness tracking.

## Features
- **Journal Entry**: Users can write free-text journal entries anytime during the day.
Each entry is analyzed using a pre-trained emotion detection model.
The detected emotion label and a corresponding mood score (from -100 to 100) are shown immediately.



- **Dashboard with Mood Trends**:A line chart visualizes mood score trends across all past entries.
Supports multiple entries per day, offering granular mood tracking.



- **Entry History**: View a reverse chronological list of all past entries.
Each entry displays the timestamp, detected emotion, mood score, and the actual journal text.
Entries are shown in a clean, card-style format.



- **Support Resources** : If the last 3 entries all have negative mood scores, the app displays an emotional support panel with, Mental health helpline numbers,
Guided meditation links, Therapy and self-help resource

##  How It Works

User inputs a journal entry via a text area.
- The input is cleaned and vectorized using a TF-IDF vectorizer.
- A pre-trained ML classifier (here, Logistic Regression) predicts the emotion label.
- A score is assigned based on a predefined score map (joy = 80, anger = -80, etc.)
- The entry is stored in Streamlit session state and shown immediately in Dashboard (for mood score trends)
History tab (for entry-by-entry reflection)
- If the last 3 entries are negatively scored, a support panel is shown automatically.



