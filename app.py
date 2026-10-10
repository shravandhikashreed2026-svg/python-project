import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AccessAble Vellore",
    page_icon="♿",
    layout="wide"
)

st.title("♿ AccessAble Vellore")
st.subheader("Vellore Accessibility Mapping Platform")
st.write("Explore accessibility information across Vellore.")

# Demo data only; locations are not verified
data = {
    "Place": [
        "Demo Location 1",
        "Demo Location 2",
        "Demo Location 3",
        "Demo Location 4",
        "Demo Location 5",
        "Demo Location 6"
    ],
    "Latitude": [
        12.9710, 12.9750, 12.9680,
        12.9800, 12.9650, 12.9730
    ],
    "Longitude": [
        79.1550, 79.1600, 79.1500,
        79.1650, 79.1700, 79.1450
    ],
    "Ramp": [1, 1, 0, 1, 0, 0],
    "Lift": [1, 0, 0, 1, 0, 1],
    "Accessible_Toilet": [1, 0, 0, 1, 0, 0],
    "Step_Free": [1, 1, 0, 1, 1, 0],
    "Tactile_Path": [1, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

features = [
    "Ramp", "Lift", "Accessible_Toilet",
    "Step_Free", "Tactile_Path"
]

df["Score"] = df[features].sum(axis=1) / 5 * 100

X = df[features]
y = []

for score in df["Score"]:
    if score >= 70:
        y.append("High")
    elif score >= 40:
        y.append("Moderate")
    else:
        y.append("Low")

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

model.fit(X, y)
df["ML_Prediction"] = model.predict(X)

st.warning(
    "Demo only: sample coordinates and accessibility data "
    "are synthetic and unverified. Do not use for real travel."
)

# Search and filter
st.sidebar.header("Search and Filter")

search = st.sidebar.text_input("Search Vellore demo locations")

category = st.sidebar.selectbox(
    "Accessibility coverage",
    ["All", "High", "Moderate", "Low"]
)

filtered = df.copy()

if search:
    filtered = filtered[
        filtered["Place"].str.contains(
            search, case=False, na=False
        )
    ]

if category != "All":
    filtered = filtered[
        filtered["ML_Prediction"] == category
    ]

# Dashboard
col1, col2, col3 = st.columns(3)

col1.metric("Demo Locations", len(filtered))

if len(filtered) > 0:
    col2.metric(
        "Average Feature Coverage",
        str(round(filtered["Score"].mean())) + "%"
    )
else:
    col2.metric("Average Feature Coverage", "N/A")

col3.metric("ML Model", "Trained")

st.divider()

# Map
st.header("📍 Vellore Demo Map")

if len(filtered) > 0:
    map_data = filtered.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude"
        }
    )
    st.map(map_data[["latitude", "longitude"]])
else:
    st.info("No matching demo locations found.")

# Results
st.header("Accessibility Results")

if len(filtered) > 0:
    st.dataframe(
        filtered[["Place"] + features + [
            "Score", "ML_Prediction"
        ]],
        use_container_width=True
    )

# Accessibility analysis form
st.divider()
st.header("➕ Check a Place in Vellore")

place = st.text_input("Enter place name", key="place_name")

ramp = st.checkbox("Ramp available")
lift = st.checkbox("Lift available")
toilet = st.checkbox("Accessible toilet available")
stepfree = st.checkbox("Step-free entrance available")
tactile = st.checkbox("Tactile path available")

if st.button("Analyse Accessibility"):

    if place.strip() == "":
        st.error("Please enter a place name.")
    else:
        values = [
            int(ramp), int(lift), int(toilet),
            int(stepfree), int(tactile)
        ]

        score = sum(values) / 5 * 100
        prediction = model.predict([values])[0]

        st.subheader("Analysis Result")
        st.write("Place:", place)
        st.write("Feature coverage score:", round(score), "/100")
        st.write("ML model category:", prediction)

        st.caption(
            "Demo estimate only. Verify accessibility features "
            "before relying on this information."
        )