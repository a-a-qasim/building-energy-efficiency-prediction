
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Configure application
st.set_page_config(
    page_title="Building Energy Efficiency Prediction",
    page_icon="🏢",
    layout="wide"
)

# Load trained models
MODEL_DIR = Path(__file__).resolve().parent / "models"

@st.cache_resource
def load_models():
    heating_model = joblib.load(MODEL_DIR / "heating_model.pkl")
    cooling_model = joblib.load(MODEL_DIR / "cooling_model.pkl")
    return heating_model, cooling_model

heating_model, cooling_model = load_models()

# Application header
st.title("🏢 Building Energy Efficiency Prediction")

st.write(
    "Predict building heating and cooling loads "
    "using architectural characteristics and machine learning."
)

st.success("Prediction models loaded successfully!")


# Application sidebar
with st.sidebar:
    st.header("About the Project")

    st.write(
        "A machine learning application for predicting building "
        "heating and cooling loads from architectural characteristics."
    )

    st.divider()

    st.subheader("Model Information")
    st.write("**Algorithm:** Gradient Boosting Regressor")
    st.write("**Dataset:** UCI Energy Efficiency")
    st.write("**Input Features:** 8")
    st.write("**Prediction Targets:** Heating Load and Cooling Load")

    st.divider()

    st.subheader("Building Features")

    with st.expander("Feature Descriptions"):
        st.markdown("""
        - **Relative Compactness:** Compactness of the building shape.
        - **Surface Area:** Total external surface area.
        - **Wall Area:** Total wall surface area.
        - **Roof Area:** Total roof surface area.
        - **Overall Height:** Building height.
        - **Orientation:** Encoded building orientation.
        - **Glazing Area:** Proportion of glazing.
        - **Glazing Distribution:** Encoded window distribution.
        """)

    st.divider()

    st.subheader("Model Limitations")
    st.caption(
        "Predictions are based on simulated buildings in the "
        "UCI dataset. Results are not validated for real-world "
        "buildings and should not be treated as electricity bills "
        "or actual energy consumption."
    )


# Model performance dashboard
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### Heating Load Model")
        st.metric("Test R² Score", "0.9978")
        st.metric("Test MAE", "0.3464")

with col2:
    with st.container(border=True):
        st.markdown("#### Cooling Load Model")
        st.metric("Test R² Score", "0.9771")
        st.metric("Test MAE", "1.0221")

st.caption(
    "Performance metrics obtained from the held-out UCI Energy "
    "Efficiency test dataset."
)

st.divider()


# Building characteristics
st.subheader("Building Characteristics")


st.caption(
    "Enter the building's architectural characteristics. "
    "Input values should remain within the ranges represented "
    "in the training dataset."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Building Geometry")

    relative_compactness = st.number_input(
        "Relative Compactness", min_value=0.62, max_value=0.98,
        value=0.82, step=0.01
    )

    surface_area = st.number_input(
        "Surface Area (m²)", min_value=514.5, max_value=808.5,
        value=612.5, step=0.5
    )

    wall_area = st.number_input(
        "Wall Area (m²)", min_value=245.0, max_value=416.5,
        value=318.5, step=0.5
    )

    roof_area = st.number_input(
        "Roof Area (m²)", min_value=110.25, max_value=220.5,
        value=147.0, step=0.25
    )

with col2:
    st.markdown("#### Design Characteristics")
    
    overall_height = st.selectbox(
        "Overall Height (m)", [3.5, 7.0], index=1
    )

    orientation = st.selectbox(
        "Orientation", [2, 3, 4, 5]
    )

    glazing_area = st.selectbox(
        "Glazing Area", [0.0, 0.1, 0.25, 0.4], index=2
    )
    
    distribution_options = [0] if glazing_area == 0 else [1, 2, 3, 4, 5]

    glazing_distribution = st.selectbox(
        "Glazing Area Distribution",
        distribution_options,
        index=0 if glazing_area == 0 else 1
    )

# Energy load prediction
st.divider()

if st.button("Predict Energy Loads", type="primary", use_container_width=True):

    building_data = pd.DataFrame([{
        "Relative_Compactness": relative_compactness,
        "Surface_Area": surface_area,
        "Wall_Area": wall_area,
        "Roof_Area": roof_area,
        "Overall_Height": overall_height,
        "Orientation": orientation,
        "Glazing_Area": glazing_area,
        "Glazing_Area_Distribution": glazing_distribution
    }])

    # Validate building geometry
    expected_surface_area = wall_area + (2 * roof_area)

    if abs(surface_area - expected_surface_area) > 0.01:
        st.error(
            "Invalid building dimensions. Surface Area must equal "
            "Wall Area + 2 × Roof Area."
        )
        st.stop()

    heating_prediction = heating_model.predict(building_data)[0]
    cooling_prediction = cooling_model.predict(building_data)[0]

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Heating Load",
            value=f"{heating_prediction:.2f}"
        )

    with col2:
        st.metric(
            label="Predicted Cooling Load",
            value=f"{cooling_prediction:.2f}"
        )

    st.caption(
        "Predictions are based on machine learning models trained "
        "using the UCI Energy Efficiency dataset."
    )

    # Visual comparison of predicted loads
    st.subheader("Heating vs Cooling Load")

    comparison = pd.DataFrame({
        "Predicted Load": [heating_prediction, cooling_prediction]
    }, index=["Heating Load", "Cooling Load"])

    st.bar_chart(comparison)


    # Prediction summary
    st.subheader("Prediction Summary")

    load_difference = heating_prediction - cooling_prediction

    if load_difference > 0:
        st.info(
            f"Heating load is predicted to be "
            f"{load_difference:.2f} units higher than cooling load."
        )
    elif load_difference < 0:
        st.info(
            f"Cooling load is predicted to be "
            f"{abs(load_difference):.2f} units higher than heating load."
        )
    else:
        st.info("Heating and cooling loads are predicted to be equal.")

    st.caption(
        "These predictions describe simulated building thermal loads, "
        "not actual electricity consumption or an energy efficiency rating."
    )


    # Export prediction results
    st.subheader("Export Results")

    export_data = building_data.copy()

    export_data["Predicted_Heating_Load"] = round(heating_prediction, 2)
    export_data["Predicted_Cooling_Load"] = round(cooling_prediction, 2)

    csv_data = export_data.to_csv(index=False)

    st.download_button(
        label="Download Prediction Results (CSV)",
        data=csv_data,
        file_name="building_energy_prediction.csv",
        mime="text/csv",
        on_click="ignore",
        use_container_width=True
    )
