import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ——— Load your model and scaler ——————————————————————————————
@st.cache(allow_output_mutation=True)
def load_model_and_scaler():
    model = joblib.load('final_workspace_volume_model.pkl')
    try:
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        scaler = None
    return model, scaler

model, scaler = load_model_and_scaler()


# ——— Feature‐prep & prediction functions —————————————————————————
def calculate_volume_cylinder(radius, half_length):
    """Volume of a cylinder with radius r and total height 2*h."""
    return np.pi * radius**2 * (2 * half_length)

def calculate_grasp_length(half_length):
    """Grasp length = 2 * half_length."""
    return 2 * half_length

def prepare_input_features(objA_dims, objB_dims):
    # both are [radius, half_length]
    vA = calculate_volume_cylinder(*objA_dims)
    vB = calculate_volume_cylinder(*objB_dims)
    gA = calculate_grasp_length(objA_dims[1])
    gB = calculate_grasp_length(objB_dims[1])
    # one-hot for cylinder = [0,0,1]
    return [
        vA, gA, 0,0,1,
        vB, gB, 0,0,1
    ]

def predict_sequence(objA_dims, objB_dims):
    # build features A→B and B→A
    feats_ab = prepare_input_features(objA_dims, objB_dims)
    feats_ba = prepare_input_features(objB_dims, objA_dims)
    df_ab = pd.DataFrame([feats_ab])
    df_ba = pd.DataFrame([feats_ba])
    if scaler:
        Xab = scaler.transform(df_ab)
        Xba = scaler.transform(df_ba)
    else:
        Xab, Xba = df_ab, df_ba

    # model.predict returns [[vol_after1, vol_after2]]
    va1, vab = model.predict(Xab)[0]
    vb1, vba = model.predict(Xba)[0]
    total_ab = va1 + vab
    total_ba = vb1 + vba

    return {
        "A_then_B": {"after_1": va1, "after_2": vab, "total": total_ab},
        "B_then_A": {"after_1": vb1, "after_2": vba, "total": total_ba},
        "better": "A→B" if total_ab>total_ba else ("B→A" if total_ba>total_ab else "Tie")
    }


# ——— Streamlit UI ————————————————————————————————————————————————
st.title("Grasp‐Sequence Workspace Volume Predictor")

st.markdown("Enter **diameter** and **length** for two cylinders (in mm).  The app will convert to radius & half‐length in meters, run your model, and tell you which sequence (A→B or B→A) gives the larger workspace volume.")

with st.form("input_form"):
    st.subheader("Object A")
    d1 = st.number_input("Diameter (mm)", min_value=0.0, value=20.0, step=0.1, key="d1")
    L1 = st.number_input("Length (mm)",   min_value=0.0, value=100.0, step=0.1, key="L1")

    st.subheader("Object B")
    d2 = st.number_input("Diameter (mm)", min_value=0.0, value=30.0, step=0.1, key="d2")
    L2 = st.number_input("Length (mm)",   min_value=0.0, value=120.0, step=0.1, key="L2")

    submitted = st.form_submit_button("Predict")

if submitted:
    # convert mm → m, half‐length
    objA = [(d1/2)/1000.0, (L1/2)/1000.0]
    objB = [(d2/2)/1000.0, (L2/2)/1000.0]

    res = predict_sequence(objA, objB)

    st.markdown("### Results")
    st.write(f"**Sequence A → B**  •  After 1st grasp: `{res['A_then_B']['after_1']:.10f}` m³  •  After both: `{res['A_then_B']['after_2']:.10f}` m³  •  **Total:** `{res['A_then_B']['total']:.10f}` m³")
    st.write(f"**Sequence B → A**  •  After 1st grasp: `{res['B_then_A']['after_1']:.10f}` m³  •  After both: `{res['B_then_A']['after_2']:.10f}` m³  •  **Total:** `{res['B_then_A']['total']:.10f}` m³")
    if res["better"] == "Tie":
        st.success("Both sequences yield the same total volume.")
    else:
        st.success(f"▶️ **Better sequence:** {res['better']}")
