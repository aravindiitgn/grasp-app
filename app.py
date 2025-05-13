import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('final_workspace_volume_model.pkl')

# Load the scaler if available
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    scaler = None

st.title("Workspace Volume Predictor for Grasp Sequences (Input in mm)")

# Helper functions
def calculate_volume(shape, dims_m):
    # dims_m are in meters
    if shape == 'Cuboid':
        x, y, z = dims_m
        return 2*x * 2*y * 2*z
    elif shape == 'Sphere':
        r, = dims_m
        return (4/3) * np.pi * r**3
    elif shape == 'Cylinder':
        r, h = dims_m
        return np.pi * r**2 * 2*h
    return 0

def calculate_grasp_length(shape, dims_m):
    # for cuboid & cylinder, grasp along the "y" dimension
    if shape in ['Cuboid', 'Cylinder']:
        return 2 * dims_m[1]
    return 0

def prepare_features(shapeA, dimsA_mm, shapeB, dimsB_mm):
    # convert mm → m
    dimsA = [d/1000 for d in dimsA_mm]
    dimsB = [d/1000 for d in dimsB_mm]

    volA = calculate_volume(shapeA, dimsA)
    gripA = calculate_grasp_length(shapeA, dimsA)
    volB = calculate_volume(shapeB, dimsB)
    gripB = calculate_grasp_length(shapeB, dimsB)

    one_hot = {'Cuboid': [1,0,0], 'Sphere': [0,1,0], 'Cylinder': [0,0,1]}
    features = [
        volA, gripA, *one_hot[shapeA],
        volB, gripB, *one_hot[shapeB]
    ]
    arr = np.array(features).reshape(1, -1)
    return scaler.transform(arr) if scaler else arr

# UI for object input
def input_object(name):
    st.subheader(name + " (all dims in mm)")
    shape = st.selectbox(f"{name} Shape", ['Cuboid', 'Sphere', 'Cylinder'], key=name)
    if shape == 'Cuboid':
        x = st.number_input(f"{name} x half-length", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key=name+'x')
        y = st.number_input(f"{name} y half-width",  min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key=name+'y')
        z = st.number_input(f"{name} z half-height", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key=name+'z')
        dims_mm = [x, y, z]
    elif shape == 'Sphere':
        r = st.number_input(f"{name} radius", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key=name+'r')
        dims_mm = [r]
    else:  # Cylinder
        r = st.number_input(f"{name} radius",     min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key=name+'r')
        h = st.number_input(f"{name} half-height", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key=name+'h')
        dims_mm = [r, h]
    return shape, dims_mm

# Layout
with st.form(key='input_form'):
    col1, col2 = st.columns(2)
    with col1:
        shapeA, dimsA_mm = input_object("Object A")
    with col2:
        shapeB, dimsB_mm = input_object("Object B")
    submitted = st.form_submit_button("Predict Workspace Volumes")

if submitted:
    feat_ab = prepare_features(shapeA, dimsA_mm, shapeB, dimsB_mm)
    feat_ba = prepare_features(shapeB, dimsB_mm, shapeA, dimsA_mm)

    pred_ab = model.predict(feat_ab)[0]
    pred_ba = model.predict(feat_ba)[0]

    total_ab = pred_ab[0] + pred_ab[1]
    total_ba = pred_ba[0] + pred_ba[1]

    st.write("## Results")
    st.write(f"**A → B Total Workspace Volume:** {total_ab:.6f} m³")
    st.write(f"**B → A Total Workspace Volume:** {total_ba:.6f} m³")

    if total_ab < total_ba:
        st.success("Optimal sequence: A first, then B")
    elif total_ba < total_ab:
        st.success("Optimal sequence: B first, then A")
    else:
        st.info("Both sequences yield equal total volumes")
