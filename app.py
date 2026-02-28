import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Page Configuration
st.set_page_config(page_title="J1-J2 Heisenberg Model Explorer", layout="wide")

st.title("6x6 Heisenberg $J_1-J_2$ Model Ground State")
st.markdown("""
This app visualizes the spin-spin correlation functions $C(r)$ for the 2D square lattice, 
interpolated from Exact Diagonalization (ED) benchmarks by Richter & Schulenburg.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")
user_alpha = st.sidebar.slider(
    "Select $J_2/J_1$ ratio:", 
    min_value=0.0, 
    max_value=0.9, 
    value=0.52, 
    step=0.01
)

# --- Data Definition ---
j2_ratios = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9])
raw_data = np.array([
    [-0.3387, 0.2072, 0.1851, -0.1790, 0.1685, -0.1670, 0.1650, -0.1645, 0.1630],
    [-0.3383, 0.2000, 0.1755, -0.1661, 0.1550, -0.1550, 0.1530, -0.1520, 0.1510],
    [-0.3369, 0.1905, 0.1631, -0.1497, 0.1380, -0.1401, 0.1370, -0.1360, 0.1350],
    [-0.3334, 0.1769, 0.1467, -0.1278, 0.1150, -0.1209, 0.1130, -0.1120, 0.1110],
    [-0.3261, 0.1562, 0.1239, -0.0971, 0.0820, -0.0952, 0.0810, -0.0790, 0.0780],
    [-0.3108, 0.1225, 0.0934, -0.0538, 0.0410, -0.0620, 0.0400, -0.0390, 0.0380],
    [-0.2986, 0.0993, 0.0774, -0.0289, 0.0250, -0.0453, 0.0240, -0.0230, 0.0220],
    [-0.1295, -0.1839, 0.1456, 0.0344, 0.0150, -0.0324, 0.0140, -0.0130, 0.0120],
    [-0.0771, -0.2659, 0.1814, 0.0302, 0.0050, -0.0248, 0.0040, -0.0030, 0.0020],
    [-0.0504, -0.3017, 0.1959, 0.0253, -0.0050, -0.0202, -0.0040, -0.0030, -0.0020],
    [-0.0347, -0.3236, 0.2028, 0.0215, -0.0100, -0.0173, -0.0080, 0.0070, -0.0050]
])

# Process Data
c00 = np.full((len(j2_ratios), 1), 0.75)
full_matrix = np.hstack((c00, raw_data))
coord_labels = ["(0,0)", "(1,0)", "(1,1)", "(2,0)", "(2,1)", "(2,2)", "(3,0)", "(3,1)", "(3,2)", "(3,3)"]
dist_values = np.array([0.0, 1.0, np.sqrt(2), 2.0, np.sqrt(5), np.sqrt(8), 3.0, np.sqrt(10), np.sqrt(13), np.sqrt(18)])

# Interpolation
f_interp = interp1d(j2_ratios, full_matrix, axis=0, kind='linear')
results = f_interp(user_alpha)

# Sort for plotting
idx = np.argsort(dist_values)
d_sorted, v_sorted, c_sorted = dist_values[idx], results[idx], [coord_labels[i] for i in idx]

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Correlation Plot for $J_2/J_1 = {user_alpha}$")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.plot(d_sorted, v_sorted, marker='o', markersize=12, linewidth=3, color='#1f77b4')
    
    for i, txt in enumerate(c_sorted):
        ax.annotate(txt, (d_sorted[i], v_sorted[i]), textcoords="offset points", 
                     xytext=(0, 15), ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Distance $r$", fontsize=14)
    ax.set_ylabel("$C(r)$", fontsize=14)
    ax.set_ylim(min(v_sorted)-0.1, 0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    st.pyplot(fig)

with col2:
    st.subheader("Numerical Values")
    st.table({
        "Coordinate": [coord_labels[i] for i in idx],
        "C(x,y)": [f"{results[i]:.5f}" for i in idx]
    })

st.info("**Phase Note:** Around $J_2/J_1 \\approx 0.5$, the system enters a highly frustrated regime. Notice the sign flip of (1,1) beyond 0.6.")
