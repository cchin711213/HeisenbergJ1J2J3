import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tenpy.models.spins import SpinModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

# App Configuration
st.set_page_config(page_title="J1-J2-J3 Heisenberg DMRG", layout="wide")

st.title("⚛️ J1-J2-J3 Heisenberg Model Solver")
st.markdown("""
This app uses **Density Matrix Renormalization Group (DMRG)** to find the ground state 
correlations of a 6x6 square lattice.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")
j1 = st.sidebar.slider("J (Nearest Neighbor)", -1.0, 1.0, 1.0, 0.1)
j2 = st.sidebar.slider("J' (2nd Nearest / Diagonal)", -1.0, 1.0, 0.5, 0.1)
j3 = st.sidebar.slider("J'' (3rd Nearest)", -1.0, 1.0, 0.1, 0.1)

st.sidebar.header("DMRG Settings")
chi_max = st.sidebar.select_slider("Max Bond Dimension (χ)", options=[40, 60, 80, 100], value=80)
sweeps = st.sidebar.number_input("Max Sweeps", 5, 20, 8)

if st.button("Run Simulation"):
    with st.spinner("Calculating Ground State... (approx 30s)"):
        # 1. Model Setup
        model_params = {
            'lattice': 'Square', 'Lx': 6, 'Ly': 6, 'S': 0.5,
            'conserve': 'Sz',
            'Jx': j1, 'Jy': j1, 'Jz': j1,
            'Jx2': j2, 'Jy2': j2, 'Jz2': j2,
            'Jx3': j3, 'Jy3': j3, 'Jz3': j3,
            'bc_MPS': 'finite', 'bc_x': 'open', 'bc_y': 'open',
        }
        
        M = SpinModel(model_params)
        psi = MPS.from_product_state(M.lat.mps_sites(), ["up", "down"] * 18, bc=M.lat.bc_MPS)

        dmrg_params = {
            'mixer': True, 
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-7, 'max_trunc_err': 0.1},
            'max_sweeps': sweeps,
            'verbose': 0,
            'warn_TenpyInconsistency': True,
        }

        # 2. Execution
        try:
            dmrg.run(psi, M, dmrg_params)
            
            # 3. Measurement
            c0 = M.lat.lat2mps_idx((2, 2, 0))
            results = {}
            for i in range(M.lat.N_sites):
                (x, y, u) = M.lat.mps2lat_idx(i)
                r = np.sqrt((x-2)**2 + (y-2)**2)
                if r > 3.3: continue
                
                val = 3 * psi.expectation_value_term([('Sz', c0), ('Sz', i)])
                rk = round(r, 4)
                if rk not in results: results[rk] = []
                results[rk].append(val)

            r_list = sorted(results.keys())
            c_list = [np.mean(results[rk]) for rk in r_list]

            # --- UI Output ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Correlation Plot")
                fig, ax = plt.subplots()
                ax.plot(r_list, c_list, 'o-', color='#1f77b4', markersize=8)
                ax.axhline(0, color='black', linestyle='--', alpha=0.5)
                ax.set_xlabel("Distance r")
                ax.set_ylabel("C(r)")
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)

            with col2:
                st.subheader("Data Table")
                st.table({"r": r_list, "C(r)": c_list})

        except Exception as e:
            st.error(f"DMRG failed: {e}")

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")
