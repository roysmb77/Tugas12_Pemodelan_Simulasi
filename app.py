import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp

st.set_page_config(page_title="TA-10 Robertson Stiff ODE", layout="wide")

# =========================
# Model Robertson (stiff ODE)
# =========================
def robertson(t, y):
    y1, y2, y3 = y
    dy1 = -0.04*y1 + 1e4*y2*y3
    dy2 =  0.04*y1 - 1e4*y2*y3 - 3e7*(y2**2)
    dy3 =  3e7*(y2**2)
    return [dy1, dy2, dy3]

def euler_explicit(f, t0, y0, t_end, h):
    t_values = [t0]
    y_values = [np.array(y0, dtype=float)]
    t = t0
    y = np.array(y0, dtype=float)

    n_steps = int(np.ceil((t_end - t0) / h))
    for _ in range(n_steps):
        y = y + h * np.array(f(t, y))
        t = t + h
        t_values.append(t)
        y_values.append(y.copy())

        # stop early if numerical blow-up
        if (not np.all(np.isfinite(y))) or np.any(np.abs(y) > 1e6):
            break

    return np.array(t_values), np.vstack(y_values)

@st.cache_data(show_spinner=False)
def run_solver_cached(method, t_span, y0, t_eval, rtol, atol):
    start = time.time()
    sol = solve_ivp(
        robertson,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )
    elapsed = time.time() - start
    return sol, elapsed

st.title("TA-10 — Simulasi Sistem Kaku (Stiff ODE): Reaksi Kimia Robertson")

st.markdown(
    """
Aplikasi ini menampilkan:
- **Bukti kegagalan metode eksplisit (Euler)**
- Perbandingan **RK45 vs solver stiff (Radau/BDF)**
- Grafik hasil dan **tabel kinerja (waktu & nfev)**
"""
)

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Parameter")

    st.subheader("Kondisi awal")
    y1_0 = st.number_input("y1(0)", value=1.0, step=0.1, format="%.6f")
    y2_0 = st.number_input("y2(0)", value=0.0, step=0.1, format="%.6f")
    y3_0 = st.number_input("y3(0)", value=0.0, step=0.1, format="%.6f")
    y0 = [float(y1_0), float(y2_0), float(y3_0)]

    st.subheader("Euler eksplisit (demo gagal)")
    h = st.number_input("Step Euler (h)", value=1e-2, format="%.6f")
    t_end_euler = st.number_input("t_end Euler", value=1.0, format="%.6f")

    st.subheader("solve_ivp")
    t_end_long = st.number_input("t_end Radau/BDF", value=1e5, format="%.0f")
    t_end_short = st.number_input("t_end RK45 (pendek)", value=100.0, format="%.0f")

    rtol = st.number_input("rtol", value=1e-6, format="%.1e")
    atol = st.number_input("atol", value=1e-10, format="%.1e")

    n_points = st.slider("Jumlah titik t_eval (Radau/BDF)", min_value=200, max_value=2000, value=999, step=50)

    run_btn = st.button("Run / Refresh", type="primary")

# Auto-run first time
if "ran" not in st.session_state:
    st.session_state.ran = False
if run_btn:
    st.session_state.ran = True

st.info(f"Kondisi awal: y0={y0} | sum={sum(y0):.6f}")

if not st.session_state.ran:
    st.warning("Klik **Run / Refresh** di sidebar untuk menjalankan simulasi.")
    st.stop()

# =========================
# 2) Euler experiment
# =========================
with st.spinner("Menjalankan Euler eksplisit..."):
    t_e, y_e = euler_explicit(robertson, 0.0, y0, t_end=float(t_end_euler), h=float(h))

any_negative = bool((y_e < 0).any())
any_naninf = bool((not np.isfinite(y_e).all()))
last_sum = float(y_e[-1].sum())

colA, colB = st.columns(2)

with colA:
    st.subheader("Bukti Kegagalan Euler (Eksplisit)")
    st.write(f"Steps: **{len(t_e)-1}**")
    st.write(f"Last y: **{y_e[-1]}**")
    st.write(f"sum(last y): **{last_sum:.6f}**")
    st.write(f"Ada nilai negatif? **{any_negative}**")
    st.write(f"Ada NaN/Inf? **{any_naninf}**")

    fig1 = plt.figure(figsize=(7, 3.5))
    plt.plot(t_e, y_e[:, 0], label="y1")
    plt.plot(t_e, y_e[:, 1], label="y2")
    plt.plot(t_e, y_e[:, 2], label="y3")
    plt.title(f"Euler Eksplisit (h={h}) — indikasi gagal pada sistem stiff")
    plt.xlabel("t")
    plt.ylabel("konsentrasi")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig1)

with colB:
    st.subheader("Catatan")
    st.markdown(
        """
- Pada sistem **stiff**, metode eksplisit sering membutuhkan **h sangat kecil** agar stabil.
- Dengan **h “normal”** (mis. 0.01), solusi Euler dapat **negatif / blow-up**.
"""
    )

# =========================
# 3) solve_ivp comparisons
# =========================
t_span_long = (0.0, float(t_end_long))
t_span_short = (0.0, float(t_end_short))

# t_eval log-scale for long run (include 0)
t_eval_long = np.concatenate(([0.0], np.logspace(-6, np.log10(max(1.0, t_span_long[1])), int(n_points))))
# Make sure strictly increasing (avoid duplicates if t_end_long == 1)
t_eval_long = np.unique(t_eval_long)

with st.spinner("Menjalankan solver stiff (Radau & BDF)..."):
    sol_radau, t_radau = run_solver_cached("Radau", t_span_long, y0, t_eval_long, float(rtol), float(atol))
    sol_bdf, t_bdf     = run_solver_cached("BDF",   t_span_long, y0, t_eval_long, float(rtol), float(atol))

with st.spinner("Menjalankan RK45 (rentang pendek)..."):
    # RK45 tanpa t_eval agar lebih bebas memilih step
    start = time.time()
    sol_rk45 = solve_ivp(robertson, t_span_short, y0, method="RK45", rtol=float(rtol), atol=float(atol))
    t_rk45 = time.time() - start

st.subheader("Hasil solve_ivp")
c1, c2, c3 = st.columns(3)
c1.metric("RK45 time (s)", f"{t_rk45:.4f}")
c1.metric("RK45 nfev", f"{sol_rk45.nfev}")
c1.write(f"RK45 success: **{sol_rk45.success}**")

c2.metric("Radau time (s)", f"{t_radau:.4f}")
c2.metric("Radau nfev", f"{sol_radau.nfev}")
c2.write(f"Radau success: **{sol_radau.success}**")

c3.metric("BDF time (s)", f"{t_bdf:.4f}")
c3.metric("BDF nfev", f"{sol_bdf.nfev}")
c3.write(f"BDF success: **{sol_bdf.success}**")

# Plot Radau long (log x)
fig2 = plt.figure(figsize=(10, 4))
plt.semilogx(sol_radau.t, sol_radau.y[0], label="y1 (Radau)")
plt.semilogx(sol_radau.t, sol_radau.y[1], label="y2 (Radau)")
plt.semilogx(sol_radau.t, sol_radau.y[2], label="y3 (Radau)")
plt.title("Solusi Stiff Solver (Radau) pada Robertson")
plt.xlabel("t (log scale)")
plt.ylabel("konsentrasi")
plt.legend()
plt.tight_layout()
st.pyplot(fig2)

# Compare early time: Euler vs Radau
k = min(80, len(sol_radau.t))
fig3 = plt.figure(figsize=(10, 4))
plt.plot(t_e, y_e[:, 0], "--", label="y1 Euler")
plt.plot(t_e, y_e[:, 1], "--", label="y2 Euler")
plt.plot(t_e, y_e[:, 2], "--", label="y3 Euler")
plt.plot(sol_radau.t[:k], sol_radau.y[0][:k], label="y1 Radau (awal)")
plt.plot(sol_radau.t[:k], sol_radau.y[1][:k], label="y2 Radau (awal)")
plt.plot(sol_radau.t[:k], sol_radau.y[2][:k], label="y3 Radau (awal)")
plt.title("Perbandingan awal waktu: Euler vs Radau")
plt.xlabel("t")
plt.ylabel("konsentrasi")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
st.pyplot(fig3)

# =========================
# 5) Table performance
# =========================
rows = [
    {"Metode": f"Euler eksplisit (h={h})", "Waktu (detik)": np.nan, "nfev": len(t_e)-1, "Status": "Gagal/Instabil (indikasi)"},
    {"Metode": f"RK45 (t_end={t_span_short[1]:.0f})", "Waktu (detik)": t_rk45, "nfev": sol_rk45.nfev, "Status": "Berhasil" if sol_rk45.success else "Gagal"},
    {"Metode": f"Radau (t_end={t_span_long[1]:.0f})", "Waktu (detik)": t_radau, "nfev": sol_radau.nfev, "Status": "Berhasil" if sol_radau.success else "Gagal"},
    {"Metode": f"BDF (t_end={t_span_long[1]:.0f})", "Waktu (detik)": t_bdf, "nfev": sol_bdf.nfev, "Status": "Berhasil" if sol_bdf.success else "Gagal"},
]
df = pd.DataFrame(rows)

st.subheader("Tabel Kinerja")
st.dataframe(df, use_container_width=True)

st.caption("Catatan: RK45 sengaja dijalankan pada rentang pendek karena sistem Robertson bersifat stiff sehingga RK45 dapat sangat lambat untuk t_end besar.")
