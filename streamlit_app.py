import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Uniswap v3 Range Tool", layout="wide")

# ======================================================
# MATH FUNCTIONS
# ======================================================

def compute_L_from_x(x, p, pmin, pmax):
    denom = (1/math.sqrt(p) - 1/math.sqrt(pmax))
    if denom <= 0:
        return float("nan")
    return x / denom

def compute_L_from_y(y, p, pmin, pmax):
    denom = (math.sqrt(p) - math.sqrt(pmin))
    if denom <= 0:
        return float("nan")
    return y / denom

def x_amount(L, p, pmax):
    return L * max(0, (1/math.sqrt(p) - 1/math.sqrt(pmax)))

def y_amount(L, p, pmin):
    return L * max(0, (math.sqrt(p) - math.sqrt(pmin)))

def x_amount_future(L, p_prime, pmin, pmax):
    if p_prime <= pmin:
        return L * (1/math.sqrt(pmin) - 1/math.sqrt(pmax))
    elif p_prime >= pmax:
        return 0.0
    else:
        return L * (1/math.sqrt(p_prime) - 1/math.sqrt(pmax))

def y_amount_future(L, p_prime, pmin, pmax):
    if p_prime <= pmin:
        return 0.0
    elif p_prime >= pmax:
        return L * (math.sqrt(pmax) - math.sqrt(pmin))
    else:
        return L * (math.sqrt(p_prime) - math.sqrt(pmin))

def price_to_tick(p):
    return int(math.log(p) / math.log(1.0001))

def tick_to_price(t):
    return 1.0001 ** t


# ======================================================
# UI INPUT
# ======================================================

st.title("🦄 Uniswap v3 Range Visualizer + GBM Volume Simulator")

st.sidebar.header("Initialization")

price = st.sidebar.number_input("Current AVAX/USDC price", value=13.0)

init_mode = st.sidebar.radio(
    "Start with:",
    ["AVAX (token0)", "USDC (token1)"]
)

if init_mode == "AVAX (token0)":
    start_amount = st.sidebar.number_input("Enter AVAX amount:", value=10.0)
else:
    start_amount = st.sidebar.number_input("Enter USDC amount:", value=130.0)

st.sidebar.header("Symmetric Tick Ranges")

width1 = st.sidebar.number_input("Range 1 width (ticks)", value=300)
width2 = st.sidebar.number_input("Range 2 width (ticks)", value=800)
width3 = st.sidebar.number_input("Range 3 width (ticks)", value=1500)

tick_now = price_to_tick(price)

ranges = [
    (tick_now - width1, tick_now + width1),
    (tick_now - width2, tick_now + width2),
    (tick_now - width3, tick_now + width3),
]

# ======================================================
# COMPUTE LIQUIDITY
# ======================================================

st.write("## Liquidity Results")

rows = []
Ls = []
labels = []

for idx, (t0, t1) in enumerate(ranges, start=1):
    pmin = tick_to_price(t0)
    pmax = tick_to_price(t1)

    if init_mode == "AVAX (token0)":
        L = compute_L_from_x(start_amount, price, pmin, pmax)
    else:
        L = compute_L_from_y(start_amount, price, pmin, pmax)

    x_now = x_amount(L, price, pmax)
    y_now = y_amount(L, price, pmin)

    rows.append({
        "Range": f"{idx}",
        "Ticks": f"[{t0}, {t1}]",
        "Prices": f"[{pmin:.4f}, {pmax:.4f}]",
        "Liquidity L": L,
        "x (AVAX)": x_now,
        "y (USDC)": y_now
    })

    Ls.append((L, pmin, pmax))
    labels.append(f"Range {idx}")

st.dataframe(rows)

# ======================================================
# HISTOGRAMS
# ======================================================

st.write("## Liquidity Profiles (histogram)")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["red", "blue", "green"]

for (L, pmin, pmax), color, label in zip(Ls, colors, labels):

    sqrt_grid = np.linspace(math.sqrt(pmin), math.sqrt(pmax), 200)
    p_grid = sqrt_grid**2
    liquidity = np.ones_like(p_grid) * L

    # Use bar histogram-style visualization
    ax.bar(p_grid, liquidity, width=0.0005 * p_grid, alpha=0.4, color=color, label=label)

ax.set_title("Liquidity Distribution vs Price")
ax.set_xlabel("Price (AVAX/USDC)")
ax.set_ylabel("Liquidity")
ax.legend()

st.pyplot(fig)

# ======================================================
# GBM SIMULATION SECTION
# ======================================================

st.write("## 📈 GBM Price Simulation and Volume Accumulation")

vol = st.number_input("Daily Volatility (e.g. 0.9 = 90%)", value=0.90)
block_time = st.number_input("Avalanche Block Time (seconds)", value=1.5)
simulate = st.button("Run GBM Simulation")

if simulate:

    dt = block_time / 86400.0
    steps = int(0.5 / dt)   # half-day

    prices = np.zeros(steps)
    prices[0] = price

    # total volume per LP range
    volumes = [0.0, 0.0, 0.0]

    rng = np.random.default_rng()

    for t in range(steps-1):
        Z = rng.standard_normal()
        prices[t+1] = prices[t] * math.exp(vol * math.sqrt(dt) * Z - 0.5 * vol**2 * dt)

        p_old = prices[t]
        p_new = prices[t+1]
        p_mid = 0.5 * (p_old + p_new)

        for i, (L, pmin, pmax) in enumerate(Ls):

            # if outside range: skip
            if p_old < pmin or p_old > pmax:
                continue
            if p_new < pmin or p_new > pmax:
                continue

            # compute x liquidity difference
            dx_old = x_amount_future(L, p_old, pmin, pmax)
            dx_new = x_amount_future(L, p_new, pmin, pmax)

            traded = abs(dx_new - dx_old) * p_mid
            volumes[i] += traded

    st.write("### 📊 Final Total Volumes (Over Half Day)")
    for i, v in enumerate(volumes, start=1):
        st.write(f"**Range {i}: ${v:,.2f}**")
