import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Uniswap v3 Range Tool + GBM", layout="wide")

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

    ax.bar(p_grid, liquidity, width=0.0005 * p_grid, alpha=0.4, color=color, label=label)

ax.set_title("Liquidity Distribution vs Price")
ax.set_xlabel("Price (AVAX/USDC)")
ax.set_ylabel("Liquidity")
ax.legend()

st.pyplot(fig)

# ======================================================
# GBM SINGLE RUN
# ======================================================

st.write("## 📈 GBM Price Simulation (Single Path)")

vol = st.number_input("Daily Volatility (σ)", value=0.90)
block_time = st.number_input("Avalanche Block Time (seconds)", value=1.5)
do_single = st.button("Run Single GBM Path")

def simulate_gbm_once(price, vol, block_time, Ls):
    dt = block_time / 86400.0
    steps = int(0.5 / dt)

    # vectorized GBM
    Z = np.random.standard_normal(steps - 1)
    increments = vol * math.sqrt(dt) * Z - 0.5 * vol**2 * dt
    prices = np.zeros(steps)
    prices[0] = price
    prices[1:] = price * np.exp(np.cumsum(increments))

    volumes = [0.0, 0.0, 0.0]

    # compute y-values
    for i, (L, pmin, pmax) in enumerate(Ls):

        mask = (prices >= pmin) & (prices <= pmax)

        valid = np.where(mask)[0]
        if len(valid) < 2:
            continue

        yvals = np.array([
            y_amount_future(L, prices[k], pmin, pmax) for k in valid
        ])

        dy = np.abs(np.diff(yvals))
        volumes[i] += dy.sum()

    return prices, volumes

if do_single:
    prices, volumes = simulate_gbm_once(price, vol, block_time, Ls)

    # Plot GBM
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(prices, color='purple')
    ax2.set_title("GBM Price Path (Half-Day)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Price")
    st.pyplot(fig2)

    st.write("### 📊 Final Total Volumes (USDC)")
    for i, v in enumerate(volumes, start=1):
        st.write(f"**Range {i}: ${v:,.2f}**")

# ======================================================
# MONTE CARLO (FAST)
# ======================================================

st.write("## 📊 Monte Carlo Volume Distribution (1000 runs)")

do_mc = st.button("Run Monte Carlo")

if do_mc:

    N = 1000
    all_vols = np.zeros((3, N))

    for k in range(N):
        _, vols = simulate_gbm_once(price, vol, block_time, Ls)
        all_vols[:, k] = vols

    st.write("### Volume Statistics")
    for i in range(3):
        st.write(
            f"**Range {i+1}** — Mean: {all_vols[i].mean():,.2f},  Std: {all_vols[i].std():,.2f}"
        )

    # Histogram plot
    fig3, axes = plt.subplots(1, 3, figsize=(18, 4))

    for i in range(3):
        axes[i].hist(all_vols[i], bins=40, color=colors[i], alpha=0.7)
        axes[i].set_title(f"Range {i+1}")
        axes[i].set_xlabel("Volume (USDC)")
        axes[i].set_ylabel("Frequency")

    st.pyplot(fig3)
