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
# STREAMLIT UI
# ======================================================

st.title("🦄 Uniswap v3 Range Visualizer + GBM Simulation")

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
# COMPUTE LIQUIDITIES
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
# LIQUIDITY HISTOGRAMS AS COLUMNS
# ======================================================

st.write("## Liquidity Profiles (histogram)")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["red", "blue", "green"]

for (L, pmin, pmax), color, label in zip(Ls, colors, labels):
    sqrt_pmin = math.sqrt(pmin)
    sqrt_pmax = math.sqrt(pmax)

    # histogram bins in sqrt(p)
    sqrt_bins = np.linspace(sqrt_pmin, sqrt_pmax, 50)
    p_centers = ((sqrt_bins[:-1] + sqrt_bins[1:]) / 2)**2
    liquidity = np.ones_like(p_centers) * L

    ax.bar(p_centers, liquidity, width=(p_centers[1] - p_centers[0]), 
           color=color, alpha=0.3, label=label)

ax.set_title("Liquidity Distribution Histogram")
ax.set_xlabel("Price (AVAX/USDC)")
ax.set_ylabel("Liquidity (L)")
ax.legend()

st.pyplot(fig)

# ======================================================
# GEOMETRIC BROWNIAN MOTION SIMULATION
# ======================================================

st.header("📈 GBM Price Simulation + Volume Tracking")

daily_vol = st.number_input("Daily volatility (σ)", value=0.10)
block_time = st.number_input("Block time (seconds)", value=2.0)
T_half_day = 12 * 60 * 60  # 12 hours

steps = int(T_half_day / block_time)
dt = 1/288  # 0.5 day = 12 hours = 288 five-minute windows; but for GBM use dt = blocktime/day
dt = block_time / (24*60*60)

prices = [price]

for _ in range(steps):
    # GBM step
    dW = np.random.normal(0, math.sqrt(dt))
    new_price = prices[-1] * math.exp(daily_vol * dW - 0.5 * daily_vol**2 * dt)
    prices.append(new_price)

# ========== VOLUME TRACKING ==========
range_volumes = [0.0, 0.0, 0.0]

for i in range(len(prices)-1):
    p0 = prices[i]
    p1 = prices[i+1]

    for idx, (L, pmin, pmax) in enumerate(Ls):

        # compute token amounts before/after
        if p0 < pmin or p0 > pmax:
            x0 = x1 = x_amount_future(L, pmin if p0 < pmin else pmax, pmin, pmax)
            y0 = y1 = y_amount_future(L, pmin if p0 < pmin else pmax, pmin, pmax)
        else:
            x0 = x_amount_future(L, p0, pmin, pmax)
            y0 = y_amount_future(L, p0, pmin, pmax)

        if p1 < pmin or p1 > pmax:
            x_new = x_amount_future(L, pmin if p1 < pmin else pmax, pmin, pmax)
            y_new = y_amount_future(L, pmin if p1 < pmin else pmax, pmin, pmax)
        else:
            x_new = x_amount_future(L, p1, pmin, pmax)
            y_new = y_amount_future(L, p1, pmin, pmax)

        # volume change in USD
        dx = x_new - x0
        dy = y_new - y0
        volume_usd = abs(dx * p1 + dy)

        # Only count if inside range at BOTH steps
        if p0 >= pmin and p0 <= pmax and p1 >= pmin and p1 <= pmax:
            range_volumes[idx] += volume_usd

# ======================================================
# SHOW RESULTS
# ======================================================

st.subheader("Cumulative Volume per Range (USD)")
volume_table = {
    "Range": labels,
    "Volume USD": range_volumes
}
st.write(volume_table)

# price path plot
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(prices)
ax2.set_title("Simulated Price Path (GBM)")
ax2.set_xlabel("Step")
ax2.set_ylabel("Price")
st.pyplot(fig2)

