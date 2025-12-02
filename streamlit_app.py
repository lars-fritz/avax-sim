import streamlit as st
import numpy as np
import math

st.set_page_config(page_title="Uniswap v3 Range Comparison", layout="wide")
st.title("🦄 Uniswap v3 AVAX/USDC — Range Comparison Tool")

# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------

def price_to_tick(p):
    """Uniswap v3 tick formula: tick = log_sqrt(p) / log_sqrt(1.0001)."""
    return int(math.log(p, 1.0001))

def tick_to_price(tick):
    """Inverse: price from tick."""
    return 1.0001 ** tick

def liquidity_from_value(price, value_usd, p_min, p_max):
    """
    Approximate L given a USD deposit.
    Assume:
        - Token0 = AVAX (volatile)
        - Token1 = USDC (stable ~ $1)
    We solve for L from token amounts at initialization price.
    """
    sqrt_p = math.sqrt(price)
    sqrt_p_min = math.sqrt(p_min)
    sqrt_p_max = math.sqrt(p_max)

    # Token amounts at price p
    x = max(0, (1/sqrt_p - 1/sqrt_p_max))
    y = max(0, (sqrt_p - sqrt_p_min))

    # Avoid division by zero
    if x + price * y == 0:
        return 0

    # We scale liquidity L such that total value = value_usd
    # Value = x*price + y
    L = value_usd / (x * price + y)
    return L

def tokens_from_liquidity(L, p, p_min, p_max):
    """Compute x(p), y(p) token amounts at price p."""
    sqrt_p = math.sqrt(p)
    x = L * max(0, (1/sqrt_p - 1/math.sqrt(p_max)))
    y = L * max(0, (sqrt_p - 1/math.sqrt(p_min)))
    return x, y


# --------------------------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------------------------

st.sidebar.header("⚙️ Parameters")

price_now = st.sidebar.number_input(
    "Current AVAX/USDC Price", value=13.0, min_value=0.1, step=0.1
)

value_usd = st.sidebar.number_input(
    "Position Size (USD)", value=1000.0, min_value=10.0, step=10.0
)

tick_width = st.sidebar.number_input(
    "Symmetric Tick Width (each side)", value=500, min_value=10, step=10
)

# Pre-define 3 ranges, scaled versions
range_multipliers = [1, 2, 4]  # narrow, medium, wide

# Compute center tick
tick_center = price_to_tick(price_now)

ranges = []
for m in range_multipliers:
    lo = tick_center - m * tick_width
    hi = tick_center + m * tick_width
    ranges.append((lo, hi))


# --------------------------------------------------------------------
# Compute Liquidity for 3 ranges
# --------------------------------------------------------------------
st.subheader("📊 Range Comparison")

data = []
for (t_lo, t_hi) in ranges:
    p_min = tick_to_price(t_lo)
    p_max = tick_to_price(t_hi)

    L = liquidity_from_value(price_now, value_usd, p_min, p_max)
    x_now, y_now = tokens_from_liquidity(L, price_now, p_min, p_max)

    data.append({
        "Ticks": f"[{t_lo}, {t_hi}]",
        "Prices": f"[{p_min:.4f}, {p_max:.4f}]",
        "Liquidity L": L,
        "Token0 (AVAX)": x_now,
        "Token1 (USDC)": y_now,
    })

st.write("### 📐 Position Initialization at Current Price")
st.dataframe(data)


# --------------------------------------------------------------------
# Uniswap V3 Math Explanation
# --------------------------------------------------------------------

st.markdown(r"""
---

# 🧠 Uniswap v3 Mathematics

### 🔢 The Core Equation

Uniswap V3 expresses the liquidity relationship between two tokens with:

\[
(x + \frac{L}{\sqrt{p_{\max}}})(y + L\sqrt{p_{\min}}) = L^{2}
\]

**Where:**

- \( x \) = token0 amount  
- \( y \) = token1 amount  
- \( L \) = liquidity  
- \( p \) = price (token1 per token0)  
- \( p_{\min}, p_{\max} \) = active range boundaries  

---

### 🧮 Token Amounts as a Function of Price

Inside the active range:

\[
x(p) = L\left(\frac{1}{\sqrt{p}} - \frac{1}{\sqrt{p_{\max}}}\right)
\]

\[
y(p) = L\left(\sqrt{p} - \sqrt{p_{\min}}\right)
\]

These describe the quantities of token0 and token1 required to support liquidity at price \( p \).

Within \([p_{\min}, p_{\max}]\), liquidity is active and earns fees.

Outside the range, one of the token amounts becomes zero — the position becomes 100% one-sided.

---
""")

st.info("📌 Next steps: add charts for IL, range value, fee APR, and dynamic updating.")


# --------------------------------------------------------------------
# End App
# --------------------------------------------------------------------
