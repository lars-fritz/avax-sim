import streamlit
import math

# -------------------------------------------------------
#      L FROM STARTING AVAX ONLY (token0 = x)
# -------------------------------------------------------
def compute_L_from_x(x, p, pmin, pmax):
    """
    Compute liquidity from token0 amount (AVAX).
    From:
        x = L(1/sqrt(p) - 1/sqrt(pmax))
    """
    denom = (1/math.sqrt(p) - 1/math.sqrt(pmax))
    if denom <= 0:
        return float("nan")
    return x / denom


# -------------------------------------------------------
#      L FROM STARTING USDC ONLY (token1 = y)
# -------------------------------------------------------
def compute_L_from_y(y, p, pmin, pmax):
    """
    Compute liquidity from token1 amount (USDC).
    From:
        y = L(sqrt(p) - sqrt(pmin))
    """
    denom = (math.sqrt(p) - math.sqrt(pmin))
    if denom <= 0:
        return float("nan")
    return y / denom


# -------------------------------------------------------
# Token amounts at current price
# -------------------------------------------------------
def x_amount(L, p, pmax):
    return L * max(0, (1/math.sqrt(p) - 1/math.sqrt(pmax)))

def y_amount(L, p, pmin):
    return L * max(0, (math.sqrt(p) - math.sqrt(pmin)))


# -------------------------------------------------------
# Token amounts at future price p'
# -------------------------------------------------------
def x_amount_future(L, p_prime, pmin, pmax):
    if p_prime <= pmin:
        # all-x region
        return L * (1/math.sqrt(pmin) - 1/math.sqrt(pmax))
    elif p_prime >= pmax:
        # all-y region
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
