import math

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculates the Black-Scholes price for European options.
    
    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiration (in years).
        r (float): Risk-free rate (annualized).
        sigma (float): Implied volatility (annualized).
        option_type (str): "call" or "put".
        
    Returns:
        price (float): Option price.
    """
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    from scipy.stats import norm
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price