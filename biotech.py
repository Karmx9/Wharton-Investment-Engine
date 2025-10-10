import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import BDay

def fetch_stock_price_history(ticker, start_date, end_date):
    """
    Fetches historical stock price data for a given ticker using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        # Add a buffer to ensure we get the days we need for our window
        df = stock.history(start=pd.to_datetime(start_date) - BDay(5), 
                           end=pd.to_datetime(end_date) + BDay(5),
                           auto_adjust=True)
        if df.empty:
            print(f"Warning: No historical data found for {ticker} in the given date range.")
            return None
        return df
    except Exception as e:
        print(f"An error occurred while fetching stock data for {ticker}: {e}")
        return None

def calculate_event_move(df, event_date):
    """
    Calculates the percentage move from the close before the event to the open on the event day.
    This captures overnight news releases more effectively.
    """
    event_date = pd.to_datetime(event_date)
    
    # Get the last trading day BEFORE the event
    before_day = event_date - BDay(1)
    
    try:
        # Ensure the dates are in the dataframe index
        if before_day.date() not in df.index.date or event_date.date() not in df.index.date:
             # Find the closest available dates if exact dates are missing
             before_day = df.index[df.index <= before_day].max()
             event_date = df.index[df.index >= event_date].min()
        
        close_before = df.loc[before_day]['Close']
        open_on = df.loc[event_date]['Open']
        
        if pd.isna(close_before) or pd.isna(open_on):
            return np.nan
            
        pct_move = abs(open_on - close_before) / close_before * 100
        return pct_move
    except (KeyError, IndexError):
        # Could happen if event date is too close to the edge of the data
        return np.nan

def assess_straddle_viability(ticker, events_with_weights, current_straddle_price):
    """
    Main function to assess if a straddle is viable by comparing its breakeven point
    to the weighted historical price moves around significant clinical events.
    
    Args:
        ticker (str): The stock ticker.
        events_with_weights (list): A list of tuples, where each tuple is ('YYYY-MM-DD', weight).
                                    Weight reflects the event's significance (e.g., 1.0 for major).
        current_straddle_price (float): The current combined price of the call and put options.
    """
    if not events_with_weights:
        print(f"No clinical events provided for {ticker}.")
        return

    # Extract dates for fetching historical data
    event_dates_str = [e[0] for e in events_with_weights]
    start_date, end_date = min(event_dates_str), max(event_dates_str)

    df = fetch_stock_price_history(ticker, start_date, end_date)
    if df is None or df.empty:
        return

    moves, weights = [], []
    for date_str, weight in events_with_weights:
        move = calculate_event_move(df, date_str)
        if not np.isnan(move):
            moves.append(move)
            weights.append(weight)

    if not moves:
        print(f"Could not calculate any historical moves for the events provided for {ticker}.")
        return
        
    # --- Core Analysis ---
    current_stock_price = df['Close'][-1]
    
    # 1. Calculate the straddle's breakeven point
    breakeven_pct = (current_straddle_price / current_stock_price) * 100

    # 2. Calculate weighted statistics for historical moves
    weighted_avg_move = np.average(moves, weights=weights)
    median_move = np.median(moves)
    max_move = np.max(moves)

    # --- Print Results ---
    print("\n" + "="*50)
    print(f"Straddle Viability Analysis for: {ticker.upper()}")
    print("="*50)
    print(f"Current Stock Price: ${current_stock_price:.2f}")
    print(f"Current Straddle Price: ${current_straddle_price:.2f}\n")

    print("--- Required Performance vs. Historical Performance ---\n")
    print(f"⚠️ Straddle Breakeven Point: {breakeven_pct:.2f}%")
    print(f"(The stock must move at least this much for the trade to be profitable)\n")
    
    print(f"📈 Weighted Avg Historical Move: {weighted_avg_move:.2f}%")
    print(f"📊 Median Historical Move: {median_move:.2f}%")
    print(f"🚀 Maximum Historical Move: {max_move:.2f}%\n")

    print("--- Conclusion ---")
    if weighted_avg_move > breakeven_pct:
        print("Historical weighted average move IS GREATER than the required breakeven point.")
        print("This suggests that, based on past weighted performance, a similar move could be profitable.")
    else:
        print("Historical weighted average move IS LESS than the required breakeven point.")
        print("This suggests the straddle is priced expensively relative to past weighted performance.")
    
    print("\n\n🚨 IMPORTANT: This is not financial advice. IV crush is a major risk. Always check options liquidity (bid-ask spread and volume) before trading.")
    print("="*50 + "\n")


if __name__ == "__main__":
    # --- User Input Section ---
    # 1. Define the Ticker
    TICKER_TO_ANALYZE = 'SAVA' # Example: Cassava Sciences
    
    # 2. Define historical events and assign a significance weight to each.
    # (1.0 = High significance, 0.5 = Medium, etc.)
    # This is the most crucial manual step. Research the company's history.
    HISTORICAL_EVENTS = [
        ('2021-08-25', 1.0), # Citizen Petition filed, high impact news
        ('2022-02-10', 0.8), # News on clinical trial screening
        ('2022-12-15', 0.7), # End-of-year update
        ('2023-10-18', 0.9), # News on lawsuit dismissal
        ('2024-06-20', 1.0)  # Top-line data news
    ]
    
    # 3. Find the current price of an at-the-money (ATM) straddle for an expiration
    #    date just after the next expected event. (e.g., from your brokerage platform)
    CURRENT_ATM_STRADDLE_PRICE = 9.50 # Example price for a near-term straddle

    # --- Run Analysis ---
    assess_straddle_viability(TICKER_TO_ANALYZE, HISTORICAL_EVENTS, CURRENT_ATM_STRADDLE_PRICE)
