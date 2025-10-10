import pandas as pd
import numpy as np


def fetch_clinical_events(ticker):
    """
    Placeholder - Replace with API integration returning trial event dates (strings 'YYYY-MM-DD') for ticker.
    Example return format: ['2025-02-15', '2025-08-01']
    """
    # TODO: integrate with ClinicalTrials.gov or another clinical API
    return []  # return empty list for now


def fetch_stock_price_history(ticker):
    """
    Placeholder - Replace with real stock price data from API (Yahoo Finance, IEX, etc.)
    Return value should be a DataFrame indexed by date with at least a 'Close' column.
    Example format:
                 Close
    2025-09-01  95.23
    2025-09-02  96.15
    """
    # TODO: implement API call or local data loading here
    return pd.DataFrame()


def calculate_event_move(df, event_date, window=2):
    """
    Calculate absolute percent move from 'window' days before event to 'window' days after event
    """
    event_date = pd.to_datetime(event_date)
    before_date = event_date - pd.Timedelta(days=window)
    after_date = event_date + pd.Timedelta(days=window)

    try:
        before_price = df.loc[before_date]['Close']
        after_price = df.loc[after_date]['Close']
        pct_move = abs(after_price - before_price) / before_price * 100
        return pct_move
    except KeyError:
        # Dates not in historical data, return NaN
        return np.nan


def assess_straddle_vs_historical(ticker, current_straddle_price):
    """
    Main function to assess if straddle price is undervalued based on historical clinical trial event moves
    """
    event_dates = fetch_clinical_events(ticker)
    if not event_dates:
        print(f'No clinical events found for {ticker}.')
        return

    df = fetch_stock_price_history(ticker)
    if df.empty:
        print(f'No stock price data found for {ticker}.')
        return

    moves = [calculate_event_move(df, d) for d in event_dates]
    avg_move = np.nanmean(moves)
    current_price = df['Close'][-1]

    expected_payout = (avg_move / 100) * current_price

    print(f'Ticker: {ticker}')
    print(f'Clinical event dates: {event_dates}')
    print(f'Historical price moves around events (%): {moves}')
    print(f'Average absolute historical move: {avg_move:.2f}%')
    print(f'Current stock price: ${current_price:.2f}')
    print(f'Expected move in $ (from historical data): ${expected_payout:.2f}')
    print(f'Current straddle price: ${current_straddle_price:.2f}')

    if current_straddle_price < 0.3 * expected_payout:
        print('Potentially undervalued straddle detected.')
    else:
        print('Straddle price appears fairly valued or expensive based on historical moves.')


# Example call with placeholders:
# Replace these functions once API data is available.
def example_stub():
    global fetch_clinical_events, fetch_stock_price_history

    def fetch_clinical_events(ticker):
        return ['2023-03-15', '2023-10-01', '2024-05-10']

    def fetch_stock_price_history(ticker):
        # Simple dummy data with some made-up prices for relevant dates
        dates = pd.date_range(start='2023-03-10', end='2024-05-15')
        prices = 100 + np.sin(np.linspace(-3, 3, len(dates))) * 5  # oscillating prices
        df = pd.DataFrame(data={'Close': prices}, index=dates)
        return df

    # Run assessment with a current straddle price of $0.10
    assess_straddle_vs_historical('FAKEBIOT', 0.10)


if __name__ == "__main__":
    # Run example stub to test structure
    example_stub()
