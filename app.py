import streamlit as st
import yfinance as yf
import sqlite3
import pandas as pd
import datetime
import logging
import requests
import altair as alt
import uuid
import numpy as np
from mftool import Mftool
import time

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s:root:%(message)s")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="pages/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- USER AUTHENTICATION ---
# Hardcoded credentials for demonstration
USERNAME = "HXR"
PASSWORD = "Rossph"

def login_page():
    """Renders the login page."""
    st.title("Login to Finance Dashboard")
    st.markdown("Please enter your credentials to access the dashboard.")

    # Callback to clear the input fields
    def reset_login_fields():
        if "username" in st.session_state:
            st.session_state.username = ""
        if "password" in st.session_state:
            st.session_state.password = ""

    # The login form now only contains the submit button
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")

        submit_button = st.form_submit_button("Login")

    # The reset button is now outside the form
    if st.button("Reset", on_click=reset_login_fields):
        pass

    if submit_button:
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# --- INDICATOR FUNCTIONS (Unused but kept for reference) ---
def rsi(close, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(close, fast=12, slow=26, signal=9):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger(close, period=20, std_dev=2):
    """Calculates Bollinger Bands."""
    rolling_mean = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

# --- DATABASE SETUP ---
@st.cache_resource
def get_db_connection():
    """Establishes and caches the database connection."""
    try:
        conn = sqlite3.connect("finance.db", check_same_thread=False)
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}", exc_info=True)
        st.error("Failed to connect to the database. Please reload the app.")
        st.stop()

def initialize_database(conn):
    """Initializes all database tables if they don't exist and handles schema migrations."""
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS portfolio (ticker TEXT PRIMARY KEY, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1, sector TEXT, market_cap TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades (symbol TEXT PRIMARY KEY, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1, target_price REAL NOT NULL, stop_loss_price REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS price_history (ticker TEXT, date TEXT, close_price REAL, PRIMARY KEY (ticker, date))""")
    c.execute("""CREATE TABLE IF NOT EXISTS realized_stocks (transaction_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL, sell_price REAL NOT NULL, sell_date TEXT NOT NULL, realized_return_pct REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS exits (transaction_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL, sell_price REAL NOT NULL, sell_date TEXT NOT NULL, realized_return_pct REAL NOT NULL, target_price REAL NOT NULL, stop_loss_price REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS fund_transactions (transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, type TEXT NOT NULL, amount REAL NOT NULL, description TEXT, allocation_type TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS expenses (expense_id TEXT PRIMARY KEY, date TEXT NOT NULL, amount REAL NOT NULL, category TEXT NOT NULL, payment_method TEXT, description TEXT, type TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS budgets (budget_id INTEGER PRIMARY KEY AUTOINCREMENT, month_year TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL, UNIQUE(month_year, category))""")
    c.execute("""CREATE TABLE IF NOT EXISTS recurring_expenses (recurring_id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT NOT NULL UNIQUE, amount REAL NOT NULL, category TEXT NOT NULL, payment_method TEXT, day_of_month INTEGER NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS mf_transactions (transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, scheme_name TEXT NOT NULL, yfinance_symbol TEXT NOT NULL, type TEXT NOT NULL, units REAL NOT NULL, nav REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS mf_sips (sip_id INTEGER PRIMARY KEY AUTOINCREMENT, scheme_name TEXT NOT NULL UNIQUE, yfinance_symbol TEXT NOT NULL, amount REAL NOT NULL, day_of_month INTEGER NOT NULL)""")
    conn.commit()
    _add_missing_columns(conn)

def _add_missing_columns(conn):
    """Handles database schema migrations by adding missing columns."""
    c = conn.cursor()
    # Migration 1: Add sector and market_cap to portfolio
    c.execute("PRAGMA table_info(portfolio)")
    columns = [info[1] for info in c.fetchall()]
    if 'sector' not in columns:
        c.execute("ALTER TABLE portfolio ADD COLUMN sector TEXT")
        logging.info("Added 'sector' column to 'portfolio' table.")
    if 'market_cap' not in columns:
        c.execute("ALTER TABLE portfolio ADD COLUMN market_cap TEXT")
        logging.info("Added 'market_cap' column to 'portfolio' table.")

    # Migration 2: Add type to expenses
    c.execute("PRAGMA table_info(expenses)")
    expense_columns = [info[1] for info in c.fetchall()]
    if 'type' not in expense_columns:
        c.execute("ALTER TABLE expenses ADD COLUMN type TEXT")
        logging.info("Added 'type' column to 'expenses' table.")
        c.execute("UPDATE expenses SET type = 'Expense' WHERE type IS NULL")
        logging.info("Set 'type' to 'Expense' for existing records.")

    # FIX: Add 'allocation_type' to fund_transactions table
    c.execute("PRAGMA table_info(fund_transactions)")
    fund_cols = [info[1] for info in c.fetchall()]
    if 'allocation_type' not in fund_cols:
        c.execute("ALTER TABLE fund_transactions ADD COLUMN allocation_type TEXT")
        logging.info("Added 'allocation_type' column to 'fund_transactions' table.")
        # Attempt to categorize existing withdrawals based on description heuristics
        c.execute("UPDATE fund_transactions SET allocation_type = 'Investment' WHERE type = 'Withdrawal' AND description LIKE '%Purchase%' AND description NOT LIKE 'ALLOCATION:%'")
        c.execute("UPDATE fund_transactions SET allocation_type = 'Trading' WHERE type = 'Withdrawal' AND description LIKE '%Trading%' AND description NOT LIKE 'ALLOCATION:%'")
        c.execute("UPDATE fund_transactions SET allocation_type = 'Mutual Fund' WHERE type = 'Withdrawal' AND description LIKE 'MF %' AND description NOT LIKE 'ALLOCATION:%'")
        c.execute("UPDATE fund_transactions SET allocation_type = 'Other/Expenses' WHERE type = 'Withdrawal' AND allocation_type IS NULL")
        c.execute("UPDATE fund_transactions SET allocation_type = 'Deposit' WHERE type = 'Deposit'")
        conn.commit()
        logging.info("Attempted to back-fill 'allocation_type' for existing data.")

    conn.commit()

DB_CONN = get_db_connection()
initialize_database(DB_CONN)

def update_funds_on_transaction(transaction_type, amount, description, date, allocation_type=None):
    """Inserts a new transaction into the fund_transactions table, now writing allocation_type directly."""

    # Clean up description if it contains the old ALLOCATION tag for new entries
    # NOTE: Since the old logic prepended ALLOCATION: to description, we must strip it for clean display
    if description and description.startswith("ALLOCATION:"):
        description = description.split(' - ', 1)[-1].strip()

    # Ensure allocation type is set for Deposits too, for completeness in filtering
    if allocation_type is None and transaction_type == 'Deposit':
        allocation_type = 'Deposit'

    c = DB_CONN.cursor()
    # The SQL INSERT statement now explicitly includes the allocation_type column
    c.execute("INSERT INTO fund_transactions (transaction_id, date, type, amount, description, allocation_type) VALUES (?, ?, ?, ?, ?, ?)",
              (str(uuid.uuid4()), date, transaction_type, amount, description, allocation_type))
    DB_CONN.commit()

# --- API & DATA FUNCTIONS ---
@st.cache_data(ttl=3600)
def search_for_ticker(company_name):
    """Searches for a stock ticker using a company name via Finnhub API."""
    try:
        api_key = st.secrets["api_keys"]["finnhub"]
        url = f"https://finnhub.io/api/v1/search?q={company_name}&token={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [f"{item.get('symbol')} - {item.get('description', 'N/A')}" for item in data.get("result", [])]
    except Exception as e:
        logging.error(f"Finnhub search failed: {e}")
        return []

@st.cache_data(ttl=600)
def fetch_stock_info(symbol):
    """Fetches key stock information including price, sector, and market cap using yfinance."""
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            ticker_str = symbol.replace('XNSE:', '') + '.NS' if 'XNSE:' in symbol else symbol
            ticker_obj = yf.Ticker(ticker_str)
            info = ticker_obj.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if price is None:
                data = ticker_obj.history(period='1d', auto_adjust=True)
                if not data.empty:
                    price = data['Close'].iloc[-1]
            if price:
                price = round(price, 2)

            sector = info.get('sector', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            if sector == 'N/A' and ('fundFamily' in info or 'category' in info):
                sector = info.get('fundFamily') or info.get('category')
            return {'price': price, 'sector': sector, 'market_cap': market_cap}
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries} to fetch info for {symbol} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logging.error(f"All {max_retries} attempts failed for {symbol}.")
                return {'price': None, 'sector': 'N/A', 'market_cap': 'N/A'}

@st.cache_data(ttl=3600)
def fetch_mf_schemes():
    """Fetches all mutual fund scheme codes and names using mftool."""
    try:
        mf = Mftool()
        schemes = mf.get_scheme_codes()
        return {v: k for k, v in schemes.items()}
    except Exception as e:
        logging.error(f"Failed to fetch mutual fund schemes: {e}")
        return {}

@st.cache_data(ttl=600)
def fetch_latest_mf_nav(scheme_code):
    """Fetches the latest NAV for a given scheme code."""
    try:
        mf = Mftool()
        data = mf.get_scheme_quote(scheme_code)
        if data and 'nav' in data and data['nav']:
            return float(data['nav'])
        return None
    except Exception as e:
        logging.error(f"Failed to fetch NAV for scheme {scheme_code}: {e}")
        return None

@st.cache_data(ttl=86400)
def get_mf_historical_data(scheme_code):
    """Fetches historical NAV data for a mutual fund."""
    try:
        mf = Mftool()
        data = mf.get_scheme_historical_nav(scheme_code)
        if not data or 'data' not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data['data'])
        df.rename(columns={'nav': 'NAV', 'date': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df['NAV'] = pd.to_numeric(df['NAV'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch historical data for scheme {scheme_code}: {e}")
        return pd.DataFrame()

def update_stock_data(symbol):
    """Downloads and saves historical stock data to the database."""
    try:
        ticker_str = symbol.replace('XNSE:', '') + '.NS' if 'XNSE:' in symbol else symbol
        today = datetime.date.today()
        data = yf.download(ticker_str, start="2020-01-01", end=today + datetime.timedelta(days=1), progress=False, auto_adjust=True)
        if data.empty:
            return False
        data.reset_index(inplace=True)
        data = data[["Date", "Close"]].rename(columns={"Date": "date_col", "Close": "close_price"})
        data["ticker"] = symbol
        data["date"] = data["date_col"].dt.strftime("%Y-%m-%d")
        c = DB_CONN.cursor()
        c.executemany("INSERT OR REPLACE INTO price_history (ticker, date, close_price) VALUES (?, ?, ?)", data[["ticker", "date", "close_price"]].to_records(index=False))
        DB_CONN.commit()
        return True
    except Exception as e:
        logging.error(f"YFinance update_stock_data failed for {symbol}: {e}")
        return False

def get_holdings_df(table_name):
    """Fetches and calculates current portfolio/trade holdings from the database."""
    if table_name == "trades":
        query = "SELECT p.symbol, p.buy_price, p.buy_date, p.quantity, p.target_price, p.stop_loss_price, h.close_price AS current_price FROM trades p LEFT JOIN price_history h ON p.symbol = h.ticker WHERE h.date = (SELECT MAX(date) FROM price_history WHERE ticker = p.symbol)"
    else:
        query = "SELECT p.ticker AS symbol, p.buy_price, p.buy_date, p.quantity, p.sector, p.market_cap, h.close_price AS current_price FROM portfolio p LEFT JOIN price_history h ON p.ticker = h.ticker WHERE h.date = (SELECT MAX(date) FROM price_history WHERE ticker = p.ticker)"
    try:
        df = pd.read_sql(query, DB_CONN)
        if df.empty:
            return pd.DataFrame()
        df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce').fillna(0).round(2)
        df["return_%"] = ((df["current_price"] - df["buy_price"]) / df["buy_price"] * 100).round(2)
        df["return_amount"] = ((df["current_price"] - df["buy_price"]) * df["quantity"]).round(2)
        df["invested_value"] = (df["buy_price"] * df["quantity"]).round(2)
        df["current_value"] = (df["current_price"] * df["quantity"]).round(2)
        if table_name == "trades":
            df["Expected RRR"] = np.where((df["buy_price"] - df["stop_loss_price"]) > 0, (df["target_price"] - df["buy_price"]) / (df["buy_price"] - df["stop_loss_price"]), np.inf).round(2)
        return df
    except Exception as e:
        logging.error(f"Error querying {table_name}: {e}")
        return pd.DataFrame()

def get_realized_df(table_name):
    """Fetches and calculates realized profits/losses from the database."""
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", DB_CONN)
        if df.empty:
            return pd.DataFrame()
        df["invested_value"] = (df["buy_price"] * df["quantity"]).round(2)
        df["realized_value"] = (df["sell_price"] * df["quantity"]).round(2)
        df["realized_profit_loss"] = (df["realized_value"] - df["invested_value"]).round(2)
        df["realized_return_pct"] = df["realized_return_pct"].round(2)
        if table_name == "exits":
            df["Expected RRR"] = np.where((df["buy_price"] - df["stop_loss_price"]) > 0, (df["target_price"] - df["buy_price"]) / (df["buy_price"] - df["stop_loss_price"]), np.inf).round(2)
            df["Actual RRR"] = np.where((df["buy_price"] - df["stop_loss_price"]) > 0, (df["sell_price"] - df["buy_price"]) / (df["buy_price"] - df["stop_loss_price"]), np.inf).round(2)
        return df
    except Exception as e:
        logging.error(f"Error querying {table_name}: {e}")
        return pd.DataFrame()

def _update_existing_portfolio_info():
    """Fetches and updates missing sector and market cap data for existing stocks."""
    c = DB_CONN.cursor()
    c.execute("SELECT ticker FROM portfolio WHERE sector IS NULL OR market_cap IS NULL OR sector = 'N/A' OR market_cap = 'N/A'")
    tickers_to_update = [row[0] for row in c.fetchall()]
    if tickers_to_update:
        for ticker in tickers_to_update:
            try:
                stock_info = fetch_stock_info(ticker)
                sector, market_cap = stock_info['sector'], stock_info['market_cap']
                if sector != 'N/A' or market_cap != 'N/A':
                    c.execute("UPDATE portfolio SET sector = ?, market_cap = ? WHERE ticker = ?", (sector, _categorize_market_cap(market_cap), ticker))
                    logging.info(f"Updated sector and market cap for {ticker}.")
            except Exception as e:
                logging.error(f"Failed to update info for {ticker}: {e}")
        DB_CONN.commit()

def _categorize_market_cap(market_cap_value):
    """Categorizes a market cap value into Large, Mid, or Small Cap."""
    if isinstance(market_cap_value, (int, float)):
        if market_cap_value >= 10000000000:
            return "Large Cap"
        elif market_cap_value >= 2000000000:
            return "Mid Cap"
        else:
            return "Small Cap"
    return "N/A"

# --- HELPER FUNCTIONS ---
def _process_recurring_expenses():
    """Adds recurring expenses to the database if not already logged for the current month."""
    c = DB_CONN.cursor()
    month_year = datetime.date.today().strftime("%Y-%m")
    try:
        recurring_df = pd.read_sql("SELECT * FROM recurring_expenses", DB_CONN)
        if recurring_df.empty:
            return
        logged_expenses_df = pd.read_sql(f"SELECT description FROM expenses WHERE date LIKE '{month_year}-%'", DB_CONN)
        logged_descriptions = logged_expenses_df['description'].tolist()
        for _, row in recurring_df.iterrows():
            marker = f"Recurring: {row['description']}"
            if marker in logged_descriptions:
                continue
            day = min(row['day_of_month'], pd.Timestamp(month_year).days_in_month)
            expense_date = f"{month_year}-{day:02d}"
            c.execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (str(uuid.uuid4()), expense_date, "Expense", round(row['amount'], 2), row['category'], row['payment_method'], marker))
            DB_CONN.commit()
            logging.info(f"Logged recurring expense: {row['description']}")
    except Exception as e:
        logging.error(f"Could not process recurring expenses: {e}")

def _process_mf_sips():
    """Automatically logs mutual fund SIP transactions if the date is passed."""
    c = DB_CONN.cursor()
    today = datetime.date.today()
    month_year = today.strftime("%Y-%m")
    try:
        sips_df = pd.read_sql("SELECT * FROM mf_sips", DB_CONN)
        if sips_df.empty:
            return
        for _, sip in sips_df.iterrows():
            day = min(sip['day_of_month'], pd.Timestamp.now().days_in_month)
            sip_date_this_month = today.replace(day=day).strftime('%Y-%m-%d')
            if datetime.datetime.strptime(sip_date_this_month, '%Y-%m-%d').date() <= today:
                existing_sip_tx = pd.read_sql("SELECT * FROM mf_transactions WHERE scheme_name = ? AND date LIKE ?",
                                              (sip['scheme_name'], f"{month_year}-%"))
                if existing_sip_tx.empty:
                    try:
                        nav = fetch_latest_mf_nav(sip['yfinance_symbol'])
                        if nav:
                            units = sip['amount'] / nav
                            # Pass allocation_type directly
                            update_funds_on_transaction("Withdrawal", round(sip['amount'], 2), f"MF SIP: {sip['scheme_name']}", sip_date_this_month, allocation_type="Mutual Fund")
                            c.execute("INSERT INTO mf_transactions (transaction_id, date, scheme_name, yfinance_symbol, type, units, nav) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                      (str(uuid.uuid4()), sip_date_this_month, sip['scheme_name'], sip['yfinance_symbol'], 'Purchase', round(units, 4), round(nav, 4)))
                            DB_CONN.commit()
                            logging.info(f"Auto-logged SIP for {sip['scheme_name']}")
                            st.sidebar.success(f"Auto-logged SIP for {sip['scheme_name']}")
                        else:
                            st.sidebar.warning(f"Could not auto-log SIP for {sip['scheme_name']}. NAV fetch failed.")
                    except Exception as e:
                        st.sidebar.warning(f"Could not auto-log SIP for {sip['scheme_name']}. An error occurred: {e}")
    except Exception as e:
        logging.error(f"Failed during MF SIP processing: {e}")

@st.cache_data(ttl=3600)
def get_benchmark_comparison_data(holdings_df, benchmark_choice):
    """Generates portfolio vs. benchmark return data for charting."""
    if holdings_df.empty:
        return pd.DataFrame()
    start_date = holdings_df['buy_date'].min() if not holdings_df.empty else datetime.date.today().strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date))
    benchmark_map = {'Nifty 50': '^NSEI', 'Nifty 100': '^CNX100', 'Nifty 200': '^CNX200', 'Nifty 500': '^CRSLDX'}
    selected_ticker = benchmark_map.get(benchmark_choice)
    if not selected_ticker:
        return pd.DataFrame()
    try:
        benchmark_df = yf.download(selected_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if benchmark_df.empty or 'Close' not in benchmark_df.columns:
            logging.error(f"yfinance returned empty or invalid data for benchmark {selected_ticker}")
            return pd.DataFrame()
        benchmarks = benchmark_df['Close'].copy()
        benchmarks.ffill(inplace=True)
        benchmark_returns = ((benchmarks / benchmarks.iloc[0] - 1) * 100).round(2)
        benchmark_returns.name = benchmark_choice
        all_tickers = holdings_df['symbol'].unique().tolist()
        price_data_query = f"""SELECT date, ticker, close_price FROM price_history WHERE ticker IN ({','.join(['?']*len(all_tickers))}) AND date >= ?"""
        all_prices = pd.read_sql(price_data_query, DB_CONN, params=[*all_tickers, start_date])
        all_prices['date'] = pd.to_datetime(all_prices['date'])
        price_pivot = all_prices.pivot(index='date', columns='ticker', values='close_price').ffill()
        price_pivot = price_pivot.reindex(date_range).ffill()
        daily_units = pd.DataFrame(0.0, index=date_range, columns=all_tickers)
        daily_invested = pd.DataFrame(0.0, index=date_range, columns=all_tickers)
        for _, row in holdings_df.iterrows():
            buy_date = pd.to_datetime(row['buy_date'])
            buy_date_index = daily_units.index.searchsorted(buy_date, side='left')
            daily_units.iloc[buy_date_index:, daily_units.columns.get_loc(row['symbol'])] = row['quantity']
            daily_invested.iloc[buy_date_index:, daily_invested.columns.get_loc(row['symbol'])] = row['quantity'] * row['buy_price']
        daily_market_value = (price_pivot * daily_units).sum(axis=1)
        total_daily_invested = daily_invested.sum(axis=1)
        total_daily_invested_clean = total_daily_invested.replace(0, np.nan).ffill()
        portfolio_return = ((daily_market_value - total_daily_invested) / total_daily_invested_clean * 100).rename('Portfolio').round(2)
        final_df = pd.concat([portfolio_return, benchmark_returns], axis=1).reset_index().rename(columns={'index': 'Date'})
        final_df = final_df.melt(id_vars='Date', var_name='Type', value_name='Return %').dropna()
        return final_df
    except Exception as e:
        logging.error(f"Failed to generate benchmark data: {e}", exc_info=True)
        return pd.DataFrame()

# --- NEW METRICS CALCULATION FUNCTIONS ---

def calculate_portfolio_metrics(holdings_df, realized_df, benchmark_choice):
    """Calculates alpha, beta, and max drawdown."""
    metrics = {
        'alpha': 'N/A', 'beta': 'N/A', 'max_drawdown': 'N/A'
    }
    if holdings_df.empty:
        return metrics
    start_date = holdings_df['buy_date'].min() if not holdings_df.empty else datetime.date.today().strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    all_tickers = holdings_df['symbol'].unique().tolist()
    price_data_query = f"""SELECT date, ticker, close_price FROM price_history WHERE ticker IN ({','.join(['?']*len(all_tickers))}) AND date >= ?"""
    all_prices = pd.read_sql(price_data_query, DB_CONN, params=[*all_tickers, start_date])
    all_prices['date'] = pd.to_datetime(all_prices['date'])
    price_pivot = all_prices.pivot(index='date', columns='ticker', values='close_price').ffill()
    daily_units = pd.DataFrame(0.0, index=price_pivot.index, columns=all_tickers)
    for _, row in holdings_df.iterrows():
        buy_date = pd.to_datetime(row['buy_date'])
        buy_date_index = price_pivot.index.searchsorted(buy_date, side='left')
        daily_units.iloc[buy_date_index:, daily_units.columns.get_loc(row['symbol'])] = row['quantity']
    portfolio_value = (price_pivot * daily_units).sum(axis=1).ffill().dropna()
    if portfolio_value.empty or len(portfolio_value) < 2:
        return metrics
    portfolio_returns = portfolio_value.pct_change().dropna()
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
    metrics['max_drawdown'] = round(max_drawdown, 2)
    benchmark_map = {'Nifty 50': '^NSEI', 'Nifty 100': '^CNX100', 'Nifty 200': '^CNX200', 'Nifty 500': '^CRSLDX'}
    selected_ticker = benchmark_map.get(benchmark_choice)
    if selected_ticker:
        try:
            benchmark_df = yf.download(selected_ticker, start=portfolio_value.index.min(), end=portfolio_value.index.max(), progress=False, auto_adjust=True)
            if not benchmark_df.empty:
                benchmark_returns = benchmark_df['Close'].pct_change().dropna().squeeze()
                combined_returns = pd.DataFrame({
                    'portfolio': portfolio_returns,
                    'benchmark': benchmark_returns
                }).dropna()
                if len(combined_returns) > 1:
                    cov_matrix = np.cov(combined_returns['portfolio'], combined_returns['benchmark'])
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                    metrics['beta'] = round(beta, 2)
                    excess_portfolio_return = combined_returns['portfolio'].mean()
                    excess_benchmark_return = combined_returns['benchmark'].mean()
                    risk_free_rate = 0
                    alpha = excess_portfolio_return - (risk_free_rate + beta * (excess_benchmark_return - risk_free_rate))
                    metrics['alpha'] = round(alpha * 252 * 100, 2)
        except Exception as e:
            logging.error(f"Failed to calculate Alpha/Beta for {selected_ticker}: {e}", exc_info=True)
    return metrics

def calculate_trading_metrics(realized_df):
    """Calculates win ratio, profit factor, and expectancy."""
    metrics = {
        'win_ratio': 'N/A', 'profit_factor': 'N/A', 'expectancy': 'N/A'
    }
    if realized_df.empty:
        return metrics
    winning_trades = realized_df[realized_df['realized_profit_loss'] > 0]
    losing_trades = realized_df[realized_df['realized_profit_loss'] <= 0]
    total_trades = len(realized_df)
    if total_trades > 0:
        win_ratio = (len(winning_trades) / total_trades) * 100
        metrics['win_ratio'] = round(win_ratio, 2)
    gross_profit = winning_trades['realized_profit_loss'].sum()
    gross_loss = abs(losing_trades['realized_profit_loss'].sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
        metrics['profit_factor'] = round(profit_factor, 2)
    if total_trades > 0:
        avg_win = winning_trades['realized_profit_loss'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['realized_profit_loss'].mean() if not losing_trades.empty else 0
        expectancy = (win_ratio / 100 * avg_win) + ((1 - win_ratio / 100) * avg_loss)
        metrics['expectancy'] = round(expectancy, 2)
    return metrics

def color_return_value(val):
    """Applies color to a cell based on its numerical value."""
    if val is None or not isinstance(val, (int, float)):
        return ''
    return 'color: green' if val >= 0 else 'color: red'

def set_page(page):
    """Sets the current page in session state."""
    st.session_state.page = page

def get_combined_returns():
    """Calculates and returns combined returns for all asset types."""
    inv_df = get_holdings_df("portfolio")
    inv_invested = inv_df['invested_value'].sum() if not inv_df.empty else 0
    inv_current = inv_df['current_value'].sum() if not inv_df.empty else 0
    trade_df = get_holdings_df("trades")
    trade_invested = trade_df['invested_value'].sum() if not trade_df.empty else 0
    trade_current = trade_df['current_value'].sum() if not trade_df.empty else 0
    mf_df = get_mf_holdings_df()
    mf_invested = mf_df['Investment'].sum() if not mf_df.empty else 0
    mf_current = mf_df['Current Value'].sum() if not mf_df.empty else 0

    inv_return_amount = round(float(inv_current - inv_invested), 2)
    inv_return_pct = round(float(inv_return_amount / inv_invested * 100), 2) if inv_invested > 0 else 0

    trade_return_amount = round(float(trade_current - trade_invested), 2)
    trade_return_pct = round(float(trade_return_amount / trade_invested * 100), 2) if trade_invested > 0 else 0

    mf_return_amount = round(float(mf_current - mf_invested), 2)
    mf_return_pct = round(float(mf_return_amount / mf_invested * 100), 2) if mf_invested > 0 else 0

    total_invested = inv_invested + trade_invested + mf_invested
    total_current = inv_current + trade_current + mf_current
    total_return_amount = round(float(total_current - total_invested), 2)
    total_return_pct = round(float(total_return_amount / total_invested * 100), 2) if total_invested > 0 else 0

    # Corrected method to calculate realized amounts
    realized_stocks_df = get_realized_df("realized_stocks")
    realized_exits_df = get_realized_df("exits")

    realized_inv = realized_stocks_df['realized_profit_loss'].sum() if not realized_stocks_df.empty else 0
    realized_trade = realized_exits_df['realized_profit_loss'].sum() if not realized_exits_df.empty else 0
    realized_mf = 0  # No realized MF table, so it remains 0

    return {
        "inv_return_amount": inv_return_amount,
        "inv_return_pct": inv_return_pct,
        "trade_return_amount": trade_return_amount,
        "trade_return_pct": trade_return_pct,
        "mf_return_amount": mf_return_amount,
        "mf_return_pct": mf_return_pct,
        "total_invested_value": total_invested,
        "total_current_value": total_current,
        "total_return_amount": total_return_amount,
        "total_return_pct": total_return_pct,
        "realized_inv": round(float(realized_inv), 2),
        "realized_trade": round(float(realized_trade), 2),
        "realized_mf": round(float(realized_mf), 2)
    }

def get_mf_holdings_df():
    """Calculates current mutual fund holdings from transaction data."""
    transactions_df = pd.read_sql("SELECT * FROM mf_transactions", DB_CONN)
    if transactions_df.empty:
        return pd.DataFrame()
    holdings = []
    unique_schemes = transactions_df['scheme_name'].unique()
    latest_navs = {code: fetch_latest_mf_nav(code) for code in transactions_df['yfinance_symbol'].unique()}
    for scheme in unique_schemes:
        scheme_tx = transactions_df[transactions_df['scheme_name'] == scheme].copy()
        purchases = scheme_tx[scheme_tx['type'] == 'Purchase']
        redemptions = scheme_tx[scheme_tx['type'] == 'Redemption']
        total_units = purchases['units'].sum() - redemptions['units'].sum()
        if total_units > 0.001:
            total_investment = (purchases['units'] * purchases['nav']).sum() - (redemptions['units'] * redemptions['nav']).sum()
            avg_nav = total_investment / total_units if total_units > 0 else 0
            code = scheme_tx['yfinance_symbol'].iloc[0]
            latest_nav = latest_navs.get(code) or 0
            current_value = total_units * latest_nav
            pnl = current_value - total_investment
            pnl_pct = (pnl / total_investment) * 100 if total_investment > 0 else 0
            holdings.append({
                "Scheme": scheme, "Units": round(total_units, 4), "Avg NAV": round(avg_nav, 4),
                "Latest NAV": round(latest_nav, 4), "Investment": round(total_investment, 2),
                "Current Value": round(current_value, 2), "P&L": round(pnl, 2), "P&L %": round(pnl_pct, 2),
                "yfinance_symbol": code
            })
    return pd.DataFrame(holdings)

def _calculate_mf_cumulative_return(transactions_df):
    """Calculates the cumulative return of a mutual fund portfolio over time."""
    if transactions_df.empty:
        return pd.DataFrame()

    all_schemes_daily_returns = []
    unique_schemes = transactions_df['scheme_name'].unique()

    for scheme_name in unique_schemes:
        scheme_tx_df = transactions_df[transactions_df['scheme_name'] == scheme_name].copy()
        scheme_tx_df['date'] = pd.to_datetime(scheme_tx_df['date'], format='%Y-%m-%d', errors='coerce')
        scheme_tx_df = scheme_tx_df.sort_values('date').reset_index(drop=True)

        scheme_code = scheme_tx_df['yfinance_symbol'].iloc[0]
        historical_data = get_mf_historical_data(scheme_code)

        if historical_data.empty:
            continue

        start_date = scheme_tx_df['date'].min()
        end_date = historical_data.index.max()
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        units = 0
        invested_amount = 0
        daily_records = []

        for date in all_dates:
            if date in historical_data.index:
                nav = historical_data.loc[date]['NAV']
                todays_tx = scheme_tx_df[scheme_tx_df['date'] == date]

                for _, tx_row in todays_tx.iterrows():
                    if tx_row['type'] == 'Purchase':
                        units += tx_row['units']
                        invested_amount += (tx_row['units'] * tx_row['nav'])
                    elif tx_row['type'] == 'Redemption':
                        units -= tx_row['units']
                        invested_amount -= (tx_row['units'] * tx_row['nav'])

                current_value = units * nav if units > 0 else 0

                if invested_amount > 0:
                    cumulative_return = ((current_value - invested_amount) / invested_amount) * 100
                else:
                    cumulative_return = 0

                daily_records.append({
                    'date': date,
                    'cumulative_return': cumulative_return,
                    'scheme_name': scheme_name
                })

        all_schemes_daily_returns.append(pd.DataFrame(daily_records))

    if not all_schemes_daily_returns:
        return pd.DataFrame()

    return pd.concat(all_schemes_daily_returns)

@st.cache_data(ttl=3600)
def get_withdrawal_allocation_df(fund_transactions_df):
    """Analyzes fund withdrawals by reading the dedicated 'allocation_type' column."""

    ALLOCATION_CATEGORIES = ['Investment', 'Trading', 'Mutual Fund', 'Other Expense']

    if fund_transactions_df.empty:
        # Guarantees a valid DataFrame structure for Altair
        return pd.DataFrame({'Category': ALLOCATION_CATEGORIES, 'Amount': [0.0]*len(ALLOCATION_CATEGORIES)})

    withdrawals_df = fund_transactions_df[fund_transactions_df['type'] == 'Withdrawal'].copy()

    # 1. Use the dedicated allocation_type column (pre-filled by migration/forms)
    # Filter out empty allocation types and map to 'Other/Expenses' if necessary.
    withdrawals_df['Category'] = withdrawals_df['allocation_type'].apply(
        lambda x: x if pd.notna(x) and x in ALLOCATION_CATEGORIES else 'Other/Expenses'
    )

    # 2. Group and sum amounts
    allocation_summary = withdrawals_df.groupby('Category')['amount'].sum().reset_index()
    allocation_summary.rename(columns={'amount': 'Amount'}, inplace=True)

    # Re-index to ensure the chart structure is complete
    full_categories_df = pd.DataFrame({'Category': ALLOCATION_CATEGORIES + ['Other/Expenses']})

    final_df = full_categories_df.merge(allocation_summary, on='Category', how='left').fillna(0)

    # Filter out zero-amount entries before returning to Altair, for clarity
    return final_df[final_df['Amount'] > 0]
# -----------------------------------------------------------------------------

def main_app():
    """Renders the main dashboard pages."""
    if "page" not in st.session_state:
        st.session_state.page = "home"

    pages = {
        "home": home_page,
        "investment": lambda: render_asset_page(PAGE_CONFIGS["investment"]),
        "trading": lambda: render_asset_page(PAGE_CONFIGS["trading"]),
        "funds": funds_page,
        "expense_tracker": expense_tracker_page,
        "mutual_fund": mutual_fund_page,
    }

    pages[st.session_state.page]()

    if st.session_state.page != "home":
        st.button("Back to Home", on_click=set_page, args=("home",))
    if st.sidebar.button("Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.page = "home"
        st.rerun()
    if st.sidebar.button("Clear Session State", type="secondary"):
        current_page = st.session_state.get("page", "home")
        for key in list(st.session_state.keys()):
            if key not in ['page', 'mf_all_schemes', 'logged_in']:
                del st.session_state[key]
        st.session_state.page = current_page
        st.rerun()

def home_page():
    """Renders the main home page."""
    st.title("Finance Dashboard")
    _update_existing_portfolio_info()
    returns_data = get_combined_returns()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Total Investment Value",
            value=f"₹{returns_data['total_invested_value']:,.2f}",
        )
    with col2:
        st.metric(
            label="Total Current Value",
            value=f"₹{returns_data['total_current_value']:,.2f}",
        )
    with col3:
        st.metric(
            label="Total Portfolio Return",
            value=f"₹{returns_data['total_return_amount']:,.2f}",
            delta=f"{returns_data['total_return_pct']:.2f}%"
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(
            label="Investment Return",
            value=f"₹{returns_data['inv_return_amount']:,.2f}",
            delta=f"{returns_data['inv_return_pct']:.2f}%",
            help=f"Realized P&L: ₹{returns_data['realized_inv']:,.2f}"
        )
    with col5:
        st.metric(
            label="Trading Return",
            value=f"₹{returns_data['trade_return_amount']:,.2f}",
            delta=f"{returns_data['trade_return_pct']:.2f}%",
            help=f"Realized P&L: ₹{returns_data['realized_trade']:,.2f}"
        )
    with col6:
        st.metric(
            label="Mutual Fund Return",
            value=f"₹{returns_data['mf_return_amount']:,.2f}",
            delta=f"{returns_data['mf_return_pct']:.2f}%",
            help=f"Realized P&L: ₹{returns_data['realized_mf']:,.2f}"
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.button("📈 Investment", use_container_width=True, on_click=set_page, args=("investment",))
        st.button("💰 Funds", use_container_width=True, on_click=set_page, args=("funds",))
    with col2:
        st.button("📊 Trading", use_container_width=True, on_click=set_page, args=("trading",))
        st.button("💸 Expense Tracker", use_container_width=True, on_click=set_page, args=("expense_tracker",))
    st.button("📚 Mutual Fund", use_container_width=True, on_click=set_page, args=("mutual_fund",))

    col_refresh, _ = st.columns([0.2, 0.8])
    with col_refresh:
        if st.button("Refresh Live Data", key="refresh_all_data"):
            with st.spinner("Fetching latest stock and mutual fund prices..."):
                all_tickers = pd.read_sql("SELECT ticker FROM portfolio UNION SELECT symbol FROM trades", DB_CONN)['ticker'].tolist()
                for symbol in all_tickers:
                    update_stock_data(symbol)
                mf_symbols = pd.read_sql("SELECT DISTINCT yfinance_symbol FROM mf_transactions", DB_CONN)['yfinance_symbol'].tolist()
                for symbol in mf_symbols:
                    fetch_latest_mf_nav(symbol)
            st.success("All live data refreshed!")
            st.rerun()

def funds_page():
    """Renders the Funds Management page."""
    c = DB_CONN.cursor()
    st.title("💰 Funds Management")
    st.sidebar.header("Add Transaction")

    # ---------------------------------------------
    # ALLOCATION OPTIONS FOR WITHDRAWAL FORM AND EDITING
    ALLOCATION_OPTIONS = ["Investment", "Trading", "Mutual Fund", "Other Expense"]
    # The complete list of values used in the allocation_type column (Deposits use 'Deposit')
    ALL_ALLOCATION_TYPES = ALLOCATION_OPTIONS + ["Deposit"]
    # ---------------------------------------------

    with st.sidebar.form("deposit_form", clear_on_submit=True):
        st.subheader("Add Deposit")
        deposit_date = st.date_input("Date", max_value=datetime.date.today(), key="deposit_date")
        deposit_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None)
        deposit_desc = st.text_input("Description", placeholder="e.g., Salary", value="")
        if st.form_submit_button("Add Deposit"):
            if deposit_amount and deposit_amount > 0:
                # Deposit: Pass allocation type as 'Deposit'
                update_funds_on_transaction("Deposit", round(deposit_amount, 2), deposit_desc, deposit_date.strftime("%Y-%m-%d"), allocation_type="Deposit")
                st.success("Deposit recorded!")
                st.rerun()
            else:
                st.warning("Deposit amount must be greater than zero.")

    with st.sidebar.form("withdrawal_form", clear_on_submit=True):
        st.subheader("Record Withdrawal")
        wd_date = st.date_input("Date", max_value=datetime.date.today(), key="wd_date")
        wd_amount = st.number_input("Amount", min_value=0.01, format="%.2f", key="wd_amount", value=None)

        # FIX: ADD ALLOCATION TYPE SELECTOR
        wd_allocation = st.selectbox("Allocation Type", options=["Select..."] + ALLOCATION_OPTIONS, index=0)

        wd_desc = st.text_input("Description", placeholder="e.g., Personal Use", value="")

        if st.form_submit_button("Record Withdrawal"):
            if wd_amount and wd_amount > 0 and wd_allocation != "Select...":
                # Withdrawal: Pass selected allocation type
                update_funds_on_transaction("Withdrawal", round(wd_amount, 2), wd_desc, wd_date.strftime("%Y-%m-%d"), allocation_type=wd_allocation)
                st.success("Withdrawal recorded!")
                st.rerun()
            elif wd_allocation == "Select...":
                st.warning("Please select an Allocation Type.")
            else:
                st.warning("Withdrawal amount must be greater than zero.")

    # Sorts by date (newest first), then by transaction_id (NEWEST entry first on the same date)
    # Changed query to select the new allocation_type column
    fund_df = pd.read_sql("SELECT transaction_id, date, type, amount, description, allocation_type FROM fund_transactions ORDER BY date DESC, transaction_id DESC", DB_CONN)

    if not fund_df.empty:
        fund_df['date'] = pd.to_datetime(fund_df['date'], format='%Y-%m-%d', errors='coerce')
        fund_df['balance'] = fund_df.apply(lambda row: row['amount'] if row['type'] == 'Deposit' else -row['amount'], axis=1)

        # NOTE: Cumulative sum still requires strict chronological (oldest to newest) order for calculation.
        chronological_df = fund_df.copy()
        # Primary sort by date (ASC), secondary sort by ID (ASC) to ensure correct chronological cumulative sum
        chronological_df.sort_values(['date', 'transaction_id'], ascending=[True, True], inplace=True)

        # Calculate cumulative balance on the chronologically sorted data
        chronological_df['cumulative_balance'] = chronological_df['balance'].cumsum()

        # Merge the cumulative balance back onto the display DF using the unique ID
        fund_df = fund_df.merge(
            chronological_df[['transaction_id', 'cumulative_balance']],
            on='transaction_id',
            how='left'
        )

        # Final sort for display (newest day first, NEWEST entry first within the day)
        fund_df.sort_values(['date', 'transaction_id'], ascending=[False, False], inplace=True)

        total_deposits, total_withdrawals = fund_df.loc[fund_df['type'] == 'Deposit', 'amount'].sum(), fund_df.loc[fund_df['type'] == 'Withdrawal', 'amount'].sum()
        available_capital = total_deposits - total_withdrawals

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Deposits", f"₹{total_deposits:,.2f}")
        col2.metric("Total Withdrawals", f"₹{total_withdrawals:,.2f}")
        col3.metric("Available Capital", f"₹{available_capital:,.2f}")

        st.divider()

        # --- NEW: FUND ALLOCATION CHART (Reading from allocation_type column) ---
        st.subheader("Capital Deployment Breakdown (Total Withdrawals)")
        allocation_df = get_withdrawal_allocation_df(fund_df)

        # Check if there is data to display beyond the empty initialization
        if allocation_df['Amount'].sum() > 0.01:

            # Filter out categories that are zero for the chart display clarity
            chart_data = allocation_df[allocation_df['Amount'] > 0.01].copy()

            base = alt.Chart(chart_data).encode(
                theta=alt.Theta("Amount", stack=True)
            ).properties(height=350)

            pie = base.mark_arc(outerRadius=120).encode(
                color=alt.Color("Category"),
                order=alt.Order("Amount", sort="descending"),
                tooltip=["Category", alt.Tooltip("Amount", format="₹,.2f")]
            )

            text = base.mark_text(radius=140).encode(
                text=alt.Text("Category:N"),
                order=alt.Order("Amount", sort="descending"),
                color=alt.value("black")
            ).transform_filter(alt.datum.Amount > chart_data['Amount'].sum() * 0.05) # Filter small slices

            st.altair_chart(pie + text, use_container_width=True)

            with st.expander("Show Withdrawal Allocation Data"):
                st.dataframe(chart_data, hide_index=True)
        else:
            st.info("No substantial withdrawal data available to create allocation breakdown. Please log a withdrawal using the sidebar and select an Allocation Type.")

        st.divider()
        st.subheader("Cumulative Fund Flow")


        # Cumulative Fund Flow Chart
        chart_df = chronological_df[['date', 'cumulative_balance']].drop_duplicates(subset=['date'], keep='last')
        chart = alt.Chart(chart_df).mark_line().encode(
            x=alt.X('date', title='Date'),
            y=alt.Y('cumulative_balance', title='Cumulative Balance (₹)'),
            tooltip=['date', 'cumulative_balance']
        ).properties(
            height=400
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        st.subheader("Transaction History")

        # --- FIX: RE-ENABLE EDITING FOR ALLOCATION_TYPE COLUMN ---
        edited_df = st.data_editor(
            fund_df[['transaction_id', 'date', 'type', 'amount', 'description', 'allocation_type']],
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "transaction_id": st.column_config.TextColumn("ID", disabled=True),
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["Deposit", "Withdrawal"], required=True), # Allow type editing
                # RE-ENABLE EDITING via selectbox for allocation_type
                "allocation_type": st.column_config.SelectboxColumn(
                    "Allocation Type",
                    options=ALL_ALLOCATION_TYPES,
                    required=True
                ),
            }
        )

        if st.button("Save Changes to Transactions"):
            # 1. Delete all existing records
            c.execute("DELETE FROM fund_transactions")

            # 2. Prepare the DataFrame for SQL insertion
            edited_df['date'] = edited_df['date'].astype(str)

            # 3. Insert the full, edited data back into the table
            edited_df.to_sql('fund_transactions', DB_CONN, if_exists='append', index=False)
            DB_CONN.commit()

            st.success("Funds transactions updated successfully! Rerunning to update the chart.")
            st.rerun()
    else:
        st.info("No fund transactions logged yet.")

def expense_tracker_page():
    """Renders the Expense Tracker page with enhanced category selection and new charts/metrics."""
    st.title("💸 Expense Tracker")
    _process_recurring_expenses()
    c = DB_CONN.cursor()

    # --- 1. Define and Fetch Categories (Using Session State for Persistence) ---
    if 'expense_categories_list' not in st.session_state:
        try:
            expense_categories = pd.read_sql("SELECT DISTINCT category FROM expenses WHERE type='Expense'", DB_CONN)['category'].tolist()
            default_categories = ["Food", "Transport", "Rent", "Utilities", "Shopping", "Entertainment", "Health", "Groceries", "Bills", "Education", "Travel", "Other"]
            all_categories = list(set([c for c in expense_categories if c and c != 'N/A'] + default_categories))
            st.session_state.expense_categories_list = sorted(all_categories)
        except pd.io.sql.DatabaseError:
            st.session_state.expense_categories_list = sorted(["Food", "Transport", "Rent", "Utilities", "Shopping", "Entertainment", "Health", "Groceries", "Bills", "Education", "Travel", "Other"])

    CATEGORIES = st.session_state.expense_categories_list

    PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking", "N/A"]

    view = st.radio("Select View", ["Dashboard", "Transaction History", "Manage Budgets", "Manage Recurring"], horizontal=True, label_visibility="hidden")

    st.sidebar.header("Add Transaction")
    with st.sidebar.form("new_transaction_form", clear_on_submit=True):
        trans_type = st.radio("Transaction Type", ["Expense", "Income"], key="trans_type")
        trans_date = st.date_input("Date", max_value=datetime.date.today(), value=datetime.date.today())
        trans_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None)

        # --- Enhanced Category Selection ---
        category_options = ['Select Category...'] + CATEGORIES

        selected_category = st.selectbox(
            "Select Category",
            options=category_options,
            index=0,
            key="selected_cat"
        )

        custom_category = st.text_input(
            "Or Enter New Category",
            help="Enter a custom category name, this will override the selection.",
            value="",
            key="custom_cat"
        )

        # Determine the final category used in the transaction
        if custom_category:
            final_cat = custom_category
        elif selected_category and selected_category != 'Select Category...':
            final_cat = selected_category
        else:
            final_cat = None
        # ------------------------------------

        if trans_type == "Income":
            trans_pm = "N/A"
        else:
            trans_pm = st.selectbox("Payment Method", options=[pm for pm in PAYMENT_METHODS if pm != 'N/A'], index=None)

        trans_desc = st.text_input("Description", value="")

        if st.form_submit_button("Add Transaction"):
            if trans_amount and final_cat and (trans_pm or trans_type == "Income"):
                c.execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (str(uuid.uuid4()), trans_date.strftime("%Y-%m-%d"), trans_type, round(trans_amount, 2), final_cat, trans_pm, trans_desc))
                DB_CONN.commit()
                st.success(f"{trans_type} added! Category: **{final_cat}**")

                # Add new custom category to session state immediately for future dropdown use
                if final_cat not in st.session_state.expense_categories_list:
                    st.session_state.expense_categories_list.append(final_cat)
                    st.session_state.expense_categories_list.sort()

                # Clear all general data caches to force dashboard charts (which read the DB) to refresh.
                st.cache_data.clear()

                # Clear input keys and rerun
                if "selected_cat" in st.session_state: del st.session_state["selected_cat"]
                if "custom_cat" in st.session_state: del st.session_state["custom_cat"]
                st.rerun()
            else:
                st.warning("Please fill all required fields (Amount and Category).")

    if view == "Dashboard":
        today = datetime.date.today()
        start_date_7days = today - datetime.timedelta(days=6)

        # Fetch all expense data for the current month and last 7 days
        month_year = today.strftime("%Y-%m")
        # Reading data directly from DB to ensure it's fresh after cache clear/rerun
        expenses_df = pd.read_sql(f"SELECT * FROM expenses WHERE date LIKE '{month_year}-%'", DB_CONN)
        all_time_expenses_df = pd.read_sql("SELECT * FROM expenses", DB_CONN)

        if all_time_expenses_df.empty:
            st.info("No expenses logged yet to display the dashboard.")
            return

        budgets_df = pd.read_sql(f"SELECT category, amount FROM budgets WHERE month_year = '{month_year}'", DB_CONN).set_index('category')
        inflows_df = expenses_df[expenses_df['type'] == 'Income']
        outflows_df = expenses_df[expenses_df['type'] == 'Expense']
        total_spent = outflows_df['amount'].sum()
        total_income = inflows_df['amount'].sum()
        total_budget = budgets_df['amount'].sum()
        net_flow = total_income - total_spent

        # --- Metric Calculation and Formatting ---

        # 1. Spent Breakdown (Outflows)
        spent_breakdown_df = outflows_df.groupby('payment_method')['amount'].sum().reset_index()
        spent_help_text = "\n".join([f"{row['payment_method']}: ₹{row['amount']:,.2f}" for _, row in spent_breakdown_df.iterrows()])

        # 2. Remaining Breakdown (Net Flow per Payment Method)
        flow_df = expenses_df.groupby(['type', 'payment_method'])['amount'].sum().unstack(level=0, fill_value=0).fillna(0)
        flow_df['Remaining'] = flow_df['Income'] - flow_df['Expense']

        # Filter out 'N/A' for payment methods and methods with near zero remaining
        remaining_breakdown_df = flow_df[(flow_df['Remaining'].abs() > 0.01) & (flow_df.index != 'N/A')].sort_values('payment_method').reset_index()

        # Build remaining help text
        remaining_help_text = "\n".join([f"{row['payment_method']}: ₹{row['Remaining']:,.2f}" for _, row in remaining_breakdown_df.iterrows()])
        if not remaining_help_text:
             remaining_help_text = "All flows balanced, or no categorized transactions."


        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Income", f"₹{total_income:,.2f}")

        # UPDATED: Total Spent with hover breakdown
        col2.metric("Total Spent this Month", f"₹{total_spent:,.2f}",
                     help=f"**Spent Breakdown (This Month):**\n{spent_help_text}")

        # Remaining Amount metric
        col3.metric("Remaining Amount", f"₹{net_flow:,.2f}",
                     delta_color="inverse" if net_flow >= 0 else "normal",
                     help=f"**Remaining Breakdown (This Month):**\n{remaining_help_text}")

        col4.metric("Total Budget for Month", f"₹{total_budget:,.2f}")
        st.divider()

        # --- 1. Daily Bar Chart (Last 7 Days) ---
        st.subheader("Daily Spending: Last 7 Days")

        # Prepare the 7-day data
        all_time_expenses_df['date'] = pd.to_datetime(all_time_expenses_df['date'])
        daily_spending = all_time_expenses_df[
            (all_time_expenses_df['type'] == 'Expense') &
            (all_time_expenses_df['date'].dt.date >= start_date_7days)
        ].groupby(all_time_expenses_df['date'].dt.date)['amount'].sum().reset_index()
        daily_spending.rename(columns={'date': 'Date', 'amount': 'Spent'}, inplace=True)

        # Create a full 7-day range for display purposes
        date_range = pd.date_range(start_date_7days, today)
        daily_df_full = pd.DataFrame({'Date': date_range.date})
        daily_df_full = daily_df_full.merge(daily_spending, on='Date', how='left').fillna(0)
        daily_df_full['Spent'] = daily_df_full['Spent'].round(2)
        daily_df_full['DayLabel'] = daily_df_full['Date'].apply(lambda x: "Today" if x == today else x.strftime('%a'))

        if not daily_df_full.empty:
            # Sort by date for display order (left to right, past to present)
            daily_df_full.sort_values('Date', ascending=True, inplace=True)

            bar_chart = alt.Chart(daily_df_full).mark_bar().encode(
                x=alt.X('DayLabel:N', sort=daily_df_full['DayLabel'].tolist(), title='Day'),
                y=alt.Y('Spent:Q', title='Amount Spent (₹)'),
                tooltip=['Date', alt.Tooltip('Spent', format='.2f', title='Total Spent (₹)')],
                color=alt.condition(
                    alt.datum.Date == today.strftime('%Y-%m-%d'), # Highlight today
                    alt.value('orange'),
                    alt.value('#4c78a8')
                )
            ).properties(height=300)
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No expense data for the last 7 days.")

        st.divider()

        # --- 2. Pie Chart Labels Update (FIXED) ---
        st.subheader(f"Category-wise Spending (Current Month: {month_year})")
        spending_by_category = outflows_df.groupby('category')['amount'].sum().reset_index()

        if not spending_by_category.empty:
            base = alt.Chart(spending_by_category).encode(
                theta=alt.Theta("amount", stack=True)
            )

            pie = base.mark_arc(outerRadius=120).encode(
                color=alt.Color("category"),
                # Tooltip corrected: shows category and amount/percentage on hover
                tooltip=["category", alt.Tooltip('amount', format='.2f', title='Amount (₹)')],
                order=alt.Order("amount", sort="descending")
            )

            # Text layer to show ONLY the category name (as requested)
            text = base.mark_text(radius=140).encode(
                text=alt.Text("category:N"), # Display ONLY the category name
                order=alt.Order("amount", sort="descending"),
                color=alt.value("black") # Set the color of the labels
            ).transform_filter(alt.datum.amount > spending_by_category['amount'].sum() * 0.05) # Only show labels for slices > 5%

            st.altair_chart(pie + text, use_container_width=True)
        else:
            st.info("No expenses logged for this month to plot.")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Monthly Spending Trend")
            monthly_spending_df = pd.read_sql("SELECT SUBSTR(date, 1, 7) AS month, SUM(amount) AS amount FROM expenses WHERE type='Expense' GROUP BY month ORDER BY month DESC", DB_CONN)
            if not monthly_spending_df.empty:
                bar_chart = alt.Chart(monthly_spending_df).mark_bar().encode(
                    x=alt.X('month', title='Month', sort='-x'),
                    y=alt.Y('amount', title='Total Spent (₹)'),
                    tooltip=[alt.Tooltip('month', title='Month'), alt.Tooltip('amount', format='.2f', title='Amount')]
                ).properties(height=350)
                st.altair_chart(bar_chart, use_container_width=True)
            else:
                st.info("No expenses logged for this month.")
        with col2:
            st.subheader("Inflow vs. Outflow")
            monthly_flows_df = pd.read_sql("SELECT SUBSTR(date, 1, 7) AS month, type, SUM(amount) AS amount FROM expenses GROUP BY month, type ORDER BY month DESC", DB_CONN)
            if not monthly_flows_df.empty:
                bar_chart = alt.Chart(monthly_flows_df).mark_bar().encode(
                    x=alt.X('month', title='Month', sort='-x'),
                    y=alt.Y('amount', title='Amount (₹)'),
                    color=alt.Color('type', title='Type', scale=alt.Scale(domain=['Income', 'Expense'], range=['#2ca02c', '#d62728'])),
                    tooltip=['month', alt.Tooltip('type', title='Type'), alt.Tooltip('amount', format='.2f', title='Amount')]
                ).properties(height=300)
                st.altair_chart(bar_chart, use_container_width=True)
            else:
                st.info("No income or expenses to compare.")

    elif view == "Transaction History":
        st.header("Transaction History")
        # Sorts by date (newest first), then by expense_id (NEWEST entry first on the same date)
        all_expenses_df = pd.read_sql("SELECT expense_id, date, type, amount, category, payment_method, description FROM expenses ORDER BY date DESC, expense_id DESC", DB_CONN)
        if not all_expenses_df.empty:
            all_expenses_df['date'] = pd.to_datetime(all_expenses_df['date'], format='%Y-%m-%d', errors='coerce')

            # The st.data_editor will display the data in this sorted order
            edited_df = st.data_editor(all_expenses_df, use_container_width=True, hide_index=True, num_rows="dynamic",
                                         column_config={"expense_id": st.column_config.TextColumn("ID", disabled=True),
                                                        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                                                        "type": st.column_config.SelectboxColumn("Type", options=["Expense", "Income"], required=True),
                                                        "category": st.column_config.TextColumn("Category", required=True),
                                                        "payment_method": st.column_config.SelectboxColumn("Payment Method", options=[pm for pm in PAYMENT_METHODS if pm != 'N/A'], required=True)})

            if st.button("Save Changes to Transactions"):
                c.execute("DELETE FROM expenses")
                edited_df['date'] = edited_df['date'].astype(str) # Convert back to string for SQLite
                edited_df.to_sql('expenses', DB_CONN, if_exists='append', index=False)
                DB_CONN.commit()
                st.success("Expenses updated successfully!")
                st.rerun()
        else:
            st.info("No transaction history to display.")
    elif view == "Manage Budgets":
        st.header("Set Your Monthly Budgets")
        budget_month_str = datetime.date.today().strftime("%Y-%m")
        st.info(f"You are setting the budget for: **{datetime.datetime.strptime(budget_month_str, '%Y-%m').strftime('%B %Y')}**")
        expense_categories_for_budget = sorted(list(pd.read_sql("SELECT DISTINCT category FROM expenses WHERE type='Expense'", DB_CONN)['category']) or CATEGORIES)
        existing_budgets = pd.read_sql(f"SELECT category, amount FROM budgets WHERE month_year = '{budget_month_str}'", DB_CONN)
        budget_df = pd.DataFrame({'category': expense_categories_for_budget, 'amount': [0.0] * len(expense_categories_for_budget)})
        if not existing_budgets.empty:
            budget_df = budget_df.set_index('category').combine_first(existing_budgets.set_index('category')).reset_index()
        edited_budgets = st.data_editor(budget_df, num_rows="dynamic", use_container_width=True, column_config={
            "category": st.column_config.TextColumn(label="Category", disabled=True),
            "amount": st.column_config.NumberColumn(label="Amount", min_value=0.0)
        })
        if st.button("Save Budgets"):
            for _, row in edited_budgets.iterrows():
                if row['amount'] >= 0 and row['category']:
                    c.execute("INSERT OR REPLACE INTO budgets (month_year, category, amount) VALUES (?, ?, ?)", (budget_month_str, row['category'], round(row['amount'], 2)))
            DB_CONN.commit()
            st.success("Budgets saved!")
            st.rerun()
    elif view == "Manage Recurring":
        st.header("Manage Recurring Expenses")
        st.info("Set up expenses that occur every month (e.g., rent, subscriptions). They will be logged automatically.")
        recurring_df = pd.read_sql("SELECT recurring_id, description, amount, category, payment_method, day_of_month FROM recurring_expenses", DB_CONN)
        edited_recurring = st.data_editor(recurring_df, num_rows="dynamic", use_container_width=True, column_config={
            "recurring_id": st.column_config.NumberColumn(disabled=True),
            "category": st.column_config.TextColumn("Category", required=True),
            "payment_method": st.column_config.SelectboxColumn("Payment Method", options=[pm for pm in PAYMENT_METHODS if pm != 'N/A'], required=True),
            "day_of_month": st.column_config.NumberColumn("Day of Month (1-31)", min_value=1, max_value=31, step=1, required=True)
        })
        if st.button("Save Recurring Rules"):
            c.execute("DELETE FROM recurring_expenses")
            for _, row in edited_recurring.iterrows():
                if row['description'] and row['amount'] > 0:
                    c.execute("INSERT INTO recurring_expenses (description, amount, category, payment_method, day_of_month) VALUES (?, ?, ?, ?, ?)", (row['description'], round(row['amount'], 2), row['category'], row['payment_method'], row['day_of_month']))
            DB_CONN.commit()
            st.success("Recurring expense rules saved!")
            st.rerun()

def mutual_fund_page():
    """Renders the Mutual Fund tracker page."""
    c = DB_CONN.cursor()
    st.title("📚 Mutual Fund Tracker")
    _process_mf_sips()
    key_prefix = "mf"

    # Read transactions_df at the beginning of the function to avoid UnboundLocalError
    transactions_df = pd.read_sql("SELECT transaction_id, date, scheme_name, yfinance_symbol, type, units, nav FROM mf_transactions ORDER BY date DESC", DB_CONN)

    view_options = ["Holdings", "Transaction History"]
    table_view = st.selectbox("View Options", view_options, key=f"{key_prefix}_table_view", label_visibility="hidden")

    if table_view == "Holdings":
        st.sidebar.header("Add Transaction")
        if f"{key_prefix}_all_schemes" not in st.session_state:
            st.session_state[f"{key_prefix}_all_schemes"] = fetch_mf_schemes()
            st.session_state[f"{key_prefix}_search_results"] = []
        with st.sidebar.form(f"{key_prefix}_search_form"):
            company_name = st.text_input("Search Fund Name", key=f"{key_prefix}_search_term_input")
            search_button = st.form_submit_button("Search")
            if search_button:
                if company_name:
                    filtered_schemes = {name: code for name, code in st.session_state[f"{key_prefix}_all_schemes"].items() if company_name.lower() in name.lower()}
                    st.session_state[f"{key_prefix}_search_results"] = [f"{name} ({code})" for name, code in filtered_schemes.items()]
                else:
                    st.session_state[f"{key_prefix}_search_results"] = []
                st.session_state[f"{key_prefix}_selected_scheme_code"] = None
                st.rerun()
        if st.session_state.get(f"{key_prefix}_search_results"):
            selected_result = st.sidebar.selectbox("Select Mutual Fund", options=[None] + st.session_state[f"{key_prefix}_search_results"], index=0, format_func=lambda x: "Select a fund..." if x is None else x)
            if selected_result and selected_result != st.session_state.get(f"{key_prefix}_selected_result"):
                st.session_state[f"{key_prefix}_selected_result"] = selected_result
                st.rerun()
        if st.session_state.get(f"{key_prefix}_selected_result"):
            selected_result = st.session_state[f"{key_prefix}_selected_result"]
            selected_name = selected_result.split(" (")[0]
            selected_code = selected_result.split(" (")[-1].replace(")", "")
            st.session_state[f"{key_prefix}_selected_scheme_code"] = selected_code
            with st.sidebar.form(f"{key_prefix}_add_details_form"):
                st.write(f"Selected: **{selected_name}**")
                mf_date = st.date_input("Date", max_value=datetime.date.today())
                mf_type = st.selectbox("Type", ["Purchase", "Redemption"])
                mf_units = st.number_input("Units", min_value=0.001, format="%.4f")
                mf_nav = st.number_input("NAV (Net Asset Value)", min_value=0.01, format="%.4f")
                mf_fee = st.number_input("Transaction Fee (₹)", min_value=0.00, format="%.2f", value=0.0)
                if st.form_submit_button("Add Transaction"):
                    if not (mf_units and mf_units > 0 and mf_nav and mf_nav > 0):
                        st.warning("Please fill all fields.")
                    else:
                        amount = mf_units * mf_nav
                        funds_change_type = "Withdrawal" if mf_type == "Purchase" else "Deposit"

                        # Use allocation_type for Funds page chart
                        allocation_type = "Mutual Fund" if funds_change_type == "Withdrawal" else "Deposit"
                        description = f"MF {mf_type}: {selected_name}" + (" (incl. fees)" if mf_type == "Purchase" else " (after fees)")

                        update_funds_on_transaction(funds_change_type, round(amount + (mf_fee if mf_type == "Purchase" else -mf_fee), 2), description, mf_date.strftime("%Y-%m-%d"), allocation_type=allocation_type) # Updated
                        c.execute("INSERT INTO mf_transactions (transaction_id, date, scheme_name, yfinance_symbol, type, units, nav) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                  (str(uuid.uuid4()), mf_date.strftime('%Y-%m-%d'), selected_name, selected_code, mf_type, round(mf_units, 4), round(mf_nav, 4)))
                        DB_CONN.commit()
                        st.success(f"{mf_type} of {selected_name} logged!")
                        st.session_state[f"{key_prefix}_selected_result"] = None
                        st.rerun()
        st.divider()

        holdings_df = get_mf_holdings_df()
        if not holdings_df.empty:
            total_investment = holdings_df['Investment'].sum()
            total_current_value = holdings_df['Current Value'].sum()
            total_pnl = holdings_df['P&L'].sum()
            total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Investment", f"₹{total_investment:,.2f}")
            col2.metric("Current Value", f"₹{total_current_value:,.2f}")
            col3.metric("Total P&L", f"₹{total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")
            st.divider()
            with st.expander("View Detailed Holdings"):
                styled_df = holdings_df.drop(columns=['yfinance_symbol']).style.map(color_return_value, subset=['P&L %']).format({
                    "Avg NAV": "₹{:.4f}", "Latest NAV": "₹{:.4f}", "Investment": "₹{:.2f}",
                    "Current Value": "₹{:.2f}", "P&L": "₹{:.2f}", "P&L %": "{:.2f}%"
                })
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.header("Return Chart")

            all_schemes = transactions_df['scheme_name'].unique().tolist() if not transactions_df.empty else []
            selected_schemes = st.multiselect("Select schemes to compare", options=all_schemes, default=all_schemes)

            if selected_schemes:
                filtered_transactions = transactions_df[transactions_df['scheme_name'].isin(selected_schemes)]
                cumulative_return_df = _calculate_mf_cumulative_return(filtered_transactions)

                if not cumulative_return_df.empty:
                    cumulative_return_df['date'] = pd.to_datetime(cumulative_return_df['date'])

                    line_chart = alt.Chart(cumulative_return_df).mark_line().encode(
                        x=alt.X('date:T', title='Date'),
                        y=alt.Y('cumulative_return:Q', title='Cumulative Return (%)'),
                        color='scheme_name:N',
                        tooltip=[
                            alt.Tooltip('scheme_name', title='Scheme'),
                            alt.Tooltip('date', title='Date', format='%Y-%m-%d'),
                            alt.Tooltip('cumulative_return', title='Return %', format=".2f")
                        ]
                    ).properties(height=400).interactive()

                    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="gray", strokeDash=[3,3]).encode(y='y')
                    st.altair_chart(line_chart + zero_line, use_container_width=True)
                else:
                    st.info("Not enough data to generate the chart for the selected schemes.")
            else:
                st.info("No schemes selected to display the chart.")
        else:
            st.info("No mutual fund holdings to display.")

    elif table_view == "Transaction History":
        if not transactions_df.empty:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'], format='%Y-%m-%d', errors='coerce')
            st.subheader("Edit Mutual Fund Transactions")
            edited_df = st.data_editor(transactions_df, use_container_width=True, hide_index=True, num_rows="dynamic",
                                         column_config={"transaction_id": st.column_config.TextColumn("ID", disabled=True),
                                                        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                                                        "scheme_name": st.column_config.TextColumn("Scheme Name", required=True),
                                                        "yfinance_symbol": st.column_config.TextColumn("YF Symbol", required=True),
                                                        "type": st.column_config.SelectboxColumn("Type", options=["Purchase", "Redemption"], required=True),
                                                        "units": st.column_config.NumberColumn("Units", min_value=0.0001, required=True),
                                                        "nav": st.column_config.NumberColumn("NAV", min_value=0.01, required=True)
                                                         })

            if st.button("Save Mutual Fund Changes"):
                c.execute("DELETE FROM mf_transactions")
                edited_df['date'] = edited_df['date'].astype(str)
                edited_df.to_sql('mf_transactions', DB_CONN, if_exists='append', index=False)
                DB_CONN.commit()
                st.success("Mutual Fund transactions updated successfully!")
                st.rerun()
        else:
            st.info("No mutual fund transactions logged yet.")


PAGE_CONFIGS = {
    "investment": {
        "title": "📈 Investment Portfolio",
        "asset_table": "portfolio",
        "realized_table": "realized_stocks",
        "asset_col": "ticker",
        "asset_name": "Stock",
        "asset_name_plural": "Stocks",
        "key_prefix": "inv"
    },
    "trading": {
        "title": "📊 Trading Book",
        "asset_table": "trades",
        "realized_table": "exits",
        "asset_col": "symbol",
        "asset_name": "Trade",
        "asset_name_plural": "Trades",
        "key_prefix": "trade"
    }
}

def render_asset_page(config):
    """Renders the Investment and Trading pages."""
    c = DB_CONN.cursor()
    key_prefix = config['key_prefix']
    is_trading_section = key_prefix == 'trade'
    st.title(config["title"])

    st.sidebar.header(f"Add {config['asset_name']}")
    with st.sidebar.form(f"{key_prefix}_add_form"):
        company_name = st.text_input(f"{config['asset_name']} Name", value="", key=f"{key_prefix}_add_company_name")
        search_button = st.form_submit_button("Search")
    if search_button and company_name:
        st.session_state[f"{key_prefix}_search_results"] = search_for_ticker(company_name)
        st.session_state[f"{key_prefix}_selected_symbol"] = None
        st.rerun()
    if st.session_state.get(f"{key_prefix}_search_results"):
        results = st.session_state[f"{key_prefix}_search_results"]
        symbols_only = [res.split(" - ")[0] for res in results]
        selected_symbol_from_search = st.sidebar.selectbox(
            f"Select {config['asset_name']} Symbol",
            options=[None] + symbols_only,
            index=0,
            key=f"{key_prefix}_select_symbol",
            format_func=lambda x: "Select a stock..." if x is None else x
        )
        if selected_symbol_from_search and selected_symbol_from_search != st.session_state.get(f"{key_prefix}_selected_symbol"):
            st.session_state[f"{key_prefix}_selected_symbol"] = selected_symbol_from_search
            st.rerun()
    if st.session_state.get(f"{key_prefix}_selected_symbol"):
        with st.sidebar.form(f"{key_prefix}_add_details_form"):
            symbol = st.session_state[f"{key_prefix}_selected_symbol"]
            st.write(f"Selected: **{symbol}**")
            stock_info = fetch_stock_info(symbol)
            current_price = stock_info['price']
            sector = stock_info['sector']
            market_cap = stock_info['market_cap']
            currency = "₹" if ".NS" in symbol else "$"
            if current_price:
                st.info(f"Current Price: {currency}{current_price:,.2f}")
            else:
                st.warning("Could not fetch current price.")
            buy_price = st.number_input(f"Buy Price ({currency})", min_value=0.01, format="%.2f", key=f"{key_prefix}_buy_price")
            buy_date = st.date_input("Buy Date", max_value=datetime.date.today(), key=f"{key_prefix}_buy_date")
            quantity = st.number_input("Quantity", min_value=1, step=1, key=f"{key_prefix}_buy_quantity")
            transaction_fee = st.number_input("Transaction Fee (₹)", min_value=0.00, format="%.2f", key=f"{key_prefix}_buy_transaction_fee", value=0.0)
            if not is_trading_section:
                st.text_input("Sector", value=sector, key=f"{key_prefix}_sector", disabled=True)
                st.text_input("Market Cap", value=_categorize_market_cap(market_cap) if market_cap != 'N/A' else 'N/A', key=f"{key_prefix}_market_cap", disabled=True)
            else:
                target_price = st.number_input("Target Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_target_price")
                stop_loss_price = st.number_input("Stop Loss Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_stop_loss_price")
            add_button = st.form_submit_button(f"Add to {config['asset_name_plural']}")
            if add_button:
                if not (buy_price and buy_price > 0 and quantity and quantity > 0):
                    st.error("Buy Price and Quantity must be positive.")
                elif update_stock_data(symbol):
                    total_cost = (buy_price * quantity) + transaction_fee
                    c.execute(f"SELECT * FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol,))
                    existing = c.fetchone()
                    if existing:
                        old_buy_price, old_quantity = existing[1], existing[3]
                        old_total_cost = old_buy_price * old_quantity
                        new_quantity = old_quantity + quantity
                        new_avg_price = (old_total_cost + total_cost - transaction_fee) / new_quantity
                        if is_trading_section:
                            c.execute(f"UPDATE {config['asset_table']} SET buy_price=?, quantity=?, target_price=?, stop_loss_price=? WHERE {config['asset_col']}=?", (round(new_avg_price, 2), new_quantity, round(target_price, 2), round(stop_loss_price, 2), symbol))
                            allocation_type = "Trading"
                        else:
                            c.execute(f"UPDATE {config['asset_table']} SET buy_price=?, quantity=?, sector=?, market_cap=? WHERE {config['asset_col']}=?", (round(new_avg_price, 2), new_quantity, sector, _categorize_market_cap(market_cap), symbol))
                            allocation_type = "Investment"
                        update_funds_on_transaction("Withdrawal", round(total_cost, 2), f"Purchase {quantity} units of {symbol}", buy_date.strftime("%Y-%m-%d"), allocation_type=allocation_type) # Updated
                        st.success(f"Updated {symbol}. New quantity: {new_quantity}, New avg. price: {currency}{new_avg_price:,.2f}")
                    else:
                        if is_trading_section:
                            c.execute(f"INSERT INTO {config['asset_table']} ({config['asset_col']}, buy_price, buy_date, quantity, target_price, stop_loss_price) VALUES (?, ?, ?, ?, ?, ?)", (symbol, round(buy_price, 2), buy_date.strftime("%Y-%m-%d"), quantity, round(target_price, 2), round(stop_loss_price, 2)))
                            allocation_type = "Trading"
                        else:
                            c.execute(f"INSERT INTO {config['asset_table']} ({config['asset_col']}, buy_price, buy_date, quantity, sector, market_cap) VALUES (?, ?, ?, ?, ?, ?)", (symbol, round(buy_price, 2), buy_date.strftime("%Y-%m-%d"), quantity, sector, _categorize_market_cap(market_cap)))
                            allocation_type = "Investment"
                        update_funds_on_transaction("Withdrawal", round(total_cost, 2), f"Purchase {quantity} units of {symbol}", buy_date.strftime("%Y-%m-%d"), allocation_type=allocation_type) # Updated
                        st.success(f"{symbol} added successfully!")
                    DB_CONN.commit()
                    st.rerun()
                else:
                    st.error(f"Failed to fetch historical data for {symbol}. Cannot add.")

    st.sidebar.header(f"Sell {config['asset_name']}")
    all_symbols = pd.read_sql(f"SELECT {config['asset_col']} FROM {config['asset_table']}", DB_CONN)[config['asset_col']].tolist()
    if all_symbols:
        selected_option = st.sidebar.selectbox(
            f"Select {config['asset_name']} to Sell",
            options=[None] + all_symbols,
            index=0,
            key=f"{key_prefix}_sell_symbol_selector",
            format_func=lambda x: "Select a stock..." if x is None else x
        )
        available_qty = 1
        if selected_option:
            symbol_to_sell = selected_option
            c.execute(f"SELECT quantity FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol_to_sell,))
            result = c.fetchone()
            if result:
                available_qty = result[0]
                st.sidebar.info(f"Available to sell: {available_qty} units of {symbol_to_sell}")
        else:
            symbol_to_sell = None
        with st.sidebar.form(f"{key_prefix}_sell_form"):
            is_disabled = not symbol_to_sell
            sell_qty = st.number_input("Quantity to Sell", min_value=1, max_value=available_qty, step=1, key=f"{key_prefix}_sell_qty", disabled=is_disabled)
            sell_price = st.number_input("Sell Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_sell_price", disabled=is_disabled)
            sell_date = st.date_input("Sell Date", max_value=datetime.date.today(), key=f"{key_prefix}_sell_date", disabled=is_disabled)
            sell_transaction_fee = st.number_input("Transaction Fee (₹)", min_value=0.00, format="%.2f", key=f"{key_prefix}_sell_transaction_fee", disabled=is_disabled, value=0.0)
            sell_button = st.form_submit_button(f"Sell {config['asset_name']}")
            if sell_button:
                if not symbol_to_sell:
                    st.warning(f"Please select a {config['asset_name']} to sell.")
                elif not (sell_price and sell_price > 0):
                    st.error("Sell price must be greater than zero.")
                elif not (sell_qty and sell_qty > 0):
                    st.error("Quantity to sell must be positive.")
                else:
                    if is_trading_section:
                        c.execute(f"SELECT buy_price, buy_date, quantity, target_price, stop_loss_price FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol_to_sell,))
                        buy_price, buy_date, current_qty, target_price, stop_loss_price = c.fetchone()
                        allocation_type = "Trading"
                    else:
                        c.execute(f"SELECT buy_price, buy_date, quantity FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol_to_sell,))
                        buy_price, buy_date, current_qty = c.fetchone()
                        allocation_type = "Investment"

                    realized_return = ((sell_price - buy_price) / buy_price * 100)
                    transaction_id = str(uuid.uuid4())

                    if is_trading_section:
                        c.execute(f"INSERT INTO {config['realized_table']} (transaction_id, {config['asset_col']}, buy_price, buy_date, quantity, sell_price, sell_date, realized_return_pct, target_price, stop_loss_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                  (transaction_id, symbol_to_sell, round(buy_price, 2), buy_date, sell_qty, round(sell_price, 2), sell_date.strftime("%Y-%m-%d"), round(realized_return, 2), round(target_price, 2), round(stop_loss_price, 2)))
                    else:
                        c.execute(f"INSERT INTO {config['realized_table']} (transaction_id, {config['asset_col']}, buy_price, buy_date, quantity, sell_price, sell_date, realized_return_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                  (transaction_id, symbol_to_sell, round(buy_price, 2), buy_date, sell_qty, round(sell_price, 2), sell_date.strftime("%Y-%m-%d"), round(realized_return, 2)))

                    # Note: Sales are Deposits to the cash account, allocation type is 'Deposit'
                    update_funds_on_transaction("Deposit", round((sell_price * sell_qty) - sell_transaction_fee, 2), f"Sale of {sell_qty} units of {symbol_to_sell}", sell_date.strftime("%Y-%m-%d"), allocation_type="Deposit")

                    if sell_qty == current_qty:
                        c.execute(f"DELETE FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol_to_sell,))
                    else:
                        c.execute(f"UPDATE {config['asset_table']} SET quantity=? WHERE {config['asset_col']}=?", (current_qty - sell_qty, symbol_to_sell))
                    DB_CONN.commit()
                    st.success(f"Sold {sell_qty} units of {symbol_to_sell}.")
                    st.rerun()
    else:
        st.sidebar.info(f"No open {config['asset_name_plural'].lower()}.")

    view_options = ["Holdings", "Exited Positions"] if not is_trading_section else ["Open Trades", "Closed Trades"]
    table_view = st.selectbox("View Options", view_options, key=f"{key_prefix}_table_view", label_visibility="hidden")
    if table_view == view_options[0]:
        holdings_df = pd.read_sql(f"SELECT * FROM {config['asset_table']}", DB_CONN)

        if not holdings_df.empty:
            df_to_display = get_holdings_df(config['asset_table'])

            total_invested, total_current = df_to_display['invested_value'].sum(), df_to_display['current_value'].sum()
            total_return_amount = (total_current - total_invested).round(2)
            total_return_percent = (total_return_amount / total_invested * 100).round(2) if total_invested > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Investment", f"₹{total_invested:,.2f}")
            with col2:
                st.metric("Current Value", f"₹{total_current:,.2f}")
            with col3:
                st.metric("Total Return", f"₹{total_return_amount:,.2f}", f"{total_return_percent:.2f}%")

            if not is_trading_section:
                benchmark_choice = 'Nifty 50'
                metrics = calculate_portfolio_metrics(df_to_display, pd.DataFrame(), benchmark_choice)
                col_alpha, col_beta, col_drawdown = st.columns(3)
                with col_alpha:
                    st.metric("Alpha", f"{metrics['alpha']}%")
                with col_beta:
                    st.metric("Beta", f"{metrics['beta']}")
                with col_drawdown:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']}%")

            st.divider()
            with st.expander("View Detailed Holdings"):
                column_rename = {
                    'symbol': 'Stock Name', 'ticker': 'Stock Name', 'buy_price': 'Buy Price', 'buy_date': 'Buy Date', 'quantity': 'Quantity',
                    'sector': 'Sector', 'market_cap': 'Market Cap', 'current_price': 'Current Price', 'return_%': 'Return (%)',
                    'return_amount': 'Return (Amount)', 'invested_value': 'Investment Value', 'current_value': 'Current Value',
                    'target_price': 'Target Price', 'stop_loss_price': 'Stop Loss'
                }
                df_to_style = df_to_display.rename(columns=column_rename)
                if not is_trading_section:
                    df_to_style = df_to_style.drop(columns=['Target Price', 'Stop Loss', 'Expected RRR'], errors='ignore')
                styled_holdings_df = df_to_style.style.map(color_return_value, subset=['Return (%)']).format({
                    'Buy Price': '₹{:.2f}', 'Current Price': '₹{:.2f}', 'Return (Amount)': '₹{:.2f}',
                    'Investment Value': '₹{:.2f}', 'Current Value': '₹{:.2f}', 'Return (%)': '{:.2f}%',
                    'Target Price': '₹{:.2f}', 'Stop Loss': '₹{:.2f}', 'Buy Date': lambda t: datetime.datetime.strptime(t, "%Y-%m-%d").strftime("%d/%m/%Y"),
                    'Expected RRR': '{:.2f}'
                })
                st.dataframe(styled_holdings_df, use_container_width=True, hide_index=True)
            st.header("Return Chart")
            all_symbols_list = df_to_display["symbol"].tolist()
            selected_symbols = st.multiselect("Select assets for return chart", all_symbols_list, default=all_symbols_list, key=f"{key_prefix}_perf_symbols")
            chart_data = []
            for symbol in selected_symbols:
                asset_info = df_to_display.loc[df_to_display["symbol"] == symbol].iloc[0]
                history_df = pd.read_sql("SELECT date, close_price FROM price_history WHERE ticker=? AND date>=? ORDER BY date ASC", DB_CONN, params=(symbol, asset_info["buy_date"]))
                if not history_df.empty:
                    history_df["return_%"] = ((history_df["close_price"] - asset_info["buy_price"]) / asset_info["buy_price"] * 100).round(2)
                    history_df["symbol"] = symbol
                    history_df['date'] = pd.to_datetime(history_df['date'])
                    chart_data.append(history_df)
            if chart_data:
                full_chart_df = pd.concat(chart_data)
                chart = alt.Chart(full_chart_df).mark_line().encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('return_%:Q', title='Return %'),
                    color='symbol:N',
                    tooltip=['symbol', 'date', alt.Tooltip('return_%', format=".2f")]
                ).properties(height=300).interactive()
                zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="gray", strokeDash=[3,3]).encode(y='y')
                st.altair_chart(chart + zero_line, use_container_width=True)
            else:
                st.info("No data to display for selected assets.")
        else:
            st.info(f"No {view_options[0].lower()} to display. Add a {config['asset_name'].lower()} from the sidebar.")
    elif table_view == view_options[1]:
        realized_df = get_realized_df(config['realized_table'])
        if not realized_df.empty:
            if is_trading_section:
                trading_metrics = calculate_trading_metrics(realized_df)
                col1, col2, col3 = st.columns(3)
                col1.metric("Win Ratio", f"{trading_metrics['win_ratio']}%")
                col2.metric("Profit Factor", f"{trading_metrics['profit_factor']}")
                col3.metric("Expectancy", f"₹{trading_metrics['expectancy']}")
            st.divider()
            with st.expander("View Detailed Realized Positions"):
                df_to_style = realized_df.drop(columns=['transaction_id'], errors='ignore')
                column_rename = {
                    'ticker': 'Stock Name', 'symbol': 'Stock Name', 'buy_price': 'Buy Price', 'buy_date': 'Buy Date',
                    'sell_price': 'Sell Price', 'sell_date': 'Sell Date', 'quantity': 'Quantity',
                    'realized_return_pct': 'Return (%)', 'realized_profit_loss': 'P/L (Amount)',
                    'invested_value': 'Investment Value', 'realized_value': 'Realized Value',
                    'target_price': 'Target Price', 'stop_loss_price': 'Stop Loss'
                }
                styled_realized_df = df_to_style.rename(columns=column_rename).style.map(color_return_value, subset=['Return (%)']).format({
                    'Buy Price': '₹{:.2f}', 'Sell Price': '₹{:.2f}', 'P/L (Amount)': '₹{:.2f}',
                    'Investment Value': '₹{:.2f}', 'Realized Value': '₹{:.2f}', 'Return (%)': '{:.2f}%',
                    'Target Price': '₹{:.2f}', 'Stop Loss': '₹{:.2f}', 'Buy Date': lambda t: datetime.datetime.strptime(t, "%Y-%m-%d").strftime("%d/%m/%Y"),
                    'Sell Date': lambda t: datetime.datetime.strptime(t, "%Y-%m-%d").strftime("%d/%m/%Y"),
                    'Expected RRR': '{:.2f}', 'Actual RRR': '{:.2f}'
                })
                st.dataframe(styled_realized_df, use_container_width=True, hide_index=True)
            st.header("Return Chart")
            realized_df['color'] = realized_df['realized_return_pct'].apply(lambda x: 'Profit' if x >= 0 else 'Loss')
            base = alt.Chart(realized_df).encode(
                x=alt.X(config['asset_col'], sort=None, title="Stock Name"),
                tooltip=[config['asset_col'], alt.Tooltip('realized_return_pct', title='Return %', format=".2f"), alt.Tooltip('realized_profit_loss', title='P/L (₹)', format=".2f")]
            )
            bars = base.mark_bar().encode(
                y=alt.Y('realized_return_pct', title='Return (%)'),
                color=alt.Color('color', scale=alt.Scale(domain=['Profit', 'Loss'], range=['#2ca02c', '#d62728']), legend=None)
            )
            st.altair_chart(bars, use_container_width=True)
        else:
            st.info(f"No {view_options[1].lower()} to display.")

    current_holdings = get_holdings_df(config['asset_table'])
    if (table_view == view_options[0] and not current_holdings.empty) or (is_trading_section and table_view == view_options[1] and not get_realized_df(config['realized_table']).empty):
        st.divider()
        st.header("Performance vs Benchmark")
        benchmark_choice = st.selectbox("Select Benchmark", ['Nifty 50', 'Nifty 100', 'Nifty 200', 'Nifty 500'], key=f"{key_prefix}_benchmark_choice")
        with st.spinner("Loading Benchmark Data..."):
            if table_view == view_options[0]:
                holdings_df = current_holdings
            else:
                realized_df = get_realized_df(config['realized_table'])
                holdings_df = realized_df.rename(columns={'symbol': 'ticker', 'realized_value': 'current_value', 'realized_profit_loss': 'return_amount', 'realized_return_pct': 'return_%'})
                holdings_df['buy_price'] = holdings_df['invested_value'] / holdings_df['quantity']
                holdings_df['symbol'] = holdings_df['ticker']

            benchmark_data = get_benchmark_comparison_data(holdings_df, benchmark_choice)
            if not benchmark_data.empty:
                benchmark_chart = alt.Chart(benchmark_data).mark_line().encode(
                    x=alt.X('Date:T', title='Date'),
                    y=alt.Y('Return %:Q', title='Total Return %'),
                    color=alt.Color('Type:N', title='Legend')
                ).properties(height=300).interactive()
                zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="gray", strokeDash=[3,3]).encode(y='y')
                st.altair_chart(benchmark_chart + zero_line, use_container_width=True)
            else:
                st.warning("Could not generate benchmark data. Ensure you have at least one position.")
    else:
        st.info("No data to compare against a benchmark.")

# --- MAIN APP LOGIC ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
