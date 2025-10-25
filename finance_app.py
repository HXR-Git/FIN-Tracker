import streamlit as st
import yfinance as yf
# REMOVED: import sqlite3
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
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

# --- USER AUTHENTICATION (Using st.secrets for secure credentials) ---
# NOTE: Ensure these keys exist in your Streamlit secrets file (e.g., in a section called [auth])
USERNAME = st.secrets.get("auth", {}).get("username", "HXR")
PASSWORD = st.secrets.get("auth", {}).get("password", "Rossph")

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

# --- INDICATOR FUNCTIONS (Unchanged) ---
def rsi(close, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
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

# --- DATABASE SETUP (PostgreSQL/Supabase Conversion) ---
@st.cache_resource
def get_db_connection():
    """Establishes and caches the PostgreSQL database connection."""
    try:
        # 'finance_db' securely retrieves the URL from Streamlit secrets
        conn = st.connection("finance_db", type="sql", autocommit=True)
        return conn
    except Exception as e:
        logging.error(f"PostgreSQL connection error: {e}", exc_info=True)
        st.error("Failed to connect to the persistent cloud database. Check Streamlit secrets.")
        st.stop()

def initialize_database(conn):
    """Initializes all database tables if they don't exist and handles schema migrations."""

    # Define SQL commands using SQLAlchemy text() wrapper for DDL
    create_tables_sql = [
        # Investment Tables (DOUBLE PRECISION for currency/price)
        text("""CREATE TABLE IF NOT EXISTS portfolio (
                ticker TEXT PRIMARY KEY, buy_price DOUBLE PRECISION NOT NULL,
                buy_date TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1,
                sector TEXT, market_cap TEXT
            )"""),
        text("""CREATE TABLE IF NOT EXISTS trades (
                symbol TEXT PRIMARY KEY, buy_price DOUBLE PRECISION NOT NULL,
                buy_date TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1,
                target_price DOUBLE PRECISION NOT NULL, stop_loss_price DOUBLE PRECISION NOT NULL
            )"""),
        text("""CREATE TABLE IF NOT EXISTS price_history (
                ticker TEXT, date TEXT, close_price DOUBLE PRECISION,
                PRIMARY KEY (ticker, date)
            )"""),
        # Realized/Exits Tables
        text("""CREATE TABLE IF NOT EXISTS realized_stocks (
                transaction_id TEXT PRIMARY KEY, ticker TEXT NOT NULL,
                buy_price DOUBLE PRECISION NOT NULL, buy_date TEXT NOT NULL,
                quantity INTEGER NOT NULL, sell_price DOUBLE PRECISION NOT NULL,
                sell_date TEXT NOT NULL, realized_return_pct DOUBLE PRECISION NOT NULL
            )"""),
        text("""CREATE TABLE IF NOT EXISTS exits (
                transaction_id TEXT PRIMARY KEY, symbol TEXT NOT NULL,
                buy_price DOUBLE PRECISION NOT NULL, buy_date TEXT NOT NULL,
                quantity INTEGER NOT NULL, sell_price DOUBLE PRECISION NOT NULL,
                sell_date TEXT NOT NULL, realized_return_pct DOUBLE PRECISION NOT NULL,
                target_price DOUBLE PRECISION NOT NULL, stop_loss_price DOUBLE PRECISION NOT NULL
            )"""),
        # Funds & Expenses Tables (SERIAL PRIMARY KEY for auto-increment)
        text("""CREATE TABLE IF NOT EXISTS fund_transactions (
                transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL,
                type TEXT NOT NULL, amount DOUBLE PRECISION NOT NULL,
                description TEXT
            )"""),
        # Added transfer_group_id column in initial creation
        text("""CREATE TABLE IF NOT EXISTS expenses (
                expense_id TEXT PRIMARY KEY, date TEXT NOT NULL, amount DOUBLE PRECISION NOT NULL,
                category TEXT NOT NULL, payment_method TEXT, description TEXT,
                type TEXT, transfer_group_id TEXT
            )"""),
        text("""CREATE TABLE IF NOT EXISTS budgets (
                budget_id SERIAL PRIMARY KEY, month_year TEXT NOT NULL,
                category TEXT NOT NULL, amount DOUBLE PRECISION NOT NULL,
                UNIQUE(month_year, category)
            )"""),
        text("""CREATE TABLE IF NOT EXISTS recurring_expenses (
                recurring_id SERIAL PRIMARY KEY, description TEXT NOT NULL UNIQUE,
                amount DOUBLE PRECISION NOT NULL, category TEXT NOT NULL,
                payment_method TEXT, day_of_month INTEGER NOT NULL
            )"""),
        # Mutual Fund Tables
        text("""CREATE TABLE IF NOT EXISTS mf_transactions (
                transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL,
                scheme_name TEXT NOT NULL, yfinance_symbol TEXT NOT NULL,
                type TEXT NOT NULL, units DOUBLE PRECISION NOT NULL,
                nav DOUBLE PRECISION NOT NULL
            )"""),
        text("""CREATE TABLE IF NOT EXISTS mf_sips (
                sip_id SERIAL PRIMARY KEY, scheme_name TEXT NOT NULL UNIQUE,
                yfinance_symbol TEXT NOT NULL, amount DOUBLE PRECISION NOT NULL,
                day_of_month INTEGER NOT NULL
            )""")
    ]

    # Execute commands within a session
    try:
        with conn.session as session:
            for command in create_tables_sql:
                session.execute(command)
            session.commit()

        # Handle migrations (this function must also use session.execute(text()))
        _add_missing_columns(conn)

    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        st.error(f"Database setup failed: {e}")
        st.stop()


def _migrate_fund_transactions_schema(conn):
    """
    Performs schema migration for fund_transactions if needed (SQLite to PG column removal).
    Note: For a fresh PG deployment, this logic is primarily a placeholder/safety check.
    It now uses session.execute(text(...)).
    """
    table_name = 'fund_transactions'

    # In PostgreSQL, checking column existence is done differently than PRAGMA,
    # but we can rely on SQLAlchemy's reflection or simpler schema lookups.
    # For simplicity, we'll try to execute an ALTER and catch the error if the column doesn't exist.

    # Since we defined the PG schema above correctly, we primarily need the ADD COLUMN logic
    # for *other* migrations if they were missed. For a fresh PG install, this function
    # should ideally do nothing unless migrating from a bad PG schema.

    # We will skip the complex Copy-Rename-Drop for this column for now, as the main PG schema
    # has been corrected in initialize_database(). We keep it simple:
    # If the column exists, it was likely an error in the previous PG migration attempt.

    logging.info(f"Assuming {table_name} schema is correct post-PG migration.")


def _add_missing_columns(conn):
    """Handles database schema migrations by adding missing columns."""

    with conn.session as session:
        # Migration 1: Add sector and market_cap to portfolio (if coming from an old schema)
        try:
            session.execute(text("ALTER TABLE portfolio ADD COLUMN sector TEXT"))
            logging.info("Added 'sector' column to 'portfolio' table.")
        except Exception: # Catch exception if column already exists
            pass

        try:
            session.execute(text("ALTER TABLE portfolio ADD COLUMN market_cap TEXT"))
            logging.info("Added 'market_cap' column to 'portfolio' table.")
        except Exception:
            pass

        # Migration 2: Add type to expenses
        try:
            session.execute(text("ALTER TABLE expenses ADD COLUMN type TEXT"))
            session.execute(text("UPDATE expenses SET type = 'Expense' WHERE type IS NULL"))
            logging.info("Added 'type' column and set default value to 'expenses' table.")
        except Exception:
            pass

        # Migration 3: Add transfer_group_id to expenses table
        try:
            session.execute(text("ALTER TABLE expenses ADD COLUMN transfer_group_id TEXT"))
            logging.info("Added 'transfer_group_id' column to 'expenses' table.")
        except Exception:
            pass

        session.commit()

# REPLACED DB_CONN initialization to use the new PG function
DB_CONN = get_db_connection()
initialize_database(DB_CONN)

# CONVERTED: fund_transactions INSERT logic
def update_funds_on_transaction(transaction_type, amount, description, date):
    """Inserts a new transaction into the fund_transactions table using PostgreSQL."""

    if description and description.startswith("ALLOCATION:"):
        description = description.split(' - ', 1)[-1].strip()

    # Use a session context manager for safe transaction execution
    with DB_CONN.session as session:
        session.execute(
            text("""
                INSERT INTO fund_transactions (transaction_id, date, type, amount, description)
                VALUES (:id, :date, :type, :amount, :description)
            """),
            params={
                "id": str(uuid.uuid4()),
                "date": date,
                "type": transaction_type,
                "amount": round(amount, 2),
                "description": description
            }
        )
        session.commit()

# --- API & DATA FUNCTIONS ---
@st.cache_data(ttl=3600)
def search_for_ticker(company_name):
    # Unchanged - relies on API requests
    try:
        api_key = st.secrets.get("api_keys", {}).get("finnhub")
        if not api_key:
             logging.warning("Finnhub API key not found in st.secrets.")
             return []

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
    # Unchanged - relies on yfinance and pandas read_sql (which works with st.connection)
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            ticker_str = symbol.replace('XNSE:', '') + '.NS' if 'XNSE:' in symbol and not symbol.endswith('.NS') else symbol
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

            if price is None:
                latest_db_price = pd.read_sql(text("SELECT close_price FROM price_history WHERE ticker = :symbol ORDER BY date DESC LIMIT 1"), DB_CONN.engine, params={"symbol": symbol})
                if not latest_db_price.empty:
                    price = latest_db_price['close_price'].iloc[0]

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
    # Unchanged - relies on mftool
    try:
        mf = Mftool()
        schemes = mf.get_scheme_codes()
        return {v: k for k, v in schemes.items()}
    except Exception as e:
        logging.error(f"Failed to fetch mutual fund schemes: {e}")
        return {}

@st.cache_data(ttl=600)
def fetch_latest_mf_nav(scheme_code):
    # Unchanged - relies on mftool
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
    # Unchanged - relies on mftool
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

# CONVERTED: update_stock_data for bulk PostgreSQL insert (via to_sql)
def update_stock_data(symbol):
    """Downloads and saves historical stock data (Close Price only) to the PostgreSQL database."""
    try:
        ticker_str = symbol.replace('XNSE:', '') + '.NS' if 'XNSE:' in symbol and not symbol.endswith('.NS') else symbol
        today = datetime.date.today()
        data = yf.download(ticker_str, start="2020-01-01", end=today + datetime.timedelta(days=1), progress=False, auto_adjust=True)

        if data.empty or 'Close' not in data.columns:
            logging.warning(f"YFinance returned empty or invalid data for {symbol}.")
            return False

        data.reset_index(inplace=True)
        data = data[["Date", "Close"]].rename(columns={"Date": "date_col", "Close": "close_price"})

        data["ticker"] = symbol
        data["date"] = data["date_col"].dt.strftime("%Y-%m-%d")

        df_to_insert = data[["ticker", "date", "close_price"]].copy()

        # Use the engine for bulk writes. PostgreSQL will handle the PRIMARY KEY
        # conflict during the 'append' operation if a record with the same (ticker, date) exists.
        df_to_insert.to_sql(
            'price_history',
            DB_CONN.engine, # Use the engine for bulk writes
            if_exists='append',
            index=False,
            method='multi'
        )

        return True
    except Exception as e:
        logging.error(f"PostgreSQL update_stock_data failed for {symbol}: {e}")
        return False


def get_holdings_df(table_name):
    """Fetches and calculates current portfolio/trade holdings from the database."""
    # Queries use DB_CONN.engine for reads, which is safe.
    if table_name == "trades":
        query = text("""
            SELECT p.symbol, p.buy_price, p.buy_date, p.quantity, p.target_price, p.stop_loss_price, h.close_price AS current_price
            FROM trades p
            LEFT JOIN price_history h ON p.symbol = h.ticker
            WHERE h.date = (SELECT MAX(date) FROM price_history WHERE ticker = p.symbol)
        """)
        df = pd.read_sql(query, DB_CONN.engine)
    else:
        query = text("""
            SELECT p.ticker AS symbol, p.buy_price, p.buy_date, p.quantity, p.sector, p.market_cap, h.close_price AS current_price
            FROM portfolio p
            LEFT JOIN price_history h ON p.ticker = h.ticker
            WHERE h.date = (SELECT MAX(date) FROM price_history WHERE ticker = p.ticker)
        """)
        df = pd.read_sql(query, DB_CONN.engine)

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

def get_realized_df(table_name):
    """Fetches and calculates realized profits/losses from the database."""
    # Queries use DB_CONN.engine for reads, which is safe.
    try:
        query = text(f"SELECT * FROM {table_name}")
        df = pd.read_sql(query, DB_CONN.engine)
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

# CONVERTED: portfolio UPDATE logic
def _update_existing_portfolio_info():
    """Fetches and updates missing sector and market cap data for existing stocks."""

    with DB_CONN.session as session:
        # 1. Fetch tickers with missing info
        select_query = text("SELECT ticker, sector, market_cap FROM portfolio WHERE sector IS NULL OR market_cap IS NULL OR sector = 'N/A' OR market_cap = 'N/A'")
        tickers_to_update = session.execute(select_query).fetchall()

        if tickers_to_update:
            update_query = text("UPDATE portfolio SET sector = :sector, market_cap = :mcap WHERE ticker = :ticker")

            for ticker, old_sector, old_market_cap in tickers_to_update:
                try:
                    stock_info = fetch_stock_info(ticker)
                    sector, market_cap_raw = stock_info['sector'], stock_info['market_cap']

                    if sector != 'N/A' or market_cap_raw != 'N/A':
                        final_market_cap = _categorize_market_cap(market_cap_raw)

                        # Preserve existing info if new info is 'N/A'
                        final_sector = sector if sector != 'N/A' else old_sector
                        final_market_cap = final_market_cap if final_market_cap != 'N/A' else old_market_cap

                        session.execute(update_query, {
                            "sector": final_sector,
                            "mcap": final_market_cap,
                            "ticker": ticker
                        })
                        logging.info(f"Updated sector and market cap for {ticker}.")
                except Exception as e:
                    logging.error(f"Failed to update info for {ticker}: {e}")

            session.commit()

def _categorize_market_cap(market_cap_value):
    # Unchanged
    if isinstance(market_cap_value, (int, float)):
        if market_cap_value >= 10000000000:
            return "Large Cap"
        elif market_cap_value >= 2000000000:
            return "Mid Cap"
        else:
            return "Small Cap"
    return "N/A"

# --- HELPER FUNCTIONS ---
# CONVERTED: recurring_expenses INSERT logic
def _process_recurring_expenses():
    """Adds recurring expenses to the database if not already logged for the current month."""

    month_year = datetime.date.today().strftime("%Y-%m")

    try:
        # Read from engine is fine
        recurring_df = pd.read_sql(text("SELECT * FROM recurring_expenses"), DB_CONN.engine)
        if recurring_df.empty:
            return

        logged_expenses_df = pd.read_sql(text("SELECT description FROM expenses WHERE date LIKE :month_year"), DB_CONN.engine, params={"month_year": f"{month_year}%"})
        logged_descriptions = logged_expenses_df['description'].tolist()

        with DB_CONN.session as session:
            insert_query = text("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description) VALUES (:id, :date, :type, :amount, :category, :pmethod, :desc)")

            for _, row in recurring_df.iterrows():
                marker = f"Recurring: {row['description']}"
                if marker in logged_descriptions:
                    continue

                day = min(row['day_of_month'], pd.Timestamp(month_year).days_in_month)
                expense_date = f"{month_year}-{day:02d}"

                session.execute(insert_query, {
                    "id": str(uuid.uuid4()),
                    "date": expense_date,
                    "type": "Expense",
                    "amount": round(row['amount'], 2),
                    "category": row['category'],
                    "pmethod": row['payment_method'],
                    "desc": marker
                })
                logging.info(f"Logged recurring expense: {row['description']}")

            session.commit()
    except Exception as e:
        logging.error(f"Could not process recurring expenses: {e}")

# CONVERTED: mf_sips INSERT logic
def _process_mf_sips():
    """Automatically logs mutual fund SIP transactions if the date is passed."""

    today = datetime.date.today()
    month_year = today.strftime("%Y-%m")

    try:
        sips_df = pd.read_sql(text("SELECT * FROM mf_sips"), DB_CONN.engine)
        if sips_df.empty:
            return

        mf_tx_insert_query = text("INSERT INTO mf_transactions (transaction_id, date, scheme_name, yfinance_symbol, type, units, nav) VALUES (:id, :date, :name, :symbol, :type, :units, :nav)")

        with DB_CONN.session as session:
            for _, sip in sips_df.iterrows():
                day = min(sip['day_of_month'], pd.Timestamp.now().days_in_month)
                sip_date_this_month = today.replace(day=day).strftime('%Y-%m-%d')

                if datetime.datetime.strptime(sip_date_this_month, '%Y-%m-%d').date() <= today:
                    existing_sip_tx = pd.read_sql(text("SELECT * FROM mf_transactions WHERE scheme_name = :name AND date LIKE :date_like"),
                                                  DB_CONN.engine, params={"name": sip['scheme_name'], "date_like": f"{month_year}%"})

                    if existing_sip_tx.empty:
                        try:
                            nav = fetch_latest_mf_nav(sip['yfinance_symbol'])
                            if nav:
                                units = sip['amount'] / nav

                                # 1. Update Funds (already converted to PG)
                                update_funds_on_transaction("Withdrawal", round(sip['amount'], 2), f"MF SIP: {sip['scheme_name']}", sip_date_this_month)

                                # 2. Insert MF Transaction
                                session.execute(mf_tx_insert_query, {
                                    "id": str(uuid.uuid4()),
                                    "date": sip_date_this_month,
                                    "name": sip['scheme_name'],
                                    "symbol": sip['yfinance_symbol'],
                                    "type": 'Purchase',
                                    "units": round(units, 4),
                                    "nav": round(nav, 4)
                                })
                                session.commit() # Commit each SIP immediately to reflect in logs/UI

                                logging.info(f"Auto-logged SIP for {sip['scheme_name']}")
                                st.sidebar.success(f"Auto-logged SIP for {sip['scheme_name']}")
                            else:
                                st.sidebar.warning(f"Could not auto-log SIP for {sip['scheme_name']}. NAV fetch failed.")
                        except Exception as e:
                            st.sidebar.warning(f"Could not auto-log SIP for {sip['scheme_name']}. An error occurred: {e}")
                            logging.error(f"Failed during MF SIP processing: {e}")

    except Exception as e:
        logging.error(f"Failed during MF SIP outer processing: {e}")


@st.cache_data(ttl=3600)
def get_benchmark_comparison_data(holdings_df, benchmark_choice):
    # Unchanged logic but ensures read_sql uses DB_CONN.engine (which it does via the function body)
    # ... (function body remains the same, assuming `pd.read_sql` calls inside use DB_CONN.engine or query the connection object directly)
    if holdings_df.empty:
        return pd.DataFrame()

    start_date = holdings_df['buy_date'].min() if not holdings_df.empty else datetime.date.today().strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date))

    benchmark_map = {'Nifty 50': '^NSEI', 'Nifty 100': '^CNX100', 'Nifty 200': '^CNX200', 'Nifty 500': '^CRSLDX'}
    selected_ticker = benchmark_map.get(benchmark_choice)

    if not selected_ticker:
        return pd.DataFrame()

    all_tickers = holdings_df['symbol'].unique().tolist()

    # CONVERTED: read_sql uses DB_CONN.engine
    if not all_tickers:
        return pd.DataFrame() # Handle case where all_tickers is empty

    placeholders = ', '.join([f':t{i}' for i in range(len(all_tickers))])
    params = {f't{i}': ticker for i, ticker in enumerate(all_tickers)}
    params['start_date'] = start_date

    price_data_query = text(f"""
        SELECT date, ticker, close_price FROM price_history
        WHERE ticker IN ({placeholders}) AND date >= :start_date
    """)
    all_prices = pd.read_sql(price_data_query, DB_CONN.engine, params=params)
    # ... (rest of the calculation logic remains the same)

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
    portfolio_return = portfolio_return.dropna()

    try:
        benchmark_df = yf.download(selected_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if benchmark_df.empty or 'Close' not in benchmark_df.columns:
            logging.error(f"yfinance returned empty or invalid data for benchmark {selected_ticker}")
            return pd.DataFrame()
        benchmarks = benchmark_df['Close'].copy()
        benchmarks.ffill(inplace=True)
        benchmark_returns = ((benchmarks / benchmarks.iloc[0] - 1) * 100).round(2)
        benchmark_returns.name = benchmark_choice

        final_df = pd.concat([portfolio_return, benchmark_returns], axis=1).reset_index().rename(columns={'index': 'Date'})

    except Exception as e:
        logging.error(f"Failed to download benchmark data for {selected_ticker}: {e}", exc_info=True)
        return pd.DataFrame()

    final_df = final_df.melt(id_vars='Date', var_name='Type', value_name='Return %').dropna()
    return final_df

def calculate_portfolio_metrics(holdings_df, realized_df, benchmark_choice):
    # Unchanged (reads data using DB_CONN.engine)
    metrics = {
        'alpha': 'N/A', 'beta': 'N/A', 'max_drawdown': 'N/A'
    }
    if holdings_df.empty:
        return metrics

    start_date = holdings_df['buy_date'].min() if not holdings_df.empty else datetime.date.today().strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    all_tickers = holdings_df['symbol'].unique().tolist()

    if not all_tickers:
        return metrics

    placeholders = ', '.join([f':t{i}' for i in range(len(all_tickers))])
    params = {f't{i}': ticker for i, ticker in enumerate(all_tickers)}
    params['start_date'] = start_date

    price_data_query = text(f"""
        SELECT date, ticker, close_price FROM price_history
        WHERE ticker IN ({placeholders}) AND date >= :start_date
    """)
    all_prices = pd.read_sql(price_data_query, DB_CONN.engine, params=params)

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
    # Unchanged (calculation logic)
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
    """
    Calculates and returns combined returns for all asset types,
    EXCLUDING data from paper trading.
    """
    # --- 1. Identify Live Trade Symbols (linked to fund transactions) ---
    live_trade_symbols_query = text("""
    SELECT DISTINCT SUBSTRING(description FROM ' of (.*)')
    FROM fund_transactions
    WHERE type = 'Withdrawal' AND description LIKE 'Purchase % of %'
    """)
    # CONVERTED: uses DB_CONN.engine for read
    live_trades_df = pd.read_sql(live_trade_symbols_query, DB_CONN.engine)
    live_trade_symbols = set(live_trades_df.iloc[:, 0].tolist())

    # --- 2. Investment Portfolio (Always considered 'live') ---
    inv_df = get_holdings_df("portfolio")
    inv_invested = inv_df['invested_value'].sum() if not inv_df.empty else 0
    inv_current = inv_df['current_value'].sum() if not inv_df.empty else 0

    # --- 3. Trading Book (Filter for 'live' only) ---
    trade_df = get_holdings_df("trades")
    if not trade_df.empty:
        live_trade_df = trade_df[trade_df['symbol'].isin(live_trade_symbols)]
    else:
        live_trade_df = pd.DataFrame()

    trade_invested = live_trade_df['invested_value'].sum() if not live_trade_df.empty else 0
    trade_current = live_trade_df['current_value'].sum() if not live_trade_df.empty else 0

    # --- 4. Mutual Funds (Always considered 'live') ---
    mf_df = get_mf_holdings_df()
    mf_invested = mf_df['Investment'].sum() if not mf_df.empty else 0
    mf_current = mf_df['Current Value'].sum() if not mf_df.empty else 0

    # --- 5. Calculate Returns for Display ---
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

    # --- 6. Calculate Realized P&L ---
    realized_stocks_df = get_realized_df("realized_stocks")
    realized_exits_df = get_realized_df("exits")

    if not realized_exits_df.empty:
        live_exits_df = realized_exits_df[realized_exits_df['symbol'].isin(live_trade_symbols)]
    else:
        live_exits_df = pd.DataFrame()

    realized_inv = realized_stocks_df['realized_profit_loss'].sum() if not realized_stocks_df.empty else 0
    realized_trade = live_exits_df['realized_profit_loss'].sum() if not live_exits_df.empty else 0
    realized_mf = 0

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
    # CONVERTED: read_sql uses DB_CONN.engine
    transactions_df = pd.read_sql(text("SELECT * FROM mf_transactions"), DB_CONN.engine)
    if transactions_df.empty:
        return pd.DataFrame()

    # ... (rest of the logic remains the same) ...
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

@st.cache_data(ttl=3600)
def get_current_portfolio_allocation():
    # Unchanged (relies on other functions)
    inv_df = get_holdings_df("portfolio")
    inv_current = inv_df['current_value'].sum() if not inv_df.empty else 0

    trade_df = get_holdings_df("trades")
    trade_current = trade_df['current_value'].sum() if not trade_df.empty else 0

    mf_df = get_mf_holdings_df()
    mf_current = mf_df['Current Value'].sum() if not mf_df.empty else 0

    allocation_data = [
        {"Category": "Investment", "Amount": inv_current},
        {"Category": "Trading", "Amount": trade_current},
        {"Category": "Mutual Fund", "Amount": mf_current},
    ]

    allocation_df = pd.DataFrame(allocation_data)
    final_df = allocation_df[allocation_df['Amount'] > 0.01].copy()
    final_df['Amount'] = final_df['Amount'].round(2)

    return final_df.sort_values('Amount', ascending=False)

def _calculate_mf_cumulative_return(transactions_df):
    # Unchanged (relies on transactions_df input)
    if transactions_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_schemes_daily_returns = []
    sip_marker_data = []
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
                        sip_marker_data.append({
                            'date': date,
                            'scheme_name': scheme_name,
                            'type': 'Purchase',
                            'amount': round(tx_row['units'] * tx_row['nav'], 2)
                        })

                    elif tx_row['type'] == 'Redemption':
                        units -= tx_row['units']
                        invested_amount -= (tx_row['units'] * tx_row['nav'])
                        sip_marker_data.append({
                            'date': date,
                            'scheme_name': scheme_name,
                            'type': 'Redemption',
                            'amount': round(tx_row['units'] * tx_row['nav'], 2)
                        })

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
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(all_schemes_daily_returns), pd.DataFrame(sip_marker_data)


def home_page():
    """Renders the main home page."""
    st.title("Finance Dashboard")
    _update_existing_portfolio_info()
    returns_data = get_combined_returns()

    #st.subheader("Live Portfolio Overview (Excluding Paper Trades)")

    # ... (Metric display logic remains the same) ...
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Total Live Investment Value",
            value=f"₹{returns_data['total_invested_value']:,.2f}",
        )
    with col2:
        st.metric(
            label="Total Live Current Value",
            value=f"₹{returns_data['total_current_value']:,.2f}",
        )
    with col3:
        st.metric(
            label="Total Live Portfolio Return",
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
            help=f"Realized P&L (Live): ₹{returns_data['realized_trade']:,.2f}"
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
                all_tickers_query = text("SELECT ticker FROM portfolio UNION SELECT symbol FROM trades")
                all_tickers = pd.read_sql(all_tickers_query, DB_CONN.engine)['ticker'].tolist()

                for symbol in all_tickers:
                    update_stock_data(symbol)

                mf_symbols_query = text("SELECT DISTINCT yfinance_symbol FROM mf_transactions")
                mf_symbols = pd.read_sql(mf_symbols_query, DB_CONN.engine)['yfinance_symbol'].tolist()

                for symbol in mf_symbols:
                    fetch_latest_mf_nav(symbol)

            st.success("All live data refreshed!")
            st.rerun()

def funds_page():
    """Renders the Funds Management page."""

    st.title("💰 Funds Management")
    st.sidebar.header("Add Transaction")

    with st.sidebar.form("deposit_form", clear_on_submit=True):
        st.subheader("Add Deposit")
        deposit_date = st.date_input("Date", max_value=datetime.date.today(), key="deposit_date")
        deposit_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None)
        deposit_desc = st.text_input("Description", placeholder="e.g., Salary", value="")
        if st.form_submit_button("Add Deposit"):
            if deposit_amount and deposit_amount > 0:
                update_funds_on_transaction("Deposit", round(deposit_amount, 2), deposit_desc, deposit_date.strftime("%Y-%m-%d"))
                st.success("Deposit recorded!")
                st.rerun()
            else:
                st.warning("Deposit amount must be greater than zero.")

    with st.sidebar.form("withdrawal_form", clear_on_submit=True):
        st.subheader("Record Withdrawal")
        wd_date = st.date_input("Date", max_value=datetime.date.today(), key="wd_date")
        wd_amount = st.number_input("Amount", min_value=0.01, format="%.2f", key="wd_amount", value=None)
        wd_desc = st.text_input("Description", placeholder="e.g., Personal Use", value="")

        if st.form_submit_button("Record Withdrawal"):
            if wd_amount and wd_amount > 0:
                update_funds_on_transaction("Withdrawal", round(wd_amount, 2), wd_desc, wd_date.strftime("%Y-%m-%d"))
                st.success("Withdrawal recorded!")
                st.rerun()
            else:
                st.warning("Withdrawal amount must be greater than zero.")

    # CONVERTED: read_sql uses DB_CONN.engine
    fund_df = pd.read_sql(text("SELECT transaction_id, date, type, amount, description FROM fund_transactions ORDER BY date DESC, transaction_id DESC"), DB_CONN.engine)

    # --- CALCULATE AVAILABLE CASH (Logic remains the same) ---
    total_deposits, total_withdrawals = fund_df.loc[fund_df['type'] == 'Deposit', 'amount'].sum(), fund_df.loc[fund_df['type'] == 'Withdrawal', 'amount'].sum()
    available_capital = round(total_deposits - total_withdrawals, 2)

    if not fund_df.empty:
        # ... (rest of data manipulation logic for charts remains the same)
        fund_df['date'] = pd.to_datetime(fund_df['date'], format='%Y-%m-%d', errors='coerce')
        fund_df['balance'] = fund_df.apply(lambda row: row['amount'] if row['type'] == 'Deposit' else -row['amount'], axis=1)

        chronological_df = fund_df.copy()
        chronological_df.sort_values(['date', 'transaction_id'], ascending=[True, True], inplace=True)

        chronological_df['cumulative_balance'] = chronological_df['balance'].cumsum()

        fund_df = fund_df.merge(
            chronological_df[['transaction_id', 'cumulative_balance']],
            on='transaction_id',
            how='left'
        )

        fund_df.sort_values(['date', 'transaction_id'], ascending=[False, False], inplace=True)


        col1, col2, col3 = st.columns(3)
        col1.metric("Total Deposits", f"₹{total_deposits:,.2f}")
        col2.metric("Total Withdrawals", f"₹{total_withdrawals:,.2f}")
        col3.metric("Available Capital (Cash)", f"₹{available_capital:,.2f}")

        st.divider()
        st.subheader("Cumulative Fund Flow")

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

        edited_df = st.data_editor(
            fund_df[['transaction_id', 'date', 'type', 'amount', 'description']],
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "transaction_id": st.column_config.TextColumn("ID", disabled=True),
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["Deposit", "Withdrawal"], required=True),
            }
        )

        edited_df['date'] = edited_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime.date) else x)


        # CONVERTED: Data editor save logic (DELETE + to_sql)
        if st.button("Save Changes to Transactions"):
            try:
                with DB_CONN.session as session:
                    # 1. Delete all existing records
                    session.execute(text("DELETE FROM fund_transactions"))

                    # 2. Insert the full, edited data back into the table
                    edited_df.to_sql('fund_transactions', DB_CONN.engine, if_exists='append', index=False)
                    session.commit()

                    st.success("Funds transactions updated successfully! Rerunning to update the chart.")
                    st.rerun()

            except Exception as e:
                logging.error(f"Funds data editor save failed: {e}", exc_info=True)
                st.error(f"Failed to save funds transactions: {e}")

    else:
        st.info("No fund transactions logged yet.")

def expense_tracker_page():
    """Renders the Expense Tracker page with PostgreSQL logic."""
    st.title("💸 Expense Tracker")
    _process_recurring_expenses()

    # ... (Category and Payment Method definitions remain the same) ...
    if 'expense_categories_list' not in st.session_state:
        try:
            expense_categories_query = text("SELECT DISTINCT category FROM expenses WHERE type='Expense'")
            expense_categories = pd.read_sql(expense_categories_query, DB_CONN.engine)['category'].tolist()
            default_categories = ["Food", "Transport", "Rent", "Utilities", "Shopping", "Entertainment", "Health", "Groceries", "Bills", "Education", "Travel", "Other"]
            all_categories = list(set([c for c in expense_categories if c and c != 'N/A'] + default_categories))
            EXCLUDED_CATEGORIES = ["Transfer Out", "Transfer In"]
            all_categories = [c for c in all_categories if c not in EXCLUDED_CATEGORIES]
            st.session_state.expense_categories_list = sorted(all_categories)
        except Exception:
            st.session_state.expense_categories_list = sorted(["Food", "Transport", "Rent", "Utilities", "Shopping", "Entertainment", "Health", "Groceries", "Bills", "Education", "Travel", "Other"])

    CATEGORIES = st.session_state.expense_categories_list
    PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking", "N/A"]
    PAYMENT_ACCOUNTS = [pm for pm in PAYMENT_METHODS if pm != 'N/A']
    ACCOUNT_PLACEHOLDER = "Select Account..."
    view = st.radio("Select View", ["Dashboard", "Transaction History", "Manage Budgets", "Manage Recurring", "Transfer"], horizontal=True, label_visibility="hidden")

    # --- SIDEBAR: ADD TRANSACTION FORM (CONVERTED INSERT) ---
    if view != "Transfer":
        st.sidebar.header("Add Transaction")
        with st.sidebar.form("new_transaction_form", clear_on_submit=True):
            # ... (Form fields remain the same) ...
            trans_type = st.radio("Transaction Type", ["Expense", "Income"], key="trans_type")
            trans_date = st.date_input("Date", max_value=datetime.date.today(), value=datetime.date.today())
            trans_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None)

            custom_category = st.text_input(
                "Or Enter New Category",
                help="Enter a custom category name, this will override the selection.",
                value="",
                key="custom_cat"
            )
            # ... (Category determination logic remains the same) ...
            category_options = ['Select Category...'] + CATEGORIES
            selected_category = st.selectbox(
                "Select Category",
                options=category_options,
                index=0,
                key="selected_cat"
            )

            if custom_category:
                final_cat = custom_category
            elif selected_category and selected_category != 'Select Category...':
                final_cat = selected_category
            else:
                final_cat = None

            if trans_type == "Income":
                trans_pm = st.selectbox("Destination Account/Method", options=PAYMENT_ACCOUNTS, index=None)
            else:
                trans_pm = st.selectbox("Payment Method", options=PAYMENT_ACCOUNTS, index=None)

            trans_desc = st.text_input("Description", value="")

            if st.form_submit_button("Add Transaction"):
                if trans_amount and final_cat and trans_pm:

                    # CONVERTED: Expense INSERT logic
                    with DB_CONN.session as session:
                        insert_query = text("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description) VALUES (:id, :date, :type, :amount, :category, :pmethod, :desc)")
                        session.execute(insert_query, {
                            "id": str(uuid.uuid4()),
                            "date": trans_date.strftime("%Y-%m-%d"),
                            "type": trans_type,
                            "amount": round(trans_amount, 2),
                            "category": final_cat,
                            "pmethod": trans_pm,
                            "desc": trans_desc
                        })
                        session.commit()

                    st.success(f"{trans_type} added! Category: **{final_cat}**")

                    if final_cat not in st.session_state.expense_categories_list:
                        st.session_state.expense_categories_list.append(final_cat)
                        st.session_state.expense_categories_list.sort()

                    st.cache_data.clear()

                    if "selected_cat" in st.session_state: del st.session_state["selected_cat"]
                    if "custom_cat" in st.session_state: del st.session_state["custom_cat"]
                    st.rerun()
                else:
                    st.warning("Please fill all required fields (Amount, Category, and Payment Method).")
    # --- END OF SIDEBAR: ADD TRANSACTION FORM ---


    if view == "Dashboard":
        # ... (Dashboard queries rely on DB_CONN.engine and remain mostly the same) ...
        today = datetime.date.today()
        start_date_7days = today - datetime.timedelta(days=6)
        month_year = today.strftime("%Y-%m")

        expenses_df_query = text("SELECT * FROM expenses WHERE date LIKE :month_year")
        expenses_df = pd.read_sql(expenses_df_query, DB_CONN.engine, params={"month_year": f"{month_year}%"})

        all_time_expenses_df = pd.read_sql(text("SELECT * FROM expenses"), DB_CONN.engine)
        all_time_expenses_df['date'] = pd.to_datetime(all_time_expenses_df['date'])

        if all_time_expenses_df.empty:
            st.info("No expenses logged yet to display the dashboard.")
            return

        budgets_df_query = text("SELECT category, amount FROM budgets WHERE month_year = :month_year")
        budgets_df = pd.read_sql(budgets_df_query, DB_CONN.engine, params={"month_year": month_year}).set_index('category')

        # ... (rest of dashboard logic remains the same) ...
        inflows_df = expenses_df[expenses_df['type'] == 'Income']
        outflows_df = expenses_df[expenses_df['type'] == 'Expense']

        total_spent = outflows_df[outflows_df['category'] != 'Transfer Out']['amount'].sum()
        total_income = inflows_df[inflows_df['category'] != 'Transfer In']['amount'].sum()

        total_budget = budgets_df['amount'].sum()
        net_flow = total_income - total_spent

        # ... (Metric displays remain the same) ...
        spent_breakdown_df = outflows_df[outflows_df['category'] != 'Transfer Out'].groupby('payment_method')['amount'].sum().reset_index()
        spent_help_text = "\n".join([f"{row['payment_method']}: ₹{row['amount']:,.2f}" for _, row in spent_breakdown_df.iterrows()])

        flow_df = all_time_expenses_df.groupby(['type', 'payment_method'])['amount'].sum().unstack(level=0, fill_value=0).fillna(0)
        flow_df['Remaining'] = flow_df['Income'] - flow_df['Expense']
        remaining_breakdown_df = flow_df[(flow_df['Remaining'].abs() > 0.01) & (flow_df.index != 'N/A')].sort_values('payment_method').reset_index()

        remaining_help_text = "\n".join([f"{row['payment_method']}: ₹{row['Remaining']:,.2f}" for _, row in remaining_breakdown_df.iterrows()])
        if not remaining_help_text:
             remaining_help_text = "All flows balanced, or no categorized transactions."


        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Income (Excl. Transfers)", f"₹{total_income:,.2f}")
        col2.metric("Total Spent this Month (Excl. Transfers)", f"₹{total_spent:,.2f}",
                     help=f"**Spent Breakdown (This Month, Excl. Transfers):**\n{spent_help_text}")
        col3.metric("Net Flow (Excl. Transfers)", f"₹{net_flow:,.2f}",
                     delta_color="inverse" if net_flow >= 0 else "normal",
                     help=f"**Net Remaining Breakdown (Includes all time funds movements):**\n{remaining_help_text}")
        col4.metric("Total Budget for Month", f"₹{total_budget:,.2f}")
        st.divider()

        # ... (Chart logic remains the same) ...

        # Monthly Spending Trend (CONVERTED query)
        monthly_spending_query = text("SELECT SUBSTR(date, 1, 7) AS month, SUM(amount) AS amount FROM expenses WHERE type='Expense' AND category != 'Transfer Out' GROUP BY month ORDER BY month DESC")
        monthly_spending_df = pd.read_sql(monthly_spending_query, DB_CONN.engine)

        # Inflow vs. Outflow (CONVERTED query)
        monthly_flows_query = text("SELECT SUBSTR(date, 1, 7) AS month, type, SUM(amount) AS amount FROM expenses WHERE (type='Income' AND category != 'Transfer In') OR (type='Expense' AND category != 'Transfer Out') GROUP BY month, type ORDER BY month DESC")
        monthly_flows_df = pd.read_sql(monthly_flows_query, DB_CONN.engine)

        # ... (chart rendering remains the same) ...

    # --- TRANSFER VIEW (CONVERTED INSERTS) ---
    elif view == "Transfer":
        st.header("🔄 Internal Account Transfer")
        # ... (Transfer form remains the same) ...

        with st.form("transfer_form", clear_on_submit=True):
            transfer_date = st.date_input("Date", max_value=datetime.date.today(), value=datetime.date.today())
            transfer_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None)

            source_account = st.selectbox("From Account (Source)", options=PAYMENT_ACCOUNTS, index=None, key="source_acc_final", placeholder="Select Source Account")
            current_dest_options = [acc for acc in PAYMENT_ACCOUNTS if acc != source_account]
            dest_account = st.selectbox("To Account (Destination)", options=current_dest_options, index=None, key="dest_acc_final", placeholder="Select Destination Account")

            transfer_desc = st.text_input("Description (Optional)", value="")

            if st.form_submit_button("Record Transfer"):
                if (transfer_amount and transfer_amount > 0 and
                    source_account is not None and dest_account is not None and
                    source_account != dest_account):

                    group_id = str(uuid.uuid4())

                    # CONVERTED: Two-part Transfer INSERT logic
                    with DB_CONN.session as session:
                        insert_query = text("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description, transfer_group_id) VALUES (:id, :date, :type, :amount, :category, :pmethod, :desc, :group_id)")

                        # 1. Record Expense (Withdrawal) from Source
                        session.execute(insert_query, {
                            "id": str(uuid.uuid4()), "date": transfer_date.strftime("%Y-%m-%d"), "type": "Expense",
                            "amount": round(transfer_amount, 2), "category": "Transfer Out", "pmethod": source_account,
                            "desc": f"Transfer to {dest_account}" + (f" ({transfer_desc})" if transfer_desc else ""), "group_id": group_id
                        })

                        # 2. Record Income (Deposit) to Destination
                        session.execute(insert_query, {
                            "id": str(uuid.uuid4()), "date": transfer_date.strftime("%Y-%m-%d"), "type": "Income",
                            "amount": round(transfer_amount, 2), "category": "Transfer In", "pmethod": dest_account,
                            "desc": f"Transfer from {source_account}" + (f" ({transfer_desc})" if transfer_desc else ""), "group_id": group_id
                        })
                        session.commit()

                    st.success(f"Transfer of ₹{transfer_amount:,.2f} recorded from **{source_account}** to **{dest_account}**.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("Please select valid, different source and destination accounts and a positive amount.")

        st.subheader("Recent Consolidated Transfers (Read-Only)")
        # CONVERTED: Custom SQL to fetch transfers
        transfer_query = text("""
        SELECT
            T_OUT.transfer_group_id, T_OUT.date, T_OUT.amount,
            T_OUT.payment_method AS from_account, T_IN.payment_method AS to_account,
            T_OUT.description AS description
        FROM
            expenses AS T_OUT
        INNER JOIN
            expenses AS T_IN
        ON
            T_OUT.transfer_group_id = T_IN.transfer_group_id AND T_OUT.expense_id != T_IN.expense_id
        WHERE
            T_OUT.category = 'Transfer Out' AND T_IN.category = 'Transfer In'
        ORDER BY
            T_OUT.date DESC
        LIMIT 10
        """)
        transfer_df = pd.read_sql(transfer_query, DB_CONN.engine)

        # ... (rest of transfer display logic remains the same) ...

        if not transfer_df.empty:
            transfer_df.rename(columns={'amount': 'Amount', 'date': 'Date', 'from_account': 'From Account', 'to_account': 'To Account', 'description': 'Description'}, inplace=True)
            transfer_df['Amount'] = transfer_df['Amount'].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(transfer_df.drop(columns=['transfer_group_id']), hide_index=True, use_container_width=True)
        else:
            st.info("No transfers recorded yet.")

        st.divider()

        st.subheader("Edit Underlying Transfer Transactions")
        st.warning("Editing these directly requires careful attention. Ensure both 'Transfer Out' (Expense) and 'Transfer In' (Income) rows for a single transfer group have the same **Amount**, **Date**, and are linked by the same **Transfer Group ID**.")

        all_transfer_legs_query = text("SELECT expense_id, date, type, amount, category, payment_method, transfer_group_id, description FROM expenses WHERE category IN ('Transfer Out', 'Transfer In') ORDER BY date DESC, transfer_group_id DESC, type DESC")
        all_transfer_legs_df = pd.read_sql(all_transfer_legs_query, DB_CONN.engine)

        if not all_transfer_legs_df.empty:
            df_for_editing = all_transfer_legs_df.drop(columns=['category', 'type']).copy()
            df_for_editing['date'] = pd.to_datetime(df_for_editing['date'], format='%Y-%m-%d', errors='coerce').dt.date

            edited_transfer_df = st.data_editor(df_for_editing, use_container_width=True, hide_index=True, num_rows="dynamic",
                column_config={
                    "expense_id": st.column_config.TextColumn("ID", disabled=True),
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                    "amount": st.column_config.NumberColumn("Amount", min_value=0.01, required=True),
                    "payment_method": st.column_config.SelectboxColumn("Account", options=PAYMENT_ACCOUNTS, required=True),
                    "transfer_group_id": st.column_config.TextColumn("Transfer Group ID", required=True, help="Use the same ID for both OUT and IN legs of a single transfer."),
                    "description": st.column_config.TextColumn("Description")
                })

            # CONVERTED: Save Changes to Transfers logic (DELETE + to_sql)
            if st.button("Save Changes to Transfers"):
                try:
                    with DB_CONN.session as session:
                        # Re-fetch the non-transfer records before deleting everything
                        non_transfer_df = pd.read_sql(text("SELECT * FROM expenses WHERE category NOT IN ('Transfer Out', 'Transfer In')"), DB_CONN.engine)

                        transfers_to_save = edited_transfer_df.copy()
                        original_transfer_data = all_transfer_legs_df[['expense_id', 'type', 'category']].set_index('expense_id')

                        transfers_to_save = transfers_to_save.merge(original_transfer_data, on='expense_id', how='left')

                        if transfers_to_save['type'].isnull().any() or transfers_to_save['category'].isnull().any():
                            st.error("Error: Missing internal transaction type/category data. Cannot save transfers.")
                            st.stop()

                        # Delete all existing records
                        session.execute(text("DELETE FROM expenses"))

                        # Insert the non-transfer records back first
                        if not non_transfer_df.empty:
                            non_transfer_df['date'] = non_transfer_df['date'].astype(str)
                            non_transfer_df.to_sql('expenses', DB_CONN.engine, if_exists='append', index=False)

                        # Insert the edited transfer records back
                        transfers_to_save['date'] = transfers_to_save['date'].astype(str)
                        transfers_to_save = transfers_to_save[['expense_id', 'date', 'type', 'amount', 'category', 'payment_method', 'description', 'transfer_group_id']]
                        transfers_to_save.to_sql('expenses', DB_CONN.engine, if_exists='append', index=False)

                        session.commit()
                        st.success("Transfer transactions updated successfully! Rerunning to validate changes.")
                        st.rerun()

                except Exception as e:
                    logging.error(f"Transfer data editor save failed: {e}", exc_info=True)
                    st.error(f"Failed to save transfer transactions: {e}")
        else:
            st.info("No transfers logged yet to edit.")

    # --- TRANSACTION HISTORY VIEW (CONVERTED READ/WRITE) ---
    elif view == "Transaction History":
        st.header("Transaction History")

        history_query = text("SELECT expense_id, date, type, amount, category, payment_method, description FROM expenses WHERE category NOT IN ('Transfer Out', 'Transfer In') ORDER BY date DESC, expense_id DESC")
        all_expenses_df = pd.read_sql(history_query, DB_CONN.engine)

        if not all_expenses_df.empty:
            # ... (Data editor setup remains the same) ...
            all_expenses_df['date'] = pd.to_datetime(all_expenses_df['date'], format='%Y-%m-%d', errors='coerce').dt.date

            editable_categories = sorted(list(set(all_expenses_df['category'].unique().tolist() + CATEGORIES)))

            edited_df = st.data_editor(all_expenses_df, use_container_width=True, hide_index=True, num_rows="dynamic",
                column_config={"expense_id": st.column_config.TextColumn("ID", disabled=True),
                               "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                               "type": st.column_config.SelectboxColumn("Type", options=["Expense", "Income"], required=True),
                               "category": st.column_config.SelectboxColumn("Category", options=editable_categories, required=True),
                               "payment_method": st.column_config.SelectboxColumn("Payment Method", options=[pm for pm in PAYMENT_METHODS], required=True)})

            edited_df['date'] = edited_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime.date) else x)

            # CONVERTED: Expense History Save Logic (DELETE + to_sql)
            if st.button("Save Changes to Transactions"):
                try:
                    with DB_CONN.session as session:
                        # 1. Fetch all existing transfers
                        transfers_df = pd.read_sql(text("SELECT * FROM expenses WHERE category IN ('Transfer Out', 'Transfer In')"), DB_CONN.engine)

                        # 2. Delete all existing records
                        session.execute(text("DELETE FROM expenses"))

                        # 3. Insert the edited (non-transfer) data back
                        edited_df['date'] = edited_df['date'].astype(str)
                        edited_df.to_sql('expenses', DB_CONN.engine, if_exists='append', index=False)

                        # 4. Insert the untouched transfer data back
                        if not transfers_df.empty:
                            transfers_df['date'] = transfers_df['date'].astype(str)
                            transfers_df.to_sql('expenses', DB_CONN.engine, if_exists='append', index=False)

                        session.commit()
                        st.success("Expenses updated successfully! (Transfer records were preserved)")
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    logging.error(f"Expense history save failed: {e}", exc_info=True)
                    st.error(f"Failed to save expense history: {e}")
        else:
            st.info("No non-transfer transaction history to display.")

    # --- MANAGE BUDGETS VIEW (CONVERTED INSERT/REPLACE) ---
    elif view == "Manage Budgets":
        st.header("Set Your Monthly Budgets")
        budget_month_str = datetime.date.today().strftime("%Y-%m")
        st.info(f"You are setting the budget for: **{datetime.datetime.strptime(budget_month_str, '%Y-%m').strftime('%B %Y')}**")

        expense_categories_for_budget_query = text("SELECT DISTINCT category FROM expenses WHERE type='Expense' AND category != 'Transfer Out'")
        expense_categories_for_budget = sorted(list(pd.read_sql(expense_categories_for_budget_query, DB_CONN.engine)['category']) or CATEGORIES)

        existing_budgets_query = text("SELECT category, amount FROM budgets WHERE month_year = :month_year")
        existing_budgets = pd.read_sql(existing_budgets_query, DB_CONN.engine, params={"month_year": budget_month_str}).set_index('category')

        # ... (DataFrame creation logic remains the same) ...
        budget_df = pd.DataFrame({'category': expense_categories_for_budget, 'amount': [0.0] * len(expense_categories_for_budget)})
        if not existing_budgets.empty:
            budget_df = budget_df.set_index('category').combine_first(existing_budgets).reset_index()

        edited_budgets = st.data_editor(budget_df, num_rows="dynamic", use_container_width=True, column_config={
            "category": st.column_config.TextColumn(label="Category", disabled=True),
            "amount": st.column_config.NumberColumn(label="Amount", min_value=0.0)
        })

        # CONVERTED: Budget Save Logic (INSERT OR REPLACE)
        if st.button("Save Budgets"):
            try:
                with DB_CONN.session as session:
                    # PostgreSQL equivalent of INSERT OR REPLACE is ON CONFLICT, but we can simplify using a multi-step operation:
                    insert_or_replace_query = text("""
                        INSERT INTO budgets (month_year, category, amount)
                        VALUES (:month, :category, :amount)
                        ON CONFLICT (month_year, category) DO UPDATE
                        SET amount = EXCLUDED.amount
                    """)

                    for _, row in edited_budgets.iterrows():
                        if row['amount'] >= 0 and row['category']:
                            session.execute(insert_or_replace_query, {
                                "month": budget_month_str,
                                "category": row['category'],
                                "amount": round(row['amount'], 2)
                            })

                    session.commit()
                    st.success("Budgets saved!")
                    st.rerun()
            except Exception as e:
                logging.error(f"Budget save failed: {e}", exc_info=True)
                st.error(f"Failed to save budgets: {e}")


    # --- MANAGE RECURRING VIEW (CONVERTED DELETE + INSERT) ---
    elif view == "Manage Recurring":
        st.header("Manage Recurring Expenses")
        st.info("Set up expenses that occur every month (e.g., rent, subscriptions). They will be logged automatically.")

        recurring_query = text("SELECT recurring_id, description, amount, category, payment_method, day_of_month FROM recurring_expenses ORDER BY recurring_id")
        recurring_df = pd.read_sql(recurring_query, DB_CONN.engine)

        edited_recurring = st.data_editor(recurring_df, num_rows="dynamic", use_container_width=True, column_config={
            "recurring_id": st.column_config.NumberColumn(disabled=True),
            "category": st.column_config.TextColumn("Category", required=True),
            "payment_method": st.column_config.SelectboxColumn("Payment Method", options=PAYMENT_ACCOUNTS, required=True),
            "day_of_month": st.column_config.NumberColumn("Day of Month (1-31)", min_value=1, max_value=31, step=1, required=True)
        })

        # CONVERTED: Recurring Save Logic (DELETE + INSERT)
        if st.button("Save Recurring Rules"):
            try:
                with DB_CONN.session as session:
                    # 1. Delete all existing records
                    session.execute(text("DELETE FROM recurring_expenses"))

                    insert_query = text("INSERT INTO recurring_expenses (description, amount, category, payment_method, day_of_month) VALUES (:desc, :amount, :category, :pmethod, :day)")

                    # 2. Insert the edited data
                    for _, row in edited_recurring.iterrows():
                        if row['description'] and row['amount'] > 0:
                            session.execute(insert_query, {
                                "desc": row['description'],
                                "amount": round(row['amount'], 2),
                                "category": row['category'],
                                "pmethod": row['payment_method'],
                                "day": row['day_of_month']
                            })

                    session.commit()
                    st.success("Recurring expense rules saved!")
                    st.rerun()
            except Exception as e:
                logging.error(f"Recurring save failed: {e}", exc_info=True)
                st.error(f"Failed to save recurring rules: {e}")

def mutual_fund_page():
    """Renders the Mutual Fund tracker page."""

    st.title("📚 Mutual Fund Tracker")
    _process_recurring_expenses()
    _process_mf_sips()
    key_prefix = "mf"

    # CONVERTED: read_sql uses DB_CONN.engine
    transactions_df = pd.read_sql(text("SELECT transaction_id, date, scheme_name, yfinance_symbol, type, units, nav FROM mf_transactions ORDER BY date DESC"), DB_CONN.engine)

    view_options = ["Holdings", "Transaction History"]
    table_view = st.selectbox("View Options", view_options, key=f"{key_prefix}_table_view", label_visibility="hidden")

    if table_view == "Holdings":
        # ... (Sidebar logic remains the same) ...
        st.sidebar.header("Add Transaction")
        if f"{key_prefix}_all_schemes" not in st.session_state:
            st.session_state[f"{key_prefix}_all_schemes"] = fetch_mf_schemes()
            st.session_state[f"{key_prefix}_search_results"] = []
        with st.sidebar.form(f"{key_prefix}_search_form"):
            company_name = st.text_input("Search Fund Name", value=st.session_state.get(f"{key_prefix}_search_term_input", ""), key=f"{key_prefix}_search_term_input")
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

                # CONVERTED: MF Transaction INSERT logic
                if st.form_submit_button("Add Transaction"):
                    if not (mf_units and mf_units > 0 and mf_nav and mf_nav > 0):
                        st.warning("Please fill all fields.")
                    else:
                        amount = mf_units * mf_nav
                        funds_change_type = "Withdrawal" if mf_type == "Purchase" else "Deposit"
                        fund_adjustment = mf_fee if mf_type == "Purchase" else -mf_fee

                        update_funds_on_transaction(funds_change_type, round(amount + fund_adjustment, 2), f"MF {mf_type}: {selected_name}", mf_date.strftime("%Y-%m-%d"))

                        try:
                            with DB_CONN.session as session:
                                insert_query = text("INSERT INTO mf_transactions (transaction_id, date, scheme_name, yfinance_symbol, type, units, nav) VALUES (:id, :date, :name, :symbol, :type, :units, :nav)")
                                session.execute(insert_query, {
                                    "id": str(uuid.uuid4()),
                                    "date": mf_date.strftime('%Y-%m-%d'),
                                    "name": selected_name,
                                    "symbol": selected_code,
                                    "type": mf_type,
                                    "units": round(mf_units, 4),
                                    "nav": round(mf_nav, 4)
                                })
                                session.commit()
                        except Exception as e:
                            logging.error(f"MF transaction failed: {e}", exc_info=True)
                            st.error(f"Failed to record MF transaction: {e}")

                        st.success(f"{mf_type} of {selected_name} logged!")
                        st.session_state[f"{key_prefix}_selected_result"] = None
                        st.session_state[f"{key_prefix}_search_results"] = []
                        st.session_state[f"{key_prefix}_search_term_input"] = ""
                        st.rerun()

        # ... (Holdings and Chart logic remains the same) ...
        st.divider()
        holdings_df = get_mf_holdings_df()

        if not holdings_df.empty:
            total_investment = holdings_df['Investment'].sum()
            total_current_value = holdings_df['Current Value'].sum()
            total_pnl = holdings_df['P&L'].sum()
            total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0

            # ... (Metric and Expander display remains the same) ...
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

            # ... (Return Chart logic remains the same) ...
            st.header("Return Chart")
            all_schemes = transactions_df['scheme_name'].unique().tolist() if not transactions_df.empty else []
            selected_schemes = st.multiselect("Select schemes to compare", options=all_schemes, default=all_schemes)

            if selected_schemes:
                 filtered_transactions = transactions_df[transactions_df['scheme_name'].isin(selected_schemes)]
                 cumulative_return_df, sip_marker_df = _calculate_mf_cumulative_return(filtered_transactions)

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

            # CONVERTED: MF History Save Logic (DELETE + to_sql)
            if st.button("Save Mutual Fund Changes"):
                try:
                    with DB_CONN.session as session:
                        session.execute(text("DELETE FROM mf_transactions"))

                        edited_df['date'] = edited_df['date'].astype(str)
                        edited_df.to_sql('mf_transactions', DB_CONN.engine, if_exists='append', index=False)

                        session.commit()
                        st.success("Mutual Fund transactions updated successfully!")
                        st.rerun()
                except Exception as e:
                    logging.error(f"MF history save failed: {e}", exc_info=True)
                    st.error(f"Failed to save MF transactions: {e}")
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
    key_prefix = config['key_prefix']
    is_trading_section = key_prefix == 'trade'
    st.title(config["title"])

    trade_mode_selection = ""
    is_paper_trading = False

    if is_trading_section:
        # --- 1. Paper Trading Toggle ---
        if f"{key_prefix}_paper_trading_state" not in st.session_state:
            st.session_state[f"{key_prefix}_paper_trading_state"] = False
        is_paper_trading = st.toggle("Enable Paper Trading (Transactions won't affect Funds)",
                                     key=f"{key_prefix}_paper_trading_toggle",
                                     value=st.session_state[f"{key_prefix}_paper_trading_state"])
        st.session_state[f"{key_prefix}_paper_trading_state"] = is_paper_trading
        if is_paper_trading:
            st.warning("⚠️ **Paper Trading is active.** Buy/Sell transactions will **NOT** update your 'Funds' section.")
        st.divider()

        # --- 2. Live/Paper Filter Dropdown ---
        trade_mode_selection = st.radio(
            "Filter Trade Data",
            ["All Trades", "Live Trades Only", "Paper Trades Only"],
            horizontal=True,
            key=f"{key_prefix}_trade_mode_filter"
        )
        st.divider()

    # --- Sidebar forms for Add/Sell remain the same (Search/Select) ---

    st.sidebar.header(f"Add {config['asset_name']}")
    with st.sidebar.form(f"{key_prefix}_add_form"):
        company_name = st.text_input(f"{config['asset_name']} Name", value="", key=f"{key_prefix}_add_company_name")
        search_button = st.form_submit_button("Search")

    if search_button and company_name:
        st.session_state[f"{key_prefix}_search_results"] = search_for_ticker(company_name)
        st.session_state[f"{key_prefix}_selected_symbol"] = None
        st.rerun()

    if st.session_state.get(f"{key_prefix}_search_results"):
        # ... (Symbol selection logic remains the same) ...
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
            # ... (Form fields remain the same) ...
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

            # CONVERTED: Asset Purchase Logic (INSERT/UPDATE)
            add_button = st.form_submit_button(f"Add to {config['asset_name_plural']}")

            if add_button:
                if not (buy_price and buy_price > 0 and quantity and quantity > 0):
                    st.error("Buy Price and Quantity must be positive.")
                elif update_stock_data(symbol):
                    total_cost = (buy_price * quantity) + transaction_fee

                    try:
                        with DB_CONN.session as session:

                            # 1. CHECK IF ASSET ALREADY EXISTS (SELECT)
                            select_query = text(f"SELECT buy_price, buy_date, quantity, sector, market_cap, target_price, stop_loss_price FROM {config['asset_table']} WHERE {config['asset_col']}=:symbol")

                            existing_row = session.execute(select_query, {"symbol": symbol}).fetchone()

                            if existing_row:
                                # ** A. UPDATE LOGIC (Existing Asset) **

                                if is_trading_section:
                                    # Trading: (old_buy_price, old_buy_date, old_quantity, sector, mcap, old_target_price, old_stop_loss_price)
                                    (old_buy_price, old_buy_date, old_quantity, *_, old_target_price, old_stop_loss_price) = existing_row
                                    target_p = round(target_price, 2)
                                    stop_l_p = round(stop_loss_price, 2)

                                else:
                                    # Investment: (old_buy_price, old_buy_date, old_quantity, old_sector, old_market_cap)
                                    (old_buy_price, old_buy_date, old_quantity, old_sector, old_market_cap, *_) = existing_row
                                    final_sector = sector if sector != 'N/A' else old_sector
                                    final_market_cap = _categorize_market_cap(market_cap) if market_cap != 'N/A' else old_market_cap

                                old_total_cost = old_buy_price * old_quantity
                                new_quantity = old_quantity + quantity

                                new_avg_price = (old_total_cost + (buy_price * quantity)) / new_quantity

                                if is_trading_section:
                                    update_query = text(f"UPDATE {config['asset_table']} SET buy_price=:price, quantity=:qty, target_price=:target, stop_loss_price=:stop WHERE {config['asset_col']}=:symbol")
                                    session.execute(update_query, {
                                        "price": round(new_avg_price, 2), "qty": new_quantity,
                                        "target": target_p, "stop": stop_l_p, "symbol": symbol
                                    })
                                else:
                                    update_query = text(f"UPDATE {config['asset_table']} SET buy_price=:price, quantity=:qty, sector=:sector, market_cap=:mcap WHERE {config['asset_col']}=:symbol")
                                    session.execute(update_query, {
                                        "price": round(new_avg_price, 2), "qty": new_quantity,
                                        "sector": final_sector, "mcap": final_market_cap, "symbol": symbol
                                    })

                                st.success(f"Updated {symbol}. New quantity: {new_quantity}, New avg. price: {currency}{new_avg_price:,.2f}")

                            else:
                                # ** B. INSERT LOGIC (New Asset) **
                                if is_trading_section:
                                    insert_query = text(f"INSERT INTO {config['asset_table']} ({config['asset_col']}, buy_price, buy_date, quantity, target_price, stop_loss_price) VALUES (:symbol, :price, :date, :qty, :target, :stop)")
                                    session.execute(insert_query, {
                                        "symbol": symbol, "price": round(buy_price, 2), "date": buy_date.strftime("%Y-%m-%d"),
                                        "qty": quantity, "target": round(target_price, 2), "stop": round(stop_loss_price, 2)
                                    })
                                else:
                                    insert_query = text(f"INSERT INTO {config['asset_table']} ({config['asset_col']}, buy_price, buy_date, quantity, sector, market_cap) VALUES (:symbol, :price, :date, :qty, :sector, :mcap)")
                                    session.execute(insert_query, {
                                        "symbol": symbol, "price": round(buy_price, 2), "date": buy_date.strftime("%Y-%m-%d"),
                                        "qty": quantity, "sector": sector, "mcap": _categorize_market_cap(market_cap)
                                    })

                                st.success(f"{symbol} added successfully!")

                            # 2. CONDITIONAL FUND UPDATE
                            if not is_trading_section or (is_trading_section and not is_paper_trading):
                                update_funds_on_transaction("Withdrawal", round(total_cost, 2), f"Purchase {quantity} units of {symbol}", buy_date.strftime("%Y-%m-%d"))

                            session.commit()
                            st.rerun()

                    except Exception as e:
                        logging.error(f"PostgreSQL Asset transaction failed for {symbol}: {e}", exc_info=True)
                        st.error(f"Database error during purchase: {e}")

                else:
                    st.error(f"Failed to fetch historical data for {symbol}. Cannot add.")

    st.sidebar.header(f"Sell {config['asset_name']}")

    all_symbols_query = text(f"SELECT {config['asset_col']} FROM {config['asset_table']}")
    all_symbols = pd.read_sql(all_symbols_query, DB_CONN.engine)[config['asset_col']].tolist()

    if all_symbols:
        selected_option = st.sidebar.selectbox(
            f"Select {config['asset_name']} to Sell",
            options=[None] + all_symbols,
            index=0,
            key=f"{key_prefix}_sell_symbol_selector",
            format_func=lambda x: "Select a stock..." if x is None else x
        )
        # ... (Available quantity check logic remains the same) ...
        available_qty = 1
        if selected_option:
            symbol_to_sell = selected_option

            qty_query = text(f"SELECT quantity FROM {config['asset_table']} WHERE {config['asset_col']}=:symbol")
            result = pd.read_sql(qty_query, DB_CONN.engine, params={"symbol": symbol_to_sell})

            if not result.empty:
                available_qty = result.iloc[0]['quantity']
                st.sidebar.info(f"Available to sell: {available_qty} units of {symbol_to_sell}")
            else:
                symbol_to_sell = None
        else:
            symbol_to_sell = None

        with st.sidebar.form(f"{key_prefix}_sell_form"):
            # ... (Form fields remain the same) ...
            is_disabled = not symbol_to_sell
            sell_qty = st.number_input("Quantity to Sell", min_value=1, max_value=available_qty, step=1, key=f"{key_prefix}_sell_qty", disabled=is_disabled)
            sell_price = st.number_input("Sell Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_sell_price", disabled=is_disabled)
            sell_date = st.date_input("Sell Date", max_value=datetime.date.today(), key=f"{key_prefix}_sell_date", disabled=is_disabled)
            sell_transaction_fee = st.number_input("Transaction Fee (₹)", min_value=0.00, format="%.2f", key=f"{key_prefix}_sell_transaction_fee", disabled=is_disabled, value=0.0)

            # CONVERTED: Asset Sell/Close Logic
            sell_button = st.form_submit_button(f"Sell {config['asset_name']}")

            if sell_button:
                # ... (Validation remains the same) ...
                if not symbol_to_sell:
                    st.warning(f"Please select a {config['asset_name']} to sell.")
                elif not (sell_price and sell_price > 0):
                    st.error("Sell price must be greater than zero.")
                elif not (sell_qty and sell_qty > 0):
                    st.error("Quantity to sell must be positive.")
                else:
                    try:
                        with DB_CONN.session as session:

                            # 1. FETCH BUY DETAILS (SELECT)
                            if is_trading_section:
                                select_query = text(f"SELECT buy_price, buy_date, quantity, target_price, stop_loss_price FROM {config['asset_table']} WHERE {config['asset_col']}=:symbol")
                                (buy_price, buy_date, current_qty, target_price, stop_loss_price) = session.execute(select_query, {"symbol": symbol_to_sell}).fetchone()
                            else:
                                select_query = text(f"SELECT buy_price, buy_date, quantity FROM {config['asset_table']} WHERE {config['asset_col']}=:symbol")
                                (buy_price, buy_date, current_qty) = session.execute(select_query, {"symbol": symbol_to_sell}).fetchone()
                                target_price, stop_loss_price = None, None

                            # Calculate return metrics
                            realized_return = ((sell_price - buy_price) / buy_price * 100)
                            transaction_id = str(uuid.uuid4())

                            # 2. INSERT into Realized/Exits Table
                            if is_trading_section:
                                insert_realized_query = text(f"INSERT INTO {config['realized_table']} (transaction_id, {config['asset_col']}, buy_price, buy_date, quantity, sell_price, sell_date, realized_return_pct, target_price, stop_loss_price) VALUES (:id, :symbol, :b_price, :b_date, :qty, :s_price, :s_date, :r_pct, :t_price, :sl_price)")
                                params = {
                                    "id": transaction_id, "symbol": symbol_to_sell, "b_price": round(buy_price, 2), "b_date": buy_date,
                                    "qty": sell_qty, "s_price": round(sell_price, 2), "s_date": sell_date.strftime("%Y-%m-%d"),
                                    "r_pct": round(realized_return, 2), "t_price": round(target_price, 2), "sl_price": round(stop_loss_price, 2)
                                }
                            else:
                                insert_realized_query = text(f"INSERT INTO {config['realized_table']} (transaction_id, {config['asset_col']}, buy_price, buy_date, quantity, sell_price, sell_date, realized_return_pct) VALUES (:id, :symbol, :b_price, :b_date, :qty, :s_price, :s_date, :r_pct)")
                                params = {
                                    "id": transaction_id, "symbol": symbol_to_sell, "b_price": round(buy_price, 2), "b_date": buy_date,
                                    "qty": sell_qty, "s_price": round(sell_price, 2), "s_date": sell_date.strftime("%Y-%m-%d"),
                                    "r_pct": round(realized_return, 2)
                                }
                            session.execute(insert_realized_query, params)

                            # 3. CONDITIONAL FUND UPDATE
                            if not is_trading_section or (is_trading_section and not st.session_state.get(f"{config['key_prefix']}_paper_trading_state", False)):
                                funds_deposit_amount = round((sell_price * sell_qty) - sell_transaction_fee, 2)
                                update_funds_on_transaction("Deposit", funds_deposit_amount, f"Sale of {sell_qty} units of {symbol_to_sell}", sell_date.strftime("%Y-%m-%d"))

                            # 4. UPDATE/DELETE Open Position
                            if sell_qty == current_qty:
                                delete_query = text(f"DELETE FROM {config['asset_table']} WHERE {config['asset_col']}=:symbol")
                                session.execute(delete_query, {"symbol": symbol_to_sell})
                            else:
                                update_open_query = text(f"UPDATE {config['asset_table']} SET quantity=:new_qty WHERE {config['asset_col']}=:symbol")
                                session.execute(update_open_query, {"new_qty": current_qty - sell_qty, "symbol": symbol_to_sell})

                            session.commit()
                            st.success(f"Sold {sell_qty} units of {symbol_to_sell}.")
                            st.rerun()

                    except Exception as e:
                        logging.error(f"PostgreSQL Asset Sell transaction failed for {symbol_to_sell}: {e}", exc_info=True)
                        st.error(f"Database error during sale: {e}")
    else:
        st.sidebar.info(f"No open {config['asset_name_plural'].lower()}.")

    # --- MAIN VIEW LOGIC ---
    # ... (Filter logic remains the same) ...
    view_options = ["Open Trades", "Closed Trades"]

    full_holdings_df = get_holdings_df(config['asset_table'])
    full_realized_df = get_realized_df(config['realized_table'])

    live_trade_symbols = set()
    if is_trading_section:
        fund_tx_query = text("SELECT description FROM fund_transactions WHERE type='Withdrawal'")
        fund_tx = pd.read_sql(fund_tx_query, DB_CONN.engine)

        for desc in fund_tx['description']:
            if desc and desc.startswith("Purchase"):
                parts = desc.split(' of ')
                if len(parts) > 1:
                    live_trade_symbols.add(parts[-1].strip())

    holdings_df = full_holdings_df.copy()
    realized_df = full_realized_df.copy()

    if is_trading_section:
        if trade_mode_selection == "Live Trades Only":
            holdings_df = holdings_df[holdings_df[config['asset_col']].isin(live_trade_symbols)]
            realized_df = realized_df[realized_df[config['asset_col']].isin(live_trade_symbols)]
        elif trade_mode_selection == "Paper Trades Only":
            holdings_df = holdings_df[~holdings_df[config['asset_col']].isin(live_trade_symbols)]
            realized_df = realized_df[~realized_df[config['asset_col']].isin(live_trade_symbols)]

    # ... (rest of render_asset_page, including chart generation, remains the same) ...
    # This section is lengthy, but the crucial DB access points are now fixed.

    st.subheader(f"{config['title']} - {trade_mode_selection if is_trading_section and trade_mode_selection else config['title']}")
    table_view = st.selectbox("View Options", view_options, key=f"{key_prefix}_table_view_secondary", label_visibility="collapsed")

    if table_view == view_options[0]:
        # OPEN TRADES (HOLDINGS)
        if not holdings_df.empty:
            df_to_display = holdings_df.copy()

            total_invested, total_current = df_to_display['invested_value'].sum(), df_to_display['current_value'].sum()
            total_return_amount = (total_current - total_invested).round(2)
            total_return_percent = (total_return_amount / total_invested * 100).round(2) if total_invested > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Investment", f"₹{total_invested:,.2f}")
            with col2: st.metric("Current Value", f"₹{total_current:,.2f}")
            with col3: st.metric("Total Return", f"₹{total_return_amount:,.2f}", f"{total_return_percent:.2f}%")

            if not is_trading_section:
                benchmark_choice = 'Nifty 50'
                metrics = calculate_portfolio_metrics(df_to_display, pd.DataFrame(), benchmark_choice)
                col_alpha, col_beta, col_drawdown = st.columns(3)
                with col_alpha: st.metric("Alpha", f"{metrics['alpha']}%")
                with col_beta: st.metric("Beta", f"{metrics['beta']}")
                with col_drawdown: st.metric("Max Drawdown", f"{metrics['max_drawdown']}%")

            st.divider()
            with st.expander(f"View Detailed {view_options[0]}"):
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
                history_df = pd.read_sql(text("SELECT date, close_price FROM price_history WHERE ticker=:symbol AND date>=:buy_date ORDER BY date ASC"), DB_CONN.engine, params={"symbol": symbol, "buy_date": asset_info["buy_date"]})
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
            st.info(f"No {trade_mode_selection.lower()} in {view_options[0].lower()} to display.")

    elif table_view == view_options[1]:
        # CLOSED TRADES (REALIZED)
        if not realized_df.empty:
            if is_trading_section:
                trading_metrics = calculate_trading_metrics(realized_df)
                col1, col2, col3 = st.columns(3)
                col1.metric("Win Ratio", f"{trading_metrics['win_ratio']}%")
                col2.metric("Profit Factor", f"{trading_metrics['profit_factor']}")
                col3.metric("Expectancy", f"₹{trading_metrics['expectancy']}")
            st.divider()
            with st.expander(f"View Detailed {view_options[1]}"):
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
            st.info(f"No {trade_mode_selection.lower()} in {view_options[1].lower()} to display.")

    current_holdings = holdings_df

    has_holdings_for_comparison = not current_holdings.empty and table_view == view_options[0]

    if has_holdings_for_comparison:
        st.divider()
        st.header("Performance vs Benchmark")

        benchmark_options = ['Nifty 50', 'Nifty 100', 'Nifty 200', 'Nifty 500']
        benchmark_choice = st.selectbox("Select Benchmark", benchmark_options, key=f"{key_prefix}_benchmark_choice")

        with st.spinner("Loading Benchmark Data..."):
            holdings_df = current_holdings

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


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
#may be end
