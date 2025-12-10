import streamlit as st
import pandas as pd
import datetime
import logging
import requests
import altair as alt
import uuid
import numpy as np
from mftool import Mftool
import time
import yfinance as yf

# SQLALCHEMY imports are here
from sqlalchemy import create_engine, text as _sql_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

# helper context manager for sessions
from contextlib import contextmanager

# ------------------ CONFIG & LOGGING ------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:root:%(message)s")

# ------------------ END NAVIGATION HELPER ------------------------------------


st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="pages/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CACHE CLEARING FUNCTIONS ---

def clear_mf_caches():
    """Clears caches related to Mutual Funds (MF transactions, NAVs, holdings, returns)."""
    if 'mf_all_schemes' in st.session_state:
        st.session_state.pop('mf_all_schemes')
    st.cache_data(fetch_mf_schemes).clear()
    st.cache_data(fetch_latest_mf_nav).clear()
    st.cache_data(get_mf_historical_data).clear()
    st.cache_data(get_mf_holdings_df).clear()
    st.cache_data(_calculate_mf_cumulative_return).clear()
    st.cache_data(get_combined_returns).clear()
    # Clear the fundamental transaction data which affects MF P&L calculations
    st.cache_data(get_fund_transactions_df).clear()
    st.cache_data(get_live_trade_symbols).clear()

def clear_all_data_caches():
    """Clears the most common data caches (DB reads and computed holdings)."""
    st.cache_data(db_query).clear()
    st.cache_data(get_holdings_df).clear()
    st.cache_data(get_realized_df).clear()
    st.cache_data(get_current_portfolio_allocation).clear()
    st.cache_data(get_combined_returns).clear()

    # Clear the core transaction and expense data caches
    st.cache_data(get_fund_transactions_df).clear()
    st.cache_data(get_all_expenses_df).clear()
    st.cache_data(get_live_trade_symbols).clear()

    # Also clear MF caches, just in case
    clear_mf_caches()


# ------------------ DATABASE (NEON / SQLALCHEMY) ------------------

@st.cache_resource
def get_engine_and_sessionmaker():
    # discover connection url
    candidates = [
        ("neon_db", "sqlalchemy_url"),
        ("database", "sqlalchemy_url"),
        ("supabase_db", "sqlalchemy_url"),
    ]
    sa_url = None
    for key, sub in candidates:
        sa_url = st.secrets.get(key, {}).get(sub)
        if sa_url:
            logging.info(f"Using SQLAlchemy URL from st.secrets['{key}']['{sub}']")
            break
    if not sa_url:
        sa_url = st.secrets.get("DATABASE_URL")
        if sa_url:
            logging.info("Using SQLAlchemy URL from st.secrets['DATABASE_URL']")

    if not sa_url:
        st.error("No SQLAlchemy URL found in Streamlit secrets. Please add your Neon/Postgres URL under st.secrets['neon_db']['sqlalchemy_url'] or st.secrets['DATABASE_URL'].")
        st.stop()

    try:
        engine = create_engine(sa_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # quick connectivity check
        with engine.connect() as conn:
            conn.execute(_sql_text("SELECT 1"))
        logging.info("Successfully created SQLAlchemy engine and verified connectivity.")
        return engine, SessionLocal
    except SQLAlchemyError as e:
        logging.error(f"Failed to create SQLAlchemy engine or connect: {e}")
        st.error(f"Database connection failed: {e}")
        st.stop()

DB_ENGINE, SessionLocal = get_engine_and_sessionmaker()

# helper context manager for sessions


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# ------------------ DB helpers ------------------

@st.cache_data(show_spinner=False, ttl=60) # Reduced cache TTL for faster data updates
def db_query(sql: str, params: dict = None) -> pd.DataFrame:
    """Run a read query and return a pandas DataFrame using SQLAlchemy engine."""
    try:
        if params:
            return pd.read_sql_query(_sql_text(sql), DB_ENGINE, params=params)
        else:
            return pd.read_sql_query(_sql_text(sql), DB_ENGINE)
    except Exception as e:
        logging.error(f"db_query failed: {e}\nSQL: {sql}")
        return pd.DataFrame()


def db_execute(sql: str, params: dict = None, cache_clear_type='general'):
    """Execute DML/DDL using a session, clearing cache selectively."""
    if cache_clear_type == 'general':
        clear_all_data_caches()
    elif cache_clear_type == 'mf':
        clear_mf_caches()
    elif cache_clear_type == 'none':
        # Don't clear caches, used primarily for DDL in init
        pass

    try:
        with get_session() as session:
            if params:
                res = session.execute(_sql_text(sql), params)
            else:
                res = session.execute(_sql_text(sql))
            return res
    except Exception as e:
        logging.error(f"db_execute failed: {e}\nSQL: {sql}")
        # Note: DDL errors are caught and logged inside initialize_database
        if 'ALTER TABLE' not in sql: # Allow DDL errors to pass through in init, but raise for DML/DQL
             raise


def df_to_table(df: pd.DataFrame, table_name: str, if_exists: str = 'append', cache_clear_type='general'):
    """Insert DataFrame into a SQL table using pandas.to_sql via SQLAlchemy engine.
Falls back to row-by-row on failure."""
    if df is None or df.empty:
        return

    if cache_clear_type == 'general':
        clear_all_data_caches()
    elif cache_clear_type == 'mf':
        clear_mf_caches()

    # --- FIX 1A: Ensure column names are flat strings for pandas.to_sql (Primary method) ---
    # Aggressive column flattening to handle yfinance MultiIndex columns (e.g., ('Close', 'TICKER'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip('_') for col in df.columns.values]

    # Standardize column names for 'price_history' before attempting to_sql
    if table_name == 'price_history':
        df_columns = df.columns.tolist()
        new_columns = {}
        found_close_price = False

        for col in df_columns:
            # Standard names
            if col == 'ticker' or col == 'date':
                new_columns[col] = col
            # The name we want in the DB
            elif col == 'close_price' or col == 'Close':
                new_columns[col] = 'close_price'
                found_close_price = True
            # Catch the problematic ticker-appended names (e.g., 'Close_GROWW.NS', 'close_price_VBL.NS')
            # and force them to be 'close_price'
            elif ('close_price' in col.lower() or 'close' in col.lower()) and any(s in col for s in ['.NS', '.BO']):
                new_columns[col] = 'close_price'
                found_close_price = True
            else:
                # Keep other non-essential columns if they exist, but don't force 'close_price'
                new_columns[col] = col

        # Apply renaming to standardize the key column
        df.rename(columns=new_columns, inplace=True)

        if not found_close_price and 'close_price' not in df.columns:
            logging.error(f"Price data conversion failed in df_to_table: No standardized 'close_price' found in {df.columns.tolist()}")
            # Skip primary insert but allow fallback to clean up

        # Ensure we only keep the three required columns for price_history table structure
        if {'ticker', 'date', 'close_price'}.issubset(df.columns):
            df = df[['ticker', 'date', 'close_price']].copy()
        else:
            logging.warning("Skipping primary to_sql for price_history as columns are incorrect for insertion.")
            # Move on to fallback if primary to_sql fails below

    # --------------------------------------------------------------------------------------

    try:
        # Check if the DataFrame is empty after filtering/preparation
        if df.empty:
            return

        df.to_sql(table_name, DB_ENGINE, if_exists=if_exists, index=False)
        return
    except Exception as e:
        # We only log this as a warning because the row-by-row fallback is designed to handle it.
        logging.warning(f"pandas.to_sql failed for {table_name}: {e}. Falling back to row inserts.")

    # --- FIX 1B: Aggressively Prepare DataFrame for row-by-row insert fallback ---
    df_fallback = df.copy()

    if table_name == 'price_history':
        # The dataframe should already be standardized to ['ticker', 'date', 'close_price']
        if not {'ticker', 'date', 'close_price'}.issubset(df_fallback.columns):
            logging.error(f"Fallback setup failed: Price history DataFrame lacks required columns: {df_fallback.columns.tolist()}")
            return

    # ----------------------------------------------------------------------

    # fallback (Row-by-row insertion)
    records = df_fallback.to_dict(orient='records')
    with get_session() as session:
        for rec in records:
            cols = ', '.join(rec.keys())
            # Use safe bind parameter naming (PostgreSQL/SQLAlchemy style: :col_name)
            vals = ', '.join(':' + k for k in rec.keys())
            try:
                # This fixes the binding issue by ensuring 'cols' and the keys in 'rec' are simple strings (e.g., 'close_price')
                session.execute(_sql_text(f"INSERT INTO {table_name} ({cols}) VALUES ({vals})"), rec)
            except Exception as ex:
                logging.warning(f"Failed inserting row into {table_name}: {ex}")

# ------------------ APP CONFIG ------------------
# NOTE: Credentials now rely solely on st.secrets, removing hardcoded defaults for security.
USERNAME = st.secrets.get("auth", {}).get("username")
PASSWORD = st.secrets.get("auth", {}).get("password")

# --- VIEWER CREDENTIALS ---
# Viewer uses VIEWER_USERNAME as the Access Code, password is ignored for UI simplicity
VIEWER_USERNAME = st.secrets.get("auth", {}).get("viewer_username")
VIEWER_PASSWORD = st.secrets.get("auth", {}).get("viewer_password")
# --------------------------

def login_page():
    # --- Dark Color Schema ---
    BACKGROUND_COLOR = "#0F172A"  # Dark Slate Blue
    CARD_COLOR = "#1F2937"        # Dark Grey/Slate for Card
    PRIMARY_COLOR = "#10B981"      # Emerald Green Accent
    TEXT_COLOR = "#F3F4F6"        # Light Text

    # Inject custom CSS for a superb dark login page
    st.markdown(f"""
        <style>
        .stApp {{
            background: {BACKGROUND_COLOR};
        }}
        .stApp > header {{
            display: none; /* Hide Streamlit header */
        }}
        /* Top title style */
        .main-title {{
            color: {TEXT_COLOR};
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            margin-top: 5vh;
            margin-bottom: 30px;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
        }}
        /* Style for the centered login card */
        .login-card {{
            background-color: {CARD_COLOR};
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
            max-width: 400px;
            margin: 0 auto;
            /* Center horizontally */
            border: 1px solid #374151;
        }}
        /* Streamlit input overrides */
        .stTextInput > label, .stToggle > label, .stSubheader {{
            color: {TEXT_COLOR} !important;
            font-weight: 500;
        }}
        .stTextInput div > div > input {{
            background-color: #374151;
            /* Dark Input Background */
            color: {TEXT_COLOR};
            border: 1px solid #4B5563;
            border-radius: 6px;
            padding: 10px;
        }}
        .stButton button {{
            width: 100%;
            background-color: {PRIMARY_COLOR};
            color: {BACKGROUND_COLOR}; /* Dark text on bright button */
            font-weight: bold;
            border-radius: 6px;
            transition: background-color 0.3s;
            border: none;
            padding: 10px;
        }}
        .stButton button:hover {{
            background-color: #059669;
            /* Darker green on hover */
            color: {TEXT_COLOR};
        }}
        /* FIX: Ensure the toggle visuals contrast */
        .stToggle label, .stToggle label span {{
            color: {TEXT_COLOR} !important;
            /* Set label text color */
        }}
        /* Specific contrast fix for the Streamlit toggle background */
        .stToggle div:has(input[type="checkbox"]) {{
            border-color: #4B5563 !important;
            /* Border color for the toggle box */
            background-color: #374151 !important;
            /* Dark background for contrast */
        }}
        .stToggle input[type="checkbox"]:checked + div {{
            background-color: {PRIMARY_COLOR} !important;
            /* Background color when checked */
            border-color: {PRIMARY_COLOR} !important;
        }}
        .stToggle {{
            display: flex;
            justify-content: flex-start;
            padding-bottom: 20px;
            color: {TEXT_COLOR}; /* Ensuring the "Login as Viewer" label text is light */
        }}
        /* Subheaders inside the card */
        .stContainer > h3 {{
            color: {PRIMARY_COLOR};
        }}
        /* Remove the horizontal line separating the toggle/header block from the inputs */
        hr {{
            display: none;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Top title outside the login box
    st.markdown('<p class="main-title">FIN-Tracker</p>', unsafe_allow_html=True)

    # Use columns to contain the content in the center
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Centered Login Card
        with st.container(border=False):
            st.markdown('<div class="login-card">', unsafe_allow_html=True)

            # Use a toggle for role selection
            is_viewer = st.toggle("Login as Viewer", key="login_as_viewer")

            # Use st.form to debounce inputs and prevent script re-runs while typing
            with st.form(key='login_form', clear_on_submit=False):

                if not is_viewer:
                    st.subheader("Owner Login")
                    # Use unique keys for inputs within the form
                    st.text_input("Username", placeholder="Owner Username", key="form_owner_username")
                    st.text_input("Password", type="password", placeholder="Owner Password", key="form_owner_password")
                else:
                    st.subheader("Viewer Access")
                    # Only prompt for Access Code
                    st.text_input("Access Code", placeholder="Viewer Access Code (e.g., viewer)", key="form_viewer_access_code")

                # Use st.form_submit_button to trigger the check
                submitted = st.form_submit_button("Login")

            if submitted:
                login_successful = False
                role = "owner" # Default role

                if not is_viewer:
                    # Owner login attempt
                    # USERname and PASSWORD are read from st.secrets.
                    if st.session_state.get("form_owner_username") == USERNAME and st.session_state.get("form_owner_password") == PASSWORD:
                        login_successful = True
                        role = "owner"
                else:
                    # Viewer login attempt (using access code only)
                    # VIEWER_USERNAME is read from st.secrets.
                    if st.session_state.get("form_viewer_access_code") == VIEWER_USERNAME:
                        login_successful = True
                        role = "viewer"

                if login_successful:
                    st.session_state.logged_in = True
                    st.session_state.role = role # Store role
                    st.success(f"Login successful as {role.capitalize()}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials or access code.")

            st.markdown('</div>', unsafe_allow_html=True)


# ------------------ NAVIGATION HELPER ------------------
def set_page(page_name):
    """Simple function to set the current page in session state."""
    st.session_state.page = page_name
# ------------------ END NAVIGATION HELPER ------------------------------------


# ------------------ DB SCHEMA INIT ------------------
def initialize_database():
    ddl_commands = [
        "CREATE TABLE IF NOT EXISTS portfolio (ticker TEXT PRIMARY KEY, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1, sector TEXT, market_cap TEXT)",
        "CREATE TABLE IF NOT EXISTS trades (symbol TEXT PRIMARY KEY, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1, target_price REAL NOT NULL, stop_loss_price REAL NOT NULL)",
        "CREATE TABLE IF NOT EXISTS price_history (ticker TEXT, date TEXT, close_price REAL, PRIMARY KEY (ticker, date))",
        "CREATE TABLE IF NOT EXISTS realized_stocks (transaction_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL, sell_price REAL NOT NULL, sell_date TEXT NOT NULL, realized_return_pct REAL NOT NULL)",
        "CREATE TABLE IF NOT EXISTS exits (transaction_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL, sell_price REAL NOT NULL, sell_date TEXT NOT NULL, realized_return_pct REAL NOT NULL, target_price REAL NOT NULL, stop_loss_price REAL NOT NULL)",
        "CREATE TABLE IF NOT EXISTS fund_transactions (transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, type TEXT NOT NULL, amount REAL NOT NULL, description TEXT, transfer_group_id TEXT)",
        "CREATE TABLE IF NOT EXISTS expenses (expense_id TEXT PRIMARY KEY, date TEXT NOT NULL, amount REAL NOT NULL, category TEXT NOT NULL, payment_method TEXT, description TEXT, type TEXT, transfer_group_id TEXT)",
        "CREATE TABLE IF NOT EXISTS budgets (budget_id SERIAL PRIMARY KEY, month_year TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL, UNIQUE(month_year, category))",
        "CREATE TABLE IF NOT EXISTS recurring_expenses (recurring_id SERIAL PRIMARY KEY, description TEXT NOT NULL UNIQUE, amount REAL NOT NULL, category TEXT NOT NULL, payment_method TEXT, day_of_month INTEGER NOT NULL)",
        "CREATE TABLE IF NOT EXISTS mf_transactions (transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, scheme_name TEXT NOT NULL, yfinance_symbol TEXT NOT NULL, type TEXT NOT NULL, units REAL NOT NULL, nav REAL NOT NULL, amount REAL)",
        "CREATE TABLE IF NOT EXISTS mf_sips (sip_id SERIAL PRIMARY KEY, scheme_name TEXT NOT NULL UNIQUE, yfinance_symbol TEXT NOT NULL, amount REAL NOT NULL, day_of_month INTEGER NOT NULL)",
    ]

    # MIGRATION COMMANDS: Fixes the 'transfer_group_id does not exist' error on existing tables
    migration_commands = [
        "ALTER TABLE fund_transactions ADD COLUMN IF NOT EXISTS transfer_group_id TEXT",
        "ALTER TABLE expenses ADD COLUMN IF NOT EXISTS transfer_group_id TEXT",
        # Add 'amount' to mf_transactions for simpler historical sync (will default to null if it existed previously)
        "ALTER TABLE mf_transactions ADD COLUMN IF NOT EXISTS amount REAL",
    ]

    for sql in ddl_commands + migration_commands:
        try:
            # We use cache_clear_type='none' because we don't want to clear main caches during init
            db_execute(sql, cache_clear_type='none')
        except Exception as e:
            logging.warning(f"DDL/Migration execution failed for SQL: {sql[:60]}... Error: {e}")

initialize_database()

# ------------------ FUND / TRANSACTION HELPERS ------------------

def update_funds_on_transaction(transaction_type, amount, description, date):
    if description and description.startswith("ALLOCATION:"):
        description = description.split(' - ', 1)[-1].strip()
    sql = "INSERT INTO fund_transactions (transaction_id, date, type, amount, description, transfer_group_id) VALUES (:id, :date, :type, :amount, :desc, :tg_id)"
    params = {'id': str(uuid.uuid4()), 'date': date, 'type': transaction_type, 'amount': amount, 'desc': description, 'tg_id': None}
    db_execute(sql, params, cache_clear_type='general')

# ------------------ TECHNICAL INDICATORS (Unused in final UI but kept) ------------------

def rsi(close, period=14):
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(close, period=20, std_dev=2):
    rolling_mean = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

# ------------------ CACHED TRANSACTION DATA FETCHERS (NEW) ------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_fund_transactions_df():
    """Fetches and processes fund transactions for display and balance calculation."""
    # Query all fund transactions
    fund_df = db_query("SELECT transaction_id, date, type, amount, description, transfer_group_id FROM fund_transactions")
    if fund_df.empty:
        return pd.DataFrame()

    fund_df['date'] = pd.to_datetime(fund_df['date'], format='%Y-%m-%d', errors='coerce')
    fund_df['balance'] = fund_df.apply(lambda row: row['amount'] if row['type'] == 'Deposit' else -row['amount'], axis=1)

    # Calculate cumulative balance on a chronologically sorted copy for accuracy
    chronological_df = fund_df.copy()
    # Primary sort by date (ASC), secondary sort by ID (ASC) to ensure correct chronological cumulative sum
    chronological_df.sort_values(['date', 'transaction_id'], ascending=[True, True], inplace=True)
    chronological_df['cumulative_balance'] = chronological_df['balance'].cumsum()

    # Merge the cumulative balance back onto the display DF
    fund_df = fund_df.merge(
        chronological_df[['transaction_id', 'cumulative_balance']],
        on='transaction_id',
        how='left'
    )
    # Final sort for display (newest day first, OLDEST entry first within the day - File 2 logic)
    fund_df.sort_values(['date', 'transaction_id'], ascending=[False, True], inplace=True)
    return fund_df

@st.cache_data(ttl=60, show_spinner=False)
def get_all_expenses_df():
    """Fetches all raw expense/income/transfer data for dashboard and history views."""
    all_expenses_df = db_query("SELECT * from expenses")
    if all_expenses_df.empty:
        return pd.DataFrame()
    all_expenses_df['date'] = pd.to_datetime(all_expenses_df['date'])
    return all_expenses_df

@st.cache_data(ttl=60, show_spinner=False)
def get_live_trade_symbols():
    """Identifies and caches the set of stock symbols linked to fund withdrawals."""
    fund_tx = db_query("SELECT description FROM fund_transactions WHERE type='Withdrawal'")
    live_trade_symbols = set()
    if not fund_tx.empty:
        for desc in fund_tx['description']:
            # Example: "Purchase 10 units of AAPL"
            if desc and desc.startswith("Purchase"):
                parts = desc.split(' of ')
                if len(parts) > 1:
                    live_trade_symbols.add(parts[-1].strip())
    return live_trade_symbols
# ------------------ END CACHED TRANSACTION DATA FETCHERS ------------------


# ------------------ DATA FETCHERS ------------------

@st.cache_data(ttl=3600)
def search_for_ticker(company_name):
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

@st.cache_data(ttl=180) # Increased TTL for external market price API calls
def fetch_stock_info(symbol):
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
            # fallback to DB price if available
            if price is None:
                try:
                    df = db_query("SELECT close_price FROM price_history WHERE ticker = :symbol ORDER BY date DESC LIMIT 1", params={'symbol': symbol})
                    if not df.empty:
                        price = df['close_price'].iloc[0]
                except Exception:
                    pass
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
    try:
        mf = Mftool()
        schemes = mf.get_scheme_codes()
        return {v: k for k, v in schemes.items()}
    except Exception as e:
        logging.error(f"Failed to fetch mutual fund schemes: {e}")
        return {}

@st.cache_data(ttl=180) # Increased TTL for external NAV API calls
def fetch_latest_mf_nav(scheme_code):
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

# ------------------ PRICE HISTORY UPDATES ------------------

def update_stock_data(symbol):
    try:
        # FIX 2A: Ensure symbol is a string before proceeding
        if not isinstance(symbol, str):
            raise TypeError(f"Expected str for ticker symbol, got {type(symbol)}")

        ticker_str = symbol.replace('XNSE:', '') + '.NS' if 'XNSE:' in symbol and not symbol.endswith('.NS') else symbol
        today = datetime.date.today()
        # Use a reasonable lookback period, e.g., 5 years
        start_date = today - datetime.timedelta(days=5 * 365)

        # NOTE: yfinance.download sometimes returns multi-indexed columns if pandas is older.
        data = yf.download(ticker_str, start=start_date, end=today + datetime.timedelta(days=1), progress=False, auto_adjust=True)

        if data.empty:
            logging.warning(f"YFinance returned empty data for {symbol}.")
            return False

        data.reset_index(inplace=True)
        data["ticker"] = symbol
        data["date"] = data["Date"].dt.strftime("%Y-%m-%d")

        # --- FIX: Aggressively rename columns here before passing to df_to_table ---

        # 1. Handle MultiIndex (if any) - df_to_table handles this too, but redundancy is safe.
        if isinstance(data.columns, pd.MultiIndex):
             data.columns = ['_'.join(map(str, col)).strip('_') for col in data.columns.values]

        # 2. Rename 'Close' to 'close_price' regardless of ticker suffix
        rename_map = {}
        for col in data.columns:
            if col == 'Close':
                rename_map[col] = 'close_price'
            elif col.lower().startswith('close_') and symbol.replace('.', '_') in col.replace('.', '_'):
                # Catch specific ticker-suffixed columns (e.g., 'Close_VBL.NS' -> 'close_price')
                rename_map[col] = 'close_price'

        data.rename(columns=rename_map, inplace=True)

        if 'close_price' not in data.columns:
            raise KeyError(f"close_price not found in columns after rename: {data.columns.tolist()}")

        # We only pass the required columns to df_to_table
        write_df = data[["ticker", "date", "close_price"]].copy()

        df_to_table(write_df, 'price_history', cache_clear_type='general')
        return True
    except KeyError as e:
        # Catch the explicit KeyError we raised or if Pandas failed to find the column
        logging.error(f"YFinance update_stock_data failed for {symbol}: \"{e}\"")
        return False
    except TypeError as e:
        # Catch the explicit TypeError we raised
        logging.error(f"YFinance update_stock_data failed for {symbol}: {e}")
        return False
    except Exception as e:
        logging.error(f"YFinance update_stock_data failed for {symbol}: {e}")
        return False

# ------------------ PORTFOLIO / MF CALCULATIONS ------------------

@st.cache_data(ttl=60, show_spinner=False) # Reduced cache TTL for faster data updates
def get_holdings_df(table_name):
    if table_name == "trades":
        query = ("SELECT p.symbol, p.buy_price, p.buy_date, p.quantity, p.target_price, p.stop_loss_price, "
                 "h.close_price AS current_price FROM trades p LEFT JOIN price_history h ON p.symbol = h.ticker "
                 "WHERE h.date = (SELECT MAX(date) FROM price_history WHERE ticker = p.symbol)")
    else:
        # --- FIX APPLIED: Use CTE and ROW_NUMBER to ensure only ONE latest price per ticker ---
        query = ("""
        WITH latest_prices AS (
            SELECT
                ticker,
                close_price,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM price_history
        )
        SELECT
            p.ticker AS symbol,
            p.buy_price,
            p.buy_date,
            p.quantity,
            p.sector,
            p.market_cap,
            lp.close_price AS current_price
        FROM portfolio p
        LEFT JOIN latest_prices lp ON p.ticker = lp.ticker
        WHERE lp.rn = 1 OR lp.close_price IS NULL; -- Ensures only the single latest price or NULL if no price exists
        """)
    try:
        # Optimized: db_query is cached, ensuring fast read
        df = db_query(query)
        if df.empty:
            return pd.DataFrame()

        # FIX: Ensure buy_date is a datetime object immediately after loading from DB
        df['buy_date'] = pd.to_datetime(df['buy_date'], format='%Y-%m-%d', errors='coerce')

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


@st.cache_data(ttl=60, show_spinner=False) # Reduced cache TTL for faster data updates
def get_realized_df(table_name):
    try:
        # Optimized: db_query is cached, ensuring fast read
        df = db_query(f"SELECT * FROM {table_name}")
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

# placeholder functions removed to clean up console warnings
def _update_existing_portfolio_info():
    pass

# --- FEATURE IMPLEMENTATION: Recurring Expenses ---
def _process_recurring_expenses():
    """
    Checks the recurring_expenses table and automatically inserts missing
    expense entries for the current month into the expenses table.
    """
    today = datetime.date.today()
    current_month_year = today.strftime("%Y-%m")

    # 1. Fetch all active recurring expense rules
    recurring_rules = db_query("SELECT description, amount, category, payment_method, day_of_month FROM recurring_expenses")
    if recurring_rules.empty:
        return

    # 2. Fetch all expenses logged this month (to avoid duplicates)
    month_expenses = db_query(f"SELECT description FROM expenses WHERE date LIKE '{current_month_year}%'")

    logged_descriptions = set(month_expenses['description'].tolist())

    new_expenses_to_log = []

    for _, rule in recurring_rules.iterrows():
        description = rule['description']

        # Check if an expense matching this description is already logged this month
        if description in logged_descriptions:
            continue

        # Determine the target date for the recurring expense
        # Use the current month/year and the configured day_of_month
        day = min(rule['day_of_month'], today.day) # Ensure we only log for today or previous days this month

        # Determine the date of the recurring expense
        try:
            expense_date = datetime.date(today.year, today.month, rule['day_of_month'])
        except ValueError:
            # Handle cases where day_of_month is 30/31 and the month doesn't have it (e.g., Feb 30)
            expense_date = datetime.date(today.year, today.month, 1) + datetime.timedelta(days=32)
            expense_date = expense_date.replace(day=1) - datetime.timedelta(days=1)

        # Only log if the target date is today or in the past
        if expense_date <= today:
            new_expenses_to_log.append({
                'expense_id': str(uuid.uuid4()),
                'date': expense_date.strftime("%Y-%m-%d"),
                'type': 'Expense',
                'amount': round(rule['amount'], 2),
                'category': rule['category'],
                'payment_method': rule['payment_method'],
                'description': description,
                'transfer_group_id': None
            })

    if new_expenses_to_log:
        new_df = pd.DataFrame(new_expenses_to_log)
        # Use df_to_table to insert all new expenses in a batch and clear caches
        df_to_table(new_df, 'expenses', if_exists='append', cache_clear_type='general')
        logging.info(f"Logged {len(new_expenses_to_log)} recurring expenses for {current_month_year}.")
# --- END FEATURE IMPLEMENTATION ---

def _process_mf_sips():
    pass

# --- UTILITY FOR HISTORICAL MF SYNC ---
def _sync_mf_to_expenses():
    """
    Utility function to retrospectively create 'Investment' expense/income
    entries in the expenses table for all existing mf_transactions.
    """

    # 1. Fetch all MF transactions
    # Note: We include 'amount' column now, which is the cash value including fees (from new input/edit).
    mf_tx_df = db_query("SELECT transaction_id, date, scheme_name, type, amount FROM mf_transactions")
    if mf_tx_df.empty:
        return

    # 2. Get the descriptions of MF transactions already logged in expenses
    # We look for transactions that were likely created by the MF section.
    existing_expense_descriptions_df = db_query("SELECT description FROM expenses WHERE category = 'Investment'")
    existing_descriptions = set(existing_expense_descriptions_df['description'].tolist())

    new_investment_entries = []

    for _, tx_row in mf_tx_df.iterrows():
        # Derive the description that the live transaction logic uses
        expense_type = "Expense" if tx_row['type'] == "Purchase" else "Income"
        expense_desc = f"{tx_row['type']} {tx_row['scheme_name']} units"

        # Skip if this transaction description already exists
        if expense_desc in existing_descriptions:
            continue

        cash_value = tx_row['amount']

        if pd.isna(cash_value) or cash_value is None:
            # Skip corrupted/old entries where cash amount wasn't saved.
            # User must fix these manually.
            logging.warning(f"Skipping MF transaction sync for {tx_row['scheme_name']} on {tx_row['date']} due to missing cash amount.")
            continue

        new_investment_entries.append({
            'expense_id': str(uuid.uuid4()),
            'date': tx_row['date'],
            'type': expense_type,
            'amount': round(cash_value, 2),
            'category': 'Investment',
            'payment_method': 'N/A',
            'description': expense_desc,
            'transfer_group_id': None
        })

    if new_investment_entries:
        new_df = pd.DataFrame(new_investment_entries)
        df_to_table(new_df, 'expenses', if_exists='append', cache_clear_type='general')
        logging.info(f"Synced {len(new_investment_entries)} historical MF transactions to the Expense Tracker.")
# --- END UTILITY FOR HISTORICAL MF SYNC ---


# ------------------ MORE CALC / UI HELPERS ------------------

def _categorize_market_cap(market_cap_value):
    if isinstance(market_cap_value, (int, float)):
        if market_cap_value >= 10_000_000_000:
            return "Large Cap"
        elif market_cap_value >= 2_000_000_000:
            return "Mid Cap"
        else:
            return "Small Cap"
    return "N/A"

def color_return_value(val):
    if val is None or not isinstance(val, (int, float)):
        return ''
    return 'color: green' if val >= 0 else 'color: red'

# ------------------ COMBINED RETURNS & MF HELPERS ------------------

@st.cache_data(ttl=60, show_spinner=False) # Reduced cache TTL for faster data updates
def get_mf_holdings_df():
    # Optimized: transactions_df read is cached via db_query
    transactions_df = db_query("SELECT * FROM mf_transactions")
    if transactions_df.empty:
        return pd.DataFrame()

    # Efficiently gather latest NAVs, potentially triggering API calls (which are cached)
    unique_codes = transactions_df['yfinance_symbol'].unique()
    # Optimized: fetch_latest_mf_nav is highly cached (180s), limiting API impact
    latest_navs = {code: fetch_latest_mf_nav(code) for code in unique_codes}

    holdings = []
    unique_schemes = transactions_df['scheme_name'].unique()

    for scheme in unique_schemes:
        scheme_tx = transactions_df[transactions_df['scheme_name'] == scheme].copy()
        purchases = scheme_tx[scheme_tx['type'] == 'Purchase']
        redemptions = scheme_tx[scheme_tx['type'] == 'Redemption']

        total_units = purchases['units'].sum() - redemptions['units'].sum()

        if total_units > 0.001:
            total_investment = (purchases['units'] * purchases['nav']).sum() - (redemptions['units'] * redemptions['nav']).sum()
            avg_nav = total_investment / total_units if total_investment > 0 else 0

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


@st.cache_data(ttl=60, show_spinner=False) # Reduced cache TTL for faster data updates
def get_combined_returns():
    try:
        # Optimized: Use cached function for live trade symbols
        live_trade_symbols = get_live_trade_symbols()

    except Exception as e:
        logging.error(f"Error loading live trade symbols in get_combined_returns: {e}")
        live_trade_symbols = set()

    # Optimized: Use cached dataframes
    inv_df = get_holdings_df("portfolio")
    inv_invested = float(inv_df['invested_value'].sum()) if not inv_df.empty else 0
    inv_current = float(inv_df['current_value'].sum()) if not inv_df.empty else 0

    trade_df = get_holdings_df("trades")

    # Filter trades based on fund linkage
    live_trade_df = trade_df[trade_df['symbol'].isin(live_trade_symbols)] if not trade_df.empty else pd.DataFrame()
    trade_invested = float(live_trade_df['invested_value'].sum()) if not live_trade_df.empty else 0
    trade_current  = float(live_trade_df['current_value'].sum()) if not live_trade_df.empty else 0

    mf_df = get_mf_holdings_df()
    mf_invested = float(mf_df['Investment'].sum()) if not mf_df.empty else 0
    mf_current  = float(mf_df['Current Value'].sum()) if not mf_df.empty else 0

    inv_return_amount = round(inv_current - inv_invested, 2)
    inv_return_pct    = round((inv_return_amount / inv_invested) * 100, 2) if inv_invested > 0 else 0

    trade_return_amount = round(trade_current - trade_invested, 2)
    trade_return_pct    = round((trade_return_amount / trade_invested) * 100, 2) if trade_invested > 0 else 0

    mf_return_amount = round(mf_current - mf_invested, 2)
    mf_return_pct     = round((mf_return_amount / mf_invested) * 100, 2) if mf_invested > 0 else 0

    total_invested = inv_invested + trade_invested + mf_invested
    total_current  = inv_current + trade_current + mf_current
    total_return_amount = round(total_current - total_invested, 2)
    total_return_pct        = round((total_return_amount / total_invested) * 100, 2) if total_invested > 0 else 0

    # Optimized: Use cached realized dataframes
    realized_stocks_df = get_realized_df("realized_stocks")
    realized_exits_df  = get_realized_df("exits")

    # Filter realized exits based on live trade symbols
    live_exits_df = realized_exits_df[realized_exits_df['symbol'].isin(live_trade_symbols)] if not realized_exits_df.empty else pd.DataFrame()
    realized_inv         = float(realized_stocks_df['realized_profit_loss'].sum()) if not realized_stocks_df.empty else 0
    realized_trade = float(live_exits_df['realized_profit_loss'].sum())            if not live_exits_df.empty    else 0
    realized_mf    = 0

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
        "realized_inv": round(realized_inv, 2),
        "realized_trade": round(realized_trade, 2),
        "realized_mf": round(realized_mf, 2)
    }

# ------------------ PORTFOLIO VS BENCHMARK CHART FUNCTIONS (FIXED) ------------------

@st.cache_data(ttl=3600)
def get_benchmark_data(benchmark_choice, start_date):
    """Fetch benchmark data using yfinance for robustness."""

    # Map the user's choice to the correct yfinance ticker
    ticker_map = {
        "Nifty 50": "^NSEI",
        "Nifty 100": "^CNX100",
        "Nifty 200": "^CNX200",
        "Nifty 500": "^CNX500",
    }

    ticker = ticker_map.get(benchmark_choice, "^NSEI")

    # Use yfinance.download for reliable data fetching
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=datetime.date.today() + datetime.timedelta(days=1),
            progress=False,
            auto_adjust=True
        )

        if data.empty or 'Close' not in data.columns:
            logging.error(f"yfinance returned empty or invalid data for benchmark {ticker}.")
            return pd.DataFrame()

        # --- FIX APPLIED: Aggressively flatten MultiIndex columns to fix MergeError ---
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten to the column name (e.g., ('Close', '^NSEI') -> 'Close')
            data.columns = [col[0] for col in data.columns]
        # --- END FIX ---


        # Keep only the 'Close' column and ensure it is named 'Close'
        return data[['Close']].rename(columns={'Close': 'Close'})

    except Exception as e:
            logging.error(f"Failed to fetch benchmark data for {ticker} using yfinance: {e}")
            return pd.DataFrame()


def calculate_portfolio_metrics(holdings_df, portfolio_data, benchmark_choice):
    # This function relies on the benchmark comparison logic but is stubbed/simplified

    metrics = {
        "alpha": 0.00,
        "beta": 1.00,
        "max_drawdown": 0.00
    }

    if not portfolio_data.empty and 'Return %' in portfolio_data.columns:
        portfolio_returns = portfolio_data[portfolio_data['Type'] == 'Portfolio'].set_index('Date')['Return %']

        if not portfolio_returns.empty:
            # Calculate daily returns (change in cumulative return)
            # Must first convert the percentage returns back to index values (1 + returns/100)
            index_values = 1 + (portfolio_returns / 100)

            # Calculate max drawdown
            peak = index_values.expanding(min_periods=1).max()
            drawdown = (index_values - peak) / peak
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0.0

            metrics["max_drawdown"] = round(max_drawdown, 2)

    return metrics

@st.cache_data(ttl=3600)
def get_benchmark_comparison_data(holdings_df, benchmark_choice):
    """Calculates the cumulative return for the portfolio and the selected benchmark."""

    if holdings_df.empty:
        return pd.DataFrame()

    # 1. Determine the earliest buy date (start of calculation)
    # The date column is already loaded as datetime in get_holdings_df
    start_date = holdings_df['buy_date'].min()

    if pd.isna(start_date):
        return pd.DataFrame()

    # 2. Map benchmark choice to a YFinance ticker
    benchmark_map = {
        'Nifty 50': '^NSEI',
        'Nifty 100': '^CNX100',
        'Nifty 200': '^CNX200',
        'Nifty 500': '^CNX500'
    }
    benchmark_ticker = benchmark_map.get(benchmark_choice, '^NSEI')

    # 3. Fetch all price history data (Portfolio + Benchmark)
    all_tickers = holdings_df['symbol'].unique().tolist()

    # Portfolio prices from DB (only the held assets)
    # The start_date must be passed as a string to match the 'TEXT' type in the DB.
    start_date_str = start_date.strftime('%Y-%m-%d')

    # Optimized: db_query is cached
    # FIX: Use GROUP BY and aggregation to ensure no duplicate (ticker, date) pairs remain,
    # which caused the "Index contains duplicate entries" error on pivot.
    portfolio_prices_df = db_query(
        "SELECT ticker, date, MAX(close_price) AS close_price FROM price_history WHERE ticker IN :tickers AND date >= :start_date GROUP BY ticker, date",
        params={'tickers': tuple(all_tickers), 'start_date': start_date_str}
    ).pivot(index='date', columns='ticker', values='close_price')

    if portfolio_prices_df.empty:
        return pd.DataFrame()

    portfolio_prices_df.index = pd.to_datetime(portfolio_prices_df.index)

    # Benchmark prices from YF (start_date is handled correctly by YF if passed as datetime)
    # Optimized: get_benchmark_data is highly cached
    benchmark_data = get_benchmark_data(benchmark_choice, start_date)
    if benchmark_data.empty:
        return pd.DataFrame()

    # 4. Align data frames based on price index
    # This join now succeeds because benchmark_data columns are flattened
    combined_prices = portfolio_prices_df.join(benchmark_data, how='inner')

    # --- FIX APPLIED: Replace deprecated fillna(method=...) with dedicated methods ---
    combined_prices = combined_prices.ffill()
    combined_prices = combined_prices.bfill()
    # --- END FIX ---

    # 5. Calculate Portfolio Value over Time

    # Extract holding details from the first date to calculate initial weights
    holdings_at_start = holdings_df.set_index('symbol')[['buy_price', 'quantity']]

    # Calculate initial investment value for each asset
    holdings_at_start['Initial_Value'] = holdings_at_start['buy_price'] * holdings_at_start['quantity']
    total_initial_investment = holdings_at_start['Initial_Value'].sum()

    if total_initial_investment == 0:
        return pd.DataFrame()

    daily_portfolio_value = 0

    if not combined_prices.empty:
        # Calculate daily portfolio value based on current prices and initial quantities
        daily_portfolio_value = (combined_prices[all_tickers].mul(holdings_at_start['quantity'].to_dict(), axis=1)).sum(axis=1)

        # Calculate Cumulative Portfolio Return
        portfolio_return = ((daily_portfolio_value - total_initial_investment) / total_initial_investment) * 100
    else:
        portfolio_return = pd.Series()

    # 6. Calculate Cumulative Benchmark Return (rebased to the start date)

    # The benchmark price on the first trading day of the portfolio
    first_benchmark_price = combined_prices.iloc[0]['Close']

    # Calculate Cumulative Benchmark Return
    benchmark_return = ((combined_prices['Close'] - first_benchmark_price) / first_benchmark_price) * 100

    # 7. Combine and format for Altair chart

    if portfolio_return.empty or benchmark_return.empty:
        return pd.DataFrame()

    portfolio_df = portfolio_return.to_frame(name='Return %').reset_index()
    portfolio_df.rename(columns={'index': 'Date'}, inplace=True)
    portfolio_df['Type'] = 'Portfolio'

    benchmark_df = benchmark_return.to_frame(name='Return %').reset_index()
    benchmark_df.rename(columns={'index': 'Date'}, inplace=True)
    benchmark_df['Type'] = benchmark_choice

    final_df = pd.concat([portfolio_df, benchmark_df])
    final_df['Return %'] = final_df['Return %'].round(2)

    return final_df

@st.cache_data(ttl=60, show_spinner=False) # Reduced cache TTL for faster data updates
def _calculate_mf_cumulative_return(transactions_df):
    """Calculates the cumulative return of a mutual fund portfolio over time."""

    if transactions_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_schemes_daily_returns = []
    sip_marker_data = []
    unique_schemes = transactions_df['scheme_name'].unique()

    for scheme_name in unique_schemes:
        scheme_tx = transactions_df[transactions_df['scheme_name'] == scheme_name].copy()
        scheme_tx['date'] = pd.to_datetime(scheme_tx['date'], format='%Y-%m-%d', errors='coerce')
        scheme_tx = scheme_tx.sort_values('date').reset_index(drop=True)

        scheme_code = scheme_tx['yfinance_symbol'].iloc[0]
        # Use cached function
        historical_data = get_mf_historical_data(scheme_code)


        if historical_data.empty:
            continue

        start_date = scheme_tx['date'].min()
        end_date = historical_data.index.max()
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        units = 0
        invested_amount = 0
        daily_records = []

        for date in all_dates:
            if date in historical_data.index:
                nav = historical_data.loc[date]['NAV']
                todays_tx = scheme_tx[scheme_tx['date'] == date]

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
                        # IMPORTANT: When calculating cumulative return/investment, redemption amounts should ideally be
                        # calculated based on the *average cost* of the redeemed units.
                        # For simplicity here, we assume the redeemed units' cost basis equals their NAV at transaction time,
                        # which is an approximation.
                        # The overall P&L calculation handles this better.
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

# ------------------ TRADING METRICS ------------------

def calculate_trading_metrics(realized_df):
    if realized_df.empty:
        return {'win_ratio': 0.0, 'profit_factor': 0.0, 'expectancy': 0.0}

    winning_trades = realized_df[realized_df['realized_profit_loss'] >= 0]
    losing_trades = realized_df[realized_df['realized_profit_loss'] < 0]

    total_trades = len(realized_df)
    win_ratio = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

    total_profit = winning_trades['realized_profit_loss'].sum()
    total_loss = losing_trades['realized_profit_loss'].sum()

    profit_factor = round(total_profit / abs(total_loss), 2) if total_loss != 0 else np.inf

    expectancy = round(realized_df['realized_profit_loss'].mean(), 2)

    return {
        'win_ratio': round(win_ratio, 2),
        'profit_factor': profit_factor,
        'expectancy': expectancy
    }

# ------------------ MORE CHART / CALC HELPERS ------------------

@st.cache_data(ttl=3600)
def get_current_portfolio_allocation():
    # Optimized: Use cached holdings functions
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

# ------------------ UI PAGES (home, funds, expense, mf, assets) ------------------

def home_page():
    """Renders the main home page."""
    st.title("Finance Dashboard")
    _update_existing_portfolio_info()
    returns_data = get_combined_returns()

    st.subheader("Live Portfolio Overview (Excluding Paper Trades)")

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
        # NOTE: Refresh button should be functional for all users as it updates view data
        if st.button("Refresh Live Data", key="refresh_all_data"):
            with st.spinner("Fetching latest stock and mutual fund prices..."):
                # FIX 2C: Explicitly select the column as 'ticker' to ensure it returns strings/simple series
                all_tickers_df = db_query("SELECT ticker FROM portfolio UNION SELECT symbol AS ticker FROM trades")
                all_tickers = [str(t) for t in all_tickers_df['ticker'].tolist()] if not all_tickers_df.empty else []

                for symbol in all_tickers:
                    update_stock_data(symbol)
                # Ensure cache is cleared for MF NAVs which were fetched directly
                st.cache_data(fetch_latest_mf_nav).clear()
            st.success("All live data refreshed!")
            st.rerun()

def funds_page():
    """Renders the Funds Management page. (Restored to File 1 logic)"""
    st.title("💰 Funds Management")
    is_viewer = st.session_state.get("role") == "viewer"
    disabled = is_viewer

    st.sidebar.header("Add Transaction")

    with st.sidebar.form("deposit_form", clear_on_submit=True):
        st.subheader("Add Deposit")
        deposit_date = st.date_input("Date", max_value=datetime.date.today(), key="deposit_date", disabled=disabled)
        deposit_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None, disabled=disabled)
        deposit_desc = st.text_input("Description", placeholder="e.g., Salary", value="", disabled=disabled)
        if st.form_submit_button("Add Deposit", disabled=disabled):
            if deposit_amount and deposit_amount > 0:
                update_funds_on_transaction("Deposit", round(deposit_amount, 2), deposit_desc, deposit_date.strftime("%Y-%m-%d"))
                st.success("Deposit recorded!")
                st.rerun()
            else:
                st.warning("Deposit amount must be greater than zero.")

    with st.sidebar.form("withdrawal_form", clear_on_submit=True):
        st.subheader("Record Withdrawal")
        wd_date = st.date_input("Date", max_value=datetime.date.today(), key="wd_date", disabled=disabled)
        wd_amount = st.number_input("Amount", min_value=0.01, format="%.2f", key="wd_amount", value=None, disabled=disabled)

        wd_desc = st.text_input("Description", placeholder="e.g., Personal Use", value="", disabled=disabled)

        if st.form_submit_button("Record Withdrawal", disabled=disabled):
            if wd_amount and wd_amount > 0:
                update_funds_on_transaction("Withdrawal", round(wd_amount, 2), wd_desc, wd_date.strftime("%Y-%m-%d"))
                st.success("Withdrawal recorded!")
                st.rerun()
            else:
                st.warning("Withdrawal amount must be greater than zero.")

    # Optimized: Use cached function to retrieve pre-processed DataFrame
    fund_df = get_fund_transactions_df()
    returns_data = get_combined_returns()

    # --- CALCULATE CAPITAL METRICS (Restored to File 1 logic) ---

    # 1. Available Cash Balance (Cash In - Cash Out)
    total_deposits = fund_df.loc[fund_df['type'] == 'Deposit', 'amount'].sum()
    total_withdrawals = fund_df.loc[fund_df['type'] == 'Withdrawal', 'amount'].sum()
    available_capital_cash = round(total_deposits - total_withdrawals, 2)

    # 2. Total Investment (Your definition: Invested Amount)
    # This is the initial cost of all currently held assets.
    total_investment_deployed = returns_data['total_invested_value']

    # 3. Total Corpus (Your definition: Invested Amount + Available Amount)
    # This is Total Current Market Value (total_current_value) + Available Cash Balance (available_capital_cash)
    total_corpus_net_worth = returns_data['total_current_value'] + available_capital_cash

    if not fund_df.empty:
        # Calculate chronological_df for the chart only, as the full fund_df is already sorted and merged
        chronological_df = fund_df.copy()
        chronological_df.sort_values(['date', 'transaction_id'], ascending=[True, True], inplace=True) # Ensure chronological sort for cumsum

        col1, col2, col3 = st.columns(3)
        # FIX APPLIED: Metrics updated to reflect user's conceptual definition of corpus and investment
        col1.metric("Total Corpus (Net Worth)", f"₹{total_corpus_net_worth:,.2f}")
        col2.metric("Total Investment (Cost Basis)", f"₹{total_investment_deployed:,.2f}")
        col3.metric("Available Capital (Cash)", f"₹{available_capital_cash:,.2f}")

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

        # Disable entire data editor for viewer
        edited_df = st.data_editor(
            fund_df[['transaction_id', 'date', 'type', 'amount', 'description', 'cumulative_balance']],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed", # Prevent new row entry in viewer mode
            disabled=disabled,
            column_config={
                "transaction_id": st.column_config.TextColumn("ID", disabled=True),
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["Deposit", "Withdrawal"], required=True),
                "cumulative_balance": st.column_config.TextColumn("Balance", disabled=True)
            }
        )

        edited_df['date'] = edited_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime.date) else x)


        if st.button("Save Changes to Transactions", disabled=disabled):
            # Using session.execute for DML
            with get_session() as session:
                session.execute(_sql_text('DELETE FROM fund_transactions'))

            # Re-insert the data using df_to_table which manages the connection/session
            df_to_save = edited_df[['transaction_id', 'date', 'type', 'amount', 'description']].copy()
            # Ensure transfer_group_id is included as None for non-transfer edits
            df_to_save['transfer_group_id'] = None
            df_to_table(df_to_save, 'fund_transactions', cache_clear_type='general')

            st.success("Funds transactions updated successfully! Rerunning to update the chart.")
            st.rerun()
    else:
        st.info("No fund transactions logged yet.")

def expense_tracker_page():
    """Renders the Expense Tracker page. (Restored to File 1 logic and structure)"""
    st.title("💸 Expense Tracker")
    is_viewer = st.session_state.get("role") == "viewer"
    disabled = is_viewer

    _process_recurring_expenses()


    @st.cache_data(ttl=3600) # Cache expense categories list
    def get_expense_categories_list():
        try:
            # Optimized: db_query is cached
            expense_categories = db_query("SELECT DISTINCT category FROM expenses WHERE type='Expense'")['category'].tolist()
            default_categories = ["Food", "Transport", "Rent", "Utilities", "Shopping", "Entertainment", "Health", "Groceries", "Bills", "Education", "Travel", "Other"]
            all_categories = list(set([c for c in expense_categories if c and c != 'N/A'] + default_categories))

            EXCLUDED_CATEGORIES = ["Transfer Out", "Transfer In"]
            all_categories = [c for c in all_categories if c not in EXCLUDED_CATEGORIES]
            return sorted(all_categories)
        except Exception:
            return sorted(["Food", "Transport", "Rent", "Utilities", "Shopping", "Entertainment", "Health", "Groceries", "Bills", "Education", "Travel", "Other"])

    if 'expense_categories_list' not in st.session_state:
        st.session_state.expense_categories_list = get_expense_categories_list()

    CATEGORIES = st.session_state.expense_categories_list

    PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking", "N/A"]

    PAYMENT_ACCOUNTS = [pm for pm in PAYMENT_METHODS if pm != 'N/A']

    # Define excluded categories for Dashboard metrics
    TRANSFER_CATEGORIES = ['Transfer Out', 'Transfer In']
    INVESTMENT_CATEGORY = 'Investment'
    SAVINGS_CATEGORY = 'Savings'

    # Exclude Transfers and Investment from ALL PFM views
    EXCLUDED_PFM_CATEGORIES = TRANSFER_CATEGORIES + [INVESTMENT_CATEGORY]

    # Exclude Transfers, Investment, and Savings from CONSUMPTION metrics (like expenses/charts)
    CONSUMPTION_EXCLUDED_CATEGORIES = EXCLUDED_PFM_CATEGORIES + [SAVINGS_CATEGORY]

    # Helper to get *Actual* cumulative balance (kept separate for help text)
    @st.cache_data(ttl=60) # Reduced cache TTL for faster data updates
    def get_cumulative_cash_balance():
        balance_query = """
        SELECT
            payment_method,
            SUM(CASE WHEN type = 'Income' THEN amount ELSE -amount END) AS balance
        FROM expenses
        WHERE payment_method != 'N/A'
        GROUP BY payment_method
        HAVING SUM(CASE WHEN type = 'Income' THEN amount ELSE -amount END) != 0;
        """
        balance_df = db_query(balance_query)
        if balance_df.empty:
            return 0.0, "No account balances tracked."

        total_balance = balance_df['balance'].sum()

        balance_help_text = "Actual Cumulative Cash Breakdown (Total Funds):\n"
        for _, row in balance_df.iterrows():
            balance_help_text += f"*{row['payment_method']}*: ₹{row['balance']:,.2f}\n"

        return total_balance, balance_help_text


    # MUST be defined immediately before use in the main function body
    view = st.radio("Select View", ["Dashboard", "Transaction History", "Manage Budgets", "Manage Recurring", "Transfer", "Savings Balances"], horizontal=True, label_visibility="hidden", disabled=disabled)


    # --- CONDITIONAL SIDEBAR FORMS ---

    if view == "Savings Balances":
        st.sidebar.header("Add Savings Transfer")
        st.sidebar.info("Transfers here update your savings balance only. Choose if they debit your liquid cash.")

        with st.sidebar.form("new_savings_transfer_form", clear_on_submit=True):

            # OPTION 1: Toggle to control debiting liquid funds
            debit_liquid_funds = st.toggle(
                "Transfer from Liquid Funds? (Debits liquid cash amount)",
                key="debit_liquid_funds_toggle",
                value=True,
                disabled=disabled
            )
            st.markdown("---")

            # Common Fields for Savings Transfer (Always Expense)
            transfer_type = "Expense" # Savings accumulation is always an outflow from liquid PFM
            st.markdown(f"**Transaction Type:** **{transfer_type}** (Fixed)")

            trans_date = st.date_input("Date", max_value=datetime.date.today(), value=datetime.date.today(), disabled=disabled)
            trans_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None, disabled=disabled)

            # Conditional Payment Method Selection
            if debit_liquid_funds:
                trans_pm = st.selectbox("Source Account (Liquid Funds)", options=PAYMENT_ACCOUNTS, index=None, disabled=disabled)
                pm_help = "Required if debiting liquid funds."
            else:
                # If not debiting liquid funds, assume N/A for payment method in the expense tracker logic,
                # but allow user to record the source for reference.
                trans_pm = st.selectbox("Source/Method (Optional)", options=['N/A'] + PAYMENT_ACCOUNTS, index=0, disabled=disabled)
                pm_help = "Optional, for internal reference only."
            transfer_category = SAVINGS_CATEGORY
            st.markdown(f"**Category:** **{transfer_category}** (Fixed)")

            trans_desc = st.text_input("Description", placeholder="e.g., Monthly SIP, Emergency fund top-up", value="", disabled=disabled)

            # Withdrawal option for Savings (reverse flow)
            is_savings_withdrawal = st.checkbox("This is a **Withdrawal** from Savings", key="is_savings_wd", disabled=disabled)

            if st.form_submit_button("Record Savings Transfer", disabled=disabled):
                if trans_amount and trans_amount > 0 and (trans_pm != 'N/A' or not debit_liquid_funds):

                    if is_savings_withdrawal:
                        # Withdrawal: Income transaction in PFM (reverts to liquid cash)
                        # The savings balance calculation will handle the sign change.
                        transaction_type_to_save = "Income" if debit_liquid_funds else "Expense"
                        description_to_save = f"SAVINGS WITHDRAWAL: {trans_desc}" if trans_desc else "SAVINGS WITHDRAWAL"
                        # When withdrawing to liquid funds, use the selected payment method as the destination for the Income.
                        # When manually recording, payment method is N/A (or other selected method).
                        pm_to_save = trans_pm if debit_liquid_funds else trans_pm
                    else:
                        # Transfer IN: Expense transaction in PFM (debits liquid cash)
                        transaction_type_to_save = "Expense" if debit_liquid_funds else "Expense"
                        description_to_save = f"SAVINGS TRANSFER: {trans_desc}" if trans_desc else "SAVINGS TRANSFER"
                        # If not debiting liquid funds, the payment method is irrelevant to the overall cash balance, but we save it.
                        pm_to_save = trans_pm

                    # Insert into expenses table with SAVINGS_CATEGORY
                    db_execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description, transfer_group_id) VALUES (:id, :date, :type, :amount, :cat, :pm, :desc, :tg_id)",
                             params={
                                'id': str(uuid.uuid4()),
                                'date': trans_date.strftime("%Y-%m-%d"),
                                'type': transaction_type_to_save,
                                'amount': round(trans_amount, 2),
                                'cat': transfer_category,
                                'pm': pm_to_save,
                                'desc': description_to_save,
                                'tg_id': None
                             }, cache_clear_type='general'
                    )
                    st.success(f"Savings {'Withdrawal' if is_savings_withdrawal else 'Transfer'} of ₹{trans_amount:,.2f} recorded!")
                    st.rerun()
                else:
                    st.warning("Please fill the Amount and select a Source Account if transferring from liquid funds.")


    elif view != "Transfer" and view != "Savings Balances":
        # --- ORIGINAL EXPENSE/INCOME FORM ---
        st.sidebar.header("Add Transaction")
        with st.sidebar.form("new_transaction_form", clear_on_submit=True):

            st.markdown("---") # Visual separator

            # Determine transaction type and category selection method
            trans_type = st.radio("Transaction Type", ["Expense", "Income"], key="trans_type", disabled=disabled)


            trans_date = st.date_input("Date", max_value=datetime.date.today(), value=datetime.date.today(), disabled=disabled)
            trans_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None, disabled=disabled)

            category_options = ['Select Category...'] + CATEGORIES

            selected_category = st.selectbox(
                "Select Category",
                options=category_options,
                index=0,
                key="selected_cat",
                disabled=disabled
            )

            custom_category = st.text_input(
                "Or Enter New Category",
                help="Enter a custom category name, this will override the selection.",
                value="",
                key="custom_cat",
                disabled=disabled
            )

            if custom_category:
                final_cat = custom_category
            elif selected_category and selected_category != 'Select Category...':
                final_cat = selected_category
            else:
                final_cat = None

            # ------------------------------------

            if trans_type == "Income":
                trans_pm = st.selectbox("Destination Account/Method", options=PAYMENT_ACCOUNTS, index=None, disabled=disabled)
                trans_desc_placeholder = "e.g., Salary, Refund"
            else:
                trans_pm = st.selectbox("Payment Method", options=PAYMENT_ACCOUNTS, index=None, disabled=disabled)
                trans_desc_placeholder = "e.g., Groceries, Rent, Investment Purchase"


            trans_desc = st.text_input("Description", placeholder=trans_desc_placeholder, value="", disabled=disabled)

            if st.form_submit_button("Add Transaction", disabled=disabled):
                if trans_amount and final_cat and trans_pm:

                    description_to_save = trans_desc

                    # Use session for execution via db_execute helper
                    db_execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description, transfer_group_id) VALUES (:id, :date, :type, :amount, :cat, :pm, :desc, :tg_id)",
                             params={
                                'id': str(uuid.uuid4()),
                                'date': trans_date.strftime("%Y-%m-%d"),
                                'type': trans_type,
                                'amount': round(trans_amount, 2),
                                'cat': final_cat,
                                'pm': trans_pm,
                                'desc': description_to_save,
                                'tg_id': None
                             }, cache_clear_type='general'
                    )
                    st.success(f"{trans_type} added! Category: **{final_cat}**")


                    if final_cat not in st.session_state.expense_categories_list:
                        st.session_state.expense_categories_list.append(final_cat)
                        st.session_state.expense_categories_list.sort()
                        st.cache_data(get_expense_categories_list).clear() # Clear cache for category list

                    st.rerun()
                else:
                    st.warning("Please fill all required fields (Amount, Category, and Payment Method).")


    if view == "Dashboard":
        today = datetime.date.today()
        start_date_7days = today - datetime.timedelta(days=6)
        month_year = today.strftime("%Y-%m")

        # Optimized: Use cached function for all raw expense data
        all_time_expenses_df = get_all_expenses_df()

        if all_time_expenses_df.empty:
            st.info("No expenses logged yet to display the dashboard.")
            return

        # Filter the cached full dataframe for the current month
        expenses_df = all_time_expenses_df[all_time_expenses_df['date'].dt.strftime('%Y-%m') == month_year]

        # --- Rollover Logic: Calculate previous month's net flow ---
        # (This logic is retained from File 1 / merged with File 2 fixes for cleanliness)

        first_day_of_current_month = datetime.date(today.year, today.month, 1)
        last_day_of_previous_month = first_day_of_current_month - datetime.timedelta(days=1)
        previous_month_year = last_day_of_previous_month.strftime("%Y-%m")

        # Exclude 'Savings' category from rollover calculation as it's an internal movement
        rollover_excluded_categories = CONSUMPTION_EXCLUDED_CATEGORIES # Includes: Transfers, Investment, Savings

        prev_month_df = all_time_expenses_df[
            (all_time_expenses_df['date'].dt.strftime('%Y-%m') == previous_month_year) &
            (~all_time_expenses_df['category'].isin(rollover_excluded_categories))
        ]

        prev_income = prev_month_df[prev_month_df['type'] == 'Income']['amount'].sum()
        prev_expense = prev_month_df[prev_month_df['type'] == 'Expense']['amount'].sum()
        rollover_amount = round(prev_income - prev_expense, 2)
        # --- End Rollover Logic ---

        # NOTE: Current month income/expense includes only actual PFM consumption transactions (excluding transfers, investment, and savings)
        inflows_df = expenses_df[(expenses_df['type'] == 'Income') & (~expenses_df['category'].isin(CONSUMPTION_EXCLUDED_CATEGORIES))]
        outflows_df = expenses_df[(expenses_df['type'] == 'Expense') & (~expenses_df['category'].isin(CONSUMPTION_EXCLUDED_CATEGORIES))]

        total_spent = outflows_df['amount'].sum()
        # This is the pure cash inflow for the current month (for consumption budget)
        total_income_new = inflows_df['amount'].sum()

        # 1. CALCULATE TOTAL INCOME (New)
        total_monthly_income = round(total_income_new, 2)

        # 2. Calculate the total budget (New Income + Rollover) for the Expense Delta calculation
        total_budget_for_delta = total_monthly_income + rollover_amount


        # 3. Get Actual Cumulative Balance for Help Text
        actual_cumulative_balance, balance_help_text = get_cumulative_cash_balance()

        # 4. USER REQUESTED CALCULATION for Available Amount
        # Available Amount = Total Income (This month) - Expenses (This month)
        # Note: This net flow now correctly EXCLUDES 'Savings' outflows.
        total_cash_online_balance = total_monthly_income - total_spent
        total_cash_online_balance = round(total_cash_online_balance, 2)


        spent_breakdown_df = outflows_df.groupby('payment_method')['amount'].sum().reset_index()
        spent_help_text = "\n".join([f"*{row['payment_method']}*: ₹{row['amount']:,.2f}" for _, row in spent_breakdown_df.iterrows()])

        # MODIFIED: Using 3 columns for the main metrics (Restored File 1 layout)
        col_income, col_expenses, col_balance = st.columns(3)

        # 1. Total Income (This month) - Restored File 1 content
        with col_income:
            st.metric("1. Total Income (This month)", f"₹{total_monthly_income:,.2f}",
                             help="This is the sum of all 'Income' transactions recorded this month (excluding transfers/investment returns/Savings).")

        # 2. Expenses (This month) -> Total Spending - Restored File 1 content
        with col_expenses:
            # Calculate the delta based on the monthly budget (New Income + Rollover - Spending)
            delta_budget_flow = total_budget_for_delta - total_spent
            delta_label = f"₹{delta_budget_flow:,.2f} Remaining" if delta_budget_flow >= 0 else f"₹{-delta_budget_flow:,.2f} Over Budget"

            st.metric("2. Expenses (This month)", f"₹{total_spent:,.2f}",
                             help=f"Total consumption expenses logged this month (excluding transfers, investments, and savings). Monthly Budget Flow (Income + Rollover - Spent) is ₹{delta_budget_flow:,.2f}.\n\n(Budget: Income: ₹{total_monthly_income:,.2f} + Rollover: ₹{rollover_amount:,.2f}).")

        # 3. Available Amount -> Displays Current Month's Net Flow - Restored File 1 content
        with col_balance:
            st.metric("3. Available Amount (Current Month Net Flow)", f"₹{total_cash_online_balance:,.2f}",
                             help=f"This amount is calculated as: **Total Income (This month)** minus **Expenses (This month)**. This is NOT your cumulative bank balance across all time (which is ₹{actual_cumulative_balance:,.2f}).\n\n**Cumulative Account Breakdown:**\n{balance_help_text}")


        st.divider()

        st.subheader("Daily Spending: Last 7 Days (Excl. Transfers)")


        daily_spending = all_time_expenses_df[
            (all_time_expenses_df['type'] == 'Expense') &
            (~all_time_expenses_df['category'].isin(CONSUMPTION_EXCLUDED_CATEGORIES)) & # Exclude Investment, Transfers, and Savings
            (all_time_expenses_df['date'].dt.date >= start_date_7days)
        ].groupby(all_time_expenses_df['date'].dt.date)['amount'].sum().reset_index()
        daily_spending.rename(columns={'date': 'Date', 'amount': 'Spent'}, inplace=True)


        date_range = pd.date_range(start_date_7days, today)
        daily_df_full = pd.DataFrame({'Date': date_range.date})
        daily_df_full = daily_df_full.merge(daily_spending, on='Date', how='left').fillna(0)
        daily_df_full['Spent'] = daily_df_full['Spent'].round(2)
        daily_df_full['DayLabel'] = daily_df_full['Date'].apply(lambda x: "Today" if x == today else x.strftime('%a'))

        if not daily_df_full.empty:

            daily_df_full.sort_values('Date', ascending=True, inplace=True)

            bar_chart = alt.Chart(daily_df_full).mark_bar().encode(
                x=alt.X('DayLabel:N', sort=daily_df_full['DayLabel'].tolist(), title='Day'),
                y=alt.Y('Spent:Q', title='Amount Spent (₹)'),
                tooltip=['Date', alt.Tooltip('Spent', format=".2f", title='Total Spent (₹)')],
                color=alt.condition(
                    alt.datum.Date == today.strftime('%Y-%m-%d'), # Highlight today
                    alt.value('orange'),
                    alt.value('#4c78a8')
                )
            ).properties(height=300).interactive()
            st.altair_chart(bar_chart, use_container_width=True)

        else:
            st.info("No expense data for the last 7 days (excluding transfers).")

        st.divider()


        st.subheader(f"Category-wise Spending (Current Month: {month_year}, Excl. Transfers)")

        # Exclude Transfers, Investment, and Savings from the pie chart
        spending_by_category = outflows_df[~outflows_df['category'].isin(CONSUMPTION_EXCLUDED_CATEGORIES)].groupby('category')['amount'].sum().reset_index()

        if not spending_by_category.empty:

            total_spent_for_chart = spending_by_category['amount'].sum()
            spending_by_category['percentage'] = (spending_by_category['amount'] / total_spent_for_chart * 100).round(2)

            base = alt.Chart(spending_by_category).encode(
                # FIX APPLIED: Explicitly define data types for robustness
                theta=alt.Theta(field="amount", type="quantitative", stack=True),
            )

            pie = base.mark_arc(outerRadius=120).encode(
                color=alt.Color(field="category", type="nominal"),

                tooltip=["category",
                         alt.Tooltip('amount', format='.2f', title='Amount (₹)'),
                          alt.Tooltip('percentage', format='.2f', title='Percentage (%)')],
                order=alt.Order("amount", sort="descending")
            )


            text = base.mark_text(radius=140).encode(
                text=alt.Text("category:N"),
                order=alt.Order("amount", sort="descending"),
                color=alt.value("black")
            ).transform_filter(alt.datum.amount > total_spent_for_chart * 0.05)

            st.altair_chart(pie + text, use_container_width=True)
        else:
            st.info("No expenses logged for this month to plot (excluding transfers).")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            # FIX APPLIED: Chart now shows only Expenses of each month (excluding transfers/investment/savings)
            st.subheader("Monthly Spending Trend (Expenses Only)")

            # Query should also exclude 'Savings'
            monthly_expense_df = db_query("SELECT SUBSTR(date, 1, 7) AS month, SUM(amount) AS amount FROM expenses WHERE type='Expense' AND category NOT IN ('Transfer Out', 'Transfer In', 'Investment', 'Savings') GROUP BY month ORDER BY month DESC")

            if not monthly_expense_df.empty:
                bar_chart = alt.Chart(monthly_expense_df).mark_bar().encode(
                    x=alt.X('month', title='Month', sort='-x'),
                    y=alt.Y('amount', title='Total Spent (₹)'),
                    tooltip=['month', alt.Tooltip('amount', format='.2f', title='Amount')]
                ).properties(height=350)
                st.altair_chart(bar_chart, use_container_width=True)
            else:
                st.info("No expenses logged for this month.")

        with col2:
            st.subheader("Inflow vs. Outflow (Excl. Transfers)")

            # Chart should also exclude 'Savings'
            monthly_flows_chart_agg = db_query("SELECT SUBSTR(date, 1, 7) AS month, type, SUM(amount) AS amount FROM expenses WHERE category NOT IN ('Transfer Out', 'Transfer In', 'Investment', 'Savings') GROUP BY month, type ORDER BY month DESC")

            if not monthly_flows_chart_agg.empty:
                bar_chart = alt.Chart(monthly_flows_chart_agg).mark_bar().encode(
                    x=alt.X('month', title='Month', sort='-x'),
                    y=alt.Y('amount', title='Amount (₹)'),
                    color=alt.Color('type', title='Type', scale=alt.Scale(domain=['Income', 'Expense'], range=['#2ca02c', '#d62728'])),
                    tooltip=['month', alt.Tooltip('type', title='Type'), alt.Tooltip('amount', format='.2f', title='Amount')]
                ).properties(height=300)
                st.altair_chart(bar_chart, use_container_width=True)
            else:
                st.info("No income or expenses to compare.")


    elif view == "Transfer":
        st.header("🔄 Internal Account Transfer")
        st.info("Record a transfer of funds between your payment methods (e.g., from Net Banking to UPI).")

        with st.form("transfer_form", clear_on_submit=True):
            transfer_date = st.date_input("Date", max_value=datetime.date.today(), value=datetime.date.today(), disabled=disabled)
            transfer_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None, disabled=disabled)


            source_account = st.selectbox("From Account (Source)", options=PAYMENT_ACCOUNTS, index=None, key="source_acc_final", placeholder="Select Source Account", disabled=disabled)


            current_dest_options = [acc for acc in PAYMENT_ACCOUNTS if acc != source_account]
            dest_account = st.selectbox("To Account (Destination)", options=current_dest_options, index=None, key="dest_acc_final", placeholder="Select Destination Account", disabled=disabled)

            transfer_desc = st.text_input("Description (Optional)", value="", disabled=disabled)

            if st.form_submit_button("Record Transfer", disabled=disabled):

                if (transfer_amount and transfer_amount > 0 and
                    source_account is not None and dest_account is not None and
                    source_account != dest_account):


                    group_id = str(uuid.uuid4())

                    # Use session for both DML operations
                    db_execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description, transfer_group_id) VALUES (:id1, :date, 'Expense', :amount, 'Transfer Out', :source, :desc1, :group_id)",
                             params={'id1': str(uuid.uuid4()), 'date': transfer_date.strftime("%Y-%m-%d"), 'amount': round(transfer_amount, 2), 'source': source_account, 'desc1': f"Transfer to {dest_account}" + (f" ({transfer_desc})" if transfer_desc else ""), 'group_id': group_id}, cache_clear_type='general'
                    )
                    db_execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description, transfer_group_id) VALUES (:id2, :date, 'Income', :amount, 'Transfer In', :dest, :desc2, :group_id)",
                              params={'id2': str(uuid.uuid4()), 'date': transfer_date.strftime("%Y-%m-%d"), 'amount': round(transfer_amount, 2), 'dest': dest_account, 'desc2': f"Transfer from {source_account}" + (f" ({transfer_desc})" if transfer_desc else ""), 'group_id': group_id}, cache_clear_type='general'
                    )

                    st.success(f"Transfer of ₹{transfer_amount:,.2f} recorded from **{source_account}** to **{dest_account}**.")
                    st.rerun()
                else:
                    st.warning("Please select valid, different source and destination accounts and a positive amount.")

        st.subheader("Recent Consolidated Transfers (Read-Only)")

        transfer_query = """
        SELECT
            T_OUT.transfer_group_id,
            T_OUT.date,
            T_OUT.amount,
            T_OUT.payment_method AS from_account,
            T_IN.payment_method AS to_account,
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
        """
        transfer_df = db_query(transfer_query)

        if not transfer_df.empty:
            transfer_df.rename(columns={'amount': 'Amount', 'date': 'Date', 'from_account': 'From Account', 'to_account': 'To Account', 'description': 'Description'}, inplace=True)
            transfer_df['Amount'] = transfer_df['Amount'].apply(lambda x: f"₹{x:,.2f}")

            st.dataframe(transfer_df.drop(columns=['transfer_group_id']), hide_index=True, use_container_width=True)
        else:
            st.info("No transfers recorded yet.")

        st.divider()

        st.subheader("Edit Underlying Transfer Transactions")
        st.warning("Editing these directly requires careful attention. Ensure both 'Transfer Out' (Expense) and 'Transfer In' (Income) rows for a single transfer group have the same **Amount**, **Date**, and are linked by the same **Transfer Group ID**.")


        all_transfer_legs_df = db_query("SELECT expense_id, date, type, amount, category, payment_method, transfer_group_id, description FROM expenses WHERE category IN ('Transfer Out', 'Transfer In') ORDER BY date DESC, transfer_group_id DESC, type DESC")

        if not all_transfer_legs_df.empty:

            df_for_editing = all_transfer_legs_df.drop(columns=['category', 'type']).copy()


            df_for_editing['date'] = pd.to_datetime(df_for_editing['date'], format='%Y-%m-%d', errors='coerce').dt.date

            # Disable entire data editor for viewer
            edited_transfer_df = st.data_editor(df_for_editing, use_container_width=True, hide_index=True, num_rows="fixed", disabled=disabled,
                column_config={
                    "expense_id": st.column_config.TextColumn("ID", disabled=True),
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                    "amount": st.column_config.NumberColumn("Amount", min_value=0.01, required=True),
                    "payment_method": st.column_config.SelectboxColumn("Account", options=PAYMENT_ACCOUNTS, required=True),
                    "transfer_group_id": st.column_config.TextColumn("Transfer Group ID", required=True, help="Use the same ID for both OUT and IN legs of a single transfer."),
                    "description": st.column_config.TextColumn("Description")
                })

            if st.button("Save Changes to Transfers", disabled=disabled):

                non_transfer_df = db_query("SELECT * FROM expenses WHERE category NOT IN ('Transfer Out', 'Transfer In')")

                transfers_to_save = edited_transfer_df.copy()


                original_transfer_data = all_transfer_legs_df[['expense_id', 'type', 'category']].set_index('expense_id')


                transfers_to_save = transfers_to_save.merge(original_transfer_data, on='expense_id', how='left')

                if transfers_to_save['type'].isnull().any() or transfers_to_save['category'].isnull().any():
                    st.error("Error: Missing internal transaction type/category data. Cannot save transfers.")
                    st.stop()


                with get_session() as session:
                    session.execute(_sql_text('DELETE FROM expenses'))

                if not non_transfer_df.empty:
                    non_transfer_df['date'] = non_transfer_df['date'].astype(str)
                    df_to_table(non_transfer_df, 'expenses', cache_clear_type='general')


                transfers_to_save['date'] = transfers_to_save['date'].astype(str)

                transfers_to_save = transfers_to_save[['expense_id', 'date', 'type', 'amount', 'category', 'payment_method', 'description', 'transfer_group_id']]

                df_to_table(transfers_to_save, 'expenses', cache_clear_type='general')

                st.success("Transfer transactions updated successfully! Rerunning to validate changes.")
                st.rerun()

        else:
            st.info("No transfers logged yet to edit.")



    elif view == "Transaction History":
        st.header("Transaction History")

        # Optimized: Use cached function and filter
        all_expenses_df = get_all_expenses_df()

        # Exclude transfers for this view
        non_transfer_expenses_df = all_expenses_df[all_expenses_df['category'].isin(['Transfer Out', 'Transfer In']) == False].sort_values(['date', 'expense_id'], ascending=[False, False])

        if not non_transfer_expenses_df.empty:
            all_expenses_df = non_transfer_expenses_df.copy()
            all_expenses_df['date'] = all_expenses_df['date'].dt.date

            editable_categories = sorted(list(set(all_expenses_df['category'].unique().tolist() + CATEGORIES)))

            # Disable entire data editor for viewer
            edited_df = st.data_editor(all_expenses_df[['expense_id', 'date', 'type', 'amount', 'category', 'payment_method', 'description']],
                                             use_container_width=True, hide_index=True, num_rows="fixed", disabled=disabled,
                                             column_config={"expense_id": st.column_config.TextColumn("ID", disabled=True),
                                                             "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                                                             "type": st.column_config.SelectboxColumn("Type", options=["Expense", "Income"], required=True),
                                                              # Use a selectbox for categories allowing user input of new ones
                                                             "category": st.column_config.SelectboxColumn("Category", options=editable_categories, required=True),
                                                             "payment_method": st.column_config.SelectboxColumn("Payment Method", options=[pm for pm in PAYMENT_METHODS], required=True)})

            # Manually convert 'date' column back to string for SQL insertion
            edited_df['date'] = edited_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime.date) else x)

            if st.button("Save Changes to Transactions", disabled=disabled):
                # 1. Fetch all existing transfers (which are being excluded from the editor)
                transfers_df = db_query("SELECT * FROM expenses WHERE category IN ('Transfer Out', 'Transfer In')")

                # 2. Delete all existing records
                with get_session() as session:
                    session.execute(_sql_text('DELETE FROM expenses'))

                # 3. Insert the edited (non-transfer) data back
                edited_df['date'] = edited_df['date'].astype(str) # Convert back to string for SQL
                # Need to manually add transfer_group_id as None since it was dropped from the editor DF
                edited_df['transfer_group_id'] = None
                df_to_table(edited_df, 'expenses', cache_clear_type='general')

                # 4. Insert the untouched transfer data back
                if not transfers_df.empty:
                    transfers_df['date'] = transfers_df['date'].astype(str)
                    df_to_table(transfers_df, 'expenses', cache_clear_type='general')

                st.success("Expenses updated successfully! (Transfer records were preserved)")
                st.cache_data(get_expense_categories_list).clear() # Clear cache in case categories changed
                st.rerun()
        else:
            st.info("No non-transfer transaction history to display.")

    elif view == "Manage Budgets":
        st.header("Set Your Monthly Budgets")
        budget_month_str = datetime.date.today().strftime("%Y-%m")
        st.info(f"You are setting the budget for: **{datetime.datetime.strptime(budget_month_str, '%Y-%m').strftime('%B %Y')}**")

        # Exclude transfer categories from budgeting
        expense_categories_for_budget = db_query("SELECT DISTINCT category FROM expenses WHERE type='Expense' AND category != 'Transfer Out'")['category'].tolist()
        expense_categories_for_budget = sorted(list(expense_categories_for_budget or CATEGORIES))

        existing_budgets = db_query(f"SELECT category, amount FROM budgets WHERE month_year = '{budget_month_str}'").set_index('category')
        budget_df = pd.DataFrame({'category': expense_categories_for_budget, 'amount': [0.0] * len(expense_categories_for_budget)})

        if not existing_budgets.empty:
            budget_df = budget_df.set_index('category').combine_first(existing_budgets).reset_index()

        # Disable entire data editor for viewer
        edited_budgets = st.data_editor(budget_df, num_rows="fixed", disabled=disabled, use_container_width=True,
            column_config={
                "category": st.column_config.TextColumn(label="Category", disabled=True),
                "amount": st.column_config.NumberColumn(label="Amount", min_value=0.0)
            })

        if st.button("Save Budgets", disabled=disabled):

            with get_session() as session:
                # Delete existing budgets for the month and re-insert the updated list
                session.execute(_sql_text("DELETE FROM budgets WHERE month_year = :month"), params={'month': budget_month_str})

                for _, row in edited_budgets.iterrows():
                    if row['amount'] >= 0 and row['category']:
                        # Using 'budgets' table defined with SERIAL PRIMARY KEY
                        session.execute(_sql_text("INSERT INTO budgets (month_year, category, amount) VALUES (:month, :cat, :amount)"),
                                             params={'month': budget_month_str, 'cat': row['category'], 'amount': round(row['amount'], 2)})

            st.success("Budgets saved!")
            st.rerun()

    elif view == "Manage Recurring":
        st.header("Manage Recurring Expenses")
        st.info("Set up expenses that occur every month (e.g., rent, subscriptions). They will be logged automatically.")

        recurring_df = db_query("SELECT recurring_id, description, amount, category, payment_method, day_of_month FROM recurring_expenses")

        # Disable entire data editor for viewer
        edited_recurring = st.data_editor(recurring_df, num_rows="fixed", use_container_width=True, disabled=disabled,
            column_config={
                "recurring_id": st.column_config.NumberColumn(disabled=True),
                "category": st.column_config.TextColumn("Category", required=True),
                "payment_method": st.column_config.SelectboxColumn("Payment Method", options=PAYMENT_ACCOUNTS, required=True),
                "day_of_month": st.column_config.NumberColumn("Day of Month (1-31)", min_value=1, max_value=31, step=1, required=True)
            })

        if st.button("Save Recurring Rules", disabled=disabled):
            # DML update needs to be session based.
            with get_session() as session:
                session.execute(_sql_text('DELETE FROM recurring_expenses'))
                for _, row in edited_recurring.iterrows():
                    if row['description'] and row['amount'] > 0:
                        # Inserting new rows, relying on SERIAL PRIMARY KEY for recurring-id
                        session.execute(_sql_text("INSERT INTO recurring_expenses (description, amount, category, payment_method, day_of_month) VALUES (:desc, :amount, :cat, :pm, :day)"),
                                             params={'desc': row['description'], 'amount': round(row['amount'], 2), 'cat': row['category'], 'pm': row['payment_method'], 'day': row['day_of_month']})
            st.success("Recurring expense rules saved!")
            st.rerun()

    elif view == "Savings Balances":
        st.header("💵 Savings Balances")
        is_viewer = st.session_state.get("role") == "viewer"
        disabled = is_viewer

        # 1. Fetch all savings-related transactions
        # --- FIX APPLIED: Used expense_id and aliased it to transaction_id ---
        SAVINGS_CATEGORY = 'Savings'
        savings_df = db_query(
            "SELECT expense_id AS transaction_id, date, amount, description, payment_method, type FROM expenses WHERE category = :cat ORDER BY date ASC",
            params={'cat': SAVINGS_CATEGORY}
        )
        # ---------------------------------------------------------------------

        if savings_df.empty:
            st.info("No savings transfers recorded yet.")
            return

        # 2. Calculate cumulative savings balance
        savings_df['date'] = pd.to_datetime(savings_df['date'], format='%Y-%m-%d', errors='coerce')

        # New logic: 'Expense' (Savings Transfer IN) increases savings (asset flow),
        # 'SAVINGS WITHDRAWAL' decreases savings.
        savings_df['is_withdrawal'] = savings_df['description'].str.startswith("SAVINGS WITHDRAWAL:")

        # Calculate balance change:
        # If it's a Withdrawal (OUT from savings): -Amount
        # If it's a Transfer (IN to savings): +Amount
        savings_df['balance_change'] = np.where(savings_df['is_withdrawal'], -savings_df['amount'], savings_df['amount'])

        # Recalculate cumulative balance based on chronological order
        savings_df.sort_values('date', ascending=True, inplace=True)
        savings_df['cumulative_balance'] = savings_df['balance_change'].cumsum().round(2)

        # Filter for display (show transfers into and out of savings)
        current_balance = savings_df['cumulative_balance'].iloc[-1]

        # Inflows to Savings: Non-withdrawal transactions
        total_inflow = savings_df.loc[~savings_df['is_withdrawal'], 'amount'].sum()
        # Outflows from Savings: Withdrawal transactions
        total_outflow = savings_df.loc[savings_df['is_withdrawal'], 'amount'].sum()

        # 3. Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Savings Balance", f"₹{current_balance:,.2f}")
        col2.metric("Total Inflow to Savings", f"₹{total_inflow:,.2f}")
        col3.metric("Total Withdrawal from Savings", f"₹{total_outflow:,.2f}")

        st.divider()

        # 4. Display chart (Cumulative Savings Trend)
        st.subheader("Cumulative Savings Trend")

        chart_df = savings_df[['date', 'cumulative_balance']].drop_duplicates(subset=['date'], keep='last')

        # --- CHART MODIFICATION: Explicit line and circle layers for clear marking ---
        base = alt.Chart(chart_df).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('cumulative_balance:Q', title='Cumulative Savings (₹)'),
            tooltip=[alt.Tooltip('date', format="%Y-%m-%d", title='Date'),
                     alt.Tooltip('cumulative_balance', format=",.2f", title='Balance')]
        ).properties(
            height=400
        ).interactive()

        line_chart = base.mark_line(color='#4c78a8') # Line layer
        point_chart = base.mark_circle(size=60, color='orange') # Explicit Point layer

        st.altair_chart(line_chart + point_chart, use_container_width=True)
        # --- END CHART MODIFICATION ---

        st.subheader("Savings Transaction Log (Read-Only)")

        display_df = savings_df[['date', 'description', 'amount', 'payment_method', 'type']].copy()

        # Determine Flow Type and net impact on Savings Account (not liquid cash)
        display_df['Flow Type'] = np.where(display_df['description'].str.startswith("SAVINGS WITHDRAWAL:"), "Withdrawal (Out)", "Transfer (In)")
        display_df['Amount (Savings Net)'] = np.where(display_df['Flow Type'] == 'Withdrawal (Out)', display_df['amount'] * -1, display_df['amount'])

        # Clean up description
        display_df['Description'] = display_df['description'].apply(lambda x: x.replace("SAVINGS TRANSFER:", "").replace("SAVINGS WITHDRAWAL:", "").strip())

        # Determine Liquid Cash Impact for clarity
        display_df['Liquid Cash Impact'] = np.where(
            (display_df['type'] == 'Expense') & (display_df['payment_method'] != 'N/A'), "Debited Liquid Cash",
            np.where((display_df['type'] == 'Income') & (display_df['payment_method'] != 'N/A'), "Credited Liquid Cash", "Manual Entry (No Liquid Cash Impact)")
        )


        st.dataframe(
            display_df[['date', 'Flow Type', 'Amount (Savings Net)', 'Description', 'payment_method', 'Liquid Cash Impact']].sort_values('date', ascending=False),
            hide_index=True,
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "Amount (Savings Net)": st.column_config.NumberColumn("Amount (Net to Savings)", format="₹%,.2f"),
                "payment_method": st.column_config.TextColumn("Liquid Account Used"),
                "Liquid Cash Impact": st.column_config.TextColumn("Liquid Cash Impact")
            }
        )

def mutual_fund_page():
    """Renders the Mutual Fund tracker page."""
    st.title("📚 Mutual Fund Tracker")
    is_viewer = st.session_state.get("role") == "viewer"
    disabled = is_viewer

    _process_recurring_expenses()
    _process_mf_sips()

    # --- Sync Historical MF Data to Expenses on load ---
    if st.session_state.get("mf_sync_run") != True:
        _sync_mf_to_expenses()
        st.session_state["mf_sync_run"] = True
    # ---------------------------------------------------

    key_prefix = "mf"

    # Read transactions_df from DB (not cached here, relies on core cache)
    transactions_df = db_query("SELECT transaction_id, date, scheme_name, yfinance_symbol, type, units, nav, amount FROM mf_transactions ORDER BY date DESC")

    # Select box is enabled for viewer
    view_options = ["Holdings", "Transaction History"]
    table_view = st.selectbox("View Options", view_options, key=f"{key_prefix}_table_view", label_visibility="hidden", disabled=False) # Always enable for navigation

    if table_view == "Holdings":
        st.sidebar.header("Add Transaction")

        # --- MF Search/Selection Block ---
        # Optimized: Use cached function for schemes
        if f"{key_prefix}_all_schemes" not in st.session_state:
            st.session_state[f"{key_prefix}_all_schemes"] = fetch_mf_schemes()
            st.session_state[f"{key_prefix}_search_results"] = []
            st.session_state[f"{key_prefix}_search_term_input"] = ""

        with st.sidebar.form(f"{key_prefix}_search_form"):
            company_name = st.text_input("Search Fund Name", value=st.session_state.get(f"{key_prefix}_search_term_input", ""), key=f"{key_prefix}_search_term_input_live", disabled=disabled)
            search_button = st.form_submit_button("Search", disabled=disabled)

        if search_button and company_name and not disabled:
            filtered_schemes = {name: code for name, code in st.session_state[f"{key_prefix}_all_schemes"].items() if company_name.lower() in name.lower()}
            st.session_state[f"{key_prefix}_search_results"] = [f"{name} ({code})" for name, code in filtered_schemes.items()]
            st.session_state[f"{key_prefix}_selected_result"] = None
            st.session_state[f"{key_prefix}_search_term_input"] = company_name # Keep search term visible
            st.rerun()

        # Display selection box only if results exist (enabled for viewer if only viewing)
        selected_result = None
        if st.session_state.get(f"{key_prefix}_search_results"):
            results = st.session_state[f"{key_prefix}_search_results"]
            # Enabled for viewer for selection
            selected_result = st.sidebar.selectbox(
                "Select Scheme",
                options=[None] + results,
                index=0,
                key=f"{key_prefix}_select_result",
                format_func=lambda x: "Select a scheme..." if x is None else x,
                disabled=disabled
            )

            # Store selected result in session state for later use
            st.session_state[f"{key_prefix}_selected_result"] = selected_result


        # Display details form if a result is selected
        if st.session_state.get(f"{key_prefix}_selected_result") and st.session_state[f"{key_prefix}_selected_result"] is not None:
            selected_result = st.session_state[f"{key_prefix}_selected_result"]
            selected_name = selected_result.split(" (")[0]
            selected_code = selected_result.split(" (")[-1].replace(")", "")

            if disabled:
                st.sidebar.warning("Add Transaction form is disabled in Viewer mode.")

            with st.sidebar.form(f"{key_prefix}_add_details_form"):
                st.subheader(f"Transaction for: {selected_name}")
                mf_date = st.date_input("Date", max_value=datetime.date.today(), disabled=disabled)
                mf_type = st.selectbox("Type", ["Purchase", "Redemption"], disabled=disabled)
                mf_units = st.number_input("Units", min_value=0.001, format="%.4f", disabled=disabled)
                mf_nav = st.number_input("NAV (Net Asset Value)", min_value=0.01, format="%.4f", disabled=disabled)
                mf_fee = st.number_input("Transaction Fee (₹)", min_value=0.00, format="%.2f", value=0.0, disabled=disabled)

                if st.form_submit_button("Add Transaction", disabled=disabled):
                    if not (mf_units and mf_units > 0 and mf_nav and mf_nav > 0):
                        st.warning("Please fill all fields.")
                    else:
                        amount = mf_units * mf_nav
                        funds_change_type = "Withdrawal" if mf_type == "Purchase" else "Deposit"

                        fund_adjustment = mf_fee if mf_type == "Purchase" else -mf_fee
                        final_cash_amount = round(amount + fund_adjustment, 2)

                        # 1. Update Funds (Cash Balance)
                        update_funds_on_transaction(funds_change_type, final_cash_amount, f"MF {mf_type}: {selected_name}", mf_date.strftime("%Y-%m-%d"))

                        # 2. Update MF Transactions
                        db_execute("INSERT INTO mf_transactions (transaction_id, date, scheme_name, yfinance_symbol, type, units, nav, amount) VALUES (:id, :date, :scheme, :symbol, :type, :units, :nav, :amount)",
                                   params={'id': str(uuid.uuid4()), 'date': mf_date.strftime('%Y-%m-%d'), 'scheme': selected_name, 'symbol': selected_code, 'type': mf_type, 'units': round(mf_units, 4), 'nav': round(mf_nav, 4), 'amount': final_cash_amount}, # Use final cash amount for consistency
                        )

                        # 3. Update Expense Tracker (Investment Category)
                        expense_type = "Expense" if mf_type == "Purchase" else "Income"
                        expense_desc = f"{mf_type} {selected_name} units"

                        db_execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description, transfer_group_id) VALUES (:id, :date, :type, :amount, :cat, :pm, :desc, :tg_id)",
                             params={
                                         'id': str(uuid.uuid4()),
                                         'date': mf_date.strftime("%Y-%m-%d"),
                                         'type': expense_type,
                                         'amount': final_cash_amount,
                                         'cat': 'Investment',
                                         'pm': 'N/A',
                                         'desc': expense_desc,
                                         'tg_id': None
                                     }, cache_clear_type='general'
                        )

                        st.success(f"{mf_type} of {selected_name} logged and Expense Tracker updated!")
                        # Clear search state upon successful submission
                        st.session_state[f"{key_prefix}_selected_result"] = None
                        st.session_state[f"{key_prefix}_search_results"] = []
                        st.session_state[f"{key_prefix}_search_term_input"] = ""
                        st.rerun()
        # --- END MF Search/Selection Block ---

        st.divider()

        # Optimized: Use cached function for holdings
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

            st.header("Return Chart (Individual Schemes)")

            transactions_df_temp = transactions_df.copy()

            all_schemes = transactions_df_temp['scheme_name'].unique().tolist() if not transactions_df_temp.empty else []
            # Multiselect for viewing schemes is functional for all users
            selected_schemes = st.multiselect("Select schemes to compare", options=all_schemes, default=all_schemes, disabled=False) # Always ENABLED for viewing/filtering

            if selected_schemes:
                # Optimized: Use cached function for return calculation
                filtered_transactions = transactions_df_temp[transactions_df_temp['scheme_name'].isin(selected_schemes)]
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

                    # Layer and display the charts (Only line_chart + zero_line)
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

            # Disable entire data editor for viewer
            edited_df = st.data_editor(transactions_df, use_container_width=True, hide_index=True, num_rows="fixed", disabled=disabled,
                                             column_config={"transaction_id": st.column_config.TextColumn("ID", disabled=True),
                                                             "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                                                            "scheme_name": st.column_config.TextColumn("Scheme Name", required=True),
                                                            "yfinance_symbol": st.column_config.TextColumn("YF Symbol", required=True),
                                                            "type": st.column_config.SelectboxColumn("Type", options=["Purchase", "Redemption"], required=True),
                                                            "units": st.column_config.NumberColumn("Units", min_value=0.0001, required=True),
                                                            "nav": st.column_config.NumberColumn("NAV", min_value=0.01, required=True),
                                                            "amount": st.column_config.NumberColumn("Amount (Cash Value)", min_value=0.01, required=True) # Added Amount column
                                                        })

            if st.button("Save Mutual Fund Changes", disabled=disabled):
                # DML update needs to be session based.
                with get_session() as session:
                    session.execute(_sql_text('DELETE FROM mf_transactions'))

                edited_df['date'] = edited_df['date'].astype(str)
                df_to_table(edited_df, 'mf_transactions', cache_clear_type='mf')

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
    key_prefix = config['key_prefix']
    is_trading_section = key_prefix == 'trade'
    is_viewer = st.session_state.get("role") == "viewer"
    disabled = is_viewer

    st.title(config["title"])

    trade_mode_selection = "All Trades"

    is_paper_trading = False
    if is_trading_section:
        # --- 1. Paper Trading Toggle (Controls Fund Linkage) ---
        if f"{key_prefix}_paper_trading_state" not in st.session_state:
            st.session_state[f"{key_prefix}_paper_trading_state"] = False

        # Read the state from the toggle, disabled for viewer
        is_paper_trading = st.toggle("Enable Paper Trading (Transactions won't affect Funds)",
                                     key=f"{key_prefix}_paper_trading_toggle",
                                     value=st.session_state[f"{key_prefix}_paper_trading_state"],
                                     disabled=disabled)

        # Immediately save the new toggle state back to session state
        st.session_state[f"{key_prefix}_paper_trading_state"] = is_paper_trading

        if is_paper_trading:
            st.warning("⚠️ **Paper Trading is active.** Buy/Sell transactions will **NOT** update your 'Funds' section.")
        st.divider()

        # --- 2. Live/Paper Filter Dropdown (Controls View Data) ---
        trade_mode_selection = st.radio(
            "Filter Trade Data",
            ["All Trades", "Live Trades Only", "Paper Trades Only"],
            horizontal=True,
            key=f"{key_prefix}_trade_mode_filter",
            disabled=disabled
        )
        st.divider()

    # Sidebar disabled flag
    sidebar_disabled = disabled

    # --- Sidebar forms for Add/Sell remain the same ---

    st.sidebar.header(f"Add {config['asset_name']}")
    with st.sidebar.form(f"{key_prefix}_add_form"):
        company_name = st.text_input(f"{config['asset_name']} Name", value=st.session_state.get(f"{key_prefix}_add_company_name_input", ""), key=f"{key_prefix}_add_company_name_input", disabled=sidebar_disabled)
        search_button = st.form_submit_button("Search", disabled=sidebar_disabled)

    # Search logic must be conditional on button click and permission
    if search_button and company_name and not sidebar_disabled:
        st.session_state[f"{key_prefix}_search_results"] = search_for_ticker(company_name)
        st.session_state[f"{key_prefix}_selected_symbol"] = None
        st.rerun()

    if st.session_state.get(f"{key_prefix}_search_results"):
        results = st.session_state[f"{key_prefix}_search_results"]
        symbols_only = [res.split(" - ")[0] for res in results]

        # Selectbox is disabled for viewer
        selected_symbol_from_search = st.sidebar.selectbox(
            f"Select {config['asset_name']} Symbol",
            options=[None] + symbols_only,
            index=0,
            key=f"{key_prefix}_select_symbol",
            format_func=lambda x: "Select a stock..." if x is None else x,
            disabled=sidebar_disabled
        )

        # Rerun only on user action (owner mode)
        if selected_symbol_from_search and selected_symbol_from_search != st.session_state.get(f"{key_prefix}_selected_symbol") and not sidebar_disabled:
            st.session_state[f"{key_prefix}_selected_symbol"] = selected_symbol_from_search
            st.rerun()

    # Add Details Form (Only visible/functional for owner)
    if st.session_state.get(f"{key_prefix}_selected_symbol"):

        if sidebar_disabled:
            st.sidebar.warning("Add/Edit forms are disabled in Viewer mode.")

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

            buy_price = st.number_input(f"Buy Price ({currency})", min_value=0.01, format="%.2f", key=f"{key_prefix}_buy_price", disabled=sidebar_disabled)
            buy_date = st.date_input("Buy Date", max_value=datetime.date.today(), key=f"{key_prefix}_buy_date", disabled=sidebar_disabled)
            quantity = st.number_input("Quantity", min_value=1, step=1, key=f"{key_prefix}_buy_quantity", disabled=sidebar_disabled)
            transaction_fee = st.number_input("Transaction Fee (₹)", min_value=0.00, format="%.2f", key=f"{key_prefix}_buy_transaction_fee", value=0.0, disabled=sidebar_disabled)

            if not is_trading_section:
                st.text_input("Sector", value=sector, key=f"{key_prefix}_sector", disabled=True)
                st.text_input("Market Cap", value=_categorize_market_cap(market_cap) if market_cap != 'N/A' else 'N/A', key=f"{key_prefix}_market_cap", disabled=True)
            else:
                target_price = st.number_input("Target Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_target_price", disabled=sidebar_disabled)
                stop_loss_price = st.number_input("Stop Loss Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_stop_loss_price", disabled=sidebar_disabled)

            add_button = st.form_submit_button(f"Add to {config['asset_name_plural']}", disabled=sidebar_disabled)

            if add_button:
                is_paper_trading_on_submit = st.session_state.get(f"trade_paper_trading_state", False) if is_trading_section else False

                if not (buy_price and buy_price > 0 and quantity and quantity > 0):
                    st.error("Buy Price and Quantity must be positive.")
                elif update_stock_data(symbol):
                    total_cost = (buy_price * quantity) + transaction_fee

                    # SELECT/UPDATE logic simplified for PostgreSQL using session execute
                    with get_session() as session:
                        # Check for existing holding
                        existing = session.execute(_sql_text(f"SELECT buy_price, quantity FROM {config['asset_table']} WHERE {config['asset_col']}=:symbol"), params={'symbol': symbol}).fetchone()

                        if existing:
                            old_buy_price, old_quantity = existing[0], existing[1]
                            old_total_cost = old_buy_price * old_quantity
                            new_quantity = old_quantity + quantity
                            new_avg_price = (old_total_cost + (buy_price * quantity)) / new_quantity

                            if is_trading_section:
                                session.execute(_sql_text(f"UPDATE {config['asset_table']} SET buy_price=:price, quantity=:qty, target_price=:target, stop_loss_price=:stop WHERE {config['asset_col']}=:symbol"),
                                                params={'price': round(new_avg_price, 2), 'qty': new_quantity, 'target': round(target_price, 2), 'stop': round(stop_loss_price, 2), 'symbol': symbol})
                            else:
                                session.execute(_sql_text(f"UPDATE {config['asset_table']} SET buy_price=:price, quantity=:qty, sector=:sector, market_cap=:mc WHERE {config['asset_col']}=:symbol"),
                                                params={'price': round(new_avg_price, 2), 'qty': new_quantity, 'sector': sector, 'mc': _categorize_market_cap(market_cap), 'symbol': symbol})

                            st.success(f"Updated {symbol}. New quantity: {new_quantity}, New avg. price: {currency}{new_avg_price:,.2f}")
                        else:
                            if is_trading_section:
                                session.execute(_sql_text(f"INSERT INTO {config['asset_table']} ({config['asset_col']}, buy_price, buy_date, quantity, target_price, stop_loss_price) VALUES (:symbol, :price, :date, :qty, :target, :stop)"),
                                                params={'symbol': symbol, 'price': round(buy_price, 2), 'date': buy_date.strftime("%Y-%m-%d"), 'qty': quantity, 'target': round(target_price, 2), 'stop': round(stop_loss_price, 2)})
                            else:
                                session.execute(_sql_text(f"INSERT INTO {config['asset_table']} ({config['asset_col']}, buy_price, buy_date, quantity, sector, market_cap) VALUES (:symbol, :price, :date, :qty, :sector, :mc)"),
                                                params={'symbol': symbol, 'price': round(buy_price, 2), 'date': buy_date.strftime("%Y-%m-%d"), 'qty': quantity, 'sector': sector, 'mc': _categorize_market_cap(market_cap)})

                            st.success(f"{symbol} added successfully!")

                        # --- FUND UPDATE LOGIC (BUY) ---
                        if not is_trading_section or (is_trading_section and not is_paper_trading_on_submit):
                            update_funds_on_transaction("Withdrawal", round(total_cost, 2), f"Purchase {quantity} units of {symbol}", buy_date.strftime("%Y-%m-%d"))
                        # -------------------------------

                else:
                    st.error(f"Failed to fetch historical data for {symbol}. Cannot add.")


    st.sidebar.header(f"Sell {config['asset_name']}")
    # Optimized: Use cached holdings function
    all_symbols_df = db_query(f"SELECT {config['asset_col']} FROM {config['asset_table']}")
    all_symbols = all_symbols_df[config['asset_col']].tolist()

    if all_symbols:
        # Selectbox is disabled for viewer
        selected_option = st.sidebar.selectbox(
            f"Select {config['asset_name']} to Sell",
            options=[None] + all_symbols,
            index=0,
            key=f"{key_prefix}_sell_symbol_selector",
            format_func=lambda x: "Select a stock..." if x is None else x,
            disabled=sidebar_disabled
        )
        available_qty = 1
        if selected_option:
            symbol_to_sell = selected_option
            c_result = db_query(f"SELECT quantity FROM {config['asset_table']} WHERE {config['asset_col']}='{symbol_to_sell}'")
            if not c_result.empty:
                available_qty = c_result['quantity'].iloc[0]
                st.sidebar.info(f"Available to sell: {available_qty} units of {symbol_to_sell}")
            else:
                symbol_to_sell = None
        else:
            symbol_to_sell = None

        with st.sidebar.form(f"{key_prefix}_sell_form"):
            is_disabled_form = not symbol_to_sell or sidebar_disabled
            sell_qty = st.number_input("Quantity to Sell", min_value=1, max_value=available_qty, step=1, key=f"{key_prefix}_sell_qty", disabled=is_disabled_form)
            sell_price = st.number_input("Sell Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_sell_price", disabled=is_disabled_form)
            sell_date = st.date_input("Sell Date", max_value=datetime.date.today(), key=f"{key_prefix}_sell_date", disabled=is_disabled_form)
            sell_transaction_fee = st.number_input("Transaction Fee (₹)", min_value=0.00, format="%.2f", key=f"{key_prefix}_sell_transaction_fee", disabled=is_disabled_form, value=0.0)
            sell_button = st.form_submit_button(f"Sell {config['asset_name']}", disabled=is_disabled_form)

            if sell_button and not sidebar_disabled:
                is_paper_trading_on_submit = st.session_state.get(f"trade_paper_trading_state", False) if is_trading_section else False

                if not symbol_to_sell:
                    st.warning(f"Please select a {config['asset_name']} to sell.")
                elif not (sell_price and sell_price > 0):
                    st.error("Sell price must be greater than zero.")
                elif not (sell_qty and sell_qty > 0):
                    st.error("Quantity to sell must be positive.")
                else:
                    # Get required data from DB
                    if is_trading_section:
                        trade_data_df = db_query(f"SELECT buy_price, buy_date, quantity, target_price, stop_loss_price FROM {config['asset_table']} WHERE {config['asset_col']}='{symbol_to_sell}'")
                        if trade_data_df.empty:
                            st.error("Error: Holding data not found.")
                            st.stop()
                        trade_data = trade_data_df.iloc[0]
                        buy_price, buy_date, current_qty, target_price, stop_loss_price = trade_data['buy_price'], trade_data['buy_date'], trade_data['quantity'], trade_data['target_price'], trade_data['stop_loss_price']
                    else:
                        stock_data_df = db_query(f"SELECT buy_price, buy_date, quantity FROM {config['asset_table']} WHERE {config['asset_col']}='{symbol_to_sell}'")
                        if stock_data_df.empty:
                            st.error("Error: Holding data not found.")
                            st.stop()
                        stock_data = stock_data_df.iloc[0]
                        buy_price, buy_date, current_qty = stock_data['buy_price'], stock_data['buy_date'], stock_data['quantity']
                        target_price, stop_loss_price = None, None # Set to None for non-trading

                    realized_return = ((sell_price - buy_price) / buy_price * 100)
                    transaction_id = str(uuid.uuid4())

                    with get_session() as session:
                        if is_trading_section:
                            session.execute(_sql_text(f"INSERT INTO {config['realized_table']} (transaction_id, {config['asset_col']}, buy_price, buy_date, quantity, sell_price, sell_date, realized_return_pct, target_price, stop_loss_price) VALUES (:id, :symbol, :bprice, :bdate, :qty, :sprice, :sdate, :ret, :target, :stop)"),
                                            params={'id': transaction_id, 'symbol': symbol_to_sell, 'bprice': round(buy_price, 2), 'bdate': buy_date, 'qty': sell_qty, 'sprice': round(sell_price, 2), 'sdate': sell_date.strftime("%Y-%m-%d"), 'ret': round(realized_return, 2), 'target': round(target_price, 2), 'stop': round(stop_loss_price, 2)})
                        else:
                            session.execute(_sql_text(f"INSERT INTO {config['realized_table']} (transaction_id, {config['asset_col']}, buy_price, buy_date, quantity, sell_price, sell_date, realized_return_pct) VALUES (:id, :symbol, :bprice, :bdate, :qty, :sprice, :sdate, :ret)"),
                                            params={'id': transaction_id, 'symbol': symbol_to_sell, 'bprice': round(buy_price, 2), 'bdate': buy_date, 'qty': sell_qty, 'sprice': round(sell_price, 2), 'sdate': sell_date.strftime("%Y-%m-%d"), 'ret': round(realized_return, 2)})

                        # --- FUND UPDATE LOGIC (SELL) ---
                        if not is_trading_section or (is_trading_section and not is_paper_trading_on_submit):
                            update_funds_on_transaction("Deposit", round((sell_price * sell_qty) - sell_transaction_fee, 2), f"Sale of {sell_qty} units of {symbol_to_sell}", sell_date.strftime("%Y-%m-%d"))
                        # --------------------------------

                        if sell_qty == current_qty:
                            session.execute(_sql_text(f"DELETE FROM {config['asset_table']} WHERE {config['asset_col']}=:symbol"), params={'symbol': symbol_to_sell})
                        else:
                            session.execute(_sql_text(f"UPDATE {config['asset_table']} SET quantity=:qty WHERE {config['asset_col']}=:symbol"), params={'qty': current_qty - sell_qty, 'symbol': symbol_to_sell})

                    st.success(f"Sold {sell_qty} units of {symbol_to_sell}.")
                    st.rerun()
    else:
        st.sidebar.info(f"No open {config['asset_name_plural'].lower()}.")

    # --- MAIN VIEW LOGIC ---
    view_options = ["Open Trades", "Closed Trades"] # Simplified for Trading view

    # Fetch Data
    # Optimized: Use cached function
    full_holdings_df = get_holdings_df(config['asset_table'])
    full_realized_df = get_realized_df(config['realized_table']) # Fetch realized data

    # Identify Live Trade Symbols
    live_trade_symbols = set()
    if is_trading_section:
        # Optimized: Use cached function for live trade symbols
        live_trade_symbols = get_live_trade_symbols()

    # Apply Mode Filter
    holdings_df = full_holdings_df.copy()
    realized_df = full_realized_df.copy()

    if is_trading_section:
        if not holdings_df.empty:
            if trade_mode_selection == "Live Trades Only":
                holdings_df = holdings_df[holdings_df[config['asset_col']].isin(live_trade_symbols)]
            elif trade_mode_selection == "Paper Trades Only":
                holdings_df = holdings_df[~holdings_df[config['asset_col']].isin(live_trade_symbols)]

        if not realized_df.empty:
            if trade_mode_selection == "Live Trades Only":
                realized_df = realized_df[realized_df[config['asset_col']].isin(live_trade_symbols)]
            elif trade_mode_selection == "Paper Trades Only":
                realized_df = realized_df[~realized_df[config['asset_col']].isin(live_trade_symbols)]

    # --- Render Open/Closed Trades ---
    st.subheader(f"{config['title']} - {trade_mode_selection if is_trading_section and trade_mode_selection else config['title']}")
    # Selectbox is disabled for viewer
    table_view = st.selectbox("View Options", view_options, key=f"{key_prefix}_table_view_secondary", label_visibility="collapsed", disabled=disabled)

    if table_view == view_options[0]:
        # OPEN TRADES (HOLDINGS)
        if not holdings_df.empty:
            df_to_display = holdings_df.copy() # Use the already filtered data

            total_invested, total_current = df_to_display['invested_value'].sum(), df_to_display['current_value'].sum()
            total_return_amount = (total_current - total_invested).round(2)
            total_return_percent = (total_return_amount / total_invested * 100).round(2) if total_invested > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Investment", f"₹{total_invested:,.2f}")
            with col2: st.metric("Current Value", f"₹{total_current:,.2f}")
            with col3: st.metric("Total Return", f"₹{total_return_amount:,.2f}", f"{total_return_percent:.2f}%")

            if not is_trading_section:
                # --- Alpha/Beta metrics display (Investment section only) ---
                benchmark_options = ['Nifty 50', 'Nifty 100', 'Nifty 200', 'Nifty 500']

                # Use a default choice for the initial data fetch
                initial_benchmark = st.session_state.get(f"{key_prefix}_benchmark_choice", 'Nifty 50')

                # Fetch initial comparison data once.
                initial_comparison_df = get_benchmark_comparison_data(df_to_display, initial_benchmark)

                # 1. Display Metrics (Conditional)
                if not initial_comparison_df.empty:
                    metrics = calculate_portfolio_metrics(df_to_display, initial_comparison_df, initial_benchmark)
                    col_alpha, col_beta, col_drawdown = st.columns(3)
                    with col_alpha: st.metric("Alpha", f"{metrics['alpha']}%")
                    with col_beta: st.metric("Beta", f"{metrics['beta']}")
                    with col_drawdown: st.metric("Max Drawdown", f"{metrics['max_drawdown']}%")

            st.divider()

            # --- START: DETAILED HOLDINGS EXPANDER (COMMON BLOCK) ---
            with st.expander(f"View Detailed {view_options[0]}"):
                column_rename = {
                    'symbol': 'Stock Name', 'ticker': 'Stock Name', 'buy_price': 'Buy Price', 'buy_date': 'Buy Date', 'quantity': 'Quantity',
                    'sector': 'Sector', 'market_cap': 'Market Cap', 'current_price': 'Current Price', 'return_%': 'Return (%)',
                    'return_amount': 'Return (Amount)', 'invested_value': 'Investment Value', 'current_value': 'Current Value',
                    'target_price': 'Target Price', 'stop_loss_price': 'Stop Loss'
                }
                df_to_style = df_to_display.rename(columns=column_rename)

                # Drop columns specific to the *other* view, if applicable
                if not is_trading_section:
                    df_to_style = df_to_style.drop(columns=['Target Price', 'Stop Loss', 'Expected RRR'], errors='ignore')

                date_formatter = lambda t: t.strftime("%d/%m/%Y") if isinstance(t, (pd.Timestamp, datetime.date)) else datetime.datetime.strptime(t, "%Y-%m-%d").strftime("%d/%m/%Y")

                styled_holdings_df = df_to_style.style.map(color_return_value, subset=['Return (%)']).format({
                    'Buy Price': '₹{:.2f}', 'Current Price': '₹{:.2f}', 'Return (Amount)': '₹{:.2f}',
                    'Investment Value': '₹{:.2f}', 'Current Value': '₹{:.2f}', 'Return (%)': '{:.2f}%',
                    'Target Price': '₹{:.2f}', 'Stop Loss': '₹{:.2f}',
                    'Buy Date': date_formatter,
                    'Expected RRR': '{:.2f}'
                })
                # Disable the data editor for the viewer
                st.dataframe(styled_holdings_df, use_container_width=True, hide_index=True)
            # --- END: DETAILED HOLDINGS EXPANDER (COMMON BLOCK) ---

            st.divider()

            # --- MOVED: Return Chart (Individual Assets) ---
            st.header("Return Chart (Individual Assets)")
            all_symbols_list = df_to_display["symbol"].tolist()
            # Multiselect for viewing is functional for all users
            selected_symbols = st.multiselect("Select assets for return chart", all_symbols_list, default=all_symbols_list, key=f"{key_prefix}_perf_symbols", disabled=False) # Always ENABLED for viewing/filtering
            chart_data = []
            for symbol in selected_symbols:
                asset_info = df_to_display.loc[df_to_display["symbol"] == symbol].iloc[0]

                # FIX APPLIED: Ensure asset_info["buy_date"] is converted to a string format
                # that PostgreSQL (with TEXT date) can compare against.
                buy_date_str = asset_info["buy_date"].strftime("%Y-%m-%d")

                history_df = pd.read_sql(
                    _sql_text("SELECT date, close_price FROM price_history WHERE ticker=:symbol AND date>=:date ORDER BY date ASC"),
                    DB_ENGINE,
                    params={'symbol': symbol, 'date': buy_date_str}
                )

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
            # --- END MOVED: Return Chart (Individual Assets) ---

            st.divider()


            # --- START: CONDITIONAL BENCHMARK COMPARISON BLOCK (INVESTMENT ONLY) ---

            if not is_trading_section:

                if initial_comparison_df.empty:
                    # RENDER FAILURE CASE: Hide the entire block and display warning
                    st.warning("Could not generate portfolio metrics or comparison chart. Either benchmark data is unavailable or your portfolio buy dates are too recent.")
                else:
                    # RENDER SUCCESS CASE: Chart controls and chart visible

                    st.header("Portfolio vs. Benchmark Comparison")

                    default_index = benchmark_options.index(initial_benchmark)

                    # Selectbox is ENABLED for viewer as it is a viewing filter
                    benchmark_choice = st.selectbox(
                        "Select Benchmark for Chart Comparison:",
                        options=benchmark_options,
                        key=f"{key_prefix}_benchmark_selector_chart",
                        index=default_index,
                        disabled=False # Always ENABLED for viewing/filtering
                    )

                    # Update session state with the selected choice
                    st.session_state[f"{key_prefix}_benchmark_choice"] = benchmark_choice

                    # Fetch the final data based on the selection (will hit cache if same as initial)
                    comparison_df = get_benchmark_comparison_data(df_to_display, benchmark_choice)

                    comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])

                    chart = alt.Chart(comparison_df).mark_line().encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('Return %:Q', title='Cumulative Return (%)'),
                        color=alt.Color('Type:N', scale=alt.Scale(domain=['Portfolio', benchmark_choice], range=['#1f77b4', '#ff7f0e'])),
                        tooltip=['Date', 'Type', alt.Tooltip('Return %', format=".2f")]
                    ).properties(
                        height=400,
                        title=f"Portfolio vs. {benchmark_choice} Cumulative Return"
                    ).interactive()

                    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="gray", strokeDash=[3,3]).encode(y='y')

                    st.altair_chart(chart + zero_line, use_container_width=True)

                    st.divider()
            # --- END: CONDITIONAL BENCHMARK COMPARISON BLOCK (INVESTMENT ONLY) ---

        else:
            st.info(f"No {trade_mode_selection.lower()} in {view_options[0].lower()} to display.")

    elif table_view == view_options[1]:
        # CLOSED TRADES (REALIZED)
        if not realized_df.empty:
            if is_trading_section:
                trading_metrics = calculate_trading_metrics(realized_df)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Win Ratio", f"{trading_metrics['win_ratio']}%")
                with col2: st.metric("Profit Factor", f"{trading_metrics['profit_factor']}")
                with col3: st.metric("Expectancy", f"₹{trading_metrics['expectancy']}")
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

                # FIX APPLIED: Handles Timestamp objects correctly
                date_formatter = lambda t: t.strftime("%d/%m/%Y") if isinstance(t, (pd.Timestamp, datetime.date)) else datetime.datetime.strptime(t, "%Y-%m-%d").strftime("%d/%m/%Y")

                styled_realized_df = df_to_style.rename(columns=column_rename).style.map(color_return_value, subset=['Return (%)']).format({
                    'Buy Price': '₹{:.2f}', 'Sell Price': '₹{:.2f}', 'P/L (Amount)': '₹{:.2f}',
                    'Investment Value': '₹{:.2f}', 'Realized Value': '₹{:.2f}', 'Return (%)': '{:.2f}%',
                    'Target Price': '₹{:.2f}', 'Stop Loss': '₹{:.2f}',
                    'Buy Date': date_formatter,
                    'Sell Date': date_formatter,
                    'Expected RRR': '{:.2f}', 'Actual RRR': '{:.2f}'
                })
                # Disable the data editor for the viewer
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


# --- MAIN APP LOGIC ---
# Define main_app AFTER all page functions
def main_app():
    """Renders the main dashboard pages."""
    if "page" not in st.session_state:
        st.session_state.page = "home"

    user_role = st.session_state.get("role", "owner")

    # Sidebar status update
    st.sidebar.markdown(f"**Logged in as: {user_role.capitalize()}**")
    st.sidebar.markdown("---")


    # All pages must be defined before this dictionary is created
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

    # Logout button is always available
    if st.sidebar.button("Logout", type="secondary"):
        st.session_state.logged_in = False
        # Clear specific role and page info on logout
        if "role" in st.session_state: del st.session_state["role"]
        st.session_state.page = "home"
        st.rerun()

    # Clear session state button is only for the owner
    if st.sidebar.button("Clear Session State", type="secondary", disabled=user_role != "owner"):
        current_page = st.session_state.get("page", "home")
        # Optimized: Call our custom clear function instead of using Streamlit's global clear which breaks resources
        clear_all_data_caches()
        # Rerun to refresh the state fully
        st.session_state.page = current_page
        st.rerun()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    # Initialize role to 'owner' if user is not logged in yet (or on initial load)
    st.session_state.role = "owner"

if st.session_state.logged_in:
    main_app()
else:
    login_page()
#end
