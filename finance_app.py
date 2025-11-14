import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import logging
import requests
import altair as alt
import uuid
import numpy as np
from mftool import Mftool
import time

from sqlalchemy import create_engine, text as _sql_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

# ------------------ CONFIG & LOGGING ------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:root:%(message)s")

def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# ------------------ NAVIGATION HELPER ------------------
def set_page(page_name):
    """Simple function to set the current page in session state."""
    st.session_state.page = page_name
# ------------------ END NAVIGATION HELPER ------------------------------------


st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="pages/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
from contextlib import contextmanager

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


def db_execute(sql: str, params: dict = None):
    """Execute DML/DDL using a session."""
    try:
        with get_session() as session:
            if params:
                res = session.execute(_sql_text(sql), params)
            else:
                res = session.execute(_sql_text(sql))
            return res
    except Exception as e:
        logging.error(f"db_execute failed: {e}\nSQL: {sql}")
        raise


def df_to_table(df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
    """Insert DataFrame into a SQL table using pandas.to_sql via SQLAlchemy engine. Falls back to row-by-row on failure."""
    if df is None or df.empty:
        return
    try:
        df.to_sql(table_name, DB_ENGINE, if_exists=if_exists, index=False)
        return
    except Exception as e:
        logging.warning(f"pandas.to_sql failed for {table_name}: {e}. Falling back to row inserts.")
    # fallback
    records = df.to_dict(orient='records')
    with get_session() as session:
        for rec in records:
            cols = ', '.join(rec.keys())
            vals = ', '.join(':' + k for k in rec.keys())
            try:
                session.execute(_sql_text(f"INSERT INTO {table_name} ({cols}) VALUES ({vals})"), rec)
            except Exception as ex:
                logging.warning(f"Failed inserting row into {table_name}: {ex}")

# ------------------ APP CONFIG ------------------
USERNAME = st.secrets.get("auth", {}).get("username", "HXR")
PASSWORD = st.secrets.get("auth", {}).get("password", "Rossph")

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
        "CREATE TABLE IF NOT EXISTS mf_transactions (transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, scheme_name TEXT NOT NULL, yfinance_symbol TEXT NOT NULL, type TEXT NOT NULL, units REAL NOT NULL, nav REAL NOT NULL)",
        "CREATE TABLE IF NOT EXISTS mf_sips (sip_id SERIAL PRIMARY KEY, scheme_name TEXT NOT NULL UNIQUE, yfinance_symbol TEXT NOT NULL, amount REAL NOT NULL, day_of_month INTEGER NOT NULL)",
    ]
    for sql in ddl_commands:
        try:
            db_execute(sql)
        except Exception as e:
            logging.warning(f"DDL execution failed for SQL: {sql[:60]}... Error: {e}")

initialize_database()

# ------------------ FUND / TRANSACTION HELPERS ------------------

def update_funds_on_transaction(transaction_type, amount, description, date):
    if description and description.startswith("ALLOCATION:"):
        description = description.split(' - ', 1)[-1].strip()
    sql = "INSERT INTO fund_transactions (transaction_id, date, type, amount, description) VALUES (:id, :date, :type, :amount, :desc)"
    params = {'id': str(uuid.uuid4()), 'date': date, 'type': transaction_type, 'amount': amount, 'desc': description}
    db_execute(sql, params)

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

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=600)
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
        ticker_str = symbol.replace('XNSE:', '') + '.NS' if 'XNSE:' in symbol and not symbol.endswith('.NS') else symbol
        today = datetime.date.today()
        # Use a reasonable lookback period, e.g., 5 years
        start_date = today - datetime.timedelta(days=5 * 365)
        data = yf.download(ticker_str, start=start_date, end=today + datetime.timedelta(days=1), progress=False, auto_adjust=True)
        if data.empty or 'Close' not in data.columns:
            logging.warning(f"YFinance returned empty or invalid data for {symbol}.")
            return False
        data.reset_index(inplace=True)
        data["ticker"] = symbol
        data["date"] = data["Date"].dt.strftime("%Y-%m-%d")
        write_df = data[["ticker", "date", "Close"]].rename(columns={"Close": "close_price"})
        df_to_table(write_df, 'price_history')
        return True
    except Exception as e:
        logging.error(f"YFinance update_stock_data failed for {symbol}: {e}")
        return False

# ------------------ PORTFOLIO / MF CALCULATIONS ------------------

def get_holdings_df(table_name):
    if table_name == "trades":
        query = ("SELECT p.symbol, p.buy_price, p.buy_date, p.quantity, p.target_price, p.stop_loss_price, "
                 "h.close_price AS current_price FROM trades p LEFT JOIN price_history h ON p.symbol = h.ticker "
                 "WHERE h.date = (SELECT MAX(date) FROM price_history WHERE ticker = p.symbol)")
    else:
        query = ("SELECT p.ticker AS symbol, p.buy_price, p.buy_date, p.quantity, p.sector, p.market_cap, "
                 "h.close_price AS current_price FROM portfolio p LEFT JOIN price_history h ON p.ticker = h.ticker "
                 "WHERE h.date = (SELECT MAX(date) FROM price_history WHERE ticker = p.ticker)")
    try:
        df = db_query(query)
        if df.empty:
            return pd.DataFrame()
        # Ensure date columns are proper datetime objects for later comparisons/formatting
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


def get_realized_df(table_name):
    try:
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

# placeholder functions retained
def _update_existing_portfolio_info():
    logging.warning("Function _update_existing_portfolio_info requires explicit session/commit for DML. Skipping execution.")


def _process_recurring_expenses():
    logging.warning("Function _process_recurring_expenses requires explicit session/commit for DML. Skipping execution.")


def _process_mf_sips():
    logging.warning("Function _process_mf_sips requires explicit session/commit for DML. Skipping execution.")

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

def get_mf_holdings_df():
    transactions_df = db_query("SELECT * FROM mf_transactions")
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


def get_combined_returns():
    try:
        # FIX APPLIED: Fetch raw descriptions and process them in Python for reliability
        live_trades_df = db_query("""
            SELECT description
            FROM fund_transactions
            WHERE type = 'Withdrawal'
              AND description LIKE 'Purchase % of %'
        """)

        live_trade_symbols = set()
        if not live_trades_df.empty:
            for desc in live_trades_df['description']:
                parts = desc.split(' of ')
                if len(parts) > 1:
                    live_trade_symbols.add(parts[-1].strip())

    except Exception as e:
        logging.error(f"Error loading live trade symbols in get_combined_returns: {e}")
        live_trade_symbols = set()

    inv_df = get_holdings_df("portfolio")
    inv_invested = float(inv_df['invested_value'].sum()) if not inv_df.empty else 0
    inv_current = float(inv_df['current_value'].sum()) if not inv_df.empty else 0

    trade_df = get_holdings_df("trades")
    # Filter trades based on whether they are linked to a fund withdrawal (i.e., not paper trades)
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
    total_return_pct     = round((total_return_amount / total_invested) * 100, 2) if total_invested > 0 else 0

    realized_stocks_df = get_realized_df("realized_stocks")
    realized_exits_df  = get_realized_df("exits")
    # Filter realized exits based on live trade symbols
    live_exits_df = realized_exits_df[realized_exits_df['symbol'].isin(live_trade_symbols)] if not realized_exits_df.empty else pd.DataFrame()
    realized_inv     = float(realized_stocks_df['realized_profit_loss'].sum()) if not realized_stocks_df.empty else 0
    realized_trade = float(live_exits_df['realized_profit_loss'].sum())         if not live_exits_df.empty      else 0
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
def get_benchmark_data(ticker, start_date):
    """Fetches historical close price data for a benchmark ticker from YFinance."""
    today = datetime.date.today()
    try:
        data = yf.download(ticker, start=start_date, end=today + datetime.timedelta(days=1), progress=False, auto_adjust=True)
        if data.empty or 'Close' not in data.columns:
            logging.warning(f"YFinance returned empty data for benchmark {ticker}.")
            return pd.DataFrame()
        return data['Close'].rename('Close').to_frame()
    except Exception as e:
        logging.error(f"Failed to fetch benchmark data for {ticker}: {e}")
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

    portfolio_prices_df = db_query(
        "SELECT ticker, date, close_price FROM price_history WHERE ticker IN :tickers AND date >= :start_date",
        params={'tickers': tuple(all_tickers), 'start_date': start_date_str}
    ).pivot(index='date', columns='ticker', values='close_price')

    if portfolio_prices_df.empty:
        return pd.DataFrame()

    portfolio_prices_df.index = pd.to_datetime(portfolio_prices_df.index)

    # Benchmark prices from YF (start_date is handled correctly by YF if passed as datetime)
    benchmark_data = get_benchmark_data(benchmark_ticker, start_date)
    if benchmark_data.empty:
        return pd.DataFrame()

    # 4. Align data frames based on price index
    combined_prices = portfolio_prices_df.join(benchmark_data, how='inner')
    combined_prices.fillna(method='ffill', inplace=True)
    combined_prices.fillna(method='bfill', inplace=True)

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
    """Calculates the cumulative return of a mutual fund portfolio over time."""

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
        st.button("📈 Investment", width='stretch', on_click=set_page, args=("investment",))
        st.button("💰 Funds", width='stretch', on_click=set_page, args=("funds",))
    with col2:
        st.button("📊 Trading", width='stretch', on_click=set_page, args=("trading",))
        st.button("💸 Expense Tracker", width='stretch', on_click=set_page, args=("expense_tracker",))
    st.button("📚 Mutual Fund", width='stretch', on_click=set_page, args=("mutual_fund",))

    col_refresh, _ = st.columns([0.2, 0.8])
    with col_refresh:
        if st.button("Refresh Live Data", key="refresh_all_data"):
            with st.spinner("Fetching latest stock and mutual fund prices..."):
                all_tickers = db_query("SELECT ticker FROM portfolio UNION SELECT symbol FROM trades")['ticker'].tolist()
                for symbol in all_tickers:
                    update_stock_data(symbol)
                mf_symbols = db_query("SELECT DISTINCT yfinance_symbol FROM mf_transactions")['yfinance_symbol'].tolist()
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

    fund_df = db_query("SELECT transaction_id, date, type, amount, description FROM fund_transactions ORDER BY date DESC, transaction_id DESC")

    # --- CALCULATE AVAILABLE CASH ---
    total_deposits, total_withdrawals = fund_df.loc[fund_df['type'] == 'Deposit', 'amount'].sum(), fund_df.loc[fund_df['type'] == 'Withdrawal', 'amount'].sum()
    available_capital = round(total_deposits - total_withdrawals, 2)

    if not fund_df.empty:
        fund_df['date'] = pd.to_datetime(fund_df['date'], format='%Y-%m-%d', errors='coerce')
        fund_df['balance'] = fund_df.apply(lambda row: row['amount'] if row['type'] == 'Deposit' else -row['amount'], axis=1)

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

        st.altair_chart(chart, width='stretch')

        st.subheader("Transaction History")

        edited_df = st.data_editor(
            fund_df[['transaction_id', 'date', 'type', 'amount', 'description', 'cumulative_balance']],
            width='stretch',
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "transaction_id": st.column_config.TextColumn("ID", disabled=True),
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=["Deposit", "Withdrawal"], required=True),
                "cumulative_balance": st.column_config.TextColumn("Balance", disabled=True)
            }
        )

        edited_df['date'] = edited_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime.date) else x)


        if st.button("Save Changes to Transactions"):
            # Using session.execute for DML
            with get_session() as session:
                session.execute(_sql_text('DELETE FROM fund_transactions'))

            # Re-insert the data using df_to_table which manages the connection/session
            df_to_table(edited_df[['transaction_id', 'date', 'type', 'amount', 'description']], 'fund_transactions')

            st.success("Funds transactions updated successfully! Rerunning to update the chart.")
            st.rerun()
    else:
        st.info("No fund transactions logged yet.")

def expense_tracker_page():
    """Renders the Expense Tracker page with enhanced category selection, charts/metrics, and the new Transfer functionality."""
    st.title("💸 Expense Tracker")
    _process_recurring_expenses()


    if 'expense_categories_list' not in st.session_state:
        try:
            expense_categories = db_query("SELECT DISTINCT category FROM expenses WHERE type='Expense'")['category'].tolist()
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


    view = st.radio("Select View", ["Dashboard", "Transaction History", "Manage Budgets", "Manage Recurring", "Transfer"], horizontal=True, label_visibility="hidden")


    if view != "Transfer":
        st.sidebar.header("Add Transaction")
        with st.sidebar.form("new_transaction_form", clear_on_submit=True):
            trans_type = st.radio("Transaction Type", ["Expense", "Income"], key="trans_type")
            trans_date = st.date_input("Date", max_value=datetime.date.today(), value=datetime.date.today())
            trans_amount = st.number_input("Amount", min_value=0.01, format="%.2f", value=None)


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


            if custom_category:
                final_cat = custom_category
            elif selected_category and selected_category != 'Select Category...':
                final_cat = selected_category
            else:
                final_cat = None
            # ------------------------------------

            if trans_type == "Income":
                trans_pm = st.selectbox("Destination Account/Method", options=PAYMENT_ACCOUNTS, index=None)
            else:
                trans_pm = st.selectbox("Payment Method", options=PAYMENT_ACCOUNTS, index=None)

            trans_desc = st.text_input("Description", value="")

            if st.form_submit_button("Add Transaction"):
                if trans_amount and final_cat and trans_pm:

                    # Use session for execution via db_execute helper
                    db_execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description) VALUES (:id, :date, :type, :amount, :cat, :pm, :desc)",
                               params={
                                   'id': str(uuid.uuid4()),
                                   'date': trans_date.strftime("%Y-%m-%d"),
                                   'type': trans_type,
                                   'amount': round(trans_amount, 2),
                                   'cat': final_cat,
                                   'pm': trans_pm,
                                   'desc': trans_desc
                               }
                    )
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


    if view == "Dashboard":
        today = datetime.date.today()
        start_date_7days = today - datetime.timedelta(days=6)


        month_year = today.strftime("%Y-%m")

        expenses_df = db_query(f"SELECT * FROM expenses WHERE date LIKE '{month_year}-%'")
        all_time_expenses_df = db_query("SELECT * FROM expenses")
        all_time_expenses_df['date'] = pd.to_datetime(all_time_expenses_df['date'])

        if all_time_expenses_df.empty:
            st.info("No expenses logged yet to display the dashboard.")
            return

        budgets_df = db_query(f"SELECT category, amount FROM budgets WHERE month_year = '{month_year}'").set_index('category')
        inflows_df = expenses_df[expenses_df['type'] == 'Income']
        outflows_df = expenses_df[expenses_df['type'] == 'Expense']


        total_spent = outflows_df[outflows_df['category'] != 'Transfer Out']['amount'].sum()
        total_income = inflows_df[inflows_df['category'] != 'Transfer In']['amount'].sum()

        net_flow = total_income - total_spent


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


        # --- NEW METRIC: Available Amount (Budget - Spent) ---

        spent_by_category_df = outflows_df[outflows_df['category'] != 'Transfer Out'].groupby('category')['amount'].sum().reset_index().set_index('category')

        available_df = budgets_df.copy().rename(columns={'amount': 'budget'})

        available_df['spent'] = spent_by_category_df.reindex(available_df.index)['amount'].fillna(0)

        available_df['available'] = available_df['budget'] - available_df['spent']

        total_available = available_df['available'].sum()

        available_help_text = "\n".join([
            f"{cat}: Budget ₹{row['budget']:,.2f} - Spent ₹{row['spent']:,.2f} = **Available ₹{row['available']:,.2f}**"
            for cat, row in available_df.iterrows()
        ])

        if not available_help_text:
             available_help_text = "No budgets set for this month."

        col4.metric("Available Amount (Budget)", f"₹{total_available:,.2f}",
                     help=f"**Available Amount Breakdown (Budget - Spent this month):**\n{available_help_text}")

        # --- END NEW METRIC ---

        st.divider()


        st.subheader("Daily Spending: Last 7 Days (Excl. Transfers)")


        daily_spending = all_time_expenses_df[
            (all_time_expenses_df['type'] == 'Expense') &
            (all_time_expenses_df['category'] != 'Transfer Out') &
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
                tooltip=['Date', alt.Tooltip('Spent', format='.2f', title='Total Spent (₹)')],
                color=alt.condition(
                    alt.datum.Date == today.strftime('%Y-%m-%d'), # Highlight today
                    alt.value('orange'),
                    alt.value('#4c78a8')
                )
            ).properties(height=300)
            st.altair_chart(bar_chart, width='stretch')

        else:
            st.info("No expense data for the last 7 days (excluding transfers).")

        st.divider()


        st.subheader(f"Category-wise Spending (Current Month: {month_year}, Excl. Transfers)")

        spending_by_category = outflows_df[outflows_df['category'] != 'Transfer Out'].groupby('category')['amount'].sum().reset_index()

        if not spending_by_category.empty:

            total_spent_for_chart = spending_by_category['amount'].sum()
            spending_by_category['percentage'] = (spending_by_category['amount'] / total_spent_for_chart * 100).round(2)

            base = alt.Chart(spending_by_category).encode(
                theta=alt.Theta("amount", stack=True)
            )

            pie = base.mark_arc(outerRadius=120).encode(
                color=alt.Color("category"),

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

            st.altair_chart(pie + text, width='stretch')
        else:
            st.info("No expenses logged for this month to plot (excluding transfers).")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Monthly Spending Trend")

            monthly_spending_df = db_query("SELECT SUBSTR(date, 1, 7) AS month, SUM(amount) AS amount FROM expenses WHERE type='Expense' AND category != 'Transfer Out' GROUP BY month ORDER BY month DESC")
            if not monthly_spending_df.empty:
                bar_chart = alt.Chart(monthly_spending_df).mark_bar().encode(
                    x=alt.X('month', title='Month', sort='-x'),
                    y=alt.Y('amount', title='Total Spent (₹)'),
                    tooltip=[alt.Tooltip('month', title='Month'), alt.Tooltip('amount', format='.2f', title='Amount')]
                ).properties(height=350)
                st.altair_chart(bar_chart, width='stretch')
            else:
                st.info("No expenses logged for this month.")
        with col2:
            st.subheader("Inflow vs. Outflow (Excl. Transfers)")

            monthly_flows_df = db_query("SELECT SUBSTR(date, 1, 7) AS month, type, SUM(amount) AS amount FROM expenses WHERE (type='Income' AND category != 'Transfer In') OR (type='Expense' AND category != 'Transfer Out') GROUP BY month, type ORDER BY month DESC")
            if not monthly_flows_df.empty:
                bar_chart = alt.Chart(monthly_flows_df).mark_bar().encode(
                    x=alt.X('month', title='Month', sort='-x'),
                    y=alt.Y('amount', title='Amount (₹)'),
                    color=alt.Color('type', title='Type', scale=alt.Scale(domain=['Income', 'Expense'], range=['#2ca02c', '#d62728'])),
                    tooltip=['month', alt.Tooltip('type', title='Type'), alt.Tooltip('amount', format='.2f', title='Amount')]
                ).properties(height=300)
                st.altair_chart(bar_chart, width='stretch')
            else:
                st.info("No income or expenses to compare.")


    elif view == "Transfer":
        st.header("🔄 Internal Account Transfer")
        st.info("Record a transfer of funds between your payment methods (e.g., from Net Banking to UPI).")

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

                    # Use db_execute for both DML operations
                    db_execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description, transfer_group_id) VALUES (:id1, :date, 'Expense', :amount, 'Transfer Out', :source, :desc1, :group_id)",
                               params={'id1': str(uuid.uuid4()), 'date': transfer_date.strftime("%Y-%m-%d"), 'amount': round(transfer_amount, 2), 'source': source_account, 'desc1': f"Transfer to {dest_account}" + (f" ({transfer_desc})" if transfer_desc else ""), 'group_id': group_id}
                    )
                    db_execute("INSERT INTO expenses (expense_id, date, type, amount, category, payment_method, description, transfer_group_id) VALUES (:id2, :date, 'Income', :amount, 'Transfer In', :dest, :desc2, :group_id)",
                               params={'id2': str(uuid.uuid4()), 'date': transfer_date.strftime("%Y-%m-%d"), 'amount': round(transfer_amount, 2), 'dest': dest_account, 'desc2': f"Transfer from {source_account}" + (f" ({transfer_desc})" if transfer_desc else ""), 'group_id': group_id}
                    )

                    st.success(f"Transfer of ₹{transfer_amount:,.2f} recorded from **{source_account}** to **{dest_account}**.")
                    st.cache_data.clear()
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

            st.dataframe(transfer_df.drop(columns=['transfer_group_id']), hide_index=True, width='stretch')
        else:
            st.info("No transfers recorded yet.")

        st.divider()

        st.subheader("Edit Underlying Transfer Transactions")
        st.warning("Editing these directly requires careful attention. Ensure both 'Transfer Out' (Expense) and 'Transfer In' (Income) rows for a single transfer group have the same **Amount**, **Date**, and are linked by the same **Transfer Group ID**.")


        all_transfer_legs_df = db_query("SELECT expense_id, date, type, amount, category, payment_method, transfer_group_id, description FROM expenses WHERE category IN ('Transfer Out', 'Transfer In') ORDER BY date DESC, transfer_group_id DESC, type DESC")

        if not all_transfer_legs_df.empty:

            df_for_editing = all_transfer_legs_df.drop(columns=['category', 'type']).copy()


            df_for_editing['date'] = pd.to_datetime(df_for_editing['date'], format='%Y-%m-%d', errors='coerce').dt.date

            edited_transfer_df = st.data_editor(df_for_editing, width='stretch', hide_index=True, num_rows="dynamic",
                column_config={
                    "expense_id": st.column_config.TextColumn("ID", disabled=True),
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                    "amount": st.column_config.NumberColumn("Amount", min_value=0.01, required=True),
                    "payment_method": st.column_config.SelectboxColumn("Account", options=PAYMENT_ACCOUNTS, required=True),
                    "transfer_group_id": st.column_config.TextColumn("Transfer Group ID", required=True, help="Use the same ID for both OUT and IN legs of a single transfer."),
                    "description": st.column_config.TextColumn("Description")
                })

            if st.button("Save Changes to Transfers"):

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
                    df_to_table(non_transfer_df, 'expenses')


                transfers_to_save['date'] = transfers_to_save['date'].astype(str)

                transfers_to_save = transfers_to_save[['expense_id', 'date', 'type', 'amount', 'category', 'payment_method', 'description', 'transfer_group_id']]

                df_to_table(transfers_to_save, 'expenses')

                st.success("Transfer transactions updated successfully! Rerunning to validate changes.")
                st.rerun()

        else:
            st.info("No transfers logged yet to edit.")



    elif view == "Transaction History":
        st.header("Transaction History")

        all_expenses_df = db_query("SELECT expense_id, date, type, amount, category, payment_method, description FROM expenses WHERE category NOT IN ('Transfer Out', 'Transfer In') ORDER BY date DESC, expense_id DESC")

        if not all_expenses_df.empty:
            all_expenses_df['date'] = pd.to_datetime(all_expenses_df['date'], format='%Y-%m-%d', errors='coerce').dt.date

            editable_categories = sorted(list(set(all_expenses_df['category'].unique().tolist() + CATEGORIES)))

            # The st.data_editor will display the data in this sorted order
            edited_df = st.data_editor(all_expenses_df, width='stretch', hide_index=True, num_rows="dynamic",
                                         column_config={"expense_id": st.column_config.TextColumn("ID", disabled=True),
                                                         "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                                                         "type": st.column_config.SelectboxColumn("Type", options=["Expense", "Income"], required=True),
                                                         # Use a selectbox for categories allowing user input of new ones
                                                         "category": st.column_config.SelectboxColumn("Category", options=editable_categories, required=True),
                                                         "payment_method": st.column_config.SelectboxColumn("Payment Method", options=[pm for pm in PAYMENT_METHODS], required=True)})

            # Manually convert 'date' column back to string for SQL insertion
            edited_df['date'] = edited_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime.date) else x)

            if st.button("Save Changes to Transactions"):
                # 1. Fetch all existing transfers (which are being excluded from the editor)
                transfers_df = db_query("SELECT * FROM expenses WHERE category IN ('Transfer Out', 'Transfer In')")

                # 2. Delete all existing records
                with get_session() as session:
                    session.execute(_sql_text('DELETE FROM expenses'))

                # 3. Insert the edited (non-transfer) data back
                edited_df['date'] = edited_df['date'].astype(str) # Convert back to string for SQL
                df_to_table(edited_df, 'expenses')

                # 4. Insert the untouched transfer data back
                if not transfers_df.empty:
                    transfers_df['date'] = transfers_df['date'].astype(str)
                    df_to_table(transfers_df, 'expenses')

                st.success("Expenses updated successfully! (Transfer records were preserved)")
                st.cache_data.clear() # Clear cache in case categories changed
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
        edited_budgets = st.data_editor(budget_df, num_rows="dynamic", width='stretch', column_config={
            "category": st.column_config.TextColumn(label="Category", disabled=True),
            "amount": st.column_config.NumberColumn(label="Amount", min_value=0.0)
        })
        if st.button("Save Budgets"):

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
        edited_recurring = st.data_editor(recurring_df, num_rows="dynamic", width='stretch', column_config={
            "recurring_id": st.column_config.NumberColumn(disabled=True),
            "category": st.column_config.TextColumn("Category", required=True),
            "payment_method": st.column_config.SelectboxColumn("Payment Method", options=PAYMENT_ACCOUNTS, required=True),
            "day_of_month": st.column_config.NumberColumn("Day of Month (1-31)", min_value=1, max_value=31, step=1, required=True)
        })
        if st.button("Save Recurring Rules"):
            # DML update needs to be session based.
            with get_session() as session:
                session.execute(_sql_text('DELETE FROM recurring_expenses'))
                for _, row in edited_recurring.iterrows():
                    if row['description'] and row['amount'] > 0:
                        # Inserting new rows, relying on SERIAL PRIMARY KEY for recurring_id
                        session.execute(_sql_text("INSERT INTO recurring_expenses (description, amount, category, payment_method, day_of_month) VALUES (:desc, :amount, :cat, :pm, :day)"),
                                         params={'desc': row['description'], 'amount': round(row['amount'], 2), 'cat': row['category'], 'pm': row['payment_method'], 'day': row['day_of_month']})
            st.success("Recurring expense rules saved!")
            st.rerun()

def mutual_fund_page():
    """Renders the Mutual Fund tracker page."""
    st.title("📚 Mutual Fund Tracker")
    _process_recurring_expenses()
    _process_mf_sips()
    key_prefix = "mf"

    # Read transactions_df at the beginning of the function
    transactions_df = db_query("SELECT transaction_id, date, scheme_name, yfinance_symbol, type, units, nav FROM mf_transactions ORDER BY date DESC")

    view_options = ["Holdings", "Transaction History"]
    table_view = st.selectbox("View Options", view_options, key=f"{key_prefix}_table_view", label_visibility="hidden")

    if table_view == "Holdings":
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

                        fund_adjustment = mf_fee if mf_type == "Purchase" else -mf_fee

                        update_funds_on_transaction(funds_change_type, round(amount + fund_adjustment, 2), f"MF {mf_type}: {selected_name}", mf_date.strftime("%Y-%m-%d"))

                        # Use db_execute for the transaction insertion
                        db_execute("INSERT INTO mf_transactions (transaction_id, date, scheme_name, yfinance_symbol, type, units, nav) VALUES (:id, :date, :scheme, :symbol, :type, :units, :nav)",
                                   params={'id': str(uuid.uuid4()), 'date': mf_date.strftime('%Y-%m-%d'), 'scheme': selected_name, 'symbol': selected_code, 'type': mf_type, 'units': round(mf_units, 4), 'nav': round(mf_nav, 4)}
                        )

                        st.success(f"{mf_type} of {selected_name} logged!")
                        st.session_state[f"{key_prefix}_selected_result"] = None
                        st.session_state[f"{key_prefix}_search_results"] = []
                        st.session_state[f"{key_prefix}_search_term_input"] = ""
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
                st.dataframe(styled_df, width='stretch', hide_index=True)

            st.header("Return Chart (Individual Schemes)")

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

                    # Layer and display the charts (Only line_chart + zero_line)
                    st.altair_chart(line_chart + zero_line, width='stretch')
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
            edited_df = st.data_editor(transactions_df, width='stretch', hide_index=True, num_rows="dynamic",
                                         column_config={"transaction_id": st.column_config.TextColumn("ID", disabled=True),
                                                         "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                                                         "scheme_name": st.column_config.TextColumn("Scheme Name", required=True),
                                                         "yfinance_symbol": st.column_config.TextColumn("YF Symbol", required=True),
                                                         "type": st.column_config.SelectboxColumn("Type", options=["Purchase", "Redemption"], required=True),
                                                         "units": st.column_config.NumberColumn("Units", min_value=0.0001, required=True),
                                                         "nav": st.column_config.NumberColumn("NAV", min_value=0.01, required=True)
                                                            })

            if st.button("Save Mutual Fund Changes"):
                # DML update needs to be session based.
                with get_session() as session:
                    session.execute(_sql_text('DELETE FROM mf_transactions'))

                edited_df['date'] = edited_df['date'].astype(str)
                df_to_table(edited_df, 'mf_transactions')

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
    st.title(config["title"])

    trade_mode_selection = "All Trades"

    is_paper_trading = False
    if is_trading_section:
        # --- 1. Paper Trading Toggle (Controls Fund Linkage) ---
        if f"{key_prefix}_paper_trading_state" not in st.session_state:
            st.session_state[f"{key_prefix}_paper_trading_state"] = False

        # Read the state from the toggle
        is_paper_trading = st.toggle("Enable Paper Trading (Transactions won't affect Funds)",
                                     key=f"{key_prefix}_paper_trading_toggle",
                                     value=st.session_state[f"{key_prefix}_paper_trading_state"])

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
            key=f"{key_prefix}_trade_mode_filter"
        )
        st.divider()

    # --- Sidebar forms for Add/Sell remain the same ---

    st.sidebar.header(f"Add {config['asset_name']}")
    with st.sidebar.form(f"{key_prefix}_add_form"):
        company_name = st.text_input(f"{config['asset_name']} Name", value=st.session_state.get(f"{key_prefix}_add_company_name_input", ""), key=f"{key_prefix}_add_company_name_input")
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
    all_symbols_df = db_query(f"SELECT {config['asset_col']} FROM {config['asset_table']}")
    all_symbols = all_symbols_df[config['asset_col']].tolist()

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
            c_result = db_query(f"SELECT quantity FROM {config['asset_table']} WHERE {config['asset_col']}='{symbol_to_sell}'")
            if not c_result.empty:
                available_qty = c_result['quantity'].iloc[0]
                st.sidebar.info(f"Available to sell: {available_qty} units of {symbol_to_sell}")
            else:
                symbol_to_sell = None
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
    full_holdings_df = get_holdings_df(config['asset_table'])
    full_realized_df = get_realized_df(config['realized_table']) # Fetch realized data

    # Identify Live Trade Symbols
    live_trade_symbols = set()
    if is_trading_section:
        fund_tx = db_query("SELECT description FROM fund_transactions WHERE type='Withdrawal'")
        for desc in fund_tx['description']:
            if desc.startswith("Purchase"):
                parts = desc.split(' of ')
                if len(parts) > 1:
                    live_trade_symbols.add(parts[-1].strip())

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
    table_view = st.selectbox("View Options", view_options, key=f"{key_prefix}_table_view_secondary", label_visibility="collapsed") # Re-use the existing selector

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

                # Default to 'Nifty 50' if not set
                if f"{key_prefix}_benchmark_choice" not in st.session_state:
                    st.session_state[f"{key_prefix}_benchmark_choice"] = 'Nifty 50'

                # Fetch comparison data first to calculate metrics accurately
                comparison_df = get_benchmark_comparison_data(df_to_display, st.session_state[f"{key_prefix}_benchmark_choice"])

                # The portfolio_data needs to be passed to calculate_portfolio_metrics
                metrics = calculate_portfolio_metrics(df_to_display, comparison_df, st.session_state[f"{key_prefix}_benchmark_choice"])
                col_alpha, col_beta, col_drawdown = st.columns(3)
                with col_alpha: st.metric("Alpha", f"{metrics['alpha']}%")
                with col_beta: st.metric("Beta", f"{metrics['beta']}")
                with col_drawdown: st.metric("Max Drawdown", f"{metrics['max_drawdown']}%")

            st.divider()

            # --- START: DETAILED HOLDINGS (MOVED UP FOR INVESTMENT SECTION) ---
            if not is_trading_section:
                with st.expander(f"View Detailed {view_options[0]}"):
                    column_rename = {
                        'symbol': 'Stock Name', 'ticker': 'Stock Name', 'buy_price': 'Buy Price', 'buy_date': 'Buy Date', 'quantity': 'Quantity',
                        'sector': 'Sector', 'market_cap': 'Market Cap', 'current_price': 'Current Price', 'return_%': 'Return (%)',
                        'return_amount': 'Return (Amount)', 'invested_value': 'Investment Value', 'current_value': 'Current Value',
                        'target_price': 'Target Price', 'stop_loss_price': 'Stop Loss'
                    }
                    df_to_style = df_to_display.rename(columns=column_rename)
                    df_to_style = df_to_style.drop(columns=['Target Price', 'Stop Loss', 'Expected RRR'], errors='ignore')

                    # FIX APPLIED: Handles Timestamp objects correctly
                    date_formatter = lambda t: t.strftime("%d/%m/%Y") if isinstance(t, (pd.Timestamp, datetime.date)) else datetime.datetime.strptime(t, "%Y-%m-%d").strftime("%d/%m/%Y")

                    styled_holdings_df = df_to_style.style.map(color_return_value, subset=['Return (%)']).format({
                        'Buy Price': '₹{:.2f}', 'Current Price': '₹{:.2f}', 'Return (Amount)': '₹{:.2f}',
                        'Investment Value': '₹{:.2f}', 'Current Value': '₹{:.2f}', 'Return (%)': '{:.2f}%',
                        'Target Price': '₹{:.2f}', 'Stop Loss': '₹{:.2f}',
                        'Buy Date': date_formatter,
                        'Expected RRR': '{:.2f}'
                    })
                    st.dataframe(styled_holdings_df, width='stretch', hide_index=True)
            # --- END: DETAILED HOLDINGS ---

            # --- START: Portfolio vs Benchmark Chart (Investment only) ---
            if not is_trading_section and not df_to_display.empty:
                st.header("Portfolio vs. Benchmark Comparison")

                benchmark_options = ['Nifty 50', 'Nifty 100', 'Nifty 200', 'Nifty 500']

                default_benchmark = st.session_state.get(f"{key_prefix}_benchmark_choice", 'Nifty 50')
                default_index = benchmark_options.index(default_benchmark) if default_benchmark in benchmark_options else 0

                benchmark_choice = st.selectbox(
                    "Select Benchmark for Chart Comparison:",
                    options=benchmark_options,
                    key=f"{key_prefix}_benchmark_selector_chart",
                    index=default_index
                )

                st.session_state[f"{key_prefix}_benchmark_choice"] = benchmark_choice

                # Recalculate comparison data after selectbox update
                comparison_df = get_benchmark_comparison_data(df_to_display, benchmark_choice)

                if not comparison_df.empty:
                    comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])

                    chart = alt.Chart(comparison_df).mark_line().encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('Return %:Q', title='Cumulative Return (%)'),
                        color=alt.Color('Type:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e'])), # Blue for Portfolio, Orange for Benchmark
                        tooltip=['Date', 'Type', alt.Tooltip('Return %', format=".2f")]
                    ).properties(
                        height=400,
                        title=f"Portfolio vs. {benchmark_choice} Cumulative Return"
                    ).interactive()

                    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="gray", strokeDash=[3,3]).encode(y='y')

                    st.altair_chart(chart + zero_line, width='stretch')
                else:
                    st.info(f"Cannot generate benchmark chart. Either market data is unavailable for {benchmark_choice} or buy dates are too recent.")
                st.divider()
            # --- END: Portfolio vs Benchmark Chart (Investment only) ---

            # --- DETAILED HOLDINGS (TRADING SECTION ONLY, was already correct) ---
            if is_trading_section:
                with st.expander(f"View Detailed {view_options[0]}"):
                    column_rename = {
                        'symbol': 'Stock Name', 'ticker': 'Stock Name', 'buy_price': 'Buy Price', 'buy_date': 'Buy Date', 'quantity': 'Quantity',
                        'sector': 'Sector', 'market_cap': 'Market Cap', 'current_price': 'Current Price', 'return_%': 'Return (%)',
                        'return_amount': 'Return (Amount)', 'invested_value': 'Investment Value', 'current_value': 'Current Value',
                        'target_price': 'Target Price', 'stop_loss_price': 'Stop Loss'
                    }
                    df_to_style = df_to_display.rename(columns=column_rename)

                    # FIX APPLIED: Handles Timestamp objects correctly
                    date_formatter = lambda t: t.strftime("%d/%m/%Y") if isinstance(t, (pd.Timestamp, datetime.date)) else datetime.datetime.strptime(t, "%Y-%m-%d").strftime("%d/%m/%Y")


                    styled_holdings_df = df_to_style.style.map(color_return_value, subset=['Return (%)']).format({
                        'Buy Price': '₹{:.2f}', 'Current Price': '₹{:.2f}', 'Return (Amount)': '₹{:.2f}',
                        'Investment Value': '₹{:.2f}', 'Current Value': '₹{:.2f}', 'Return (%)': '{:.2f}%',
                        'Target Price': '₹{:.2f}', 'Stop Loss': '₹{:.2f}',
                        'Buy Date': date_formatter,
                        'Expected RRR': '{:.2f}'
                    })
                    st.dataframe(styled_holdings_df, width='stretch', hide_index=True)
            # --- END: DETAILED HOLDINGS (TRADING) ---


            st.header("Return Chart (Individual Assets)")
            all_symbols_list = df_to_display["symbol"].tolist()
            selected_symbols = st.multiselect("Select assets for return chart", all_symbols_list, default=all_symbols_list, key=f"{key_prefix}_perf_symbols")
            chart_data = []
            for symbol in selected_symbols:
                asset_info = df_to_display.loc[df_to_display["symbol"] == symbol].iloc[0]

                # FIX APPLIED: Ensure asset_info["buy_date"] is converted to a string format
                # that PostgreSQL (with TEXT date) can compare against.
                # This fixes the "operator does not exist: text >= timestamp" error.
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
                st.altair_chart(chart + zero_line, width='stretch')
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
                st.dataframe(styled_realized_df, width='stretch', hide_index=True)
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
            st.altair_chart(bars, width='stretch')
        else:
            st.info(f"No {trade_mode_selection.lower()} in {view_options[1].lower()} to display.")


# --- MAIN APP LOGIC ---
# Define main_app AFTER all page functions
def main_app():
    """Renders the main dashboard pages."""
    if "page" not in st.session_state:
        st.session_state.page = "home"

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
