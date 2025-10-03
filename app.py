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
import numpy_financial as npf

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s:root:%(message)s")

# NEW Correct Version
st.set_page_config(
    page_title="Finance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INDICATOR FUNCTIONS ---
def rsi(close, period=14):
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
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
    """Initializes all database tables if they don't exist."""
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS portfolio (ticker TEXT PRIMARY KEY, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades (symbol TEXT PRIMARY KEY, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL DEFAULT 1, target_price REAL NOT NULL, stop_loss_price REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS price_history (ticker TEXT, date TEXT, close_price REAL, PRIMARY KEY (ticker, date))""")
    c.execute("""CREATE TABLE IF NOT EXISTS realized_stocks (transaction_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL, sell_price REAL NOT NULL, sell_date TEXT NOT NULL, realized_return_pct REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS exits (transaction_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, buy_price REAL NOT NULL, buy_date TEXT NOT NULL, quantity INTEGER NOT NULL, sell_price REAL NOT NULL, sell_date TEXT NOT NULL, realized_return_pct REAL NOT NULL, target_price REAL NOT NULL, stop_loss_price REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS fund_transactions (transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, type TEXT NOT NULL, amount REAL NOT NULL, description TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS expenses (expense_id TEXT PRIMARY KEY, date TEXT NOT NULL, amount REAL NOT NULL, category TEXT NOT NULL, payment_method TEXT, description TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS budgets (budget_id INTEGER PRIMARY KEY AUTOINCREMENT, month_year TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL, UNIQUE(month_year, category))""")
    c.execute("""CREATE TABLE IF NOT EXISTS recurring_expenses (recurring_id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT NOT NULL UNIQUE, amount REAL NOT NULL, category TEXT NOT NULL, payment_method TEXT, day_of_month INTEGER NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS mf_transactions (transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, scheme_name TEXT NOT NULL, yfinance_symbol TEXT NOT NULL, type TEXT NOT NULL, units REAL NOT NULL, nav REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS mf_sips (sip_id INTEGER PRIMARY KEY AUTOINCREMENT, scheme_name TEXT NOT NULL UNIQUE, yfinance_symbol TEXT NOT NULL, amount REAL NOT NULL, day_of_month INTEGER NOT NULL)""")
    conn.commit()

DB_CONN = get_db_connection()
initialize_database(DB_CONN)

# --- API & DATA FUNCTIONS ---
@st.cache_data(ttl=3600)
def search_for_ticker(company_name):
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
def fetch_current_price(symbol):
    try:
        ticker_str = symbol.replace('XNSE:', '') + '.NS' if 'XNSE:' in symbol else symbol
        ticker_obj = yf.Ticker(ticker_str)
        price = ticker_obj.fast_info.get('last_price')
        if price:
            return round(price, 2)
        data = ticker_obj.history(period='2d', auto_adjust=True)
        if not data.empty:
            return round(data['Close'].iloc[-1], 2)
        return None
    except Exception as e:
        logging.error(f"YFinance fetch_current_price failed for {symbol}: {e}")
        return None

def update_stock_data(symbol):
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
    if table_name == "trades":
        query = "SELECT p.symbol, p.buy_price, p.buy_date, p.quantity, p.target_price, p.stop_loss_price, h.close_price AS current_price FROM trades p LEFT JOIN (SELECT ticker, close_price FROM price_history WHERE (ticker, date) IN (SELECT ticker, MAX(date) FROM price_history GROUP BY ticker)) h ON p.symbol = h.ticker"
    else:
        query = "SELECT p.ticker AS symbol, p.buy_price, p.buy_date, p.quantity, h.close_price AS current_price FROM portfolio p LEFT JOIN (SELECT ticker, close_price FROM price_history WHERE (ticker, date) IN (SELECT ticker, MAX(date) FROM price_history GROUP BY ticker)) h ON p.ticker = h.ticker"
    try:
        df = pd.read_sql(query, DB_CONN)
        if df.empty:
            return pd.DataFrame()

        df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce').fillna(0)
        df["return_%"] = ((df["current_price"] - df["buy_price"]) / df["buy_price"] * 100).round(2)
        df["invested_value"] = (df["buy_price"] * df["quantity"]).round(2)
        df["current_value"] = (df["current_price"] * df["quantity"]).round(2)
        if table_name == "trades":
            reward, risk = df["target_price"] - df["buy_price"], df["buy_price"] - df["stop_loss_price"]
            df["Expected RRR"] = np.where(risk > 0, (reward / risk).round(2), np.inf)
        return df
    except Exception as e:
        logging.error(f"Error querying {table_name}: {e}")
        return pd.DataFrame()

def get_realized_df(table_name):
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", DB_CONN)
        if df.empty:
            return pd.DataFrame()
        df["invested_value"] = (df["buy_price"] * df["quantity"]).round(2)
        df["realized_value"] = (df["sell_price"] * df["quantity"]).round(2)
        df["realized_profit_loss"] = df["realized_value"] - df["invested_value"]
        if table_name == "exits":
            expected_reward, original_risk, actual_reward = df["target_price"] - df["buy_price"], df["buy_price"] - df["stop_loss_price"], df["sell_price"] - df["buy_price"]
            df["Expected RRR"] = np.where(original_risk > 0, (expected_reward / original_risk).round(2), np.inf)
            df["Actual RRR"] = np.where(original_risk > 0, (actual_reward / original_risk).round(2), np.inf)
        return df
    except Exception as e:
        logging.error(f"Error querying {table_name}: {e}")
        return pd.DataFrame()

# --- HELPER FUNCTIONS ---
def _process_recurring_expenses():
    c = DB_CONN.cursor()
    today, month_year = datetime.date.today(), datetime.date.today().strftime("%Y-%m")
    try:
        recurring_df = pd.read_sql("SELECT * FROM recurring_expenses", DB_CONN)
        if recurring_df.empty:
            return
        expenses_df = pd.read_sql(f"SELECT description, date FROM expenses WHERE date LIKE '{month_year}-%'", DB_CONN)
        for _, row in recurring_df.iterrows():
            try:
                day = min(row['day_of_month'], pd.Timestamp(month_year).days_in_month)
                expense_date, marker = f"{month_year}-{day:02d}", f"Recurring: {row['description']}"
                if not expenses_df[(expenses_df['description'] == marker) & (expenses_df['date'].str.startswith(month_year))].empty:
                    continue
                c.execute("INSERT INTO expenses (expense_id, date, amount, category, payment_method, description) VALUES (?, ?, ?, ?, ?, ?)", (str(uuid.uuid4()), expense_date, row['amount'], row['category'], row['payment_method'], marker))
                DB_CONN.commit()
                logging.info(f"Logged recurring expense: {row['description']}")
            except Exception as e:
                logging.error(f"Error processing recurring row {row['description']}: {e}")
    except Exception as e:
        logging.error(f"Could not process recurring expenses: {e}")

def _calculate_xirr(transactions_df, latest_nav):
    if transactions_df.empty:
        return 0.0
    purchases = transactions_df[transactions_df['type'] == 'Purchase'].copy()
    redemptions = transactions_df[transactions_df['type'] == 'Redemption'].copy()
    purchases['cash_flow'] = -1 * purchases['units'] * purchases['nav']
    redemptions['cash_flow'] = 1 * redemptions['units'] * redemptions['nav']
    total_units = purchases['units'].sum() - redemptions['units'].sum()
    if total_units <= 0.001:
        if redemptions.empty: return 0.0
        all_flows = pd.concat([purchases[['date', 'cash_flow']], redemptions[['date', 'cash_flow']]])
        if all_flows['cash_flow'].min() >= 0 or all_flows['cash_flow'].max() <= 0: return 0.0
    else:
        current_value = total_units * latest_nav
        final_redemption = pd.DataFrame([{'date': datetime.date.today().strftime('%Y-%m-%d'), 'cash_flow': current_value}])
        all_flows = pd.concat([purchases[['date', 'cash_flow']], redemptions[['date', 'cash_flow']], final_redemption])
    all_flows['date'] = pd.to_datetime(all_flows['date'])
    dates = all_flows['date'].tolist()
    values = all_flows['cash_flow'].tolist()
    if len(values) < 2 or not any(v < 0 for v in values) or not any(v > 0 for v in values):
        return 0.0
    try:
        return npf.xirr(values, dates) * 100
    except Exception:
        return 0.0

def _process_mf_sips():
    c = DB_CONN.cursor()
    today = datetime.date.today()
    try:
        sips_df = pd.read_sql("SELECT * FROM mf_sips", DB_CONN)
        if sips_df.empty:
            return
        for _, sip in sips_df.iterrows():
            day = min(sip['day_of_month'], pd.Timestamp.now().days_in_month)
            sip_date_this_month = today.replace(day=day)
            if sip_date_this_month <= today:
                existing_sip_tx = pd.read_sql(f"SELECT * FROM mf_transactions WHERE scheme_name = ? AND date = ?", (sip['scheme_name'], sip_date_this_month.strftime('%Y-%m-%d')))
                if existing_sip_tx.empty:
                    try:
                        nav_data = yf.Ticker(sip['yfinance_symbol']).history(start=sip_date_this_month, end=sip_date_this_month + datetime.timedelta(days=1))
                        if not nav_data.empty:
                            nav = nav_data['Close'].iloc[0]
                            units = sip['amount'] / nav
                            c.execute("INSERT INTO mf_transactions (transaction_id, date, scheme_name, yfinance_symbol, type, units, nav) VALUES (?, ?, ?, ?, ?, ?, ?)", (str(uuid.uuid4()), sip_date_this_month.strftime('%Y-%m-%d'), sip['scheme_name'], sip['yfinance_symbol'], 'Purchase', units, nav))
                            DB_CONN.commit()
                            logging.info(f"Auto-logged SIP for {sip['scheme_name']}")
                            st.sidebar.success(f"Auto-logged SIP for {sip['scheme_name']}")
                    except Exception as e:
                        st.sidebar.warning(f"Could not auto-log SIP for {sip['scheme_name']}. NAV fetch failed: {e}")
    except Exception as e:
        logging.error(f"Failed during MF SIP processing: {e}")

@st.cache_data(ttl=3600)
def get_benchmark_comparison_data(holdings_df, benchmark_choice):
    if holdings_df.empty:
        return pd.DataFrame()
    start_date = holdings_df['buy_date'].min()
    end_date = datetime.date.today().strftime('%Y-%m-%d')

    benchmark_map = {
        'Nifty 50': '^NSEI',
        'Nifty 500': '^CRSLDX'
    }
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
        benchmark_returns = ((benchmarks / benchmarks.iloc[0] - 1) * 100)
        benchmark_returns.name = benchmark_choice  # Set the Series name directly

        all_tickers = holdings_df['symbol'].unique().tolist()
        price_data_query = f"""SELECT date, ticker, close_price FROM price_history WHERE ticker IN ({','.join(['?']*len(all_tickers))}) AND date >= ?"""
        all_prices = pd.read_sql(price_data_query, DB_CONN, params=[*all_tickers, start_date])
        price_pivot = all_prices.pivot(index='date', columns='ticker', values='close_price').ffill()

        date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date))
        daily_quantities = pd.DataFrame(index=date_range)
        daily_invested = pd.DataFrame(index=date_range)

        for _, row in holdings_df.iterrows():
            daily_quantities[row['symbol']] = np.where(daily_quantities.index >= pd.to_datetime(row['buy_date']), row['quantity'], 0)
            daily_invested[row['symbol']] = np.where(daily_invested.index >= pd.to_datetime(row['buy_date']), row['quantity'] * row['buy_price'], 0)

        price_pivot.index = pd.to_datetime(price_pivot.index)
        price_pivot = price_pivot.reindex(date_range).ffill()

        daily_market_value = (price_pivot * daily_quantities).sum(axis=1)
        total_daily_invested = daily_invested.sum(axis=1).replace(0, np.nan).ffill()

        portfolio_return = ((daily_market_value - total_daily_invested) / total_daily_invested * 100).rename('Portfolio')

        final_df = pd.concat([portfolio_return, benchmark_returns], axis=1).reset_index().rename(columns={'index': 'Date'})
        final_df = final_df.melt(id_vars='Date', var_name='Type', value_name='Return %').dropna()

        return final_df
    except Exception as e:
        logging.error(f"Failed to generate benchmark data: {e}", exc_info=True)
        return pd.DataFrame()

# --- PAGE RENDERERS ---
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
    st.title(config["title"])
    c = DB_CONN.cursor()
    key_prefix = config['key_prefix']
    is_trading_section = config['key_prefix'] == 'trade'
    st.sidebar.header(f"Add {config['asset_name']}")
    for key in ['search_results', 'selected_symbol', 'company_name']:
        session_key = f"{key_prefix}_{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = [] if key == 'search_results' else ""
    with st.sidebar.form(f"{key_prefix}_add_form"):
        company_name = st.text_input(f"{config['asset_name']} Name", value=st.session_state[f"{key_prefix}_company_name"], key=f"{key_prefix}_add_company_name")
        search_button = st.form_submit_button("Search")
        if search_button and company_name:
            st.session_state[f"{key_prefix}_company_name"] = company_name
            st.session_state[f"{key_prefix}_search_results"] = search_for_ticker(company_name)
            st.session_state[f"{key_prefix}_selected_symbol"] = None
    if st.session_state[f"{key_prefix}_search_results"]:
        selected_result = st.sidebar.selectbox(f"Select {config['asset_name']}", st.session_state[f"{key_prefix}_search_results"], key=f"{key_prefix}_select_symbol")
        if selected_result:
            st.session_state[f"{key_prefix}_selected_symbol"] = selected_result.split(" - ")[0]
            with st.sidebar.form(f"{key_prefix}_add_details_form"):
                symbol = st.session_state[f"{key_prefix}_selected_symbol"]
                st.write(f"Selected: **{symbol}**")
                current_price = fetch_current_price(symbol)
                currency = "₹" if ".NS" in symbol else "$"
                if current_price:
                    st.info(f"Current Price: {currency}{current_price:,.2f}")
                else:
                    st.warning("Could not fetch current price.")
                buy_price = st.number_input(f"Buy Price ({currency})", min_value=0.01, format="%.2f", key=f"{key_prefix}_buy_price")
                buy_date = st.date_input("Buy Date", max_value=datetime.date.today(), key=f"{key_prefix}_buy_date")
                quantity = st.number_input("Quantity", min_value=1, step=1, key=f"{key_prefix}_buy_quantity")
                if is_trading_section:
                    target_price = st.number_input("Target Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_target_price")
                    stop_loss_price = st.number_input("Stop Loss Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_stop_loss_price")
                add_button = st.form_submit_button(f"Add to {config['asset_name_plural']}")
                if add_button:
                    if not (buy_price > 0 and quantity > 0):
                        st.error("Buy Price and Quantity must be positive.")
                    elif update_stock_data(symbol):
                        c.execute(f"SELECT * FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol,))
                        existing = c.fetchone()
                        if existing:
                            old_buy_price, old_quantity = existing[1], existing[3]
                            new_quantity = old_quantity + quantity
                            new_avg_price = ((old_buy_price * old_quantity) + (buy_price * quantity)) / new_quantity
                            if is_trading_section:
                                c.execute(f"UPDATE {config['asset_table']} SET buy_price=?, quantity=?, target_price=?, stop_loss_price=? WHERE {config['asset_col']}=?", (new_avg_price, new_quantity, target_price, stop_loss_price, symbol))
                            else:
                                c.execute(f"UPDATE {config['asset_table']} SET buy_price=?, quantity=? WHERE {config['asset_col']}=?", (new_avg_price, new_quantity, symbol))
                            st.success(f"Updated {symbol}. New quantity: {new_quantity}, New avg. price: {currency}{new_avg_price:,.2f}")
                        else:
                            if is_trading_section:
                                c.execute(f"INSERT INTO {config['asset_table']} ({config['asset_col']}, buy_price, buy_date, quantity, target_price, stop_loss_price) VALUES (?, ?, ?, ?, ?, ?)", (symbol, buy_price, buy_date.strftime("%Y-%m-%d"), quantity, target_price, stop_loss_price))
                            else:
                                c.execute(f"INSERT INTO {config['asset_table']} ({config['asset_col']}, buy_price, buy_date, quantity) VALUES (?, ?, ?, ?)", (symbol, buy_price, buy_date.strftime("%Y-%m-%d"), quantity))
                            st.success(f"{symbol} added successfully!")
                        DB_CONN.commit()
                        for key in st.session_state:
                            if key.startswith(key_prefix):
                                del st.session_state[key]
                        st.rerun()
                    else:
                        st.error(f"Failed to fetch historical data for {symbol}. Cannot add.")
    st.sidebar.header(f"Sell {config['asset_name']}")
    all_symbols = pd.read_sql(f"SELECT {config['asset_col']} FROM {config['asset_table']}", DB_CONN)[config['asset_col']].tolist()
    if all_symbols:
        symbol_to_sell = st.sidebar.selectbox(f"Select {config['asset_name']} to Sell", options=[""] + all_symbols, key=f"{key_prefix}_sell_symbol_selector")
        available_qty = 1
        if symbol_to_sell:
            c.execute(f"SELECT quantity FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol_to_sell,))
            result = c.fetchone()
            if result:
                available_qty = result[0]
                st.sidebar.info(f"Available to sell: {available_qty} units of {symbol_to_sell}")
        with st.sidebar.form(f"{key_prefix}_sell_form"):
            is_disabled = not symbol_to_sell
            sell_qty = st.number_input("Quantity to Sell", min_value=1, max_value=available_qty, step=1, key=f"{key_prefix}_sell_qty", disabled=is_disabled)
            sell_price = st.number_input("Sell Price", min_value=0.01, format="%.2f", key=f"{key_prefix}_sell_price", disabled=is_disabled)
            sell_date = st.date_input("Sell Date", max_value=datetime.date.today(), key=f"{key_prefix}_sell_date", disabled=is_disabled)
            sell_button = st.form_submit_button(f"Sell {config['asset_name']}")
            if sell_button:
                if not symbol_to_sell:
                    st.warning(f"Please select a {config['asset_name']} to sell.")
                elif sell_price <= 0:
                    st.error("Sell price must be greater than zero.")
                else:
                    if is_trading_section:
                        c.execute(f"SELECT buy_price, buy_date, quantity, target_price, stop_loss_price FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol_to_sell,))
                        buy_price, buy_date, current_qty, target_price, stop_loss_price = c.fetchone()
                    else:
                        c.execute(f"SELECT buy_price, buy_date, quantity FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol_to_sell,))
                        buy_price, buy_date, current_qty = c.fetchone()
                    realized_return = ((sell_price - buy_price) / buy_price * 100)
                    transaction_id = str(uuid.uuid4())
                    if is_trading_section:
                        c.execute(f"INSERT INTO {config['realized_table']} (transaction_id, {config['asset_col']}, buy_price, buy_date, quantity, sell_price, sell_date, realized_return_pct, target_price, stop_loss_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (transaction_id, symbol_to_sell, buy_price, buy_date, sell_qty, sell_price, sell_date.strftime("%Y-%m-%d"), realized_return, target_price, stop_loss_price))
                    else:
                        c.execute(f"INSERT INTO {config['realized_table']} (transaction_id, {config['asset_col']}, buy_price, buy_date, quantity, sell_price, sell_date, realized_return_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (transaction_id, symbol_to_sell, buy_price, buy_date, sell_qty, sell_price, sell_date.strftime("%Y-%m-%d"), realized_return))
                    if sell_qty == current_qty:
                        c.execute(f"DELETE FROM {config['asset_table']} WHERE {config['asset_col']}=?", (symbol_to_sell,))
                    else:
                        c.execute(f"UPDATE {config['asset_table']} SET quantity=? WHERE {config['asset_col']}=?", (current_qty - sell_qty, symbol_to_sell))
                    DB_CONN.commit()
                    st.success(f"Sold {sell_qty} units of {symbol_to_sell}.")
                    st.rerun()
    else:
        st.sidebar.info(f"No open {config['asset_name_plural'].lower()}.")
    st.header("Overview")
    if not is_trading_section:
        holdings_summary_df = get_holdings_df(config['asset_table'])
        if not holdings_summary_df.empty:
            total_invested, total_current = holdings_summary_df['invested_value'].sum(), holdings_summary_df['current_value'].sum()
            total_return_amount = total_current - total_invested
            total_return_percent = (total_return_amount / total_invested) * 100 if total_invested > 0 else 0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Investment", f"₹{total_invested:,.2f}")
            with col2:
                st.metric("Current Value", f"₹{total_current:,.2f}")
            with col3:
                st.metric("Total Return", f"₹{total_return_amount:,.2f}", f"{total_return_percent:.2f}%")
            st.divider()
    if st.button("Refresh Live Data", key=f"{key_prefix}_refresh_data"):
        with st.spinner("Fetching latest prices..."):
            for symbol in all_symbols:
                update_stock_data(symbol)
        st.success("Data refreshed!")
        st.rerun()
    view_options = [f"Current {config['asset_name_plural']}", f"Realized {config['asset_name_plural']}"]
    table_view = st.selectbox("View", view_options, key=f"{key_prefix}_table_view")
    df = get_holdings_df(config['asset_table']) if table_view == view_options[0] else get_realized_df(config['realized_table'])
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data to display.")
    if table_view == view_options[1]:
        st.header(f"Realized {config['asset_name']} Analysis")
        if not df.empty:
            asset_col = config['asset_col']
            chart_df = df.copy()
            chart_df['trade_label'] = chart_df[asset_col] + '-' + chart_df['transaction_id'].str[:8]
            chart_df['color'] = chart_df['realized_return_pct'].apply(lambda x: 'Profit' if x >= 0 else 'Loss')
            def create_bar_label(row):
                currency = "₹" if ".NS" in row[asset_col] else "$"
                return f"{row['realized_return_pct']:.1f}% ({currency}{row['realized_profit_loss']:.2f})"
            chart_df['bar_top_label'] = chart_df.apply(create_bar_label, axis=1)
            bars = alt.Chart(chart_df).mark_bar().encode(x=alt.X('trade_label:N', title='Individual Sale', sort=None), y=alt.Y('realized_return_pct:Q', title='Return %'), color=alt.Color('color:N', scale=alt.Scale(domain=['Profit', 'Loss'], range=['#2ca02c', '#d62728']), legend=None), tooltip=[asset_col, 'buy_date', 'sell_date', 'realized_return_pct', 'realized_profit_loss'])
            text = bars.mark_text(align='center', baseline='bottom', dy=-5).encode(text='bar_top_label:N')
            st.altair_chart((bars + text).interactive(), use_container_width=True)
    else:
        st.header("Individual Performance Chart")
        if not df.empty:
            all_symbols_list = df["symbol"].tolist()
            selected_symbols = st.multiselect("Select assets for chart", all_symbols_list, default=all_symbols_list, key=f"{key_prefix}_perf_symbols")
            if selected_symbols:
                chart_data = []
                for symbol in selected_symbols:
                    asset_info = df.loc[df["symbol"] == symbol].iloc[0]
                    history_df = pd.read_sql("SELECT date, close_price FROM price_history WHERE ticker=? AND date>=? ORDER BY date ASC", DB_CONN, params=(symbol, asset_info["buy_date"]))
                    if not history_df.empty:
                        history_df["return_%"] = (history_df["close_price"] - asset_info["buy_price"]) / asset_info["buy_price"] * 100
                        history_df["symbol"] = symbol
                        chart_data.append(history_df)
                if chart_data:
                    full_chart_df = pd.concat(chart_data)
                    full_chart_df["date"] = pd.to_datetime(full_chart_df["date"])
                    chart = alt.Chart(full_chart_df).mark_line().encode(x=alt.X('date:T', title='Date'), y=alt.Y('return_%:Q', title='Return %'), color='symbol:N', tooltip=['symbol', 'date', 'return_%']).interactive()
                    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="gray", strokeDash=[3,3]).encode(y='y')
                    st.altair_chart(chart + zero_line, use_container_width=True)

        if not is_trading_section and not df.empty:
            st.divider()
            benchmark_choice = st.selectbox("Select Benchmark", ['Nifty 50', 'Nifty 500'], key=f"{key_prefix}_benchmark_choice")
            st.header(f"Portfolio vs. {benchmark_choice} Comparison")
            with st.spinner("Loading Benchmark Data..."):
                benchmark_data = get_benchmark_comparison_data(df, benchmark_choice)
                if not benchmark_data.empty:
                    benchmark_chart = alt.Chart(benchmark_data).mark_line().encode(x=alt.X('Date:T', title='Date'), y=alt.Y('Return %:Q', title='Total Return %'), color=alt.Color('Type:N', title='Legend')).interactive()
                    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="gray", strokeDash=[3,3]).encode(y='y')
                    st.altair_chart(benchmark_chart + zero_line, use_container_width=True)
                else:
                    st.warning("Could not generate benchmark data. Ensure you have at least one investment.")

def funds_page():
    st.title("💰 Funds Management")
    c = DB_CONN.cursor()
    st.sidebar.header("Add Transaction")
    with st.sidebar.form("deposit_form", clear_on_submit=True):
        st.subheader("Add Deposit")
        deposit_date = st.date_input("Date", max_value=datetime.date.today(), key="deposit_date")
        deposit_amount = st.number_input("Amount", min_value=0.01, format="%.2f", key="deposit_amount")
        deposit_desc = st.text_input("Description", placeholder="e.g., Salary", key="deposit_desc")
        if st.form_submit_button("Add Deposit"):
            if deposit_amount > 0:
                c.execute("INSERT INTO fund_transactions (transaction_id, date, type, amount, description) VALUES (?, ?, ?, ?, ?)", (str(uuid.uuid4()), deposit_date.strftime("%Y-%m-%d"), "Deposit", deposit_amount, deposit_desc))
                DB_CONN.commit()
                st.success("Deposit recorded!")
            else:
                st.warning("Deposit amount must be greater than zero.")
    with st.sidebar.form("withdrawal_form", clear_on_submit=True):
        st.subheader("Record Withdrawal")
        wd_date = st.date_input("Date", max_value=datetime.date.today(), key="wd_date")
        wd_amount = st.number_input("Amount", min_value=0.01, format="%.2f", key="wd_amount")
        wd_desc = st.text_input("Description", placeholder="e.g., Personal Use", key="wd_desc")
        if st.form_submit_button("Record Withdrawal"):
            if wd_amount > 0:
                c.execute("INSERT INTO fund_transactions (transaction_id, date, type, amount, description) VALUES (?, ?, ?, ?, ?)", (str(uuid.uuid4()), wd_date.strftime("%Y-%m-%d"), "Withdrawal", wd_amount, wd_desc))
                DB_CONN.commit()
                st.success("Withdrawal recorded!")
            else:
                st.warning("Withdrawal amount must be greater than zero.")
    fund_df = pd.read_sql("SELECT * FROM fund_transactions ORDER BY date DESC", DB_CONN)
    total_deposits, total_withdrawals = fund_df.loc[fund_df['type'] == 'Deposit', 'amount'].sum(), fund_df.loc[fund_df['type'] == 'Withdrawal', 'amount'].sum()
    available_capital = total_deposits - total_withdrawals
    st.header("Capital Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Deposits", f"₹{total_deposits:,.2f}")
    col2.metric("Total Withdrawals", f"₹{total_withdrawals:,.2f}")
    col3.metric("Available Capital", f"₹{available_capital:,.2f}")
    st.divider()
    st.header("Transaction History")
    st.dataframe(fund_df, use_container_width=True)

def expense_tracker_page():
    st.title("💸 Expense Tracker")
    _process_recurring_expenses()
    c = DB_CONN.cursor()
    CATEGORIES = sorted(["Food", "Transport", "Rent", "Utilities", "Shopping", "Entertainment", "Health", "Groceries", "Bills", "Education", "Travel", "Other"])
    PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]
    st.sidebar.header("Add Expense")
    with st.sidebar.form("new_expense_form", clear_on_submit=True):
        exp_date = st.date_input("Date", max_value=datetime.date.today())
        exp_amount = st.number_input("Amount", min_value=0.01, format="%.2f")
        exp_cat = st.selectbox("Category", CATEGORIES)
        exp_pm = st.selectbox("Payment Method", PAYMENT_METHODS)
        exp_desc = st.text_input("Description")
        if st.form_submit_button("Add Expense"):
            c.execute("INSERT INTO expenses (expense_id, date, amount, category, payment_method, description) VALUES (?, ?, ?, ?, ?, ?)", (str(uuid.uuid4()), exp_date.strftime("%Y-%m-%d"), exp_amount, exp_cat, exp_pm, exp_desc))
            DB_CONN.commit()
            st.success("Expense added!")
            st.rerun()
    st.sidebar.header("Select View")
    view = st.sidebar.radio("Navigation", ["Dashboard", "Transaction History", "Manage Budgets", "Manage Recurring"])
    if view == "Dashboard":
        st.header("Monthly Dashboard")
        month_year = datetime.date.today().strftime("%Y-%m")
        expenses_df = pd.read_sql(f"SELECT * FROM expenses WHERE date LIKE '{month_year}-%'", DB_CONN)
        budgets_df = pd.read_sql(f"SELECT category, amount FROM budgets WHERE month_year = '{month_year}'", DB_CONN).set_index('category')
        total_spent, total_budget = expenses_df['amount'].sum(), budgets_df['amount'].sum()
        budget_remaining = total_budget - total_spent
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spent this Month", f"₹{total_spent:,.2f}")
        col2.metric("Total Budget for Month", f"₹{total_budget:,.2f}")
        col3.metric("Budget Remaining", f"₹{budget_remaining:,.2f}", delta_color="inverse" if budget_remaining >= 0 else "normal")
        st.divider()
        if not expenses_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Spending by Category")
                category_spending = expenses_df.groupby('category')['amount'].sum().reset_index()
                pie_chart = alt.Chart(category_spending).mark_arc(innerRadius=50).encode(theta=alt.Theta(field="amount", type="quantitative"), color=alt.Color(field="category", type="nominal", title="Category"), tooltip=['category', 'amount']).properties(height=350)
                st.altair_chart(pie_chart, use_container_width=True)
            with col2:
                st.subheader("Budget vs. Actual Spending")
                spending_by_cat = expenses_df.groupby('category')['amount'].sum().rename('actual')
                budget_analysis_df = pd.concat([budgets_df, spending_by_cat], axis=1).fillna(0).rename(columns={'amount':'budget'}).reset_index().melt(id_vars='category', value_vars=['budget', 'actual'], var_name='Type', value_name='Amount')
                bar_chart = alt.Chart(budget_analysis_df).mark_bar(opacity=0.8).encode(x=alt.X('Amount:Q', title='Amount (₹)'), y=alt.Y('category:N', title='Category', sort='-x'), color='Type:N', xOffset='Type:N').properties(height=350)
                st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No expenses logged for this month to display charts.")
    elif view == "Transaction History":
        st.header("Transaction History")
        all_expenses_df = pd.read_sql("SELECT * FROM expenses ORDER BY date DESC", DB_CONN)
        st.dataframe(all_expenses_df, use_container_width=True)
    elif view == "Manage Budgets":
        st.header("Set Your Monthly Budgets")
        budget_month_str = datetime.date.today().strftime("%Y-%m")
        st.info(f"You are setting the budget for: **{datetime.datetime.strptime(budget_month_str, '%Y-%m').strftime('%B %Y')}**")
        existing_budgets = pd.read_sql(f"SELECT category, amount FROM budgets WHERE month_year = '{budget_month_str}'", DB_CONN)
        budget_df = pd.DataFrame({'category': CATEGORIES, 'amount': 0.0})
        if not existing_budgets.empty:
            budget_df = budget_df.set_index('category')
            budget_df.update(existing_budgets.set_index('category'))
            budget_df = budget_df.reset_index()
        edited_budgets = st.data_editor(budget_df, num_rows="dynamic", use_container_width=True)
        if st.button("Save Budgets"):
            for _, row in edited_budgets.iterrows():
                if row['amount'] >= 0 and row['category']:
                    c.execute("INSERT OR REPLACE INTO budgets (month_year, category, amount) VALUES (?, ?, ?)", (budget_month_str, row['category'], row['amount']))
            DB_CONN.commit()
            st.success("Budgets saved!")
            st.rerun()
    elif view == "Manage Recurring":
        st.header("Manage Recurring Expenses")
        st.info("Set up expenses that occur every month (e.g., rent, subscriptions). They will be logged automatically.")
        recurring_df = pd.read_sql("SELECT recurring_id, description, amount, category, payment_method, day_of_month FROM recurring_expenses", DB_CONN)
        edited_recurring = st.data_editor(recurring_df, num_rows="dynamic", use_container_width=True, column_config={"category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True), "payment_method": st.column_config.SelectboxColumn("Payment Method", options=PAYMENT_METHODS, required=True), "day_of_month": st.column_config.NumberColumn("Day of Month (1-31)", min_value=1, max_value=31, step=1, required=True)})
        if st.button("Save Recurring Rules"):
            c.execute("DELETE FROM recurring_expenses")
            for _, row in edited_recurring.iterrows():
                if row['description'] and row['amount'] > 0:
                    c.execute("INSERT INTO recurring_expenses (description, amount, category, payment_method, day_of_month) VALUES (?, ?, ?, ?, ?)", (row['description'], row['amount'], row['category'], row['payment_method'], row['day_of_month']))
            DB_CONN.commit()
            st.success("Recurring expense rules saved!")
            st.rerun()

def mutual_fund_page():
    st.title("📚 Mutual Fund Tracker")
    c = DB_CONN.cursor()
    _process_mf_sips()

    st.sidebar.header("Add Transaction")
    with st.sidebar.form("mf_transaction_form", clear_on_submit=True):
        st.subheader("Log a Transaction")
        mf_date = st.date_input("Date", max_value=datetime.date.today())
        mf_scheme = st.text_input("Scheme Name (e.g., Axis Bluechip Fund)")
        mf_symbol = st.text_input("Yahoo Finance Symbol (e.g., 0P0000XWJ5.BO)")
        mf_type = st.selectbox("Type", ["Purchase", "Redemption"])
        mf_units = st.number_input("Units", min_value=0.001, format="%.4f")
        mf_nav = st.number_input("NAV (Net Asset Value)", min_value=0.01, format="%.4f")
        if st.form_submit_button("Add Transaction"):
            if mf_scheme and mf_symbol and mf_units > 0 and mf_nav > 0:
                c.execute("INSERT INTO mf_transactions (transaction_id, date, scheme_name, yfinance_symbol, type, units, nav) VALUES (?, ?, ?, ?, ?, ?, ?)", (str(uuid.uuid4()), mf_date.strftime('%Y-%m-%d'), mf_scheme, mf_symbol, mf_type, mf_units, mf_nav))
                DB_CONN.commit()
                st.success(f"{mf_type} of {mf_scheme} logged!")
                st.rerun()
            else:
                st.warning("Please fill all fields.")

    view = st.radio("Select View", ["Dashboard", "All Transactions", "Manage SIPs"], horizontal=True)

    if view == "Dashboard":
        st.header("Portfolio Dashboard")
        transactions_df = pd.read_sql("SELECT * FROM mf_transactions", DB_CONN)
        if transactions_df.empty:
            st.info("No mutual fund transactions logged yet. Add one from the sidebar.")
        else:
            holdings = []
            unique_schemes = transactions_df['scheme_name'].unique()
            symbols = transactions_df['yfinance_symbol'].unique()
            latest_navs = {sym: fetch_current_price(sym) for sym in symbols}
            for scheme in unique_schemes:
                scheme_tx = transactions_df[transactions_df['scheme_name'] == scheme]
                purchases = scheme_tx[scheme_tx['type'] == 'Purchase']
                redemptions = scheme_tx[scheme_tx['type'] == 'Redemption']
                total_units = purchases['units'].sum() - redemptions['units'].sum()
                if total_units > 0.001:
                    total_investment = (purchases['units'] * purchases['nav']).sum() - (redemptions['units'] * redemptions['nav']).sum()
                    avg_nav = total_investment / total_units if total_units > 0 else 0
                    symbol = scheme_tx['yfinance_symbol'].iloc[0]
                    latest_nav = latest_navs.get(symbol) or 0
                    current_value = total_units * latest_nav
                    pnl = current_value - total_investment
                    pnl_pct = (pnl / total_investment) * 100 if total_investment > 0 else 0
                    xirr = _calculate_xirr(scheme_tx, latest_nav)
                    holdings.append({"Scheme": scheme, "Units": total_units, "Avg NAV": avg_nav, "Latest NAV": latest_nav, "Investment": total_investment, "Current Value": current_value, "P&L": pnl, "P&L %": pnl_pct, "XIRR %": xirr})

            if holdings:
                holdings_df = pd.DataFrame(holdings)
                total_mf_investment = holdings_df['Investment'].sum()
                total_mf_current_value = holdings_df['Current Value'].sum()
                total_mf_pnl = total_mf_current_value - total_mf_investment
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Investment", f"₹{total_mf_investment:,.2f}")
                col2.metric("Current Value", f"₹{total_mf_current_value:,.2f}")
                col3.metric("Overall P&L", f"₹{total_mf_pnl:,.2f}")
                st.divider()
                st.subheader("Holdings Summary")
                st.dataframe(holdings_df, use_container_width=True, column_config={"XIRR %": st.column_config.ProgressColumn("XIRR %", format="%.2f%%", min_value=-25, max_value=50)})
                st.subheader("Portfolio Allocation")
                pie_chart = alt.Chart(holdings_df).mark_arc(innerRadius=50).encode(theta=alt.Theta(field="Current Value", type="quantitative"), color=alt.Color(field="Scheme", type="nominal"), tooltip=['Scheme', 'Current Value']).properties(height=400)
                st.altair_chart(pie_chart, use_container_width=True)
            else:
                st.info("You have no current mutual fund holdings.")

    elif view == "All Transactions":
        st.header("All Transactions")
        transactions_df = pd.read_sql("SELECT * FROM mf_transactions ORDER BY date DESC", DB_CONN)
        st.dataframe(transactions_df, use_container_width=True)

    elif view == "Manage SIPs":
        st.header("Manage Your SIPs")
        st.info("Set up your monthly SIPs. They will be auto-logged when you open this page after the SIP date.")
        sips_df = pd.read_sql("SELECT sip_id, scheme_name, yfinance_symbol, amount, day_of_month FROM mf_sips", DB_CONN)
        edited_sips = st.data_editor(sips_df, num_rows="dynamic", use_container_width=True, column_config={"day_of_month": st.column_config.NumberColumn("Day of Month (1-31)", min_value=1, max_value=31, step=1, required=True)})
        if st.button("Save SIP Rules"):
            c.execute("DELETE FROM mf_sips")
            for _, row in edited_sips.iterrows():
                if row['scheme_name'] and row['yfinance_symbol'] and row['amount'] > 0:
                    c.execute("INSERT INTO mf_sips (scheme_name, yfinance_symbol, amount, day_of_month) VALUES (?, ?, ?, ?)", (row['scheme_name'], row['yfinance_symbol'], row['amount'], row['day_of_month']))
            DB_CONN.commit()
            st.success("SIP rules saved!")
            st.rerun()

# --- HOME PAGE & ROUTING ---
if "page" not in st.session_state:
    st.session_state.page = "home"

def set_page(page):
    st.session_state.page = page

def home_page():
    st.title("Finance Dashboard")
    st.write("Select a section to manage your finances:")

    st.button("📈 Investment", use_container_width=True, on_click=set_page, args=("investment",))
    st.button("📊 Trading", use_container_width=True, on_click=set_page, args=("trading",))
    st.button("💰 Funds", use_container_width=True, on_click=set_page, args=("funds",))
    st.button("💸 Expense Tracker", use_container_width=True, on_click=set_page, args=("expense_tracker",))
    st.button("📚 Mutual Fund", use_container_width=True, on_click=set_page, args=("mutual_fund",))

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
if st.sidebar.button("Clear Session State", type="secondary"):
    current_page = st.session_state.get("page", "home")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.page = current_page
    st.rerun()
