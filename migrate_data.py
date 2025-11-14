import streamlit as st
import pandas as pd
import sqlite3
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import sys
import os

# --- 1. Configuration and Setup ---

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:root:%(message)s")

# Tables to migrate (based on your Streamlit app's schema)
MIGRATION_TABLES = [
    "portfolio",
    "trades",
    "price_history",
    "realized_stocks",
    "exits",
    "fund_transactions",
    "expenses",
    "budgets",
    "recurring_expenses",
    "mf_transactions",
    "mf_sips",
]

LOCAL_DB_PATH = "/Users/harshareddy/Documents/Investment/FIN-Tracker/finance.db"

# --- 2. Database Connection Logic (Copied/Adapted from your app) ---

def get_neon_engine():
    """Discovers the Neon SQLAlchemy URL from Streamlit secrets and returns the Engine."""
    # Discover connection URL candidates
    candidates = [
        ("neon_db", "sqlalchemy_url"),
        ("database", "sqlalchemy_url"),
        ("supabase_db", "sqlalchemy_url"),
    ]
    sa_url = None
    for key, sub in candidates:
        # NOTE: We assume st.secrets is accessible or secrets are mocked/environment variables are used
        sa_url = st.secrets.get(key, {}).get(sub)
        if sa_url:
            logging.info(f"Using SQLAlchemy URL from st.secrets['{key}']['{sub}']")
            break
    if not sa_url:
        sa_url = st.secrets.get("DATABASE_URL")
        if sa_url:
            logging.info("Using SQLAlchemy URL from st.secrets['DATABASE_URL']")

    if not sa_url:
        print("ERROR: No SQLAlchemy URL found in Streamlit secrets.")
        print("Please ensure your Neon/Postgres URL is set in .streamlit/secrets.toml.")
        sys.exit(1)

    try:
        engine = create_engine(sa_url, pool_pre_ping=True)
        # Quick connectivity check
        with engine.connect():
             pass
        logging.info("Successfully created SQLAlchemy engine and verified connectivity.")
        return engine
    except SQLAlchemyError as e:
        logging.error(f"Failed to create SQLAlchemy engine or connect: {e}")
        print(f"ERROR: Database connection failed. Check your credentials/URL. {e}")
        sys.exit(1)

# --- 3. Main Migration Function ---

def migrate_sqlite_to_neon():
    """Reads all data from local SQLite and writes it to Neon/PostgreSQL."""
    print(f"--- Starting Data Migration to Neon ---")

    if not os.path.exists(LOCAL_DB_PATH):
        print(f"ERROR: Local SQLite database not found at: {LOCAL_DB_PATH}")
        sys.exit(1)

    # Get the remote engine
    NEON_ENGINE = get_neon_engine()

    # Connect to the local SQLite database
    try:
        sqlite_conn = sqlite3.connect(LOCAL_DB_PATH)
    except Exception as e:
        print(f"ERROR: Failed to connect to local SQLite DB: {e}")
        sys.exit(1)

    migrated_count = 0
    print("\nStarting table migration...")

    for table_name in MIGRATION_TABLES:
        print(f"Processing table: {table_name}...")
        try:
            # 1. Fetch data from the local SQLite table
            local_df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)

            if not local_df.empty:
                # 2. Convert date/datetime columns to string for consistent writing to PostgreSQL
                for col in ['date', 'buy_date', 'sell_date']:
                    if col in local_df.columns:
                         local_df[col] = local_df[col].astype(str)

                # 3. Write data to the remote PostgreSQL table (REPLACE mode is critical here)
                # 'replace' will drop the existing remote table and recreate it with the DataFrame's schema.
                local_df.to_sql(table_name, NEON_ENGINE, if_exists='replace', index=False)
                print(f"  ✅ Migrated {len(local_df)} records to **{table_name}** (Replaced existing data).")
                migrated_count += 1
            else:
                print(f"  ⚠️ Skipped table {table_name}: No data found in local DB.")

        except Exception as e:
            print(f"  ❌ Migration failed for table **{table_name}**: {e}")
            logging.error(f"Migration error for {table_name}: {e}")

    # Cleanup
    sqlite_conn.close()

    if migrated_count > 0:
        print("\n----------------------------------------------------")
        print(f"🎉 SUCCESS! {migrated_count}/{len(MIGRATION_TABLES)} tables successfully migrated to Neon.")
        print("You can now remove this script and run your Streamlit application.")
        print("----------------------------------------------------")
    else:
        print("\n--- WARNING: No data was migrated. ---")

# --- 4. Script Execution ---

if __name__ == "__main__":
    # Load secrets outside of Streamlit context (assumes running from the directory
    # where .streamlit/secrets.toml exists, which is common for local execution).
    try:
        import toml
        with open("/Users/harshareddy/Documents/Investment/FIN-Tracker/.streamlit/secrets.toml", "r") as f:
            st.secrets = toml.load(f)
    except FileNotFoundError:
        print("WARNING: .streamlit/secrets.toml not found. Assuming environment variables/Streamlit run.")
        # If running in a separate environment, secrets must be loaded via ENV variables or passed manually.
        pass

    migrate_sqlite_to_neon()
