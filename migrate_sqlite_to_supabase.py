import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text

# -----------------------------
# 1. PATH TO YOUR SQLITE DB
# -----------------------------
SQLITE_DB_PATH = "/Users/harshareddy/Documents/Investment/finance.db"

# -----------------------------
# 2. SUPABASE DATABASE URL
# -----------------------------
SUPABASE_URL = (
    "postgresql://postgres:hr2zbZP5qrkI0uNq@db.jfozmxvuswolewlzkotm.supabase.co:6543/postgres"
)

# -------------------------------------------
# TABLES TO MIGRATE (Order matters due to FK)
# -------------------------------------------
TABLES = [
    "portfolio",
    "trades",
    "exits",
    "realized_stocks",
    "price_history",
    "fund_transactions",
    "mf_transactions",
    "mf_sips",
    "expenses",
    "budgets",
    "recurring_expenses",
    "benchmark_history",
]

# ------------------------------------------------
# STEP 1 — Connect to SQLite + Supabase Postgres
# ------------------------------------------------
sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
postgres_engine = create_engine(SUPABASE_URL)

print("\n🚀 Starting Migration From SQLite → Supabase…\n")

# ------------------------------------------------
# STEP 2 — Loop through tables & migrate
# ------------------------------------------------
for table in TABLES:
    print(f"➡ Migrating table: **{table}**")

    # Load from SQLite
    df = pd.read_sql(f"SELECT * FROM {table}", sqlite_conn)

    if df.empty:
        print(f"   ⚠️  Table `{table}` is empty, skipping.")
        continue

    # Convert SQLite datetime/text objects
    df = df.applymap(lambda x: None if x == "None" else x)

    # Upload to Supabase
    try:
        df.to_sql(
            table,
            postgres_engine,
            if_exists="append",   # append data into existing tables
            index=False
        )
        print(f"   ✅ Inserted {len(df)} rows.")
    except Exception as e:
        print(f"   ❌ Failed to insert into `{table}`: {str(e)}")

print("\n🎉 Migration Completed Successfully!\n")
