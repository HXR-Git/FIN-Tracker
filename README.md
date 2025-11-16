# 💰 FIN-Tracker: Secure Personal Finance Dashboard

FIN-Tracker is a secure, all-in-one financial dashboard built with **Streamlit** and powered by a robust **PostgreSQL (Neon DB)** backend. It provides comprehensive real-time tracking for stocks, mutual funds, trading P&L, budgeting, and overall fund flow.

The application uses a strict **Owner/Viewer access model** and relies entirely on **Streamlit Secrets** for credentials—ensuring zero exposure of sensitive data in the repository.

---

## ✨ Features at a Glance

### 1. 🔐 Secure Access & Role Management
- **Owner Access:** Full read/write access to add, edit, or delete data.
- **Viewer Access:** Read-only view of all dashboards and analytics.
- **Security:** All credentials and API keys stored securely using `st.secrets`.

---

### 2. 💸 Fund Management (Bank Account Monitoring)
- Track all deposits, withdrawals, and fund allocations.
- Calculate **Available Capital**.
- View cumulative fund flow over time.

---

### 3. 📈 Investment & Trading Portfolio
- Real-time P&L, returns, and valuation using *yfinance*.
- Long-term investment tracking with sector & market cap analysis.
- Trading book with Target, Stop Loss, and Risk-Reward Ratio (RRR).
- **Paper Trading Mode** available.

---

### 4. 📊 Mutual Fund Tracking (India)
- Fetch real-time NAVs using `mftool`.
- Track MF performance, average NAV, and XIRR-like returns.

---

### 5. 🧾 Expense Tracker & Budgeting
- Categorized income & expenses.
- Internal transfers (e.g., UPI → Credit Card) without affecting net worth.
- Monthly budget tracking.
- Manage recurring expenses/subscriptions.

---

## 🚀 Setup and Deployment

FIN-Tracker is optimized for deployment on **Streamlit Cloud**.

---

## ✅ Prerequisites
- Python 3.8+
- PostgreSQL / Neon DB
- (Optional) Finnhub API key for enhanced stock search

---

## 🛠️ Local Installation

```bash
git clone git@github.com:your_username/FIN-Tracker.git
cd FIN-Tracker
pip install -r requirements.txt
```

---

## 🔐 Security Configuration (Required)

Create the file:

```
.streamlit/secrets.toml
```

Add the following:

```toml
[api_keys]
finnhub = "YOUR_FINNHUB_API_KEY"

[auth]
username = "YOUR_OWNER_USERNAME"
password = "YOUR_OWNER_PASSWORD"
viewer_username = "YOUR_VIEWER_ACCESS_CODE"
viewer_password = "VIEWER_PASSWORD"

[neon_db]
sqlalchemy_url = "postgresql+psycopg2://user:password@host/database?sslmode=require"
```

---

## ▶️ Run Locally

```bash
streamlit run finance_app.py
```

---

