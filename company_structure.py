import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 1. CONFIG & ASSUMPTIONS (The "Digital Twin")
# ==========================================

st.set_page_config(page_title="Coffee Strategic Model", layout="wide")
st.title("☕ Coffee Business & Restructuring Engine")

# --- SIDEBAR: OPERATIONAL LEVERS ---
st.sidebar.header("⚙️ Operational Levers")

# 1. Growth Assumptions
growth_rate = st.sidebar.slider("Monthly Revenue Growth (%)", 0.0, 10.0, 2.0) / 100

# 2. Founder Farm Logic (The "Transfer Price")
st.sidebar.subheader("🇵🇦 Founder Farm Pricing")
transfer_method = st.sidebar.radio(
    "How do we pay Edgar for Green Coffee?",
    ["Market Rate ($8.00/lb)", "Cost Plus ($3.50 + 15%)"]
)
green_cost_per_lb = 8.00 if transfer_method == "Market Rate ($8.00/lb)" else 4.02

# 3. Inventory Constraints
total_panama_inventory = st.sidebar.number_input("Total Panama Inventory (Lbs)", value=5000)

# --- SIDEBAR: DEAL STRUCTURE ---
st.sidebar.header("🤝 Deal Structure")

# 4. Edgar's Salary
edgar_salary_annual = st.sidebar.number_input("Edgar's Annual Salary ($)", value=80000, step=5000)
monthly_salary = edgar_salary_annual / 12

# 5. Capital Payback Targets
target_payback = st.sidebar.number_input("Target Repayment (1.5x of $350k)", value=525000)
current_repaid = 0 # Starting from 0

# ==========================================
# 2. LOGIC ENGINE
# ==========================================

def run_projection(months=36):
    data = []
    
    # State Variables
    cumulative_repaid = current_repaid
    inventory_balance = total_panama_inventory
    is_flipped = False
    
    # Base Financials (Current State)
    base_monthly_rev = 50000
    base_cogs_pct = 0.25 # Standard cogs without green coffee variance
    
    for m in range(1, months + 1):
        # A. Growth Logic
        revenue = base_monthly_rev * ((1 + growth_rate) ** m)
        
        # B. COGS Logic (Including Transfer Price)
        # Assume 1lb roasted coffee generates $25 revenue (blended)
        # Shrinkage 18%
        lbs_sold_roasted = revenue / 25
        lbs_green_needed = lbs_sold_roasted / (1 - 0.18)
        
        # Inventory Check
        inventory_balance -= lbs_green_needed
        stock_status = "OK" if inventory_balance > 0 else "STOCKOUT"
        
        # Cost Calculation
        green_cost = lbs_green_needed * green_cost_per_lb
        other_cogs = revenue * 0.15 # Packaging, milk, etc.
        gross_profit = revenue - (green_cost + other_cogs)
        
        # C. Step Cost Logic (Labor)
        # Base labor $15k/mo. Jumps by $4k every $20k in added revenue
        base_labor = 15000
        step_labor = np.floor((revenue - 50000) / 20000) * 4000
        if step_labor < 0: step_labor = 0
        total_labor = base_labor + step_labor
        
        # D. OpEx & Salary
        rent_overhead = 8000
        # Edgar's salary reduces profit available for repayment
        net_profit_pre_tax = gross_profit - (total_labor + rent_overhead + monthly_salary)
        
        # E. THE WATERFALL (Distributions)
        # Ian gets 12% off the top? Let's assume yes based on new structure
        ian_share = max(0, net_profit_pre_tax * 0.12)
        distributable_cash = max(0, net_profit_pre_tax - ian_share)
        
        # The Flip Logic
        # Phase 1: Priority Payback (75% to You/Austin, 25% to Edgar)
        # Phase 2: Equity Split (50% You/Austin, 50% Edgar)
        
        if cumulative_repaid < target_payback:
            phase = "Phase 1: Priority Payback"
            you_austin_share = distributable_cash * 0.75
            edgar_share = distributable_cash * 0.25
            
            # Track Repayment
            cumulative_repaid += you_austin_share
        else:
            phase = "Phase 2: Equity Split"
            is_flipped = True
            you_austin_share = distributable_cash * 0.50
            edgar_share = distributable_cash * 0.50
            
        data.append({
            "Month": m,
            "Revenue": revenue,
            "Net_Profit": net_profit_pre_tax,
            "Inventory_Lbs": inventory_balance,
            "You_Austin_Payout": you_austin_share,
            "Edgar_Payout": edgar_share,
            "Ian_Payout": ian_share,
            "Cumulative_Repaid": cumulative_repaid,
            "Phase": phase,
            "Stock_Status": stock_status
        })
        
    return pd.DataFrame(data)

# Run the model
df = run_projection()

# ==========================================
# 3. DASHBOARD VISUALIZATION
# ==========================================

# METRICS ROW
col1, col2, col3 = st.columns(3)
flip_month = df[df['Phase'] == "Phase 2: Equity Split"].head(1)
if not flip_month.empty:
    flip_date = f"Month {flip_month['Month'].values[0]}"
else:
    flip_date = "Not reached in 3 years"

col1.metric("Projected Monthly Profit (Yr 1)", f"${df['Net_Profit'].iloc[0]:,.0f}")
col2.metric("Projected Payback Date", flip_date)
col3.metric("Inventory Run-out Date", f"Month {df[df['Inventory_Lbs'] < 0]['Month'].min()}")

# CHART 1: THE PAYOUT STACK (Who gets paid what?)
st.subheader("1. Partner Payout Simulation")
st.markdown("Use this to show Edgar how his **Salary** delays the **Flip**.")

chart_data = df.melt(id_vars=['Month'], value_vars=['You_Austin_Payout', 'Edgar_Payout', 'Ian_Payout'], var_name='Partner', value_name='Cash')

c = alt.Chart(chart_data).mark_bar().encode(
    x='Month',
    y=alt.Y('Cash', stack='zero'),
    color=alt.Color('Partner', scale=alt.Scale(domain=['You_Austin_Payout', 'Edgar_Payout', 'Ian_Payout'], range=['#4c78a8', '#f58518', '#54a24b'])),
    tooltip=['Month', 'Partner', 'Cash']
).interactive()
st.altair_chart(c, use_container_width=True)

# CHART 2: REPAYMENT PROGRESS VS TARGET
st.subheader("2. Capital Repayment Progress")
st.markdown("We receive 75% of cash flow until we hit **$525k** (The Red Line).")

base = alt.Chart(df).encode(x='Month')
line = base.mark_line(strokeWidth=3).encode(y='Cumulative_Repaid')
rule = base.mark_rule(color='red').encode(y=alt.value(target_payback)) # Fixed Target Line won't render correctly with alt.value in some versions, simpler:
target_line = alt.Chart(pd.DataFrame({'y': [target_payback]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')

st.altair_chart((line + target_line).interactive(), use_container_width=True)

# CHART 3: INVENTORY BURN DOWN
st.subheader("3. Panama Inventory Burn-down")
st.markdown("When do we run out of the Founder's Lot?")

inv_chart = alt.Chart(df).mark_line(color='green').encode(
    x='Month',
    y='Inventory_Lbs',
    tooltip=['Month', 'Inventory_Lbs']
)
zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red').encode(y='y')

st.altair_chart((inv_chart + zero_line), use_container_width=True)