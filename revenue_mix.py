import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import yaml
from yaml.loader import SafeLoader

# --- AUTH LIBRARIES ---
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

# ==========================================
# 0. AUTHENTICATION SETUP (MODERN SYNTAX)
# ==========================================

# 1. Define Users
names = ['Praveen R.', 'Guest User']
usernames = ['pr', 'guest']
passwords = ['abc1234', 'test']

# 2. Hash Passwords
hashed_passwords = Hasher(passwords).generate()

# 3. Create the 'credentials' dictionary (REQUIRED for new versions)
credentials = {
    "usernames": {
        usernames[0]: {"name": names[0], "password": hashed_passwords[0]},
        usernames[1]: {"name": names[1], "password": hashed_passwords[1]}
    }
}

# 4. Initialize Authenticator using the dictionary
authenticator = stauth.Authenticate(
    credentials,
    'coffee_app_cookie',  # Cookie Name
    'abcdef',             # Key (Signature)
    cookie_expiry_days=30
)

# 5. Render Login Widget
authenticator.login('main')

# Get the status directly from session state
if st.session_state["authentication_status"]:
    
    # --- SUCCESS: APP LOGIC STARTS HERE ---
    authenticator.logout('Logout', 'main')
    st.title(f"Welcome, {st.session_state['name']}!")

    tab1, tab2 = st.tabs(["📊 Scenario Planning", "📅 2025 Lookback"])

    with tab1:
        # ==========================================
        # 1. CONFIGURATION & PAGE SETUP
        # ==========================================
        st.markdown("""
        <style>
            .big-font { font-size: 24px !important; font-weight: bold; }
            .metric-container { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True)

        st.header("☕ Consolidated Coffee Business Engine")
        st.markdown("---")

        # ==========================================
        # 2. SIDEBAR - THE INPUTS
        # ==========================================
        with st.sidebar:
            st.header("1. Supply Side (Green Buying)")
            st.caption("Enter the exact Amount (KG) you plan to buy:")

            col_a, col_b = st.columns(2)
            with col_a:
                price_low = st.number_input("Low Price ($/kg)", value=6.0, step=0.5)
                kg_low = st.number_input("Low Amount (KG)", value=300, step=50)

                price_med = st.number_input("Med Price ($/kg)", value=12.0, step=0.5)
                kg_med = st.number_input("Med Amount (KG)", value=500, step=50)

                price_high = st.number_input("High Price ($/kg)", value=45.0, step=1.0)
                kg_high = st.number_input("High Amount (KG)", value=100, step=10)

            total_supply_kg = kg_low + kg_med + kg_high

            if total_supply_kg > 0:
                blended_green_price = ((price_low * kg_low) + (price_med * kg_med) + (price_high * kg_high)) / total_supply_kg
            else:
                blended_green_price = 0

            with col_b:
                st.metric("Total Supply Plan", f"{total_supply_kg:,.0f} kg")
                st.metric("Blended Cost", f"${blended_green_price:.2f}/kg")

            st.divider()

            st.header("2. Operational Costs")
            st.caption("Roasting Economics")
            base_roast_cost = st.number_input("Base Roast Labor ($/kg)", value=2.50)
            roast_step_threshold = st.number_input("Step Threshold (kg/mo)", value=1000)
            roast_step_bump = st.number_input("Step Up Cost ($/kg)", value=1.00)
            avg_shrinkage = st.slider("Roast Shrinkage (%)", 10, 25, 18) / 100

            st.divider()

            st.header("3. Demand Side (Channels)")
            st.caption("Projected Sales for January")

            with st.expander("Shop Service (Cups)", expanded=False):
                shop_vol_kg = st.slider("Shop Volume (Roasted KG)", 0, 500, 150)
                shop_rev_kg = st.number_input("Shop Rev per KG ($)", value=85.0)
                shop_cost_kg = st.number_input("Shop Labor/Pack per KG ($)", value=40.0)

            with st.expander("Retail Bags (In-Store)", expanded=False):
                retail_vol_kg = st.slider("Retail Volume (Roasted KG)", 0, 500, 100)
                retail_rev_kg = st.number_input("Retail Rev per KG ($)", value=45.0)
                retail_cost_kg = st.number_input("Retail Pack/Labor per KG ($)", value=5.0)

            with st.expander("Online Subscriptions", expanded=True):
                sub_vol_kg = st.slider("Sub Volume (Roasted KG)", 0, 1000, 200)
                sub_rev_kg = st.number_input("Sub Rev per KG ($)", value=40.0)
                sub_cost_kg = st.number_input("Sub Ship/Pack per KG ($)", value=8.0)

            with st.expander("Online Drops (High End)", expanded=True):
                drop_vol_kg = st.slider("Drop Volume (Roasted KG)", 0, 200, 50)
                drop_rev_kg = st.number_input("Drop Rev per KG ($)", value=120.0)
                drop_cost_kg = st.number_input("Drop Ship/Pack per KG ($)", value=15.0)

            with st.expander("Events / Catering", expanded=False):
                event_vol_kg = st.slider("Event Volume (Roasted KG)", 0, 500, 0)
                event_rev_kg = st.number_input("Event Rev per KG ($)", value=60.0)
                event_cost_kg = st.number_input("Event Labor per KG ($)", value=10.0)

            with st.expander("Bottled Drinks", expanded=False):
                bottle_vol_kg = st.slider("Bottle Volume (Roasted KG equiv)", 0, 500, 50)
                bottle_rev_kg = st.number_input("Bottle Rev per KG ($)", value=150.0)
                bottle_cost_kg = st.number_input("Bottle Prod Cost per KG ($)", value=80.0)

        # ==========================================
        # 3. LOGIC ENGINE
        # ==========================================
        channels = {
            "Shop": {"vol": shop_vol_kg, "rev": shop_rev_kg, "var_cost": shop_cost_kg},
            "Retail": {"vol": retail_vol_kg, "rev": retail_rev_kg, "var_cost": retail_cost_kg},
            "Subs": {"vol": sub_vol_kg, "rev": sub_rev_kg, "var_cost": sub_cost_kg},
            "Drops": {"vol": drop_vol_kg, "rev": drop_rev_kg, "var_cost": drop_cost_kg},
            "Events": {"vol": event_vol_kg, "rev": event_rev_kg, "var_cost": event_cost_kg},
            "Bottles": {"vol": bottle_vol_kg, "rev": bottle_rev_kg, "var_cost": bottle_cost_kg}
        }

        total_roasted_demand_kg = sum(c['vol'] for c in channels.values())
        total_revenue = sum(c['vol'] * c['rev'] for c in channels.values())

        base_labor_total = base_roast_cost * total_roasted_demand_kg
        step_labor_total = roast_step_bump * max(0, total_roasted_demand_kg - roast_step_threshold)
        total_roast_cost = base_labor_total + step_labor_total
        avg_roast_cost_per_kg = total_roast_cost / total_roasted_demand_kg if total_roasted_demand_kg > 0 else 0

        required_green_kg = total_roasted_demand_kg / (1 - avg_shrinkage)
        supply_gap_kg = total_supply_kg - required_green_kg
        total_green_cost_used = required_green_kg * blended_green_price

        total_channel_var_cost = sum(c['vol'] * c['var_cost'] for c in channels.values())
        gross_profit = total_revenue - (total_green_cost_used + total_roast_cost + total_channel_var_cost)
        net_margin_pct = (gross_profit / total_revenue) * 100 if total_revenue > 0 else 0

        # ==========================================
        # 4. DASHBOARD LAYOUT
        # ==========================================
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Projected Revenue", f"${total_revenue:,.0f}")
        kpi2.metric("Projected Gross Profit", f"${gross_profit:,.0f}", f"{net_margin_pct:.1f}% Margin")
        kpi3.metric("Green Needed", f"{required_green_kg:,.0f} kg", f"Supply: {total_supply_kg:,.0f} kg")
        kpi4.metric("Roast Cost Avg", f"${avg_roast_cost_per_kg:.2f}/kg", delta="Step Cost Triggered" if step_labor_total > 0 else "Efficient", delta_color="inverse")

        st.markdown("### 📦 Supply vs Demand Position")
        gap_col, supply_chart_col = st.columns([2, 1])

        with gap_col:
            if supply_gap_kg >= 0:
                st.success(f"✅ **Long Position:** You are buying **{supply_gap_kg:,.0f} kg** more green coffee than you need.")
            else:
                st.error(f"❌ **Short Position:** You need to buy **{abs(supply_gap_kg):,.0f} kg** more to fulfill sales!")
            
            mix_data = pd.DataFrame({
                "Type": ["Low", "Med", "High"],
                "KG": [kg_low, kg_med, kg_high]
            })
            st.dataframe(mix_data, hide_index=True, use_container_width=True)

        with supply_chart_col:
            base = alt.Chart(mix_data).encode(theta=alt.Theta("KG", stack=True))
            pie = base.mark_arc(outerRadius=80, innerRadius=40).encode(
                color=alt.Color("Type", scale=alt.Scale(domain=['Low', 'Med', 'High'], range=['#D7CCC8', '#795548', '#3E2723'])),
                tooltip=["Type", "KG"]
            )
            st.altair_chart(pie, use_container_width=True)

        st.divider()

        # Profitability
        st.markdown("### 💰 Channel Profitability Mixer")
        profit_data = []
        for name, data in channels.items():
            green_alloc = (data['vol'] / (1-avg_shrinkage)) * blended_green_price
            roast_alloc = data['vol'] * avg_roast_cost_per_kg
            channel_gross = (data['vol'] * data['rev']) - (green_alloc + roast_alloc + (data['vol'] * data['var_cost']))
            profit_data.append({
                "Channel": name,
                "Revenue": data['vol'] * data['rev'],
                "Gross_Profit": channel_gross,
                "Volume_KG": data['vol'],
                "Margin_Pct": (channel_gross / (data['vol'] * data['rev'])) * 100 if data['vol'] > 0 else 0
            })

        df_prof = pd.DataFrame(profit_data)

        st.markdown("**1. Profit vs Revenue by Channel**")
        melted = df_prof.melt(id_vars=['Channel'], value_vars=['Revenue', 'Gross_Profit'], var_name='Metric', value_name='Amount')
        bar_chart = alt.Chart(melted).mark_bar().encode(
            x='Channel',
            y='Amount',
            color='Metric',
            column='Metric',
            tooltip=['Channel', 'Amount']
        ).properties(height=250)
        st.altair_chart(bar_chart, use_container_width=True)

        # Growth Simulator
        st.markdown("### 📈 Growth Simulator")
        st.caption("What happens to Profit if we simply sold MORE of the current mix?")
        
        current_total_vol = total_roasted_demand_kg
        current_total_rev = total_revenue
        current_total_var_cost = total_channel_var_cost + total_green_cost_used

        if current_total_vol > 0:
            growth_steps = []
            for pct in range(0, 201, 10):
                sim_vol = current_total_vol * (pct / 100)
                sim_rev = current_total_rev * (pct / 100)
                sim_var_goods = current_total_var_cost * (pct / 100)
                sim_base_labor = base_roast_cost * sim_vol
                sim_step_labor = roast_step_bump * max(0, sim_vol - roast_step_threshold)
                sim_total_roast = sim_base_labor + sim_step_labor
                sim_total_cost = sim_var_goods + sim_total_roast
                sim_profit = sim_rev - sim_total_cost
                growth_steps.append({"Volume_KG": sim_vol, "Revenue": sim_rev, "Total_Cost": sim_total_cost, "Net_Profit": sim_profit})

            df_growth = pd.DataFrame(growth_steps)
            base_growth = alt.Chart(df_growth).encode(x=alt.X('Volume_KG', title='Roasted Volume (KG)'))
            line_rev = base_growth.mark_line(color='green').encode(y='Revenue')
            line_cost = base_growth.mark_line(color='red').encode(y='Total_Cost')
            area_profit = base_growth.mark_area(opacity=0.3, color='blue').encode(y='Net_Profit')
            final_chart = (line_rev + line_cost + area_profit).properties(height=400).interactive()
            st.altair_chart(final_chart, use_container_width=True)

        # Scenario Analysis (Scaling)
        st.markdown("---")
        st.header("📈 Scenario Analysis: Profit Scaling by Stream")
        profit_type = st.radio("Select Metric to Plot:", ("Gross Profit", "Net Profit"), horizontal=True)

        with st.expander("Adjust Scaling Assumptions", expanded=True):
            base_fixed_costs = st.slider("Base Monthly Fixed Costs ($)", 1000, 15000, 5000, 500)
            col1, col2, col3 = st.columns(3)
            with col1:
                ws_margin = st.number_input("Wholesale Margin ($/lb)", value=4.0)
                ws_share = st.slider("Wholesale % of Vol", 0, 100, 50)
            with col2:
                cafe_margin = st.number_input("Cafe Margin ($/lb)", value=18.0)
                cafe_cap_kg = st.number_input("Cafe Cap (kg/mo)", value=360)
            with col3:
                online_margin = st.number_input("Online Margin ($/lb)", value=12.0)
                remaining_share = 100 - ws_share
                st.markdown(f"**Online % of Vol:** {remaining_share}%")

        KG_TO_LBS = 2.20462
        volumes_kg = np.arange(0, 3000 + 100, 50)
        data_scaling, net_profit_data = [], []
        cafe_cap_lbs = cafe_cap_kg * KG_TO_LBS

        for v_kg in volumes_kg:
            v_lbs = v_kg * KG_TO_LBS
            ws_profit = (v_lbs * (ws_share/100)) * ws_margin
            cafe_vol_lbs = min(v_lbs * 0.30, cafe_cap_lbs)
            cafe_profit = cafe_vol_lbs * cafe_margin
            online_profit = (v_lbs * (remaining_share/100)) * online_margin
            
            overhead = base_fixed_costs + (np.floor(v_lbs / 2000) * 2000)
            net_profit = (ws_profit + cafe_profit + online_profit) - overhead
            
            data_scaling.append([v_kg, ws_profit, "Wholesale"])
            data_scaling.append([v_kg, cafe_profit, "Cafe (Retail)"])
            data_scaling.append([v_kg, online_profit, "Online/DTC"])
            net_profit_data.append([v_kg, net_profit])

        df_scaling = pd.DataFrame(data_scaling, columns=["Total Roasted Volume (kg)", "Profit ($)", "Stream"])
        df_net_profit = pd.DataFrame(net_profit_data, columns=["Total Roasted Volume (kg)", "Profit ($)"])

        if profit_type == "Gross Profit":
            fig = px.line(df_scaling, x="Total Roasted Volume (kg)", y="Profit ($)", color="Stream", title="Projected GROSS PROFIT")
        else:
            fig = px.line(df_net_profit, x="Total Roasted Volume (kg)", y="Profit ($)", title="Projected NET PROFIT")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📅 2025 Lookback: Historical Data Analysis")
        # Initialize default data
        months = pd.date_range(start='2025-01-01', periods=12, freq='MS').strftime('%Y-%m')
        monthly_data = pd.DataFrame({'Month': months, 'Net Sales ($)': [10000.0]*12, 'Total Costs ($)': [8000.0]*12}).set_index('Month')
        
        items = ['Filter (Low)', 'Filter (Medium)', 'Filter (High)', 'Retail (Low)', 'Retail (Medium)', 'Retail (High)', 'Espresso + Milk', 'Matcha', 'Tea', 'Merch']
        item_data = pd.DataFrame({'Item': items, 'Total Volume (kg)': [10.0, 50.0, 100.0, 50.0, 200.0, 300.0, 100.0, 5.0, 2.0, 10.0], 'Total Revenue ($)': [200.0, 800.0, 1500.0, 1000.0, 4000.0, 6000.0, 3000.0, 150.0, 50.0, 500.0], 'Total COGS ($)': [50.0, 200.0, 300.0, 250.0, 1000.0, 1500.0, 750.0, 30.0, 10.0, 150.0]}).set_index('Item')

        # Visuals Section
        st.header("📊 2025 Performance Visualizations")
        
        # Calculate for plots
        df_monthly = monthly_data.copy()
        df_monthly['Net Profit ($)'] = df_monthly['Net Sales ($)'] - df_monthly['Total Costs ($)']
        df_monthly['Month'] = df_monthly.index
        
        df_items = item_data.copy()
        df_items['Gross Profit ($)'] = df_items['Total Revenue ($)'] - df_items['Total COGS ($)']
        df_items.loc[df_items['Total Revenue ($)'] > 0, 'Profit Margin (%)'] = (df_items['Gross Profit ($)'] / df_items['Total Revenue ($)']) * 100
        df_items['Profit Margin (%)'] = df_items['Profit Margin (%)'].fillna(0)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            df_pnl_plot = df_monthly[['Net Sales ($)', 'Total Costs ($)', 'Net Profit ($)']].reset_index().melt(id_vars='Month', value_vars=['Net Sales ($)', 'Total Costs ($)', 'Net Profit ($)'], var_name='Metric', value_name='Value')
            fig_pnl = px.bar(df_pnl_plot, x='Month', y='Value', color='Metric', barmode='group', title="Net Sales, Costs, & Profit")
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pnl, use_container_width=True)
            
        with col_chart2:
            df_profit_rank = df_items[df_items['Gross Profit ($)'] > 0].sort_values('Gross Profit ($)', ascending=False).reset_index()
            fig_items = px.bar(df_profit_rank, x='Item', y='Gross Profit ($)', color='Profit Margin (%)', title="Item Profit Ranking")
            st.plotly_chart(fig_items, use_container_width=True)

        st.header("✍️ Update Historical Data")
        st.subheader("1. Monthly Net Sales and Total Costs")
        st.data_editor(monthly_data, num_rows="fixed", key="monthly_editor")
        
        st.subheader("2. Sales Per Item")
        st.data_editor(item_data, num_rows="fixed", key="item_editor")

# Handle Failed Authentication
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
