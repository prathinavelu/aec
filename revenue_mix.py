import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

tab1, tab2 = st.tabs(["📊 Scenario Planning", "📅 2025 Lookback"])

# ----------------------------------------------------
# TAB 1: SCENARIO PLANNING (Your existing code goes here)
# ----------------------------------------------------

with tab1:

    # ==========================================
    # 1. CONFIGURATION & PAGE SETUP
    # ==========================================
    st.set_page_config(page_title="Consolidated Coffee Engine", layout="wide")

    st.markdown("""
    <style>
        .big-font { font-size: 24px !important; font-weight: bold; }
        .metric-container { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("☕ Consolidated Coffee Business Engine")
    st.markdown("---")

    # ==========================================
    # 2. SIDEBAR - THE INPUTS
    # ==========================================

    with st.sidebar:
        st.header("1. Supply Side (Green Buying)")
        st.caption("Enter the exact Amount (KG) you plan to buy:")
        
        # Green Coffee Portfolio - NOW ABSOLUTE KG
        col_a, col_b = st.columns(2)
        with col_a:
            price_low = st.number_input("Low Price ($/kg)", value=6.0, step=0.5)
            kg_low = st.number_input("Low Amount (KG)", value=300, step=50)
            
            price_med = st.number_input("Med Price ($/kg)", value=12.0, step=0.5)
            kg_med = st.number_input("Med Amount (KG)", value=500, step=50)
            
            price_high = st.number_input("High Price ($/kg)", value=45.0, step=1.0)
            kg_high = st.number_input("High Amount (KG)", value=100, step=10)
        
        total_supply_kg = kg_low + kg_med + kg_high
        
        # Calculate Blended Price immediately for display
        if total_supply_kg > 0:
            blended_green_price = ((price_low * kg_low) + (price_med * kg_med) + (price_high * kg_high)) / total_supply_kg
        else:
            blended_green_price = 0
            
        with col_b:
            st.metric("Total Supply Plan", f"{total_supply_kg:,.0f} kg")
            st.metric("Blended Cost", f"${blended_green_price:.2f}/kg")

        st.divider()

        st.header("2. Operational Costs")
        
        # Roasting Step Function
        st.caption("Roasting Economics")
        base_roast_cost = st.number_input("Base Roast Labor ($/kg)", value=2.50)
        roast_step_threshold = st.number_input("Step Threshold (kg/mo)", value=1000)
        roast_step_bump = st.number_input("Step Up Cost ($/kg)", value=1.00, help="Extra cost per KG if volume exceeds threshold")
        
        avg_shrinkage = st.slider("Roast Shrinkage (%)", 10, 25, 18) / 100

        st.divider()

        st.header("3. Demand Side (Channels)")
        st.caption("Projected Sales for January")

        # Channel 1: In-Shop (Cups)
        with st.expander("Shop Service (Cups)", expanded=False):
            shop_vol_kg = st.slider("Shop Volume (Roasted KG)", 0, 500, 150)
            shop_rev_kg = st.number_input("Shop Rev per KG ($)", value=85.0) 
            shop_cost_kg = st.number_input("Shop Labor/Pack per KG ($)", value=40.0)

        # Channel 2: Retail Bags
        with st.expander("Retail Bags (In-Store)", expanded=False):
            retail_vol_kg = st.slider("Retail Volume (Roasted KG)", 0, 500, 100)
            retail_rev_kg = st.number_input("Retail Rev per KG ($)", value=45.0)
            retail_cost_kg = st.number_input("Retail Pack/Labor per KG ($)", value=5.0)

        # Channel 3: Online Subs
        with st.expander("Online Subscriptions", expanded=True):
            sub_vol_kg = st.slider("Sub Volume (Roasted KG)", 0, 1000, 200)
            sub_rev_kg = st.number_input("Sub Rev per KG ($)", value=40.0) 
            sub_cost_kg = st.number_input("Sub Ship/Pack per KG ($)", value=8.0)

        # Channel 4: Online Drops (High End)
        with st.expander("Online Drops (High End)", expanded=True):
            drop_vol_kg = st.slider("Drop Volume (Roasted KG)", 0, 200, 50)
            drop_rev_kg = st.number_input("Drop Rev per KG ($)", value=120.0)
            drop_cost_kg = st.number_input("Drop Ship/Pack per KG ($)", value=15.0)

        # Channel 5: Events
        with st.expander("Events / Catering", expanded=False):
            event_vol_kg = st.slider("Event Volume (Roasted KG)", 0, 500, 0)
            event_rev_kg = st.number_input("Event Rev per KG ($)", value=60.0)
            event_cost_kg = st.number_input("Event Labor per KG ($)", value=10.0)

        # Channel 6: Bottled Drinks
        with st.expander("Bottled Drinks", expanded=False):
            bottle_vol_kg = st.slider("Bottle Volume (Roasted KG equiv)", 0, 500, 50)
            bottle_rev_kg = st.number_input("Bottle Rev per KG ($)", value=150.0) 
            bottle_cost_kg = st.number_input("Bottle Prod Cost per KG ($)", value=80.0)

    # ==========================================
    # 3. LOGIC ENGINE
    # ==========================================

    # A. Calculate Demand Aggregates
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

    # B. Calculate Roast Costs (with Step Function)
    base_labor_total = base_roast_cost * total_roasted_demand_kg
    step_labor_total = roast_step_bump * max(0, total_roasted_demand_kg - roast_step_threshold)
    total_roast_cost = base_labor_total + step_labor_total
    avg_roast_cost_per_kg = total_roast_cost / total_roasted_demand_kg if total_roasted_demand_kg > 0 else 0

    # C. Green Coffee Requirements & Gap Analysis
    required_green_kg = total_roasted_demand_kg / (1 - avg_shrinkage)
    supply_gap_kg = total_supply_kg - required_green_kg
    total_green_cost_used = required_green_kg * blended_green_price # Cost of goods SOLD (not necessarily bought)

    # D. Profitability
    total_channel_var_cost = sum(c['vol'] * c['var_cost'] for c in channels.values())
    gross_profit = total_revenue - (total_green_cost_used + total_roast_cost + total_channel_var_cost)
    net_margin_pct = (gross_profit / total_revenue) * 100 if total_revenue > 0 else 0

    # ==========================================
    # 4. DASHBOARD LAYOUT
    # ==========================================

    # --- HEADLINE KPI ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Projected Revenue", f"${total_revenue:,.0f}")
    kpi2.metric("Projected Gross Profit", f"${gross_profit:,.0f}", f"{net_margin_pct:.1f}% Margin")
    kpi3.metric("Green Needed", f"{required_green_kg:,.0f} kg", f"Supply: {total_supply_kg:,.0f} kg")
    kpi4.metric("Roast Cost Avg", f"${avg_roast_cost_per_kg:.2f}/kg", delta="Step Cost Triggered" if step_labor_total > 0 else "Efficient", delta_color="inverse")

    # --- INVENTORY POSITION ---
    st.markdown("### 📦 Supply vs Demand Position")

    gap_col, supply_chart_col = st.columns([2, 1])

    with gap_col:
        if supply_gap_kg >= 0:
            st.success(f"✅ **Long Position:** You are buying **{supply_gap_kg:,.0f} kg** more green coffee than you need.")
        else:
            st.error(f"❌ **Short Position:** You need to buy **{abs(supply_gap_kg):,.0f} kg** more to fulfill sales!")
            
        # Visualize the components
        mix_data = pd.DataFrame({
            "Type": ["Low", "Med", "High"],
            "KG": [kg_low, kg_med, kg_high]
        })
        st.dataframe(mix_data, hide_index=True, use_container_width=True)

    with supply_chart_col:
        # Donut Chart for Mix
        base = alt.Chart(mix_data).encode(theta=alt.Theta("KG", stack=True))
        pie = base.mark_arc(outerRadius=80, innerRadius=40).encode(
            color=alt.Color("Type", scale=alt.Scale(domain=['Low', 'Med', 'High'], range=['#D7CCC8', '#795548', '#3E2723'])),
            tooltip=["Type", "KG"]
        )
        st.altair_chart(pie, use_container_width=True)

    st.divider()

    # --- PROFITABILITY & EFFICIENCY (Stacked Layout) ---
    st.markdown("### 💰 Channel Profitability Mixer")

    profit_data = []
    for name, data in channels.items():
        # Allocations
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

    # Chart 1: Profit Stack (Full Width)
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

    # Chart 2: Efficiency Map (Full Width)
    st.markdown("**2. Efficiency Map (Margin % vs Volume)**")
    scatter = alt.Chart(df_prof).mark_circle(size=200).encode(
        x='Volume_KG',
        y='Margin_Pct',
        color='Channel',
        tooltip=['Channel', 'Gross_Profit', 'Margin_Pct', 'Volume_KG']
    ).properties(height=300).interactive()

    rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
    st.altair_chart(scatter + rule, use_container_width=True)

    st.divider()

    # --- SENSITIVITY ANALYSIS (New Feature) ---
    st.markdown("### 📈 Growth Simulator (Volume Sensitivity)")
    st.caption("What happens to Profit if we simply sold MORE of the current mix?")

    # Generate hypothetical volumes (0% to 200% of current)
    growth_steps = []
    current_total_vol = total_roasted_demand_kg
    current_total_rev = total_revenue
    current_total_var_cost = total_channel_var_cost + total_green_cost_used

    if current_total_vol > 0:
        for pct in range(0, 201, 10): # 0% to 200%
            sim_vol = current_total_vol * (pct / 100)
            
            # Linear scaling for Rev and Var Costs
            sim_rev = current_total_rev * (pct / 100)
            sim_var_goods = current_total_var_cost * (pct / 100)
            
            # Step Function Logic for Roast Labor
            sim_base_labor = base_roast_cost * sim_vol
            sim_step_labor = roast_step_bump * max(0, sim_vol - roast_step_threshold)
            sim_total_roast = sim_base_labor + sim_step_labor
            
            sim_total_cost = sim_var_goods + sim_total_roast
            sim_profit = sim_rev - sim_total_cost
            
            growth_steps.append({
                "Growth_Pct": pct,
                "Volume_KG": sim_vol,
                "Revenue": sim_rev,
                "Total_Cost": sim_total_cost,
                "Net_Profit": sim_profit
            })

        df_growth = pd.DataFrame(growth_steps)

        # Line Chart: Revenue vs Cost
        base_growth = alt.Chart(df_growth).encode(x=alt.X('Volume_KG', title='Roasted Volume (KG)'))
        
        line_rev = base_growth.mark_line(color='green').encode(y='Revenue')
        line_cost = base_growth.mark_line(color='red').encode(y='Total_Cost')
        
        # Area Chart for Profit
        area_profit = base_growth.mark_area(opacity=0.3, color='blue').encode(y='Net_Profit')
        
        # Step Threshold Line
        rule_step = alt.Chart(pd.DataFrame({'x': [roast_step_threshold]})).mark_rule(color='orange', strokeDash=[5,5]).encode(x='x')
        
        final_chart = (line_rev + line_cost + area_profit + rule_step).properties(height=400).interactive()
        
        st.altair_chart(final_chart, use_container_width=True)
        st.caption("Green = Revenue | Red = Cost | Blue Area = Net Profit | Orange Line = Roast Step Threshold")
    else:
        st.warning("Add some volume in the sidebar to see the Growth Simulator.")




    # --- CONSTANT FOR CONVERSION ---
    KG_TO_LBS = 2.20462 
    MAX_VOLUME_KG = 3000 # Approx. 5000 lbs

    # --- START OF THE PROFIT SCALING SECTION ---

    st.markdown("---")
    st.header("📈 Scenario Analysis: Profit Scaling by Stream")
    st.write("This tool projects how profit evolves as you increase production, highlighting that some streams scale linearly, while others hit physical capacity limits. **All volumes are shown in kilograms (kg).**")

    # --- TOGGLE FOR PROFIT TYPE ---
    profit_type = st.radio(
        "Select Metric to Plot:",
        ("Gross Profit", "Net Profit"),
        horizontal=True,
        help="Gross Profit shows scaling potential. Net Profit shows actual profitability after overhead."
    )

    # --- INPUTS FOR SCALING ASSUMPTIONS ---
    with st.expander("Adjust Scaling Assumptions", expanded=True):
        
        # --- FIXED COST SLIDER ---
        base_fixed_costs = st.slider(
            "Base Monthly Fixed Costs ($)", 
            min_value=1000, 
            max_value=15000, 
            value=5000, 
            step=500,
            help="This covers rent, base salaries, utilities, etc."
        )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("Wholesale (Linear Scaling)")
            # Margin remains $/lb as this is often how prices are set
            ws_margin = st.number_input("Wholesale Margin ($/lb)", value=4.0, step=0.5)
            ws_share = st.slider("Wholesale % of Vol", 0, 100, 50, help="What % of your total roasting goes to wholesale?")
        
        with col2:
            st.caption("Cafe (Capped Capacity)")
            cafe_margin = st.number_input("Cafe Margin ($/lb equivalent)", value=18.0, step=1.0, help="High margin, but harder to scale volume.")
            # CAPACITY IS NOW IN KG
            cafe_cap_kg = st.number_input("Cafe Max Capacity (kg/mo)", value=360, step=50, help="Maximum beans your cafe can brew before it's full (approx. 800 lbs).")
            
        with col3:
            st.caption("Online/Sub (High Var Cost)")
            online_margin = st.number_input("Online Margin ($/lb)", value=12.0, step=1.0)
            # Remaining share calculation
            remaining_share = 100 - ws_share
            st.markdown(f"**Online % of Vol:** {remaining_share}%")

    # --- DATA GENERATION ---
    # Volume range is now in KG
    volumes_kg = np.arange(0, MAX_VOLUME_KG + 100, 50) 
    data = []
    net_profit_data = [] 

    # Convert the KG cap to LBS for internal calculation
    cafe_cap_lbs = cafe_cap_kg * KG_TO_LBS
    FIXED_COST_STEP_LBS = 2000 # Fixed cost steps up every 2000 lbs

    for v_kg in volumes_kg:
        # Convert total volume to LBS for margin calculation
        v_lbs = v_kg * KG_TO_LBS
        
        # --- GROSS PROFIT CALCULATION (ALL CALCULATED USING LBS VOLUME) ---
        
        # 1. Wholesale Profit
        ws_vol_lbs = v_lbs * (ws_share / 100)
        ws_profit = ws_vol_lbs * ws_margin
        
        # 2. Cafe Profit (Capped)
        potential_cafe_demand_lbs = v_lbs * 0.30 
        cafe_vol_lbs = min(potential_cafe_demand_lbs, cafe_cap_lbs) 
        cafe_profit = cafe_vol_lbs * cafe_margin
        
        # 3. Online Profit
        online_vol_lbs = v_lbs * (remaining_share / 100)
        online_profit = online_vol_lbs * online_margin
        
        # --- NET PROFIT CALCULATION ---
        # Fixed Costs: Base amount from slider, plus $2k step increase every 2000 lbs
        # We use v_lbs for the step function trigger to maintain original scenario logic
        overhead = base_fixed_costs + (np.floor(v_lbs / FIXED_COST_STEP_LBS) * 2000) 
        
        total_gross_profit = ws_profit + cafe_profit + online_profit
        net_profit = total_gross_profit - overhead
        
        # Append Gross Profit data using the original volume (KG) for plotting
        data.append([v_kg, ws_profit, "Wholesale"])
        data.append([v_kg, cafe_profit, "Cafe (Retail)"])
        data.append([v_kg, online_profit, "Online/DTC"])
        
        # Append Net Profit data using the original volume (KG) for plotting
        net_profit_data.append([v_kg, net_profit])

    # Create the main DataFrame for Gross Profit Streams
    df_scaling = pd.DataFrame(data, columns=["Total Roasted Volume (kg)", "Profit ($)", "Stream"])

    # Create the Net Profit DataFrame
    df_net_profit = pd.DataFrame(net_profit_data, columns=["Total Roasted Volume (kg)", "Profit ($)"])

    # --- CONDITIONAL PLOTTING ---
    volume_label = "Total Roasted Volume (kg)"

    if profit_type == "Gross Profit":
        
        # Plotting Gross Profit by Stream
        fig = px.line(
            df_scaling, 
            x=volume_label, 
            y="Profit ($)", 
            color="Stream",
            title="Projected GROSS PROFIT Scaling by Stream",
            labels={"Profit ($)": "Gross Profit ($)"},
            template="plotly_white"
        )
        
    else: # Net Profit
        # Plotting Net Profit
        fig = px.line(
            df_net_profit, 
            x=volume_label, 
            y="Profit ($)", 
            title="Projected NET PROFIT vs. Volume (Showing Breakeven)",
            labels={"Profit ($)": "Net Profit ($)"},
            template="plotly_white"
        )
        # Add a horizontal line at 0 for the Breakeven Point
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Breakeven Point") 
        
    # Add scaling annotations
    if profit_type == "Gross Profit":
        # Calculate annotation position in KG
        annotation_x_kg = cafe_cap_kg / 0.30 
        annotation_y = cafe_cap_lbs * cafe_margin
        fig.add_annotation(
            x=annotation_x_kg, y=annotation_y,
            text="Cafe Max Capacity Hit",
            showarrow=True, arrowhead=1
        )

    st.plotly_chart(fig, use_container_width=True)

    # --- INSIGHTS ---
    if profit_type == "Net Profit":
        # Find the breakeven volume in KG
        breakeven_vol_kg = df_net_profit[df_net_profit['Profit ($)'] > 0]['Total Roasted Volume (kg)'].min()
        
        if pd.isna(breakeven_vol_kg):
            st.error(f"""
            **Current Insight:** Based on your current margins and a Base Fixed Cost of ${base_fixed_costs:,.0f}, the projected Net Profit does not become positive within the {MAX_VOLUME_KG} kg range. You need to **increase margins** or **decrease fixed costs**!
            """)
        else:
            st.info(f"""
            **Current Base Fixed Costs:** ${base_fixed_costs:,.0f} per month.
            
            **Key Insight (Net Profit):** The total business reaches its **Breakeven Point** (Net Profit > $0) at approximately **{breakeven_vol_kg:.0f} kg** of total roasted volume per month. Use the fixed cost slider to see how easily overhead shifts the required volume for profitability.
            """)
    else:
        st.info(f"""
        **Key Insight (Gross Profit):** Notice how the **Cafe** line flattens out due to physical limitations (capacity cap of {cafe_cap_kg:.0f} kg). **Wholesale**, while lower margin per unit, continues to climb, showing it drives ultimate scale.
        """)



# ----------------------------------------------------
# TAB 2: 2025 LOOKBACK (Visuals First, then Inputs)
# ----------------------------------------------------

with tab2:
    st.header("📅 2025 Lookback: Historical Data Analysis")
    st.markdown("Review the charts below for current data performance, then use the input sections to update the historical financials.")
    
    # --- DEFAULT DATA INITIALIZATION (Must happen before plotting) ---
    
    # 1. Monthly Financials
    months = pd.date_range(start='2025-01-01', periods=12, freq='MS').strftime('%Y-%m')
    monthly_data = pd.DataFrame({
        'Month': months,
        'Net Sales ($)': [10000.0, 11000, 10500, 12000, 13000, 15000, 14000, 13500, 12500, 11500, 10800, 11200], # Varied sales for better visual
        'Total Costs ($)': [8000.0, 8500, 8300, 9000, 9500, 11000, 10500, 10000, 9300, 8800, 8600, 8900]     # Varied costs
    }).set_index('Month')
    
    # 2. Item Profitability
    items = [
        'Filter (Low)', 'Filter (Medium)', 'Filter (High)', 
        'Retail (Low)', 'Retail (Medium)', 'Retail (High)', 
        'Espresso + Milk', 'Matcha', 'Tea', 'Merch'
    ]
    item_data = pd.DataFrame({
        'Item': items,
        'Total Volume (kg)': [10.0, 50.0, 100.0, 50.0, 200.0, 300.0, 100.0, 5.0, 2.0, 10.0],
        'Total Revenue ($)': [200.0, 800.0, 1500.0, 1000.0, 4000.0, 6000.0, 3000.0, 150.0, 50.0, 500.0],
        'Total COGS ($)': [50.0, 200.0, 300.0, 250.0, 1000.0, 1500.0, 750.0, 30.0, 10.0, 150.0]
    }).set_index('Item')
    
    # 3. Fixed Costs (Default values for calculating insights)
    annual_fixed_costs = {
        'Total Labor Cost ($)': 50000.0, 
        'Rent ($)': 12000.0, 
        'Utilities ($)': 12000.0, 
        'Insurance ($)': 12000.0, 
        'Other Overhead ($)': 12000.0
    }

    # --- DATA PROCESSING (MUST run using *defaults* first, then *edited data*) ---
    
    # Calculate Margins for Item Profitability
    df_items = item_data.copy()
    df_items['Gross Profit ($)'] = df_items['Total Revenue ($)'] - df_items['Total COGS ($)']
    valid_revenue = df_items['Total Revenue ($)'] > 0
    df_items.loc[valid_revenue, 'Profit Margin (%)'] = (df_items['Gross Profit ($)'] / df_items['Total Revenue ($)']) * 100
    df_items['Profit Margin (%)'] = df_items['Profit Margin (%)'].fillna(0) 

    # Calculate Net Profit for Monthly Trend
    df_monthly = monthly_data.copy()
    df_monthly['Net Profit ($)'] = df_monthly['Net Sales ($)'] - df_monthly['Total Costs ($)']
    df_monthly['Month'] = df_monthly.index 
    
    st.markdown("---")

    # ----------------------------------------------------
    # VISUALIZATION SECTION
    # ----------------------------------------------------
    st.header("📊 2025 Performance Visualizations")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # --- CHART 1: MONTHLY NET SALES, TOTAL COSTS, AND NET PROFIT (BAR CHART) ---
        st.subheader("Monthly P&L Breakdown")
        
        # Prepare data for grouped bar chart
        df_pnl_plot = df_monthly[['Net Sales ($)', 'Total Costs ($)', 'Net Profit ($)']].reset_index().melt(
            id_vars='Month', 
            value_vars=['Net Sales ($)', 'Total Costs ($)', 'Net Profit ($)'],
            var_name='Metric',
            value_name='Value'
        )

        fig_pnl = px.bar(
            df_pnl_plot,
            x='Month',
            y='Value',
            color='Metric',
            barmode='group',
            title="Net Sales, Total Costs, & Profit by Month",
            template="plotly_white"
        )
        # Highlight the breakeven point (Y=0)
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Breakeven")
        fig_pnl.update_layout(yaxis_title="Amount ($)")
        
        st.plotly_chart(fig_pnl, use_container_width=True)

    with col_chart2:
        # --- CHART 2: ITEM PROFIT CONTRIBUTION (PARETO) ---
        st.subheader("Gross Profit Contribution by Item")
        
        # Filter for items with profit and sort
        df_profit_rank = df_items[df_items['Gross Profit ($)'] > 0].sort_values(
            'Gross Profit ($)', ascending=False
        ).reset_index()

        fig_items = px.bar(
            df_profit_rank,
            x='Item',
            y='Gross Profit ($)',
            color='Profit Margin (%)', # Use margin as color for quick insight
            title="Item Profit Ranking",
            template="plotly_white",
            labels={'Gross Profit ($)': 'Annual Gross Profit ($)'}
        )
        fig_items.update_xaxes(title_text="")
        
        st.plotly_chart(fig_items, use_container_width=True)
        
    st.markdown("---")
    
    # ----------------------------------------------------
    # DATA INPUTS (MOVED TO BOTTOM)
    # ----------------------------------------------------
    st.header("✍️ Update Historical Data")

    ## 1. Monthly Financial Summary
    st.subheader("1. Monthly Net Sales and Total Costs")
    st.write("Edit the table below to update aggregated P&L data.")
    
    # Capture the output of the data editor
    df_monthly_input = st.data_editor(
        monthly_data,
        column_config={
            "Net Sales ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Total Costs ($)": st.column_config.NumberColumn(format="$%.2f")
        },
        num_rows="fixed",
        key="monthly_data_editor_2025" 
    )
    
    st.markdown("---")

    ## 2. Item Profitability Breakdown
    st.subheader("2. Sales Per Item (Volume, Revenue, Cost)")
    st.write("Edit the table below to update detailed sales and cost data.")
    
    # Capture the output of the data editor
    df_items_input = st.data_editor(
        item_data,
        column_config={
            "Total Volume (kg)": st.column_config.NumberColumn(format="%.1f kg"),
            "Total Revenue ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Total COGS ($)": st.column_config.NumberColumn(format="$%.2f")
        },
        num_rows="fixed",
        key="item_data_editor_2025" 
    )

    # Simple Margin Calculation Display (using input data for display)
    df_items_input['Gross Profit ($)'] = df_items_input['Total Revenue ($)'] - df_items_input['Total COGS ($)']
    valid_rev_input = df_items_input['Total Revenue ($)'] > 0
    df_items_input.loc[valid_rev_input, 'Profit Margin (%)'] = (df_items_input['Gross Profit ($)'] / df_items_input['Total Revenue ($)']) * 100
    
    st.caption("Calculated Profitability (based on edited data):")
    st.dataframe(df_items_input[['Gross Profit ($)', 'Profit Margin (%)']].style.format({
        'Gross Profit ($)': "${:,.2f}", 
        'Profit Margin (%)': "{:.1f}%"
    }))
    
    st.markdown("---")

    ## 3. Fixed Cost Breakdown
    st.subheader("3. Fixed Cost Breakdown (Labor Focus)")
    st.write("Update your major fixed expenses for the year.")
    
    fixed_categories = ['Total Labor Cost ($)', 'Rent ($)', 'Utilities ($)', 'Insurance ($)', 'Other Overhead ($)']
    
    annual_fixed_costs_edited = {}
    for cost in fixed_categories:
        annual_fixed_costs_edited[cost] = st.number_input(
            cost, 
            min_value=0.0, 
            value=(50000.0 if 'Labor' in cost else 12000.0), 
            step=1000.0,
            key=f"fixed_cost_2025_{cost}" 
        )
    
    total_fixed_edited = sum(annual_fixed_costs_edited.values())
    st.markdown(f"**Total Annual Fixed Costs:** **${total_fixed_edited:,.2f}**")
    
    st.markdown("---")


    # --- FINAL INSIGHTS (Using the current state of data inputs for reporting) ---
    st.subheader("🎯 Key Insights (Based on Current Input)")
    
    # Use the output of the data editor to calculate the current profit
    total_annual_profit_input = df_monthly_input['Net Sales ($)'].sum() - df_monthly_input['Total Costs ($)'].sum()
    
    if total_annual_profit_input > 0:
        st.success(f"**Overall Annual Performance:** The current input shows a total Net Profit of **${total_annual_profit_input:,.2f}**.")
    else:
        st.error(f"**Overall Annual Performance:** The current input shows an annual loss of **${abs(total_annual_profit_input):,.2f}**.")
        
    labor_cost_input = annual_fixed_costs_edited.get('Total Labor Cost ($)', 0)
    
    if total_fixed_edited > 0:
        labor_percent_input = (labor_cost_input / total_fixed_edited) * 100
        st.info(f"**Fixed Cost Structure:** Labor accounts for **{labor_percent_input:.1f}%** of your total annual fixed costs (${total_fixed_edited:,.2f}).")
