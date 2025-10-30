# ======================================================
# DASHBOARD: Laadpalen & Elektrische Voertuigen
# ======================================================

# ------------------- Imports --------------------------
# ------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster
from streamlit_folium import st_folium
import requests
import re
import geopandas as gpd
from shapely.geometry import Point, box
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import io
import warnings
from datetime import datetime
from streamlit_option_menu import option_menu
import json
from pathlib import Path
import os

# ------------------- Dark Mode Styling ----------------
# ------------------------------------------------------
st.markdown("""
    <style>
        /* Achtergrond en tekst van hele app */
        body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: #121212;
            color: #E0E0E0;
        }

        /* Sidebar achtergrond en tekst */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }

        /* Sidebar kopjes en labels */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] span {
            color: #E0E0E0 !important;
        }

        /* Hoofdcontent tekst en headers */
        [data-testid="stMarkdownContainer"] h1, 
        [data-testid="stMarkdownContainer"] h2, 
        [data-testid="stMarkdownContainer"] h3, 
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] label, 
        [data-testid="stMarkdownContainer"] span {
            color: #E0E0E0 !important;
        }

        /* Buttons, selectboxes en sliders */
        .stButton>button, .stSelectbox>div>div>div, .stSlider>div>div>div {
            background-color: #333 !important;
            color: #E0E0E0 !important;
        }

        /* Hyperlinks */
        a {
            color: #1E90FF !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- Sidebar ---------------------------
# ------------------------------------------------------
st.markdown("""
    <style>
        /* Achtergrond en tekst van hele app */
        body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: #121212;
            color: #E0E0E0;
        }

        /* Sidebar achtergrond en tekst */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }

        /* Sidebar kopjes en labels */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] span {
            color: #E0E0E0 !important;
        }

        /* Hoofdcontent tekst en headers */
        [data-testid="stMarkdownContainer"] h1, 
        [data-testid="stMarkdownContainer"] h2, 
        [data-testid="stMarkdownContainer"] h3, 
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] label, 
        [data-testid="stMarkdownContainer"] span {
            color: #E0E0E0 !important;
        }

        /* Buttons, selectboxes en sliders */
        .stButton>button, .stSelectbox>div>div>div, .stSlider>div>div>div {
            background-color: #333 !important;
            color: #E0E0E0 !important;
        }

        /* Hyperlinks */
        a {
            color: #1E90FF !important;
        }

        /* Optiemenu minimaliseer padding tussen knoppen */
        .option-menu .nav-link {
            margin-bottom: 2px !important;  /* minder ruimte tussen items */
            padding-bottom: 4px !important;
            padding-top: 4px !important;
        }

        /* Verminder marge van de horizontale lijn */
        hr {
            margin-top: 4px !important;
            margin-bottom: 4px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Titel bovenaan sidebar
st.sidebar.markdown("## Dashboard Elektrische Voertuigen")

# Toggle voor versiekeuze
use_new_sidebar = st.sidebar.toggle("Gebruik nieuwe sidebar", value=False)

# Huidige datum automatisch ophalen
vandaag = datetime.now().strftime("%d %b %Y")

# --- Paginanamen mapping ---
page_mapping = {
    "Laadpalen": "⚡️ Laadpalen",
    "Voertuigen": "🚘 Voertuigen",
    "Voorspellend model": "📊 Voorspellend model"
}

# --- OUDE SIDEBAR ---
if not use_new_sidebar:
    st.sidebar.markdown("---")

    selected_page = st.sidebar.selectbox(
        "Selecteer een pagina",
        list(page_mapping.values())  # laat emoji zien in oude sidebar
    )

    st.sidebar.write("")
    st.sidebar.info("🔋 Data afkomstig van OpenChargeMap & RDW")
    st.sidebar.markdown("---")
    st.sidebar.write("Voor het laatst geüpdatet op:")
    st.sidebar.write(f"*{vandaag}*")
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.sidebar.markdown("---")
    nieuwe_pagina = st.sidebar.toggle("Toon nieuwe pagina", value=False)

# --- NIEUWE SIDEBAR ---
else:
    with st.sidebar:
        # Voeg "Conclusie" toe
        clean_names = list(page_mapping.keys()) + ["Conclusie"] 
        icons_list = ["lightning", "car-front", "bar-chart", "check2-circle"] 

        selected_clean_page = option_menu(
            "Navigatie", 
            clean_names,
            icons=icons_list,
            menu_icon="compass",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#1E1E1E"},
                "icon": {"color": "#00b4d8", "font-size": "20px"}, 
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "2px 0 !important",  # minder marge
                    "padding": "4px 5px !important",
                    "--hover-color": "#3a3a3a",
                    "color": "white"
                },
                "nav-link-selected": {"background-color": "#00b4d8", "color": "white"},
            }
        )

        # Zet de geselecteerde pagina om naar de originele naam met emoji
        if selected_clean_page == "Conclusie":
            selected_page = "📌 Conclusie"
        else:
            selected_page = page_mapping[selected_clean_page]

        st.markdown("---")

        st.write("Voor het laatst geüpdatet op:")
        st.write(f"*{vandaag}*")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("---")
        nieuwe_pagina = st.sidebar.toggle("Toon nieuwe pagina", value=False)

# Gebruik altijd `selected_page` verder in je code
page = selected_page

# ------------------- Data inladen -----------------------
# -------------------------------------------------------
@st.cache_data
def load_data():
    df_auto = pd.read_csv("duitse_automerken_JA.csv")
    return df_auto

@st.cache_data(ttl=86400)
def get_laadpalen_data(lat: float, lon: float, radius: float) -> pd.DataFrame:
    """Haalt laadpalen binnen een straal op."""
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "countrycode": "NL",
        "latitude": lat,
        "longitude": lon,
        "distance": radius,
        "maxresults": 5000,
        "compact": True,
        "verbose": False,
        "key": "bbc1c977-6228-42fc-b6af-5e5f71be11a5"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.json_normalize(data)
    df = df.dropna(subset=['AddressInfo.Latitude', 'AddressInfo.Longitude'])
    return df

@st.cache_data(ttl=86400)
def get_all_laadpalen_nederland() -> pd.DataFrame:
    """Haalt laadpalen van heel Nederland op (voor grafieken)."""
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "countrycode": "NL",
        "maxresults": 10000,
        "compact": True,
        "verbose": False,
        "key": "bbc1c977-6228-42fc-b6af-5e5f71be11a5"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.json_normalize(data)
    return df

df_auto = load_data()


# ======================================================
#                   PAGINA-INDELING
# ======================================================

# ------------------- Pagina 1 --------------------------
if page == "⚡️ Laadpalen":
    if not nieuwe_pagina:
        st.markdown("## Kaart Laadpalen Nederland")
        st.markdown("---")

        provincies = {
            "Heel Nederland": [52.1, 5.3, 200],
            "Groningen": [53.2194, 6.5665, 60],
            "Friesland": [53.1642, 5.7818, 60],
            "Drenthe": [52.9476, 6.6231, 60],
            "Overijssel": [52.4380, 6.5010, 60],
            "Flevoland": [52.5270, 5.5953, 60],
            "Gelderland": [52.0452, 5.8712, 60],
            "Utrecht": [52.0907, 5.1214, 60],
            "Noord-Holland": [52.5206, 4.7885, 60],
            "Zuid-Holland": [52.0116, 4.3571, 60],
            "Zeeland": [51.4940, 3.8497, 60],
            "Noord-Brabant": [51.5730, 5.0670, 60],
            "Limburg": [51.2490, 5.9330, 60],
        }

        provincie_keuze = st.selectbox("📍 Kies een provincie", provincies.keys(), index=0)
        center_lat, center_lon, radius_km = provincies[provincie_keuze]

        with st.spinner(f" Laad laadpalen voor {provincie_keuze}..."):
            df = get_laadpalen_data(center_lat, center_lon, radius_km)
            df_all = get_all_laadpalen_nederland()

            if provincie_keuze != "Heel Nederland":
                Laadpalen = df[df["AddressInfo.StateOrProvince"].str.contains(provincie_keuze, case=False, na=False)]
            else:
                Laadpalen = df

        MAX_DEFAULT = 300  
        st.write(f"Provincie: **{provincie_keuze}**")
        laad_alle = st.checkbox("Laad alle laadpalen (geen popups)", value=False)

        if len(Laadpalen) == 0:
            st.warning("Geen laadpalen gevonden voor deze locatie/provincie.")
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")
            st_folium(m, width=900, height=650)
        else:
            start_zoom = 8 if provincie_keuze == "Heel Nederland" else 10
            m = folium.Map(location=[center_lat, center_lon], zoom_start=start_zoom, tiles="OpenStreetMap")

            if laad_alle:
                coords = list(zip(Laadpalen["AddressInfo.Latitude"], Laadpalen["AddressInfo.Longitude"]))
                FastMarkerCluster(data=coords).add_to(m)
                st.info(f"Snelmodus: alle laadpalen geladen (geen popups).")
            else:
                subset_df = Laadpalen.sample(n=min(len(Laadpalen), MAX_DEFAULT), random_state=1).reset_index(drop=True)
                marker_cluster = MarkerCluster().add_to(m)
                for _, row in subset_df.iterrows():
                    lat, lon = row["AddressInfo.Latitude"], row["AddressInfo.Longitude"]
                    popup = f"""
                    <b>{row.get('AddressInfo.Title', 'Onbekend')}</b><br>
                    {row.get('AddressInfo.AddressLine1', '')}<br>
                    {row.get('AddressInfo.Town', '')}<br>
                    Kosten: {row.get('UsageCost', 'N/B')}<br>
                    Vermogen: {row.get('PowerKW', 'N/B')} kW
                    """
                    icon = folium.Icon(color="green", icon="bolt", prefix="fa")
                    folium.Marker(location=[lat, lon], popup=folium.Popup(popup, max_width=300), icon=icon).add_to(marker_cluster)

                st.success(f"{len(subset_df)} laadpalen met popups geladen.")
            st_folium(m, width=900, height=650, returned_objects=["center", "zoom"])

        st.markdown("<small>**Bron: openchargemap.org**</small>", unsafe_allow_html=True)
        #Grafiek verdeling laadpalen in nederland 
        st.markdown("---")
        st.markdown("## 📊 Verdeling laadpalen in Nederland")

        if len(df_all) > 0:
            def parse_cost(value):
                if isinstance(value, str):
                    if "free" in value.lower() or "gratis" in value.lower():
                        return 0.0
                    match = re.search(r"(\d+[\.,]?\d*)", value.replace(",", "."))
                    return float(match.group(1)) if match else np.nan
                return np.nan

            df_all["UsageCostClean"] = df_all["UsageCost"].apply(parse_cost)

            df_all.loc[
                (df_all["UsageCostClean"] < 0) | (df_all["UsageCostClean"] > 2),
                "UsageCostClean"
            ] = np.nan

            if "PowerKW" in df_all.columns:
                df_all["PowerKW_clean"] = pd.to_numeric(df_all["PowerKW"], errors="coerce")
            elif "Connections.PowerKW" in df_all.columns:
                df_all["PowerKW_clean"] = pd.to_numeric(df_all["Connections.PowerKW"], errors="coerce")
            elif "Connections[0].PowerKW" in df_all.columns:
                df_all["PowerKW_clean"] = pd.to_numeric(df_all["Connections[0].PowerKW"], errors="coerce")
            else:
                df_all["PowerKW_clean"] = np.nan

            provincie_mapping = {
                "Groningen": "Groningen",
                "Friesland": "Friesland",
                "Fryslân": "Friesland",
                "Drenthe": "Drenthe",
                "Overijssel": "Overijssel",
                "Flevoland": "Flevoland",
                "Gelderland": "Gelderland",
                "Utrecht": "Utrecht",
                "Noord-Holland": "Noord-Holland",
                "North Holland": "Noord-Holland",
                "Zuid-Holland": "Zuid-Holland",
                "South Holland": "Zuid-Holland",
                "Zeeland": "Zeeland",
                "Noord-Brabant": "Noord-Brabant",
                "North Brabant": "Noord-Brabant",
                "Limburg": "Limburg"
            }

            df_all["Provincie"] = df_all["AddressInfo.StateOrProvince"].map(provincie_mapping)
            df_all = df_all[df_all["Provincie"].isin(list(provincies.keys()))]

            df_agg = (
                df_all.groupby("Provincie")
                .agg(
                    Aantal_palen=("ID", "count"),
                    Gemiddelde_kosten=("UsageCostClean", "mean"),
                )
                .reset_index()
            )

            totaal = df_agg["Aantal_palen"].sum()
            df_agg["Percentage"] = (df_agg["Aantal_palen"] / totaal) * 100
            df_agg = df_agg.sort_values("Percentage", ascending=False)

            keuze = st.selectbox(
                "📈 Kies welke verdeling je wilt zien:",
                ["Verdeling laadpalen per provincie (%)", "Gemiddelde kosten per provincie"]
            )

            if keuze == "Verdeling laadpalen per provincie (%)":
                fig = px.bar(
                    df_agg,
                    x="Provincie",
                    y="Percentage",
                    title="Verdeling laadpalen per provincie (%)",
                    text=df_agg["Percentage"].apply(lambda x: f"{x:.1f}%")
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(yaxis_title="Percentage van totaal (%)")
            elif keuze == "Gemiddelde kosten per provincie":
                fig = px.bar(
                    df_agg,
                    x="Provincie",
                    y="Gemiddelde_kosten",
                    title="Gemiddelde kosten per provincie (€ per kWh)"
                )
                fig.update_layout(yaxis_title="€ per kWh")

            fig.update_layout(xaxis_title="Provincie", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Kon geen landelijke data laden voor de grafiek.")
    #------------NIEUWE PAGINA 1--------------

    else:

        # ------------------- Functie om provinciegrenzen te laden --------------------
        @st.cache_data(ttl=86400)
        def load_provincie_grenzen():
            """Laadt provinciegrenzen van Nederland (GeoJSON via Cartomap)."""
            import geopandas as gpd
            url = "https://cartomap.github.io/nl/wgs84/provincie_2023.geojson"
            gdf = gpd.read_file(url)
            if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs(epsg=4326)
            return gdf

        st.markdown("## 🗺️ Kaart van Nederland – Laadpalen per Provincie")
        st.markdown("---")

        with st.spinner("Provinciegrenzen laden..."):
            gdf = load_provincie_grenzen()

        # ------------------- Naam Correctie / Mapping ----------------------
        pv_to_nl = {
            "PV27": "Noord-Holland", "PV21": "Friesland", "PV20": "Groningen",
            "PV22": "Drenthe", "PV24": "Flevoland", "PV23": "Overijssel",
            "PV25": "Gelderland", "PV28": "Zuid-Holland", "PV26": "Utrecht",
            "PV29": "Zeeland", "PV30": "Noord-Brabant", "PV31": "Limburg"
        }

        mogelijke_kolommen = ["PROV_NAAM", "provincie", "Provincie", "statnaam", "naam"]
        bron_name_col = next((c for c in mogelijke_kolommen if c in gdf.columns), None)

        code_col = None
        for c in gdf.columns:
            sample_vals = gdf[c].astype(str).dropna().unique()[:10]
            if any(re.match(r"^PV\d{1,3}$", str(v)) for v in sample_vals):
                code_col = c
                break

        if code_col is not None:
            gdf["ProvCode"] = gdf[code_col].astype(str)
            gdf["Provincie_NL"] = gdf["ProvCode"].map(pv_to_nl)
            if bron_name_col is not None:
                gdf["Provincie_NL"] = gdf["Provincie_NL"].fillna(gdf[bron_name_col])
            else:
                gdf["Provincie_NL"] = gdf["Provincie_NL"].fillna(gdf[code_col])
        else:
            if bron_name_col is not None:
                gdf["Provincie_NL"] = gdf[bron_name_col]
            else:
                first_col = gdf.columns[0]
                gdf["Provincie_NL"] = gdf[first_col].astype(str)

        gdf["Provincie_NL"] = gdf["Provincie_NL"].fillna("Onbekend")
        gdf["Provincie"] = gdf["Provincie_NL"]

        provincies = {
            "Heel Nederland": [52.1, 5.3, 200],
            "Groningen": [53.2194, 6.5665, 60],
            "Friesland": [53.1642, 5.7818, 60],
            "Drenthe": [52.9476, 6.6231, 60],
            "Overijssel": [52.4380, 6.5010, 60],
            "Flevoland": [52.5270, 5.5953, 60],
            "Gelderland": [52.0452, 5.8712, 60],
            "Utrecht": [52.0907, 5.1214, 60],
            "Noord-Holland": [52.5206, 4.7885, 60],
            "Zuid-Holland": [52.0116, 4.3571, 60],
            "Zeeland": [51.4940, 3.8497, 60],
            "Noord-Brabant": [51.5730, 5.0670, 60],
            "Limburg": [51.2490, 5.9330, 60],
        }

        provincie_keuze = st.selectbox("📍 Kies een provincie", list(provincies.keys()), index=0)
        center_lat, center_lon, radius_km = provincies[provincie_keuze]

        # === NIEUW: zoomniveau op basis van provincie
        zoom_default = 7 if provincie_keuze == "Heel Nederland" else 8

        # --- Detecteer wijziging in provincie en reset kaartpositie ---
        if "last_provincie" not in st.session_state:
            st.session_state["last_provincie"] = provincie_keuze

        if "map_center" not in st.session_state:
            st.session_state["map_center"] = (center_lat, center_lon)

        if "zoom_level" not in st.session_state:
            st.session_state["zoom_level"] = zoom_default

        # Als gebruiker een andere provincie kiest → reset center, zoom en highlight
        if provincie_keuze != st.session_state["last_provincie"]:
            st.session_state["map_center"] = (center_lat, center_lon)
            st.session_state["zoom_level"] = zoom_default
            st.session_state["highlight_id"] = None
            st.session_state["last_provincie"] = provincie_keuze


        # ------------------- Data Ophalen ------------------------
        with st.spinner(f"🔌 Laadpalen laden voor {provincie_keuze}..."):
            try:
                df_all = get_all_laadpalen_nederland()
            except Exception as e:
                st.error(f"Kon laadpaaldata niet ophalen: {e}")
                df_all = pd.DataFrame()

        if df_all.empty:
            st.warning("Geen laadpalen gevonden.")
            st.stop()

        # ------------------- Kostenberekening -------------------
        def parse_cost(value):
            if isinstance(value, str):
                if "free" in value.lower() or "gratis" in value.lower():
                    return 0.0
                match = re.search(r"(\d+[\.,]?\d*)", value.replace(",", "."))
                return float(match.group(1)) if match else np.nan
            return np.nan

        df_all["UsageCostClean"] = df_all["UsageCost"].apply(parse_cost)
        df_all.loc[(df_all["UsageCostClean"] < 0) | (df_all["UsageCostClean"] > 2), "UsageCostClean"] = np.nan

        provincie_mapping = {
            "Groningen": "Groningen", "Friesland": "Friesland", "Fryslân": "Friesland",
            "Drenthe": "Drenthe", "Overijssel": "Overijssel", "Flevoland": "Flevoland",
            "Gelderland": "Gelderland", "Utrecht": "Utrecht",
            "Noord-Holland": "Noord-Holland", "North Holland": "Noord-Holland",
            "Zuid-Holland": "Zuid-Holland", "South Holland": "Zuid-Holland",
            "Zeeland": "Zeeland", "Noord-Brabant": "Noord-Brabant",
            "North Brabant": "Noord-Brabant", "Limburg": "Limburg"
        }

        df_all["Provincie"] = df_all["AddressInfo.StateOrProvince"].map(provincie_mapping)
        df_all = df_all[df_all["Provincie"].isin(list(provincies.keys()))]

        df_agg = df_all.groupby("Provincie").agg(Gemiddelde_kosten=("UsageCostClean", "mean")).reset_index()

        # ---------------- Filter data per provincie -------------------
        if provincie_keuze != "Heel Nederland":
            df_prov = df_all[df_all["Provincie"] == provincie_keuze]
            gemiddelde = df_agg.loc[df_agg["Provincie"] == provincie_keuze, "Gemiddelde_kosten"].values[0]
        else:
            gemiddelde = df_agg["Gemiddelde_kosten"].mean()
            df_prov = df_all.copy()

        goedkoopste = df_prov["UsageCostClean"].min()
        duurste = df_prov["UsageCostClean"].max()

        def fmt_cost(val):
            if pd.isna(val):
                return "N/B"
            if val == 0.0:
                return "Gratis"
            return f"€{val:.2f}"

        # ---------------- Zoek gratis laadpalen ----------------
        gratis_df = df_prov[df_prov["UsageCost"].astype(str).str.contains("jaarabonnement", case=False, na=False)]
        gratis_df = gratis_df.dropna(subset=["AddressInfo.AddressLine1", "AddressInfo.Latitude", "AddressInfo.Longitude"])
        gratis_df = gratis_df.head(10)

        # ------------------- Layout ------------------------
        col1, col2 = st.columns([2.3, 1.7], gap="large")

        if "map_center" not in st.session_state:
            st.session_state["map_center"] = (center_lat, center_lon)
        if "highlight_id" not in st.session_state:
            st.session_state["highlight_id"] = None
        if "zoom_level" not in st.session_state:
            st.session_state["zoom_level"] = 7 if provincie_keuze == "Heel Nederland" else 9

        with col1:
            center_lat, center_lon = st.session_state["map_center"]

            # Gebruik alleen één style_function (oude werkende)
            def style_function(feature):
                naam = feature["properties"].get("Provincie_NL", "Onbekend")
                base_style = {"color": "black", "weight": 1.5, "fillOpacity": 0.0, "fillColor": "#00000000"}
                if provincie_keuze != "Heel Nederland" and naam == provincie_keuze:
                    base_style.update({"color": "#b30000", "weight": 3})
                return base_style

            highlight_function = None
            if provincie_keuze == "Heel Nederland":
                highlight_function = lambda x: {"fillColor": "#4b4b4b", "fillOpacity": 0.6, "color": "#cc0000", "weight": 3}

            base_lat, base_lon = st.session_state["map_center"]

            # --- Offset enkel visueel toepassen ---
            # --- Offset enkel visueel toepassen ---
            if st.session_state.get("highlight_id") is None:
                # Initieel (geen selectie)
                if provincie_keuze == "Heel Nederland":
                    display_lon = base_lon + 2.8  # Nederland iets verschuiven voor mooi overzicht
                else:
                    display_lon = base_lon + 1.1  # GEEN verschuiving bij provincies
            else:
                if provincie_keuze == "Heel Nederland":
                    display_lon = base_lon + 0.01  
                else:
                    display_lon = base_lon + 0.13


            m = folium.Map(
                location=[base_lat, display_lon],
                zoom_start=st.session_state["zoom_level"],
                tiles="OpenStreetMap"
            )






            folium.GeoJson(
                gdf,
                name="Provinciegrenzen",
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=folium.GeoJsonTooltip(fields=["Provincie_NL"], aliases=["Provincie:"], labels=False)
            ).add_to(m)

            # Laadpalen toevoegen
            if provincie_keuze == "Heel Nederland":
                coords = [(lat, lon) for lat, lon in zip(df_all["AddressInfo.Latitude"], df_all["AddressInfo.Longitude"]) if not (pd.isna(lat) or pd.isna(lon))]
                FastMarkerCluster(data=coords).add_to(m)
            else:
                marker_cluster = MarkerCluster().add_to(m)
                for _, row in df_prov.iterrows():
                    lat, lon = row["AddressInfo.Latitude"], row["AddressInfo.Longitude"]
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    popup = f"<b>{row.get('AddressInfo.Title','Onbekend')}</b><br>{row.get('AddressInfo.AddressLine1','')}<br>{row.get('AddressInfo.Town','')}<br>Kosten: {row.get('UsageCost','N/B')}<br>Vermogen: {row.get('PowerKW','N/B')} kW"
                    folium.Marker(location=[lat, lon], popup=folium.Popup(popup,max_width=300), icon=folium.Icon(color="green", icon="bolt", prefix="fa")).add_to(marker_cluster)

            # Highlight geselecteerde marker
            if st.session_state.get("highlight_id") is not None:
                selected_row = df_prov[df_prov["ID"] == st.session_state["highlight_id"]]
                if not selected_row.empty:
                    sel = selected_row.iloc[0]
                    lat_sel, lon_sel = sel["AddressInfo.Latitude"], sel["AddressInfo.Longitude"]
                    if not (pd.isna(lat_sel) or pd.isna(lon_sel)):
                        popup_sel = f"<b>{sel.get('AddressInfo.Title','Onbekend')}</b><br>{sel.get('AddressInfo.AddressLine1','')}<br>{sel.get('AddressInfo.Town','')}<br>Kosten: {sel.get('UsageCost','N/B')}<br>Vermogen: {sel.get('PowerKW','N/B')} kW"
                        folium.Marker(
                            location=[lat_sel, lon_sel],
                            popup=folium.Popup(popup_sel, max_width=300, autoPan=False),
                            icon=folium.Icon(color="red", icon="bolt", prefix="fa")
                        ).add_to(m)


            st_data = st_folium(m, width=900, height=650)
            st.markdown("<small>Bron: Cartomap GeoJSON & OpenChargeMap API</small>", unsafe_allow_html=True)

        with col2:
            if df_prov.empty:
                st.warning("Geen laadpaaldata gevonden voor dit gebied.")
            else:
                st.metric("Gemiddelde kosten", f"{fmt_cost(gemiddelde)}/kWh")
                colc1, colc2 = st.columns(2)
                with colc1:
                    st.metric("Goedkoopste", fmt_cost(goedkoopste))
                with colc2:
                    st.metric("Duurste", fmt_cost(duurste))

                st.markdown("#### Gratis laadpalen met jaarabonnement")
                if not gratis_df.empty:
                    for i, row in gratis_df.iterrows():
                        addr = row["AddressInfo.AddressLine1"]
                        town = row.get("AddressInfo.Town", "")
                        lat, lon = row["AddressInfo.Latitude"], row["AddressInfo.Longitude"]

                        if st.button(f"📍 {addr}, {town}", key=f"btn_{i}"):
                            st.session_state["map_center"] = (lat, lon)
                            st.session_state["highlight_id"] = row["ID"]
                            st.session_state["zoom_level"] = 15 if provincie_keuze == "Heel Nederland" else 11
                            st.rerun()
                else:
                    st.info("Geen gratis laadpalen met jaarabonnement gevonden in dit gebied.")

            st.markdown("---")
            st.markdown("#### Legenda Laadpaaldichtheid")
            st.markdown("""
            <div style='padding: 10px; background-color: #1E1E1E; border-radius: 10px;'>
                <div style='display: flex; flex-direction: column; gap: 4px;'>
                    <div><span style='background-color: green; width: 20px; height: 10px; display: inline-block; margin-right: 8px;'></span>Weinig laadpalen</div>
                    <div><span style='background-color: yellow; width: 20px; height: 10px; display: inline-block; margin-right: 8px;'></span>Middelmatig aantal</div>
                    <div><span style='background-color: orange; width: 20px; height: 10px; display: inline-block; margin-right: 8px;'></span>Veel laadpalen</div>
                    <div><span style='background-color: red; width: 20px; height: 10px; display: inline-block; margin-right: 8px;'></span>Zeer veel laadpalen</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


        # --- Grafiek verdeling / kosten / verband ---
        st.markdown("---")
        st.markdown("## 📊 Verdeling & Verband tussen Laadpalen en Kosten")

        if len(df_all) > 0:
            def parse_cost(value):
                if isinstance(value, str):
                    if "free" in value.lower() or "gratis" in value.lower():
                        return 0.0
                    match = re.search(r"(\d+[\.,]?\d*)", value.replace(",", "."))
                    return float(match.group(1)) if match else np.nan
                return np.nan

            df_all["UsageCostClean"] = df_all["UsageCost"].apply(parse_cost)
            df_all.loc[
                (df_all["UsageCostClean"] < 0) | (df_all["UsageCostClean"] > 2),
                "UsageCostClean"
            ] = np.nan

            # Power kolom bepalen
            if "PowerKW" in df_all.columns:
                df_all["PowerKW_clean"] = pd.to_numeric(df_all["PowerKW"], errors="coerce")
            elif "Connections.PowerKW" in df_all.columns:
                df_all["PowerKW_clean"] = pd.to_numeric(df_all["Connections.PowerKW"], errors="coerce")
            elif "Connections[0].PowerKW" in df_all.columns:
                df_all["PowerKW_clean"] = pd.to_numeric(df_all["Connections[0].PowerKW"], errors="coerce")
            else:
                df_all["PowerKW_clean"] = np.nan

            # Provincie mapping
            provincie_mapping = {
                "Groningen": "Groningen",
                "Friesland": "Friesland",
                "Fryslân": "Friesland",
                "Drenthe": "Drenthe",
                "Overijssel": "Overijssel",
                "Flevoland": "Flevoland",
                "Gelderland": "Gelderland",
                "Utrecht": "Utrecht",
                "Noord-Holland": "Noord-Holland",
                "North Holland": "Noord-Holland",
                "Zuid-Holland": "Zuid-Holland",
                "South Holland": "Zuid-Holland",
                "Zeeland": "Zeeland",
                "Noord-Brabant": "Noord-Brabant",
                "North Brabant": "Noord-Brabant",
                "Limburg": "Limburg"
            }

            df_all["Provincie"] = df_all["AddressInfo.StateOrProvince"].map(provincie_mapping)
            df_all = df_all[df_all["Provincie"].isin(list(provincies.keys()))]

            # Aggregatie per provincie
            df_agg = (
                df_all.groupby("Provincie")
                .agg(
                    Aantal_palen=("ID", "count"),
                    Gemiddelde_kosten=("UsageCostClean", "mean"),
                )
                .reset_index()
            )

            totaal = df_agg["Aantal_palen"].sum()
            df_agg["Percentage"] = (df_agg["Aantal_palen"] / totaal) * 100
            df_agg = df_agg.sort_values("Percentage", ascending=False)

            # Oppervlakte van provincies (km²)
            oppervlakte_dict = {
                "Groningen": 2324,
                "Friesland": 3336,
                "Drenthe": 2633,
                "Overijssel": 3325,
                "Flevoland": 1412,
                "Gelderland": 4964,
                "Utrecht": 1485,
                "Noord-Holland": 2665,
                "Zuid-Holland": 2700,
                "Zeeland": 1782,
                "Noord-Brabant": 4905,
                "Limburg": 2147
            }
            df_agg["Oppervlakte_km2"] = df_agg["Provincie"].map(oppervlakte_dict)

            # Dropdown met 3 opties
            keuze = st.selectbox(
                "📈 Kies welke verdeling of verband je wilt zien:",
                [
                    "Verdeling laadpalen per provincie (%)",
                    "Gemiddelde kosten per provincie",
                    "Verband tussen beschikbaarheid en kosten"
                ]
            )

            # --- Optie 1: Verdeling laadpalen ---
            if keuze == "Verdeling laadpalen per provincie (%)":
                fig = px.bar(
                    df_agg,
                    x="Provincie",
                    y="Percentage",
                    title="Verdeling laadpalen per provincie (%)",
                    text=df_agg["Percentage"].apply(lambda x: f"{x:.1f}%")
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(yaxis_title="Percentage van totaal (%)")
                st.plotly_chart(fig, use_container_width=True)

            # --- Optie 2: Gemiddelde kosten ---
            elif keuze == "Gemiddelde kosten per provincie":
                fig = px.bar(
                    df_agg,
                    x="Provincie",
                    y="Gemiddelde_kosten",
                    title="Gemiddelde kosten per provincie (€ per kWh)"
                )
                fig.update_layout(yaxis_title="€ per kWh")
                st.plotly_chart(fig, use_container_width=True)

            # --- Optie 3: Verband tussen beschikbaarheid en kosten ---
            elif keuze == "Verband tussen beschikbaarheid en kosten":
                st.subheader("Verband tussen beschikbaarheid en kosten per provincie")

                # Basisgrafiek
                kleuren = ["rgb(0,180,255)"] * len(df_agg)

                # Checkbox 1: regressiemodel zonder oppervlakte
                gebruik_regressie = st.checkbox("Optimale provincie (regressie-analyse)")
                gebruik_oppervlakte = st.checkbox("Optimale provincie (regressie-analyse, gecorrigeerd voor oppervlakte)", disabled=not gebruik_regressie)

                optimale_provincie = None
                optimale_provincie_oppervlakte = None

                if gebruik_regressie:
                    import statsmodels.api as sm

                    # Regressiemodel: kosten t.o.v. aandeel laadpalen
                    df_agg = df_agg.reset_index(drop=True)  # <<< belangrijk
                    X = sm.add_constant(df_agg["Percentage"])
                    y = df_agg["Gemiddelde_kosten"]
                    model = sm.OLS(y, X, missing="drop").fit()
                    df_agg["Voorspeld"] = model.predict(X)

                    corr = df_agg["Percentage"].corr(df_agg["Gemiddelde_kosten"])
                    st.markdown(f"**📉 Correlatie (kosten vs beschikbaarheid):** `{corr:.2f}`")
                    st.markdown(f"**Model:** Kosten = {model.params[0]:.3f} + {model.params[1]:.3f} × Percentage")

                    # Optimale provincie volgens model (laagste voorspelde kosten)
                    idx_opt = df_agg["Voorspeld"].idxmin()
                    optimale_provincie = df_agg.loc[idx_opt, "Provincie"]

                    st.success(f"Optimale provincie: **{optimale_provincie}**")

                    kleuren[idx_opt] = "limegreen"

                    # Tweede checkbox: oppervlakte meenemen
                    if gebruik_oppervlakte:
                        df_agg["Aanpassing_oppervlakte"] = df_agg["Voorspeld"] * (df_agg["Oppervlakte_km2"] / df_agg["Oppervlakte_km2"].mean())
                        idx_opt2 = df_agg["Aanpassing_oppervlakte"].idxmin()
                        optimale_provincie_oppervlakte = df_agg.loc[idx_opt2, "Provincie"]

                        st.info(f"Optimale provincie (correctie voor oppervlakte): **{optimale_provincie_oppervlakte}**")

                        # Kleur ook deze provincie (anders dan limegreen)
                        kleuren[idx_opt2] = "orange"

                # Plot gecombineerde grafiek
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=df_agg["Provincie"],
                    y=df_agg["Percentage"],
                    name="Aandeel laadpalen (%)",
                    marker_color=kleuren,
                    opacity=0.7,
                    yaxis="y1"
                ))

                fig.add_trace(go.Scatter(
                    x=df_agg["Provincie"],
                    y=df_agg["Gemiddelde_kosten"],
                    name="Gemiddelde kosten (€ per kWh)",
                    mode="lines+markers",
                    line=dict(color="rgb(255, 100, 100)", width=3),
                    marker=dict(size=8, color="rgb(255, 150, 150)"),
                    yaxis="y2"
                ))

                fig.update_layout(
                    title="Verband tussen beschikbaarheid en kosten per provincie",
                    xaxis=dict(title="Provincie"),
                    yaxis=dict(
                        title="Aandeel laadpalen (%)",
                        showgrid=False,
                        color="rgb(0,180,255)"
                    ),
                    yaxis2=dict(
                        title="Gemiddelde kosten (€ per kWh)",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        color="rgb(255,150,150)"
                    ),
                    legend=dict(orientation="h", y=1.15, x=0.25),
                    template="plotly_dark",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Uitleg / legenda
                uitleg = "🔵 = standaard provincie • 🟩 = optimale volgens regressiemodel"
                if gebruik_oppervlakte:
                    uitleg += " • 🟧 = aangepaste optimale provincie (rekening houdend met oppervlakte)"
                st.caption(uitleg)

        else:
            st.warning("Kon geen landelijke data laden voor de grafiek.")

       

# ------------------- Pagina 2 --------------------------
elif page == "🚘 Voertuigen":
    if not nieuwe_pagina:
        st.markdown("## Elektrische Voertuigen & laadtijden")
        st.markdown("---")

        #-----Grafiek Lieke------


        # --- Functie om brandstoftype te bepalen ---
        def bepaal_type(merk, uitvoering):
            u = str(uitvoering).upper()
            m = str(merk).upper()

            elektrische_prefixen = [
                "FA1FA1CZ", "3EER", "3EDF", "3EDE", "2EER", "2EDF", "2EDE",
                "E11", "0AW5", "QE2QE2G1", "QE1QE1G1", "HE1HE1G1", "FA1FA1MD"
            ]

            # Elektrisch
            if "BMW I" in m or "PORSCHE" in m or any(u.startswith(pref) for pref in elektrische_prefixen) or "EV" in u:
                return "Elektrisch"

            # Diesel
            if "DIESEL" in u or "TDI" in u or "CDI" in u or "DPE" in u or u.startswith("D"):
                return "Diesel"

            # Benzine (default)
            return "Benzine"


        # --- Data inladen ---
        data = pd.read_csv("duitse_automerken_JA.csv")

        # --- Merknamen normaliseren ---
        merk_mapping = {
            "VW": "VOLKSWAGEN",
            "FAW-VOLKSWAGEN": "VOLKSWAGEN",
            "VOLKSWAGEN/ZIMNY": "VOLKSWAGEN",
            "BMW I": "BMW",
            "FORD-CNG-TECHNIK": "FORD"
        }
        data["Merk"] = data["Merk"].str.upper().replace(merk_mapping)

        # --- Type bepalen ---
        data["Type"] = data.apply(lambda row: bepaal_type(row["Merk"], row["Uitvoering"]), axis=1)

        # --- Datumverwerking ---
        data["Datum eerste toelating"] = (
            data["Datum eerste toelating"].astype(str).str.split(".").str[0]
        )
        data["Datum eerste toelating"] = pd.to_datetime(
            data["Datum eerste toelating"], format="%Y%m%d", errors="coerce"
        )
        data = data.dropna(subset=["Datum eerste toelating"])
        data = data[data["Datum eerste toelating"].dt.year > 2010]
        data["Maand"] = data["Datum eerste toelating"].dt.to_period("M").dt.to_timestamp()

        # ---  Keuzemenu voor merken ---
        alle_merknamen = sorted(data["Merk"].unique())
        geselecteerde_merknamen = st.multiselect(
            "*Selecteer automerken om te tonen:*",
            options=alle_merknamen,
            default=[]  # begin met geen selectie
        )

        # ---  Als geen merken geselecteerd: waarschuwing + alle merken gebruiken ---
        if not geselecteerde_merknamen:
            st.warning("⚠️ Geen merken geselecteerd. Alle merken worden getoond!")
            geselecteerde_merknamen = alle_merknamen

        # Filter data op geselecteerde merken
        data = data[data["Merk"].isin(geselecteerde_merknamen)]

        # --- Aggregatie ---
        maand_aantal = data.groupby(["Maand", "Type"]).size().unstack(fill_value=0)
        cumulatief = maand_aantal.cumsum()

        # --- 📈 Titel + Grafiek ---
        st.subheader("Cumulatief aantal voertuigen per maand")
        st.line_chart(cumulatief)

    #-------------Grafiek Ann---------


        # ---- Bestand vast instellen ----
        file_path = "Charging_data.pkl"

        # ---- FUNCTIE: x-as als hele getallen zetten ----
        def force_integer_xaxis(fig):
            fig.update_xaxes(dtick=1)
            return fig

        # ---- DATA INLADEN ----
        try:
            ev_data = pd.read_pickle(file_path)
            ev_data.columns = (
                ev_data.columns.astype(str)
                .str.strip()
                .str.replace("\u200b", "", regex=False)
                .str.lower()
            )

            # ---- DATUMCONVERSIE EN KOLOMMEN TOEVOEGEN ----
            ev_data["start_time"] = pd.to_datetime(ev_data["start_time"], errors="coerce")
            ev_data["exit_time"] = pd.to_datetime(ev_data["exit_time"], errors="coerce")
            ev_data["hour"] = ev_data["start_time"].dt.hour
            ev_data["month"] = ev_data["start_time"].dt.to_period("M").astype(str)
            ev_data["year"] = ev_data["start_time"].dt.year
            ev_data = ev_data[ev_data["year"].notna()]
            ev_data["year"] = ev_data["year"].astype(int)
            ev_data["weekday"] = ev_data["start_time"].dt.day_name()

            energy_col = "energy_delivered [kwh]"

            # ---- WEEKDAGFILTER ----
            st.subheader("🔍 Filter op weekdagen")
            weekdays_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            selected_days = st.multiselect(
                "Selecteer één of meerdere weekdagen:",
                weekdays_order,
                default=weekdays_order,
            )
            ev_data = ev_data[ev_data["weekday"].isin(selected_days)]

            # ---- HEATMAP: Laadpatronen per dag en uur ----
            st.subheader("Laadpatronen per dag en uur")
            heatmap_data = ev_data.groupby(["weekday", "hour"]).size().reset_index(name="count")
            heatmap_data["weekday"] = pd.Categorical(heatmap_data["weekday"], categories=weekdays_order, ordered=True)

            fig_hm = px.density_heatmap(
                heatmap_data,
                x="hour",
                y="weekday",
                z="count",
                color_continuous_scale=[[0.0, "#317595"], [0.5, "#fcffbf"], [1.0, "#c2242f"]],
            )
            fig_hm = force_integer_xaxis(fig_hm)
            fig_hm.update_coloraxes(colorbar_title="Aantal sessies", colorbar_title_side="top")
            st.plotly_chart(fig_hm, use_container_width=True)

            # ---- FILTER OP AANTAL FASEN ----
            phase_options = ["Alle"] + [x for x in sorted(ev_data["n_phases"].dropna().unique()) if 0 <= x <= 6]
            phase_choice = st.selectbox("**Filter op aantal fasen**", phase_options)

            ev_filtered = ev_data.copy()
            if phase_choice != "Alle":
                ev_filtered = ev_filtered[ev_filtered["n_phases"] == phase_choice]

            # ---- GRAFIEK 1: Laadsessies per uur van de dag ----
            st.subheader("Laadsessies per uur van de dag")
            hourly_counts = ev_filtered.groupby("hour").size().reset_index(name="Aantal laadsessies")
            fig1 = px.bar(hourly_counts, x="hour", y="Aantal laadsessies")
            fig1 = force_integer_xaxis(fig1)
            st.plotly_chart(fig1, use_container_width=True)

            # ---- GRAFIEK 2: Totaal geladen energie per maand ----
            st.subheader("Totaal geladen energie per maand")
            energy_by_month = ev_filtered.groupby("month")[energy_col].sum().reset_index().sort_values("month")

            # Controleer aantal unieke maanden
            unique_months = energy_by_month["month"].nunique()

            fig2 = px.bar(energy_by_month, x="month", y=energy_col)
            fig2.update_xaxes(type="category", title_text="Maand")
            fig2.update_yaxes(title_text="Totaal geladen energie (kWh)")
            st.plotly_chart(fig2, use_container_width=True)

            # ---- GRAFIEK 3: Gemiddelde sessieduur per maand ----
            st.subheader("Gemiddelde sessieduur per maand (uren)")
            ev_filtered["session_duration"] = (ev_filtered["exit_time"] - ev_filtered["start_time"]).dt.total_seconds() / 3600
            avg_duration = (
                ev_filtered.groupby("month")["session_duration"].mean().reset_index().sort_values("month")
            )
            fig3 = px.line(avg_duration, x="month", y="session_duration", markers=True)
            fig3.update_xaxes(type="category", title_text="Maand")
            fig3.update_yaxes(title_text="Gemiddelde sessieduur (uren)")
            st.plotly_chart(fig3, use_container_width=True)

            # ---- GRAFIEK 4: Boxplot energie per sessie per maand ----
            st.subheader("Verdeling van geladen energie per sessie per maand")
            fig4 = px.box(ev_filtered, x="month", y=energy_col, points="all")
            fig4.update_xaxes(type="category", title_text="Maand")
            fig4.update_yaxes(title_text="Energie per sessie (kWh)")
            st.plotly_chart(fig4, use_container_width=True)

            # ---- DATA BEKIJKEN ----
            with st.expander("📊 Bekijk gebruikte data (Charging_data.pkl)"):
                st.dataframe(ev_filtered)

        except Exception as e:
            st.error(f"Er is een fout opgetreden bij het inlezen van `{file_path}`: {e}")



    #------------NIEUWE PAGINA 2--------------

    else:

        st.markdown("## 🚘 Elektrische Voertuigen & Laadtijden")
        st.markdown("---")

        st.title("Staafdiagram: Laadtijd (uur) en Energie (kWh)")

        # --- Data inladen (altijd lokaal, geen uploader) ---
        file_path = Path(os.getcwd()) / "Charging_data.pkl"
        try:
            df = pd.read_pickle(file_path)
        except Exception as e:
            st.error(
                "Kan 'Charging_data.pkl' niet laden. Plaats het bestand in dezelfde map als dit script.\n\n"
                f"Foutmelding: {e}"
            )
            st.stop()

        # --- Verwachte kolommen checken / voorbereiden ---
        expected = {"start_time", "charging_duration", "energy_delivered [kWh]"}
        missing = expected - set(df.columns)
        if missing:
            st.error(f"Ontbrekende kolommen in dataset: {missing}.")
            st.stop()

        # Types converteren indien nodig
        if not np.issubdtype(df["start_time"].dtype, np.datetime64):
            df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        if "exit_time" in df.columns and not np.issubdtype(df["exit_time"].dtype, np.datetime64):
            df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")

        # ---- Laadtijd in uren: ALLEEN uit 'charging_duration' met robuuste parser ----
        def parse_duration_to_hours(val) -> float:
            import re
            import numpy as np
            import pandas as pd

            if val is None or (isinstance(val, float) and np.isnan(val)):
                return np.nan

            # 0) Directe Timedelta
            if isinstance(val, pd.Timedelta):
                hours = val.total_seconds() / 3600.0
                return hours if 0 <= hours < 24 * 48 else np.nan

            # 1) Probeer generiek met to_timedelta
            try:
                td = pd.to_timedelta(str(val), errors="coerce")
                if pd.notna(td):
                    hours = td.total_seconds() / 3600.0
                    return hours if 0 <= hours < 24 * 48 else np.nan
            except Exception:
                pass

            # 2) Getal → heuristiek
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                x = float(val)
                if np.isnan(x):
                    return np.nan
                if x > 1000:         # waarschijnlijk seconden
                    hours = x / 3600.0
                elif 10 < x < 1000:  # waarschijnlijk minuten
                    hours = x / 60.0
                else:                # waarschijnlijk uren
                    hours = x
                return hours if 0 <= hours < 24 * 48 else np.nan

            # 3) Strings normaliseren en parsen
            s = str(val).strip().lower()
            s = s.replace(",", ".")
            s = s.replace("±", "").replace("~", "").replace("≈", "")
            s = re.sub(r"\s+", " ", s)

            specials = {
                "an hour": 1.0, "a hour": 1.0, "one hour": 1.0,
                "half hour": 0.5, "half an hour": 0.5, "half uur": 0.5,
                "kwartier": 0.25, "quarter hour": 0.25,
                "3/4 hour": 0.75, "¾ hour": 0.75,
                "30 minutes": 0.5, "30 minute": 0.5, "an half hour": 0.5,
                "an minute": 1.0/60.0,
            }
            if s in specials:
                return specials[s]

            m_iso = re.match(r"^pt(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", s)
            if m_iso:
                h = float(m_iso.group(1) or 0)
                m = float(m_iso.group(2) or 0)
                sec = float(m_iso.group(3) or 0)
                hours = h + m/60.0 + sec/3600.0
                return hours if 0 <= hours < 24 * 48 else np.nan

            parts = re.findall(
                r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|uur|uren|m|min|mins|minute|minutes|s|sec|secs|second|seconds)",
                s
            )
            if parts:
                total_hours = 0.0
                for num, unit in parts:
                    v = float(num)
                    if unit in ["h", "hr", "hrs", "hour", "hours", "uur", "uren"]:
                        total_hours += v
                    elif unit in ["m", "min", "mins", "minute", "minutes"]:
                        total_hours += v / 60.0
                    elif unit in ["s", "sec", "secs", "second", "seconds"]:
                        total_hours += v / 3600.0
                return total_hours if 0 <= total_hours < 24 * 48 else np.nan

            m_simple = re.match(r"^(\d+(?:\.\d+)?)(h|m|s)$", s)
            if m_simple:
                v = float(m_simple.group(1)); u = m_simple.group(2)
                hours = v if u == "h" else v/60.0 if u == "m" else v/3600.0
                return hours if 0 <= hours < 24 * 48 else np.nan

            try:
                x = float(s)
                if x > 1000:
                    hours = x / 3600.0
                elif 10 < x < 1000:
                    hours = x / 60.0
                else:
                    hours = x
                return hours if 0 <= hours < 24 * 48 else np.nan
            except Exception:
                return np.nan

        # Toepassen
        df["laadtijd_uren"] = df["charging_duration"].apply(parse_duration_to_hours)
        df.loc[(df["laadtijd_uren"] < 0) | (df["laadtijd_uren"] >= 24 * 48), "laadtijd_uren"] = np.nan
        df["energie_kwh"] = pd.to_numeric(df["energy_delivered [kWh]"], errors="coerce")

        # Extra tijdsdimensies
        df["jaar"] = df["start_time"].dt.year
        df["maand_num"] = df["start_time"].dt.month
        df["dag"] = df["start_time"].dt.day

        nl_months = {
            1: "januari", 2: "februari", 3: "maart", 4: "april", 5: "mei", 6: "juni", 7: "juli",
            8: "augustus", 9: "september", 10: "oktober", 11: "november", 12: "december"
        }

        months_available_df = (
            df.dropna(subset=["start_time"])
            .sort_values(["jaar", "maand_num"])
            .drop_duplicates(["jaar", "maand_num"])[["jaar", "maand_num"]]
            .reset_index(drop=True)
        )
        month_keys = list(months_available_df.itertuples(index=False, name=None))
        month_labels = [f"{nl_months[m]} {y}" for (y, m) in month_keys]

        keuze = st.selectbox("Kies maand (of 'Alle maanden'):", options=["Alle maanden"] + month_labels, index=0)

        def plot_bars(x_values, laadtijd, energie, x_title):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x_values, y=laadtijd, name="Laadtijd (uur)",
                yaxis="y", offsetgroup="laadtijd",
                hovertemplate="%{x}<br>Laadtijd: %{y:.2f} uur<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=x_values, y=energie, name="Energie (kWh)",
                yaxis="y2", offsetgroup="energie",
                hovertemplate="%{x}<br>Energie: %{y:.2f} kWh<extra></extra>",
            ))
            fig.update_layout(
                barmode="group",
                bargap=0.15,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified",
                xaxis=dict(title=x_title, type="category"),
                yaxis=dict(title="Laadtijd (uur)"),
                yaxis2=dict(title="Energie (kWh)", overlaying="y", side="right"),
            )
            return fig

        if keuze == "Alle maanden":
            grp = (
                df
                .groupby(["jaar", "maand_num"], dropna=True)
                .agg(
                    laadtijd_uren=("laadtijd_uren", "sum"),
                    energie_kwh=("energie_kwh", "sum"),
                )
                .reset_index()
                .sort_values(["jaar", "maand_num"])
            )
            x_vals = [f"{nl_months[m]} {y}" for y, m in zip(grp["jaar"], grp["maand_num"])]
            fig = plot_bars(x_vals, grp["laadtijd_uren"], grp["energie_kwh"], x_title="Maanden")
            st.plotly_chart(fig, use_container_width=True)
        else:
            idx = month_labels.index(keuze)
            year, month = month_keys[idx]
            df_sel = df[(df["jaar"] == year) & (df["maand_num"] == month)]
            grp = (
                df_sel
                .groupby(["jaar", "maand_num", "dag"], dropna=True)
                .agg(
                    laadtijd_uren=("laadtijd_uren", "sum"),
                    energie_kwh=("energie_kwh", "sum"),
                )
                .reset_index()
                .sort_values(["dag"])
            )
            x_vals = [str(d) for d in grp["dag"].tolist()]
            fig = plot_bars(x_vals, grp["laadtijd_uren"], grp["energie_kwh"], x_title=f"Dagen in {nl_months[month]} {year}")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Toon geaggregeerde waarden"):
            if keuze == "Alle maanden":
                st.dataframe(grp.rename(columns={"maand_num": "maand"}))
            else:
                st.dataframe(grp.rename(columns={"dag": "dag van maand"}))

        # =======================
        # 📅 Vergelijk 2 dagen
        # =======================
        st.markdown("### 📅 Vergelijk twee dagen")
        df["datum"] = df["start_time"].dt.date
        available_dates = sorted(d for d in df["datum"].dropna().unique())

        if len(available_dates) < 2:
            st.info("Er zijn minder dan twee dagen met data om te vergelijken.")
        else:
            default_d1 = available_dates[-2]
            default_d2 = available_dates[-1]

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                dag1 = st.date_input(
                    "Dag 1",
                    value=default_d1,
                    min_value=available_dates[0],
                    max_value=available_dates[-1],
                    format="YYYY-MM-DD",
                    key="compare_day1",
                )
            with c2:
                dag2 = st.date_input(
                    "Dag 2",
                    value=default_d2,
                    min_value=available_dates[0],
                    max_value=available_dates[-1],
                    format="YYYY-MM-DD",
                    key="compare_day2",
                )

            keuze_dagen = [dag1, dag2]
            df_sel = df[df["datum"].isin(keuze_dagen)].copy()

            if df_sel.empty or len(set(keuze_dagen)) < 2:
                st.warning("Kies twee verschillende dagen met beschikbare data.")
            else:
                dag_agg = (
                    df_sel.groupby("datum", dropna=True)
                    .agg(
                        laadtijd_uren=("laadtijd_uren", "sum"),
                        energie_kwh=("energie_kwh", "sum"),
                        sessies=("energie_kwh", "count"),
                    )
                    .reset_index()
                )

                st.dataframe(
                    dag_agg.rename(
                        columns={
                            "datum": "Datum",
                            "laadtijd_uren": "Laadtijd (uur)",
                            "energie_kwh": "Energie (kWh)",
                            "sessies": "Aantal sessies",
                        }
                    ),
                    use_container_width=True,
                )

                metrics = ["Laadtijd (uur)", "Energie (kWh)", "Aantal sessies"]

                def vals_for_day(d):
                    row = dag_agg.loc[dag_agg["datum"] == d]
                    if row.empty:
                        return [0, 0, 0]
                    return [
                        float(row["laadtijd_uren"].iloc[0] or 0),
                        float(row["energie_kwh"].iloc[0] or 0),
                        float(row["sessies"].iloc[0] or 0),
                    ]

                y_d1 = vals_for_day(dag1)
                y_d2 = vals_for_day(dag2)

                fig_cmp = go.Figure()
                fig_cmp.add_trace(
                    go.Bar(
                        x=metrics, y=y_d1, name=str(dag1),
                        offsetgroup="d1",
                        hovertemplate="%{x}<br>%{y:.2f}<extra>" + str(dag1) + "</extra>",
                    )
                )
                fig_cmp.add_trace(
                    go.Bar(
                        x=metrics, y=y_d2, name=str(dag2),
                        offsetgroup="d2",
                        hovertemplate="%{x}<br>%{y:.2f}<extra>" + str(dag2) + "</extra>",
                    )
                )
                fig_cmp.update_layout(
                    barmode="group",
                    bargap=0.15,
                    title=dict(text="Vergelijking per metriek", font=dict(size=24), x=0.5, xanchor="center"),
                    xaxis_title="Metriek",
                    yaxis_title="Waarde",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    margin=dict(l=10, r=10, t=30, b=10),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                with st.expander("Detail: per uur binnen de twee dagen"):
                    df_detail = df_sel.copy()
                    df_detail["uur"] = df_detail["start_time"].dt.hour

                    per_uur = (
                        df_detail.groupby(["datum", "uur"])
                        .agg(
                            laadtijd_uren=("laadtijd_uren", "sum"),
                            energie_kwh=("energie_kwh", "sum"),
                        )
                        .reset_index()
                        .sort_values(["datum", "uur"])
                    )

                    fig_uur_lt = go.Figure()
                    for d in keuze_dagen:
                        ddata = per_uur[per_uur["datum"] == d]
                        fig_uur_lt.add_trace(
                            go.Scatter(x=ddata["uur"], y=ddata["laadtijd_uren"], mode="lines+markers", name=f"Laadtijd - {d}")
                        )
                    fig_uur_lt.update_layout(
                        title="Laadtijd per uur",
                        xaxis_title="Uur van de dag",
                        yaxis_title="Laadtijd (uur)",
                        hovermode="x unified",
                        margin=dict(l=10, r=10, t=30, b=10),
                    )
                    st.plotly_chart(fig_uur_lt, use_container_width=True)

                    fig_uur_en = go.Figure()
                    for d in keuze_dagen:
                        ddata = per_uur[per_uur["datum"] == d]
                        fig_uur_en.add_trace(
                            go.Scatter(x=ddata["uur"], y=ddata["energie_kwh"], mode="lines+markers", name=f"Energie - {d}")
                        )
                    fig_uur_en.update_layout(
                        title="Energie per uur",
                        xaxis_title="Uur van de dag",
                        yaxis_title="Energie (kWh)",
                        hovermode="x unified",
                        margin=dict(l=10, r=10, t=30, b=10),
                    )
                    st.plotly_chart(fig_uur_en, use_container_width=True)

        # ======================================================
        # 2e GRAFIEK: Cumulatief per jaar per brandstof + forecast t/m 2050
        # ======================================================
        st.markdown("---")
        st.subheader("Cumulatief aantal voertuigen per jaar per brandstof (historisch + voorspelling)")

        def _ev_linear_forecast_yearly(series_y: pd.Series, horizon_year: int = 2050) -> pd.Series:
            if series_y.empty:
                return pd.Series(dtype=float)
            years_hist = np.array([d.year for d in series_y.index])
            vals_hist = series_y.values.astype(float)
            mask = ~np.isnan(vals_hist)
            years_hist, vals_hist = years_hist[mask], vals_hist[mask]
            if len(vals_hist) < 2:
                last_year = int(series_y.index.max().year) if len(series_y) else 2025
                future_years = np.arange(last_year + 1, horizon_year + 1)
                last_val = float(vals_hist[-1]) if len(vals_hist) else 0.0
                preds = np.full_like(future_years, last_val, dtype=float)
                return pd.Series(preds, index=pd.to_datetime([f"{y}-12-31" for y in future_years]))
            m, b = np.polyfit(years_hist, vals_hist, 1)
            last_year = int(series_y.index.max().year)
            future_years = np.arange(last_year + 1, horizon_year + 1)
            preds = m * future_years + b
            preds = np.maximum.accumulate(preds)
            preds = np.maximum(preds, vals_hist[-1])
            return pd.Series(preds, index=pd.to_datetime([f"{y}-12-31" for y in future_years]))

        def _ev_prepare_autos_yearly(df_autos: pd.DataFrame) -> pd.DataFrame:
            data = df_autos.copy()
            if "Type" not in data.columns:
                def _bepaal_type(merk, uitvoering):
                    u = str(uitvoering).upper()
                    m = str(merk).upper()
                    elektrische_prefixen = [
                        "FA1FA1CZ","3EER","3EDF","3EDE","2EER","2EDF","2EDE","E11","0AW5","QE2QE2G1","QE1QE1G1","HE1HE1G1","FA1FA1MD"
                    ]
                    if ("BMW I" in m or "PORSCHE" in m or any(u.startswith(p) for p in elektrische_prefixen) or "EV" in u):
                        return "Elektrisch"
                    if ("DIESEL" in u or "TDI" in u or "CDI" in u or "DPE" in u or u.startswith("D")):
                        return "Diesel"
                    return "Benzine"
                data["Type"] = data.apply(lambda r: _bepaal_type(r.get("Merk",""), r.get("Uitvoering","")), axis=1)

            s = data["Datum eerste toelating"].astype(str).str.split(".").str[0]
            dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            dt = dt.fillna(pd.to_datetime(s, errors="coerce"))
            data = data.assign(_datum=dt).dropna(subset=["_datum"])
            data = data[data["_datum"].dt.year > 2010]

            monthly = (
                data.assign(_maand=data["_datum"].dt.to_period("M").dt.to_timestamp())
                    .groupby(["_maand","Type"]).size().unstack(fill_value=0).sort_index()
            )
            monthly_cumu = monthly.cumsum()
            yearly_cumu = monthly_cumu.resample("Y").last().fillna(method="ffill")
            yearly_cumu.index = yearly_cumu.index.to_period("Y").to_timestamp("Y")
            return yearly_cumu

        # We gebruiken df_auto dat je elders al inlaadt via load_data()
        yearly_cumu = _ev_prepare_autos_yearly(df_auto)
        if yearly_cumu.empty:
            st.warning("Geen bruikbare data gevonden na 2010.")
        else:
            eindjaar = st.slider("Voorspellen tot jaar", 2025, 2050, 2050, key="ev_forecast_endyear")
            last_hist_year = int(yearly_cumu.index.max().year)
            eindjaar = max(eindjaar, last_hist_year)

            # Forecast per brandstof
            fc_list = []
            for col in yearly_cumu.columns:
                fc = _ev_linear_forecast_yearly(yearly_cumu[col], horizon_year=eindjaar)
                fc_list.append(fc.rename(col))
            forecast_yearly = pd.concat(fc_list, axis=1) if fc_list else pd.DataFrame()

            # Plotly-figuur
            fig2 = go.Figure()
            for col in yearly_cumu.columns:
                fig2.add_trace(go.Scatter(
                    x=yearly_cumu.index, y=yearly_cumu[col],
                    name=f"{col} – historisch", mode="lines"
                ))
            if not forecast_yearly.empty:
                for col in forecast_yearly.columns:
                    fig2.add_trace(go.Scatter(
                        x=forecast_yearly.index, y=forecast_yearly[col],
                        name=f"{col} – voorspelling", mode="lines", line=dict(dash="dash")
                    ))
            fig2.update_layout(
                title=f"Cumulatief per jaar per brandstof (voorspelling t/m {eindjaar})",
                xaxis_title="Jaar",
                yaxis_title="Cumulatief aantal voertuigen",
                hovermode="x unified",
                height=600
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Downloadknop
            combined = pd.concat([yearly_cumu, forecast_yearly])
            st.download_button(
                "Download data (CSV)",
                combined.to_csv(index_label="datum").encode("utf-8"),
                file_name="cumulatief_per_jaar_met_voorspelling.csv",
                mime="text/csv",
                key="dl_ev_cumul_forecast"
            )











# ------------------- Pagina 3 --------------------------
elif page == "📊 Voorspellend model":
    if not nieuwe_pagina:
        st.markdown("## Voorspellend Model")
        st.markdown("---")
        st.subheader("Voorspelling auto's in Nederland per brandstofcategorie")

        warnings.filterwarnings("ignore")

        # ---------- Interactieve instellingen ----------
        eindjaar = st.slider("Voorspellen tot jaar", 2025, 2050, 2030)
        EINDDATUM = pd.Timestamp(f"{eindjaar}-12-01")

        # ---------- Kopie gebruiken ----------
        df_auto_kopie = df_auto.copy()

        # ---------- Type bepalen ----------
        def bepaal_type(merk, uitvoering):
            u = str(uitvoering).upper()
            m = str(merk).upper()
            if ("BMW I" in m or "PORSCHE" in m or
                u.startswith(("FA1FA1CZ","3EER","3EDF","3EDE","2EER","2EDF","2EDE",
                            "E11","0AW5","QE2QE2G1","QE1QE1G1","HE1HE1G1")) or
                "EV" in u or "FA1FA1MD" in u):
                return "Elektrisch"
            if "DIESEL" in u or "TDI" in u or "CDI" in u or "DPE" in u or u.startswith("D"):
                return "Diesel"
            return "Benzine"

        df_auto_kopie["Type"] = df_auto_kopie.apply(
            lambda r: bepaal_type(r.get("Merk",""), r.get("Uitvoering","")), axis=1
        )

        # ---------- Datums opschonen ----------
        df_auto_kopie["Datum eerste toelating"] = df_auto_kopie["Datum eerste toelating"].astype(str).str.split(".").str[0]
        df_auto_kopie["Datum eerste toelating"] = pd.to_datetime(
            df_auto_kopie["Datum eerste toelating"], format="%Y%m%d", errors="coerce"
        )

        # ---------- Filteren en groeperen ----------
        df_auto_kopie2 = df_auto_kopie.dropna(subset=["Datum eerste toelating"])
        df_auto_kopie2 = df_auto_kopie2[df_auto_kopie2["Datum eerste toelating"].dt.year > 2010]
        df_auto_kopie2["Maand"] = df_auto_kopie2["Datum eerste toelating"].dt.to_period("M").dt.to_timestamp()

        maand_counts_charging = df_auto_kopie2.groupby(["Maand", "Type"]).size().unstack(fill_value=0).sort_index()
        if maand_counts_charging.empty:
            st.error("⚠ Geen bruikbare data gevonden in dataset na 2010.")
            st.stop()

        # ---------- Historische cumulatieven ----------
        cumul_hist_charging = maand_counts_charging.cumsum()
        laatste_hist_maand = cumul_hist_charging.index.max()
        forecast_start = laatste_hist_maand + pd.DateOffset(months=1)
        if forecast_start > EINDDATUM:
            st.error("⚠ Het gekozen eindjaar ligt vóór de laatste beschikbare data. Kies een later jaar.")
            st.stop()

        forecast_index = pd.date_range(start=forecast_start, end=EINDDATUM, freq="MS")
        h = len(forecast_index)
        if h <= 0:
            st.error("⚠ Geen forecast-horizon (controleer eindjaar).")
            st.stop()

        # ---------- Voorspel totaal aantal maandelijkse registraties ----------
        totale_maand = maand_counts_charging.sum(axis=1).astype(float)

        # probeer SARIMAX op totaal
        try:
            if len(totale_maand) >= 24:
                model_tot = SARIMAX(totale_maand, order=(1,1,1), seasonal_order=(1,1,0,12),
                                    enforce_stationarity=False, enforce_invertibility=False)
                fit_tot = model_tot.fit(disp=False)
                pred_tot = fit_tot.get_forecast(steps=h).predicted_mean.values
            else:

                x = np.arange(len(totale_maand))
                m_tot, b_tot = np.polyfit(x, totale_maand, 1)
                future_x = np.arange(len(totale_maand), len(totale_maand) + h)
                pred_tot = b_tot + m_tot * future_x
        except Exception:

            x = np.arange(len(totale_maand))
            m_tot, b_tot = np.polyfit(x, totale_maand, 1)
            future_x = np.arange(len(totale_maand), len(totale_maand) + h)
            pred_tot = b_tot + m_tot * future_x

        pred_tot = np.maximum(pred_tot, 0.0)  
        pred_tot_series = pd.Series(pred_tot, index=forecast_index)

        types = maand_counts_charging.columns.tolist()


        last_counts = maand_counts_charging.iloc[-1].astype(float)
        last_total = last_counts.sum()
        if last_total <= 0:
            last_12 = maand_counts_charging.tail(12).sum().astype(float)
            if last_12.sum() > 0:
                current_shares = (last_12 / last_12.sum()).to_dict()
            else:

                current_shares = {t: 1.0/len(types) for t in types}
        else:
            current_shares = (last_counts / last_total).to_dict()

        non_ev_targets = {}
        for t in types:
            if t == "Elektrisch":
                continue
            cur = current_shares.get(t, 0.0)
            if t == "Benzine":
                non_ev_targets[t] = cur * 0.15   
            elif t == "Diesel":
                non_ev_targets[t] = cur * 0.10  
            else:
                non_ev_targets[t] = cur * 0.25   

        sum_non_ev_targets = sum(non_ev_targets.values())

        ev_target = max(0.75, min(0.98, 1.0 - sum_non_ev_targets))

        if "Elektrisch" not in types:

            scale = 1.0 / sum_non_ev_targets if sum_non_ev_targets > 0 else 1.0 / max(1, len(types))
            for t in non_ev_targets:
                non_ev_targets[t] = non_ev_targets[t] * scale
            ev_target = 0.0


        targets = {}
        for t in types:
            if t == "Elektrisch":
                targets[t] = ev_target
            else:
                targets[t] = non_ev_targets.get(t, 0.0)

        total_target_sum = sum(targets.values())
        if total_target_sum <= 0:

            targets = {t: 1.0/len(types) for t in types}
        else:
            targets = {t: targets[t]/total_target_sum for t in types}


        t_frac = np.linspace(0, 1, h)
        k = 7.0  
        sigmoid = 1.0 / (1.0 + np.exp(-k*(t_frac - 0.5)))  

        share_dict = {}
        for t in types:
            cur = current_shares.get(t, 0.0)
            targ = targets.get(t, 0.0)

            share_dict[t] = cur + (targ - cur) * sigmoid

        shares_df = pd.DataFrame(share_dict, index=forecast_index)


        row_sums = shares_df.sum(axis=1)
        zero_rows = row_sums == 0
        if zero_rows.any():

            fallback = pd.Series(current_shares)
            fallback = fallback / fallback.sum() if fallback.sum() > 0 else fallback.fillna(1.0/len(types))
            shares_df.loc[zero_rows, :] = fallback.values
            row_sums = shares_df.sum(axis=1)
        shares_df = shares_df.div(row_sums, axis=0)


        future_alloc = shares_df.multiply(pred_tot_series, axis=0)  # per maand counts per type


        forecast_median_charging = pd.DataFrame(index=forecast_index, columns=types)
        for col in types:
            future_monthly = future_alloc[col].fillna(0).values
            last_cumul = cumul_hist_charging[col].iloc[-1] if col in cumul_hist_charging.columns else 0
            cumul_forecast = last_cumul + np.cumsum(np.maximum(future_monthly, 0.0))
            forecast_median_charging[col] = cumul_forecast

        # ----------  selectie categorieën ----------
        categorieen = st.multiselect(
            "Kies brandstoftypes om te tonen",
            options=maand_counts_charging.columns.tolist(),
            default=maand_counts_charging.columns.tolist()
        )

        # ---------- Plotly grafiek  ----------
        fig = go.Figure()

        for col in categorieen:
            # Historisch 
            fig.add_trace(go.Scatter(
                x=cumul_hist_charging.index,
                y=cumul_hist_charging[col],
                mode="lines",
                name=f"{col} (historisch)",
                line=dict(width=2)
            ))
            # Voorspelling 
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=forecast_median_charging[col].astype(float),
                mode="lines",
                line=dict(dash="dash", width=3),
                name=f"{col} (voorspelling)"
            ))

        fig.update_layout(
            title=f"Voertuigregistraties per brandstoftype — Historisch + voorspelling tot {eindjaar}",
            xaxis_title="Jaar",
            yaxis_title="Aantal voertuigen (cumulatief)",
            hovermode="x unified",
            height=720  
        )

        st.plotly_chart(fig, use_container_width=True)

       
     #------------NIEUWE PAGINA 3--------------

    else:
        st.write('hey')


# ------------------- Pagina 3 --------------------------
elif page == "📌 Conclusie":
    st.write('hoi ')