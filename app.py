from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import io, re
import numpy as np
import requests
import streamlit.components.v1 as components

st.set_page_config(page_title='CCH Exploration of Gun Violence', page_icon='🏙️', layout='wide')

st.markdown("""
<style>

/* Main dashboard layout */
.block-container {
    padding-top: 1.4rem;
    max-width: 1320px;
}

h1 {
    font-size: 2.1rem !important;
    letter-spacing: -.035em;
}

/* Follow Streamlit's selected theme */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: var(--secondary-background-color) !important;
    border: 1px solid var(--secondary-background-color) !important;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    font-weight: 600;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: var(--text-color) !important;
}

/* Dropdowns and input boxes */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
}

[data-baseweb="select"] input,
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: var(--text-color) !important;
}

/* Sidebar text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text-color) !important;
}

</style>
""", unsafe_allow_html=True)



ROOT = Path(__file__).parent / 'data'
@st.cache_data
def load():
    yearly = pd.read_csv(ROOT/'neighborhood_by_year.csv')
    city = pd.read_csv(ROOT/'citywide_by_year.csv')
    profiles = pd.read_csv(ROOT/'neighborhood_profiles.csv')
    hospitals = pd.read_csv(ROOT/'trauma_hospitals.csv')
    geo = json.loads((ROOT/'dashboard_map.geojson').read_text())
    return yearly,city,profiles,hospitals,geo
ny,cy,profiles,hospitals,geo = load()
for d in (ny,cy,profiles,hospitals):
    d.columns=d.columns.str.strip()
# Only state-designated adult Level I trauma centers (Rush is not one)
hospitals = hospitals[hospitals.adult_level1_trauma.astype(str).str.upper().eq('YES')].reset_index(drop=True)

# ---------------------------------------------------------------- costs and thresholds
COST_PER_PATIENT = 58_000      # average hospital care cost per nonfatal gunshot patient (project benchmark)
CATCHMENT_MILES = 5            # transport beyond ~5 miles / 20 minutes: firearm mortality risk up ~21%

def circle(lat, lon, miles, n=72):
    import numpy as np
    t = np.linspace(0, 2 * np.pi, n)
    dlat = miles / 69.0
    dlon = miles / (69.0 * np.cos(np.radians(lat)))
    return list(lat + dlat * np.sin(t)), list(lon + dlon * np.cos(t))

def care_cost(nonfatal):
    return nonfatal * COST_PER_PATIENT

def money(v):
    if pd.isna(v): return '—'
    return f'${v/1e9:,.2f}B' if v >= 1e9 else f'${v/1e6:,.1f}M'

# ---------------------------------------------------------------- live data
# Chicago Data Portal (shooting victims, updated daily). Cached for 6 hours,
# so the app pulls fresh numbers a few times a day without any extra setup.
SHOOTINGS_URL = 'https://data.cityofchicago.org/resource/gumc-mgzr.csv'

def _key(name):
    return re.sub(r'[^A-Z]', '', str(name).upper())

NAME_TO_NUM = {_key(n): int(c) for n, c in ny[['ca_name', 'ca_num']].drop_duplicates().values}

@st.cache_data(ttl=6 * 3600, show_spinner='Getting the latest shooting data...')
def live_shootings():
    """Shooting victims since Jan 1 last year. Returns None if the portal can't be reached."""
    start = f'{pd.Timestamp.today().year - 1}-01-01'
    try:
        r = requests.get(SHOOTINGS_URL, timeout=30, params={
            '$select': 'date,victimization_primary,community_area',
            '$where': f"gunshot_injury_i = 'YES' AND date >= '{start}'",
            '$limit': 50000})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['fatal'] = (df['victimization_primary'] == 'HOMICIDE').astype(int)
    df['ca_num'] = df['community_area'].map(lambda n: NAME_TO_NUM.get(_key(n)))
    return df.dropna(subset=['ca_num']).astype({'ca_num': int})[['date', 'ca_num', 'fatal']]

def year_to_date(df):
    """This year so far vs the same dates last year."""
    as_of = df['date'].max()
    this_year = df[df['date'].dt.year == as_of.year]
    last_year = df[(df['date'].dt.year == as_of.year - 1) & (df['date'] <= as_of - pd.DateOffset(years=1))]
    return as_of, this_year, last_year

# ---------------------------------------------------------------- model results
# Negative binomial count model, 77 areas x 2021-2025, population offset,
# year effects, errors clustered by area. Best fit of the models tested (AIC 3042),
# all overlap checks (VIF) under 3. Tested on held-out areas: error 15.2 vs 37.9 baseline.
MODEL = pd.DataFrame([
    # label, column in profiles, rate ratio, low, high, unit text, unit size, slider max (in units)
    ('Poverty', 'pct_poverty_2024', 1.032, 1.017, 1.046, '1 percentage point', 1, 10),
    ('Vacant housing', 'pct_vacant_housing', 1.056, 1.027, 1.085, '1 percentage point', 1, 5),
    ('On Medicaid', 'pct_on_medicaid_2025', 1.023, 1.010, 1.037, '1 percentage point', 1, 10),
    ('Evictions (2019)', 'eviction_rate_2019', 1.132, 1.051, 1.219, '1 filing per 100 renter households', 1, 3),
    ('Income gap with wealthiest neighbor', 'border_income_gap', 1.098, 1.034, 1.166, '$10,000', 10_000, 3),
], columns=['factor', 'col', 'rr', 'low', 'high', 'unit', 'scale', 'max_units'])
# ---- Fit the regression live from the master dataset, so new years (2026, 2027...) update it automatically.
MODEL_VARS = {  # column in master_panel -> label used in MODEL
    'pct_poverty': 'Poverty', 'pct_vacant_housing': 'Vacant housing', 'pct_on_medicaid': 'On Medicaid',
    'eviction_rate_2019': 'Evictions (2019)', 'gap10k': 'Income gap with wealthiest neighbor'}

@st.cache_data
def fit_model():
    """Negative binomial regression: shooting victims ~ conditions + year, population offset,
    errors clustered by neighborhood. Also a plain scikit-learn linear regression for comparison."""
    import statsmodels.api as sm, statsmodels.formula.api as smf
    from sklearn.linear_model import LinearRegression
    mp = pd.read_csv(ROOT / 'master_panel.csv')
    mp['gap10k'] = mp.border_income_gap / 10_000
    X = list(MODEL_VARS)
    r = mp[mp.year >= 2021].dropna(subset=X + ['shot_total', 'population']).copy()
    f = 'shot_total ~ ' + ' + '.join(X) + ' + C(year)'
    off = np.log(r.population)
    alpha = smf.negativebinomial(f, data=r, offset=off).fit(disp=0).params['alpha']
    nb = smf.glm(f, data=r, family=sm.families.NegativeBinomial(alpha=alpha), offset=off).fit(
        cov_type='cluster', cov_kwds={'groups': r.ca_num})
    ci = np.exp(nb.conf_int())
    est = pd.DataFrame({'factor': [MODEL_VARS[x] for x in X], 'rr': np.exp(nb.params[X]).values,
                        'low': ci.loc[X, 0].values, 'high': ci.loc[X, 1].values, 'p': nb.pvalues[X].values})
    rate = r.shot_total / r.population * 10_000
    lin = LinearRegression().fit(r[X], rate)
    linear = pd.DataFrame({'factor': [MODEL_VARS[x] for x in X], 'coef': lin.coef_})
    meta = dict(n=len(r), areas=r.ca_num.nunique(), years=f'{r.year.min()}-{r.year.max()}', alpha=alpha, aic=nb.aic,
                lin_r2=lin.score(r[X], rate), lin_neg=int((lin.predict(r[X]) < 0).sum()), lin_intercept=lin.intercept_)
    return est, linear, meta

try:
    _est, LINEAR, MODEL_META = fit_model()
    MODEL = MODEL.drop(columns=['rr', 'low', 'high']).merge(_est, on='factor', how='left')
    MODEL_LIVE = True
except Exception:          # master_panel.csv missing or fit failed: keep the stored results above
    LINEAR, MODEL_META, MODEL_LIVE = None, None, False
LEVERS = MODEL[MODEL.factor != 'On Medicaid']   # Medicaid reflects need for care, not something to reduce

def whatif_ui(key):
    """Pick a condition and how much lower; returns (factor row, units lower, label text)."""
    c1, c2 = st.columns([1, 1.4])
    fac = c1.selectbox('Condition', LEVERS.factor, key=key + '_f')
    f = LEVERS[LEVERS.factor == fac].iloc[0]
    if f.scale == 1:
        amt = c2.slider(f'{fac} lower by ({f.unit.replace("1 ", "").replace("percentage point", "percentage points")})',
                        0.0, float(f.max_units), float(f.max_units) / 2, float(f.max_units) / 10, key=key + '_a')
        text = (f'{amt:g} points lower {fac.lower()}' if 'percentage' in f.unit
                else f'{amt:g} fewer eviction filings per 100 renter households')
    else:
        dollars = c2.slider(f'{fac} smaller by ($)', 0, int(f.max_units * f.scale), 10_000, 5_000, key=key + '_a')
        amt = dollars / f.scale
        text = f'a ${dollars:,} smaller income gap with the wealthiest neighbor'
    return f, amt, text


TABLEAU_URL = 'https://public.tableau.com/views/CCH_17947008091880/Sheet1'
TABLEAU_YEAR_FIELD = 'Year'      # field names exactly as they appear in the Tableau workbook
TABLEAU_AREA_FIELD = 'Ca Name'

st.sidebar.title('CCH Exploration of Gun Violence')
st.sidebar.caption('Cook County Health • Chicago community areas')
page=st.sidebar.radio('Explore', ['Neighborhood explorer','Overview','Neighborhoods','Living conditions','Gun deaths','Risk factors','Healthcare access','Neighborhood insights','Hospital finances']+(['Tableau view'] if TABLEAU_URL else [])+['Data & methods'])
years=sorted(ny.year.dropna().astype(int).unique())
year=st.sidebar.selectbox('Reporting year',years,index=len(years)-1)
areas=sorted(ny.ca_name.dropna().unique())
area=st.sidebar.selectbox('Focus community area',['All Chicago']+areas)
st.sidebar.caption('Study area: Chicago’s 77 community areas')

st.title('CCH Exploration of Gun Violence')
st.caption('Gun violence, neighborhood conditions and the cost of trauma care across Chicago’s 77 community areas')
current=ny[ny.year==year].copy()
selected=current if area=='All Chicago' else current[current.ca_name==area]
cityrow=cy[cy.year==year]

def fmt(v, decimals=0):
    return '—' if pd.isna(v) else f'{v:,.{decimals}f}'
def numsum(df,col):
    return df[col].sum(min_count=1) if col in df else float('nan')

def metrics():
    a,b,c,d=st.columns(4)
    total=numsum(selected,'shot_total'); fatal=numsum(selected,'fatal'); nonfatal=numsum(selected,'nonfatal')
    population=numsum(selected,'population')
    rate=total/population*10000 if pd.notna(population) and population>0 else float('nan')
    a.metric('Shooting victims',fmt(total))
    b.metric('Fatal victims',fmt(fatal))
    c.metric('Nonfatal victims',fmt(nonfatal))
    d.metric('Victims per 10,000',fmt(rate,1))

if page == 'Neighborhood explorer':
    st.markdown("""
<div style="border-left:4px solid #b6383a;padding:10px 16px;margin:4px 0 14px;background:var(--secondary-background-color);border-radius:6px">
<b>How this is different:</b> most gun violence dashboards count shootings after they happen; this one links them to the neighborhood conditions behind them and to what they cost Chicago’s trauma hospitals.<br>
<b>Why it matters:</b> a small set of neighborhoods carries most of the violence, and the public safety-net hospital carries most of the unpaid care.<br>
<b>How it helps hospital leaders:</b> it shows where patients come from, what their care costs, and which conditions go with more shootings, so prevention spending can be targeted and justified.
</div>""", unsafe_allow_html=True)

    layers = {
        'Shooting victims per 10,000': 'shot_rate_per_10k',
        'Unemployment (%)': 'pct_unemployed_2024',
        'Poverty (%)': 'pct_poverty_2024',
        'Vacant housing (%)': 'pct_vacant_housing',
        'On Medicaid (%)': 'pct_on_medicaid_2025',
        'Eviction filings per 100 renter households (2019)': 'eviction_rate_2019',
        'Est. transport time to trauma center (min)': 'trauma_drive_min',
        'Income gap with wealthiest neighbor ($)': 'border_income_gap',
        'Social Vulnerability Index (2022)': 'svi_2022',
    }
    ex = current[['ca_num', 'ca_name', 'shot_total', 'fatal', 'nonfatal', 'shot_rate_per_10k']].merge(
        profiles[['ca_name'] + [c for c in layers.values() if c != 'shot_rate_per_10k'] + ['nearest_center']], on='ca_name', how='left')
    ex['care_cost'] = care_cost(ex.nonfatal)

    c1, c2 = st.columns([1, 1.3])
    names = sorted(ex.ca_name)
    focus = c1.selectbox('Neighborhood', names, index=names.index(area) if area in names else names.index('Austin'))
    shade = c2.selectbox('Color map by', list(layers))
    t1, t2, t3 = st.columns(3)
    show_centers = t1.checkbox('Trauma centers + 5-mile circles', True)
    show_bubbles = t2.checkbox('Shooting victims (bubbles)', True)
    show_far = t3.checkbox('Outline areas >20 min from trauma care', False)
    col = layers[shade]
    row = ex[ex.ca_name == focus].iloc[0]

    left, right = st.columns([1.5, 1], gap='large')
    with left:
        g = json.loads(json.dumps(geo))
        cent = {}
        for f in g['features']:
            f['properties']['ca_num'] = str(f['properties']['ca_num'])
            geom = f['geometry']
            polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
            ring = max((p[0] for p in polys), key=len)
            cent[f['properties']['ca_num']] = (sum(p[1] for p in ring) / len(ring), sum(p[0] for p in ring) / len(ring))
        md = ex.assign(ca_num=ex.ca_num.astype(int).astype(str))
        fig = px.choropleth_mapbox(md, geojson=g, locations='ca_num', featureidkey='properties.ca_num', color=col,
            hover_name='ca_name', hover_data={col: ':,.1f', 'ca_num': False}, labels={col: shade},
            color_continuous_scale='YlGnBu' if col == 'border_income_gap' else 'YlOrRd',
            mapbox_style='carto-positron', center={'lat': 41.84, 'lon': -87.68}, zoom=9.6, opacity=.7, height=620)
        if show_bubbles:
            md['lat'] = md.ca_num.map(lambda k: cent[k][0]); md['lon'] = md.ca_num.map(lambda k: cent[k][1])
            fig.add_trace(go.Scattermapbox(lat=md.lat, lon=md.lon, mode='markers', name='Shooting victims',
                marker=dict(size=(md.shot_total.fillna(0) ** .5) * 2.2 + 3, color='rgba(120,20,30,.55)'),
                text=md.ca_name + ': ' + md.shot_total.fillna(0).astype(int).astype(str) + ' victims',
                hovertemplate='%{text}<extra></extra>'))
        if show_centers:
            for _, h in hospitals.dropna(subset=['lat', 'lon']).iterrows():
                la, lo = circle(h.lat, h.lon, CATCHMENT_MILES)
                fig.add_trace(go.Scattermapbox(lat=la, lon=lo, mode='lines', line=dict(width=1.5, color='#155e75'),
                                               hoverinfo='skip', showlegend=False))
            hh = hospitals.dropna(subset=['lat', 'lon'])
            fig.add_trace(go.Scattermapbox(lat=hh.lat, lon=hh.lon, mode='markers', name='Level I trauma center',
                marker=dict(size=hh.hospital.str.startswith('Stroger').map({True: 18, False: 12}), color='#155e75'),
                text=hh.hospital, hovertemplate='<b>%{text}</b><extra></extra>'))
        def outline(feats, color, width):
            if feats:
                fig.add_trace(go.Choroplethmapbox(geojson={'type': 'FeatureCollection', 'features': feats},
                    locations=[f['properties']['ca_num'] for f in feats], z=[1] * len(feats), featureidkey='properties.ca_num',
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], showscale=False,
                    marker_line_width=width, marker_line_color=color, hoverinfo='skip'))
        if show_far:
            far = set(md[md.trauma_drive_min > 20].ca_num)
            outline([f for f in g['features'] if f['properties']['ca_num'] in far], '#7c3aed', 3)
        outline([f for f in g['features'] if f['properties']['ca_num'] == str(int(row.ca_num))], '#111827', 4)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=''),
                          legend=dict(orientation='h', y=1.01, x=0, yanchor='bottom'))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(f'### {focus} · {year}')
        a, b = st.columns(2)
        a.metric('Shooting victims', fmt(row.shot_total))
        b.metric('Per 10,000 residents', fmt(row.shot_rate_per_10k, 1),
                 f'{row.shot_rate_per_10k / ex.shot_rate_per_10k.mean():.1f}x typical', delta_color='off')
        a.metric('Est. care cost', money(row.care_cost))
        b.metric('To trauma care', f'{row.trauma_drive_min:.0f} min')
        st.caption(f'Care cost = nonfatal victims × ${COST_PER_PATIENT:,} average per patient. '
                   f'Nearest Level I center: {row.nearest_center} (estimated transport time).')
        st.markdown('**All conditions vs the typical area**')
        comp = []
        for lbl, c in layers.items():
            avg = ex[c].mean()
            comp.append({'Measure': lbl.split(' (')[0], 'Ratio': row[c] / avg if avg else float('nan'),
                         'Value': row[c], 'Typical': avg})
        comp = pd.DataFrame(comp)
        bar = px.bar(comp, x='Ratio', y='Measure', orientation='h', height=330,
                     color=comp.Ratio > 1, color_discrete_map={True: '#b6383a', False: '#3989c9'},
                     hover_data={'Value': ':,.1f', 'Typical': ':,.1f', 'Ratio': ':.2f'},
                     labels={'Ratio': 'Times the typical area (1 = typical)', 'Measure': ''})
        bar.add_vline(x=1, line_dash='dot', line_color='gray')
        bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(autorange='reversed'))
        st.plotly_chart(bar, use_container_width=True)

    st.markdown(f'#### {focus} over time: shootings and the conditions you pick')
    trend_opts = {
        'Poverty (%)': 'pct_poverty',
        'Unemployment (%)': 'pct_unemployed',
        'Eviction filings per 100 renter households': 'eviction_filing_rate',
        'On Medicaid (%)': 'pct_on_medicaid',
        'Uninsured (%)': 'pct_uninsured',
        'Rent burdened (%)': 'pct_rent_burdened',
    }
    picks = st.multiselect('Add conditions to the chart', list(trend_opts),
                           default=['Poverty (%)', 'Unemployment (%)', 'Eviction filings per 100 renter households'])
    ts = ny[ny.ca_name == focus].sort_values('year')
    tf = go.Figure()
    tf.add_trace(go.Bar(x=ts.year, y=ts.shot_total, name='People shot', marker_color='rgba(182,56,58,.55)',
                        hovertemplate='%{x}: %{y} people shot<extra></extra>'))
    palette = ['#245d93', '#e08a1e', '#5b4b8a', '#2a9d8f', '#6b7280', '#c2410c']
    for i, lbl in enumerate(picks):
        c = trend_opts[lbl]
        tf.add_trace(go.Scatter(x=ts.year, y=ts[c], name=lbl, yaxis='y2', mode='lines+markers',
                                line=dict(width=2.5, color=palette[i % len(palette)]),
                                hovertemplate='%{x}: %{y:.1f}<extra>' + lbl + '</extra>'))
    tf.add_vline(x=year, line_dash='dot', line_color='gray')
    tf.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), hovermode='x unified',
                     xaxis=dict(dtick=1), yaxis=dict(title='People shot'),
                     yaxis2=dict(title='Condition (%, or per 100 renters)', overlaying='y', side='right', rangemode='tozero'),
                     legend=dict(orientation='h', y=1.08, x=0))
    st.plotly_chart(tf, use_container_width=True)
    notes = []
    if 'Eviction filings per 100 renter households' in picks:
        notes.append('eviction filings are published by neighborhood only through 2019')
    if 'On Medicaid (%)' in picks:
        notes.append('Medicaid enrollment starts in 2021')
    st.caption('Bars = people shot (left axis); lines = conditions (right axis). Dotted line = selected year.'
               + (' Note: ' + '; '.join(notes) + '.' if notes else ''))

    st.markdown(f'#### How shooting rates line up with {shade.split(" (")[0].lower()} across all 77 areas')
    if col == 'shot_rate_per_10k':
        st.caption('Pick a condition in "Color map by" to compare it with shooting rates.')
    else:
        sc = px.scatter(ex, x=col, y='shot_rate_per_10k', size='shot_total', hover_name='ca_name', size_max=38, height=380,
                        labels={col: shade, 'shot_rate_per_10k': 'Shooting victims per 10,000', 'shot_total': 'Victims'})
        sc.update_traces(marker=dict(color='rgba(36,93,147,.55)', line=dict(width=0)))
        sc.add_trace(go.Scatter(x=[row[col]], y=[row.shot_rate_per_10k], mode='markers+text', text=[focus],
                                textposition='top center', marker=dict(size=16, color='#b6383a'), showlegend=False))
        sc.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(sc, use_container_width=True)

    st.markdown(f'#### What if {focus} looked more like other areas?')
    f, amt, text = whatif_ui('ex')
    share = 1 - f.rr ** -amt
    fewer = row.shot_total * share
    fatal_share = row.fatal / row.shot_total if row.shot_total else 0
    x1, x2, x3 = st.columns(3)
    x1.metric('Fewer people shot per year', f'{fewer:,.0f}')
    x2.metric('Of those, deaths', f'{fewer * fatal_share:,.0f}')
    x3.metric('Hospital care cost avoided', money(fewer * (1 - fatal_share) * COST_PER_PATIENT))
    st.caption(f'Based on the regression model: otherwise similar areas with {text} have about {share * 100:.0f}% fewer shooting victims. '
               'This compares neighborhoods; it is not a guarantee of what a single program would achieve.')

elif page=='Overview':
    st.subheader(f'{area} · {year}')
    metrics()
    live = live_shootings()
    if live is not None and len(live):
        as_of, ty, ly = year_to_date(live)
        if area != 'All Chicago':
            num = NAME_TO_NUM.get(_key(area))
            ty, ly = ty[ty.ca_num == num], ly[ly.ca_num == num]
        st.markdown(f'#### Live: {as_of.year} so far (data through {as_of:%b %d, %Y})')
        x, y, z = st.columns(3)
        change = (len(ty) - len(ly)) / len(ly) * 100 if len(ly) else float('nan')
        x.metric('Shooting victims this year', f'{len(ty):,}',
                 None if pd.isna(change) else f'{change:+.0f}% vs same dates last year', delta_color='inverse')
        y.metric('Fatal this year', f'{int(ty.fatal.sum()):,}')
        z.metric('Same dates last year', f'{len(ly):,}')
        st.caption('From the Chicago Data Portal, refreshed every 6 hours. The portal itself runs about a week behind, and recent counts can still change.')
    else:
        st.caption('Live numbers are unavailable right now (data portal could not be reached). Showing yearly data only.')
    st.write('')
    left,right=st.columns([1.6,1],gap='large')
    with left:
        st.subheader('Where are shooting victims concentrated?')
        measure=st.radio('Map shading',['Victims per 10,000 residents','Total shooting victims'],horizontal=True)
        field='shot_rate_per_10k' if measure.startswith('Victims per') else 'shot_total'
        mapdf=current[['ca_num','ca_name',field]].copy()
        mapdf['ca_num']=mapdf.ca_num.astype(str)
        fig=px.choropleth_mapbox(mapdf,geojson=geo,locations='ca_num',featureidkey='properties.ca_num',color=field,
            hover_name='ca_name',hover_data={field:':.1f','ca_num':False},color_continuous_scale='YlOrRd',
            mapbox_style='carto-positron',center={'lat':41.84,'lon':-87.68},zoom=9.65,opacity=.78,height=520,
            labels={field:measure})
        # GeoJSON area identifiers are numeric; align with CSV identifiers.
        for feature in geo['features']:
            feature['properties']['ca_num']=str(feature['properties']['ca_num'])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),coloraxis_colorbar=dict(title='Rate' if field.endswith('10k') else 'Victims'))
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Map shows all 77 areas for the selected year. The community-area filter controls the summary and trends.')
    with right:
        st.subheader('Top 10 community areas')
        top=current.nlargest(10,field)[['ca_name',field]].sort_values(field)
        chart=px.bar(top,x=field,y='ca_name',orientation='h',labels={'ca_name':'',field:measure},height=490)
        chart.update_traces(marker_color='#245d93')
        chart.update_layout(margin=dict(l=0,r=0,t=10,b=0),showlegend=False)
        st.plotly_chart(chart,use_container_width=True)
    st.subheader('How have shootings changed over time?')
    trend=ny if area=='All Chicago' else ny[ny.ca_name==area]
    agg=trend.groupby('year',as_index=False)[['fatal','nonfatal']].sum(min_count=1)
    long=agg.melt(id_vars='year',var_name='Type',value_name='Victims')
    fig=px.bar(long,x='year',y='Victims',color='Type',color_discrete_map={'fatal':'#b6383a','nonfatal':'#3989c9'},barmode='stack',height=340)
    fig.update_layout(margin=dict(l=0,r=0,t=5,b=0),xaxis=dict(dtick=1))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Counts are shooting victims, not distinct shooting incidents. Different indicators may use different source years.')
elif page=='Neighborhoods':
    st.subheader('Compare Chicago community areas')
    metrics()
    columns=['ca_name','shot_total','shot_rate_per_10k','fatal','nonfatal','pct_poverty','pct_unemployed']
    display=selected[columns].sort_values('shot_total',ascending=False).rename(columns={'ca_name':'Community area','shot_total':'Victims','shot_rate_per_10k':'Victims / 10k','fatal':'Fatal','nonfatal':'Nonfatal','pct_poverty':'Poverty (%)','pct_unemployed':'Unemployment (%)'})
    st.dataframe(display,use_container_width=True,hide_index=True)
    st.subheader('Neighborhood trends')
    names=[area] if area!='All Chicago' else st.multiselect('Compare up to five areas',areas,default=['Austin','South Shore','West Englewood'],max_selections=5)
    if names:
        trend=ny[ny.ca_name.isin(names)]
        fig=px.line(trend,x='year',y='shot_total',color='ca_name',markers=True,labels={'year':'Year','shot_total':'Shooting victims','ca_name':'Community area'},height=410)
        st.plotly_chart(fig,use_container_width=True)
    st.info('Poverty and unemployment are descriptive indicators; associations do not establish causation.')


elif page == 'Living conditions':
    st.subheader('Living conditions by community area')
    st.caption('Census (ACS 2020-24), CMAP, Medicaid (2025) and eviction filings (2019, the last year available).')
    options = {
        'Unemployment (%)': 'pct_unemployed_2024',
        'Poverty (%)': 'pct_poverty_2024',
        'Vacant housing (%)': 'pct_vacant_housing',
        'On Medicaid (%)': 'pct_on_medicaid_2025',
        'Eviction filings per 100 renter households (2019)': 'eviction_rate_2019',
        'Uninsured (%)': 'pct_uninsured_2024',
        'Median household income ($)': 'median_hh_income',
        'Hardship index': 'hardship_index_2024',
        'Income gap with wealthiest neighbor ($)': 'border_income_gap',
        'Social Vulnerability Index (2022)': 'svi_2022',
        'Housing cost burdened (%)': 'pct_cost_burdened',
        'Single-parent households (%)': 'pct_single_parent_hh',
        'Households with no vehicle (%)': 'pct_no_vehicle',
    }
    label = st.selectbox('Show', list(options))
    col = options[label]
    lc = profiles[['ca_num', 'ca_name', col, 'shot_rate_per_10k_2025']].copy()
    lc['ca_num'] = lc.ca_num.astype(str)
    lgeo = json.loads(json.dumps(geo))
    for f in lgeo['features']:
        f['properties']['ca_num'] = str(f['properties']['ca_num'])
    left, right = st.columns([1.4, 1], gap='large')
    with left:
        scale = 'YlGnBu' if col == 'median_hh_income' else 'YlOrRd'
        fig = px.choropleth_mapbox(lc, geojson=lgeo, locations='ca_num', featureidkey='properties.ca_num', color=col,
            hover_name='ca_name', hover_data={col: ':,.1f', 'ca_num': False, 'shot_rate_per_10k_2025': False},
            color_continuous_scale=scale, mapbox_style='carto-positron', center={'lat': 41.84, 'lon': -87.68},
            zoom=9.65, opacity=.78, height=560, labels={col: label})
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=''))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('#### Alongside shooting rates (2025)')
        sc = px.scatter(lc, x=col, y='shot_rate_per_10k_2025', hover_name='ca_name', trendline=None,
            labels={col: label, 'shot_rate_per_10k_2025': 'Shooting victims per 10,000'}, height=520)
        sc.update_traces(marker=dict(size=9, color='#245d93', opacity=.75))
        if area != 'All Chicago':
            hi = lc[lc.ca_name == area]
            sc.add_trace(go.Scatter(x=hi[col], y=hi.shot_rate_per_10k_2025, mode='markers+text', text=hi.ca_name,
                textposition='top center', marker=dict(size=15, color='#b6383a'), showlegend=False))
        sc.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(sc, use_container_width=True)
    st.dataframe(profiles[['ca_name'] + list(options.values())].rename(columns={v: k for k, v in options.items()} | {'ca_name': 'Community area'})
                 .sort_values(label, ascending=False), hide_index=True, use_container_width=True)
    st.info('Each dot is one community area. Areas with harder living conditions tend to have more shootings; this shows a pattern, not a cause.')

elif page == 'Gun deaths':
    st.subheader('Gun deaths: homicide vs suicide')
    st.caption('Cook County Medical Examiner records, which include suicides (police shooting data does not). Reliable from 2015.')
    gd = cy.dropna(subset=['gun_homicides', 'gun_suicides'])[['year', 'gun_homicides', 'gun_suicides']]
    if not gd.empty:
        latest = gd.iloc[-1]
        a, b, c = st.columns(3)
        a.metric(f'Gun homicides ({int(latest.year)})', fmt(latest.gun_homicides))
        b.metric(f'Gun suicides ({int(latest.year)})', fmt(latest.gun_suicides))
        c.metric('Suicide share of gun deaths', fmt(latest.gun_suicides / (latest.gun_homicides + latest.gun_suicides) * 100, 0) + '%')
        long = gd.melt(id_vars='year', var_name='Manner', value_name='Deaths')
        long['Manner'] = long.Manner.map({'gun_homicides': 'Homicide', 'gun_suicides': 'Suicide'})
        fig = px.line(long, x='year', y='Deaths', color='Manner', markers=True, height=360,
            color_discrete_map={'Homicide': '#b6383a', 'Suicide': '#5b4b8a'})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
    top15 = profiles.hardest_hit_15.eq('Yes')
    h_share = profiles[top15].gun_homicides_2021_25.sum() / profiles.gun_homicides_2021_25.sum() * 100
    s_share = profiles[~top15].gun_suicides_2021_25.sum() / profiles.gun_suicides_2021_25.sum() * 100
    st.success(f'**{h_share:.0f}% of gun homicides** happen in the 15 hardest-hit neighborhoods, while **{s_share:.0f}% of gun suicides** happen in the other 62. '
               'They are different problems in different places and need different responses.')
    st.markdown('#### By community area, 2021-2025 (per 100,000 residents)')
    dd = profiles[['ca_name', 'gun_homicides_2021_25', 'gun_suicides_2021_25', 'gun_homicide_rate_per_100k', 'gun_suicide_rate_per_100k']].copy()
    dd['More common'] = dd.apply(lambda r: 'Suicide' if r.gun_suicides_2021_25 > r.gun_homicides_2021_25 else 'Homicide', axis=1)
    st.write(f"In **{(dd['More common'] == 'Suicide').sum()} of 77** areas, more people died by gun suicide than gun homicide over 2021-2025.")
    st.dataframe(dd.rename(columns={'ca_name': 'Community area', 'gun_homicides_2021_25': 'Gun homicides', 'gun_suicides_2021_25': 'Gun suicides',
        'gun_homicide_rate_per_100k': 'Homicide rate /100k', 'gun_suicide_rate_per_100k': 'Suicide rate /100k'})
        .sort_values('Homicide rate /100k', ascending=False), hide_index=True, use_container_width=True)

elif page == 'Risk factors':
    st.subheader('Which neighborhood conditions go with more shootings?')
    st.caption('Statistical model of shooting victims across all 77 areas, 2021-2025, comparing areas of the same population in the same year.')
    m = MODEL.copy()
    m['pct'] = (m.rr - 1) * 100
    m['err_hi'] = (m.high - m.rr) * 100
    m['err_lo'] = (m.rr - m.low) * 100
    m['label'] = m.apply(lambda r: f"+{r.pct:.1f}% per {r.unit}", axis=1)
    fig = go.Figure(go.Bar(x=m.pct, y=m.factor, orientation='h', marker_color='#245d93',
        error_x=dict(type='data', array=m.err_hi, arrayminus=m.err_lo), text=m.label, textposition='outside'))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title='Change in expected shooting victims (%)',
        yaxis=dict(autorange='reversed'), xaxis=dict(range=[0, 30]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Lines show the uncertainty range. All five conditions are statistically clear. Read each bar as: comparing two otherwise similar areas, one with this much more of the condition has this % more shooting victims.')

    if MODEL_LIVE:
        st.caption(f"Fitted live from master_panel.csv: {MODEL_META['n']} rows ({MODEL_META['areas']} areas × {MODEL_META['years']}). "
                   "Add a new year of data to the file and these numbers update automatically.")
    with st.expander('The regression equation and what the numbers mean'):
        st.markdown(r"""
**Multivariate regression (count version, negative binomial):**

$$\log(\text{expected people shot}_{n,t}) = \log(\text{population}_{n,t}) + \beta_0 + \beta_1\,\text{Poverty}_{n,t} + \beta_2\,\text{Vacancy}_n + \beta_3\,\text{Medicaid}_{n,t} + \beta_4\,\text{Evictions}_n + \beta_5\,\text{Income gap}_n + \gamma_t$$

- *n* = neighborhood, *t* = year, $\gamma_t$ = a separate baseline for each year (captures citywide spikes like 2021).
- Each coefficient $\beta$ is turned into a **rate ratio** $e^{\beta}$: e.g. 1.032 for poverty means each extra point of poverty goes with **3.2% more** people shot, holding the other conditions and the year constant.
- Errors are clustered by neighborhood because the same area appears in several years.
""")
        if MODEL_LIVE:
            tbl = MODEL[['factor', 'rr', 'low', 'high', 'p']].rename(columns={'factor': 'Condition', 'rr': 'Rate ratio', 'low': 'Low', 'high': 'High', 'p': 'p-value'})
            st.dataframe(tbl.round(3), hide_index=True, use_container_width=True)
            st.markdown('**Why not plain linear regression?** Fitted with scikit-learn on the same data '
                        f"(Y = people shot per 10,000): R² = {MODEL_META['lin_r2']:.2f}, but it predicts **negative shootings for "
                        f"{MODEL_META['lin_neg']} of {MODEL_META['n']} rows**, because shooting counts are skewed and many are near zero, "
                        'so the normal-distribution assumption does not hold. The count model never predicts below zero and does better on neighborhoods it has not seen.')
            st.dataframe(LINEAR.assign(coef=LINEAR.coef.round(2)).rename(columns={'factor': 'Condition', 'coef': 'Linear coefficient (per 10,000 per unit)'}),
                         hide_index=True, use_container_width=True)

    st.markdown('#### Why one neighborhood differs from the typical area')
    pick = area if area != 'All Chicago' else st.selectbox('Community area', sorted(profiles.ca_name), index=sorted(profiles.ca_name).index('Austin'))
    row = profiles[profiles.ca_name == pick].iloc[0]
    parts = []
    for _, f in m.iterrows():
        avg = profiles[f.col].mean()
        diff = (row[f.col] - avg) / f.scale
        parts.append({'Condition': f.factor, pick: row[f.col], 'Typical area': avg,
                      'Effect on expected shootings': (f.rr ** diff - 1) * 100})
    parts = pd.DataFrame(parts)
    bar = px.bar(parts, x='Effect on expected shootings', y='Condition', orientation='h', height=300,
        color=parts['Effect on expected shootings'] > 0, color_discrete_map={True: '#b6383a', False: '#3989c9'},
        labels={'Effect on expected shootings': f'{pick} vs typical area (%)', 'Condition': ''})
    bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(parts.round(1), hide_index=True, use_container_width=True)

    st.markdown('#### Comparison: what if conditions were better?')
    f, amt, text = whatif_ui('rf')
    low, mid, high = (1 / np.array([f.high, f.rr, f.low])) ** amt
    st.write(f'Comparing otherwise similar areas, one with **{text}** is expected to have about '
             f'**{(1 - mid) * 100:.0f}% fewer** shooting victims (range {(1 - high) * 100:.0f}-{(1 - low) * 100:.0f}%).')
    st.info('This compares neighborhoods; it is not a prediction of what one policy would do. These conditions describe '
            'long-standing differences between areas. Nearby areas also tend to share errors (spatial clustering), so treat ranges as approximate.')

elif page == 'Healthcare access':
    st.subheader('Healthcare Access in Chicago')
    st.caption(
        'Explore trauma-center locations and neighborhood healthcare access. '
        'Estimated driving times are not actual ambulance response times.'
    )

    # Neighborhood selection
    focus = st.selectbox(
        'Select a Chicago community area',
        ['All Chicago'] + sorted(profiles['ca_name'].dropna().unique()),
        index=(
            sorted(profiles['ca_name'].dropna().unique()).index(area) + 1
            if area != 'All Chicago'
            else 0
        )
    )

    # Neighborhood healthcare information
    filtered = profiles.copy()
    if focus != 'All Chicago':
        filtered = filtered[filtered['ca_name'] == focus]

    a, b, c = st.columns(3)

    if focus == 'All Chicago':
        a.metric('Community areas', len(filtered))
        b.metric('Adult Level I trauma centers', len(hospitals))
        c.metric(
            'Areas over 20 min from a trauma center',
            int((filtered['trauma_drive_min'] > 20).sum())
        )
    else:
        record = filtered.iloc[0]
        a.metric(
            'Estimated transport time',
            fmt(record['trauma_drive_min'], 1) + ' min'
            if pd.notna(record['trauma_drive_min'])
            else '—'
        )
        b.metric(
            'Clinics (source estimate)',
            fmt(record['clinics'])
        )
        c.metric(
            'Nearest listed center',
            record['nearest_center']
            if pd.notna(record['nearest_center'])
            else '—'
        )

    st.divider()
    st.subheader('Chicago Healthcare Access Map')

    map_measure = st.radio(
        'Color neighborhoods by',
        ['Estimated transport time', 'Shooting victims'],
        horizontal=True
    )

    if map_measure == 'Estimated transport time':
        map_data = profiles[['ca_name', 'trauma_drive_min']].copy()
        value_col = 'trauma_drive_min'
        legend = 'Estimated minutes'
    else:
        map_data = current[['ca_name', 'shot_total']].copy()
        value_col = 'shot_total'
        legend = 'Shooting victims'

    # Match map data to community-area identifiers
    area_ids = ny[['ca_name', 'ca_num']].drop_duplicates()
    map_data = map_data.merge(area_ids, on='ca_name', how='left')
    map_data['ca_num'] = map_data['ca_num'].astype('Int64').astype(str)

    # Work with a copy of the GeoJSON
    healthcare_geo = json.loads(json.dumps(geo))
    for feature in healthcare_geo['features']:
        feature['properties']['ca_num'] = str(
            feature['properties']['ca_num']
        )

    fig = px.choropleth_mapbox(
        map_data,
        geojson=healthcare_geo,
        locations='ca_num',
        featureidkey='properties.ca_num',
        color=value_col,
        hover_name='ca_name',
        color_continuous_scale='YlOrRd',
        mapbox_style='carto-positron',
        center={'lat': 41.84, 'lon': -87.68},
        zoom=10.2,
        opacity=0.65,
        height=650,
        labels={value_col: legend}
    )

    # Add hospital markers
    valid_hospitals = hospitals.dropna(subset=['lat', 'lon']).copy()

    for _, hrow in valid_hospitals.iterrows():
        la, lo = circle(hrow.lat, hrow.lon, CATCHMENT_MILES)
        fig.add_trace(go.Scattermapbox(lat=la, lon=lo, mode='lines', line=dict(width=1.5, color='#155e75'),
                                       hoverinfo='skip', showlegend=False))
    if not valid_hospitals.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=valid_hospitals['lat'],
                lon=valid_hospitals['lon'],
                mode='markers',
                marker=dict(
                    size=13,
                    color='#155e75',
                    opacity=0.95
                ),
                text=valid_hospitals['hospital'],
                hovertemplate=(
                    '<b>%{text}</b><extra></extra>'
                ),
                name='Listed hospitals'
            )
        )

    # Highlight selected neighborhood
    if focus != 'All Chicago':
        selected_id = ny.loc[
            ny['ca_name'] == focus, 'ca_num'
        ].dropna().iloc[0]

        selected_features = [
            f for f in healthcare_geo['features']
            if f['properties']['ca_num'] == str(int(selected_id))
        ]

        if selected_features:
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson={
                        'type': 'FeatureCollection',
                        'features': selected_features
                    },
                    locations=[str(int(selected_id))],
                    z=[1],
                    featureidkey='properties.ca_num',
                    colorscale=[
                        [0, 'rgba(0,0,0,0)'],
                        [1, 'rgba(0,0,0,0)']
                    ],
                    showscale=False,
                    marker_line_width=4,
                    marker_line_color='#17365d',
                    hoverinfo='skip',
                    name='Selected community area'
                )
            )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01,
            xanchor='left',
            x=0
        ),
        coloraxis_colorbar=dict(title=legend)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        'Map colors represent the selected indicator. '
        'Hospital markers show locations from the supplied dataset. '
        'Circles show a 5-mile radius around each Level I center.'
    )

    st.divider()
    st.subheader('Neighborhood Healthcare Comparison')

    table_columns = [
        'ca_name', 'nearest_center',
        'trauma_drive_min', 'clinics'
    ]

    display = filtered[table_columns].rename(columns={
        'ca_name': 'Community area',
        'nearest_center': 'Nearest listed trauma center',
        'trauma_drive_min': 'Est. transport time (min)',
        'clinics': 'Clinic count (source estimate)'
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    st.subheader('Hospital Information')

    hospital_columns = [
        'hospital', 'adult_level1_trauma', 'type'
    ]

    st.dataframe(
        hospitals[hospital_columns].rename(columns={
            'hospital': 'Hospital',
            'adult_level1_trauma': 'Adult Level I (source)',
            'type': 'Hospital type'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption('Transport times are estimated road travel times to the nearest Level I center. Public ambulance response times by neighborhood are not published.')

#this one 

elif page == 'Neighborhood insights':
    st.subheader('Neighborhood Health & Safety Explorer')
    st.caption(
        'Compare shooting rates with estimated access to trauma care '
        'across Chicago community areas.'
    )

    insights = current[
        ['ca_name', 'shot_total', 'shot_rate_per_10k']
    ].merge(
        profiles[
            ['ca_name', 'trauma_drive_min', 'clinics', 'nearest_center']
        ],
        on='ca_name',
        how='left'
    )

    insights['shot_rate_per_10k'] = pd.to_numeric(
        insights['shot_rate_per_10k'], errors='coerce'
    )
    insights['trauma_drive_min'] = pd.to_numeric(
        insights['trauma_drive_min'], errors='coerce'
    )

    st.markdown('#### Compare healthcare access and gun violence')

    valid = insights.dropna(
        subset=['shot_rate_per_10k', 'trauma_drive_min']
    )

    if valid.empty:
        st.warning('No matching shooting and travel-time data available.')
    else:
        fig = px.scatter(
            valid,
            x='trauma_drive_min',
            y='shot_rate_per_10k',
            size='shot_total',
            color='shot_rate_per_10k',
            hover_name='ca_name',
            hover_data=['clinics', 'nearest_center'],
            color_continuous_scale='YlOrRd',
            labels={
                'trauma_drive_min': 'Est. transport time to trauma center (minutes)',
                'shot_rate_per_10k': 'Shooting victims per 10,000',
                'shot_total': 'Shooting victims',
                'clinics': 'Clinics'
            },
            height=500
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            'Each bubble represents a Chicago community area. '
            'Bubble size represents shooting-victim counts. '
            'Travel times are estimates, not actual EMS response times.'
        )

    st.markdown('#### Neighborhood comparison')

    ranked = insights.sort_values(
        'shot_rate_per_10k', ascending=False
    )

    st.dataframe(
        ranked.rename(columns={
            'ca_name': 'Community area',
            'shot_total': 'Shooting victims',
            'shot_rate_per_10k': 'Victims per 10,000',
            'trauma_drive_min': 'Est. transport time (min)',
            'clinics': 'Clinics',
            'nearest_center': 'Nearest listed trauma center'
        }),
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        'Download neighborhood comparison',
        insights.to_csv(index=False),
        file_name=f'myu_neighborhood_insights_{year}.csv',
        mime='text/csv'
    )

    st.info(
        'This comparison identifies patterns for further investigation. '
        'It does not establish causation, measure actual ambulance response '
        'times, or provide a validated neighborhood risk score.'
    )
# to here 
elif page=='Hospital finances':
    st.subheader('Hospital financial information')
    st.caption('Hospital-wide 2024 cost-report measures, not gunshot-specific treatment costs.')
    cityh=hospitals.copy()
    a,b,c=st.columns(3)
    stg=cityh[cityh.hospital.str.startswith('Stroger')].iloc[0]
    a.metric('Stroger charity care',money(stg.charity_care_cost),f'{stg.charity_pct_of_costs:.1f}% of its costs',delta_color='off')
    b.metric('All other Level I centers combined',money(cityh[~cityh.hospital.str.startswith('Stroger')].charity_care_cost.sum()))
    losers=cityh[cityh.net_income<0]
    c.metric('Centers losing money (2024)',f'{len(losers)} of {len(cityh)}')
    c.caption(', '.join(f'{r.hospital}: {money(-r.net_income)} loss' for r in losers.itertuples()))
    st.caption('The 8 adult Level I trauma centers serving Chicago, from 2024 CMS cost reports.')
    fields=['hospital','total_costs','charity_care_cost','uncompensated_care_cost']
    tbl=cityh[fields+['charity_pct_of_costs','net_income']].copy()
    for c in ['total_costs','charity_care_cost','uncompensated_care_cost','net_income']:
        tbl[c]=tbl[c]/1e6
    st.dataframe(tbl.rename(columns={'hospital':'Hospital','total_costs':'Total costs ($M)','charity_care_cost':'Charity care ($M)','uncompensated_care_cost':'Uncompensated care ($M)','charity_pct_of_costs':'Charity % of costs','net_income':'Net income ($M)'}).round(1),hide_index=True,use_container_width=True)
    plot=cityh.dropna(subset=['total_costs']).sort_values('total_costs')
    if not plot.empty:
        fig=px.bar(plot,x='total_costs',y='hospital',orientation='h',labels={'total_costs':'Hospital-wide total costs ($)','hospital':''},height=410)
        fig.update_traces(marker_color='#245d93')
        st.plotly_chart(fig,use_container_width=True)
    share=cityh.dropna(subset=['charity_pct_of_costs']).sort_values('charity_pct_of_costs')
    if not share.empty:
        st.markdown('#### Share of costs spent on charity care')
        fig=px.bar(share,x='charity_pct_of_costs',y='hospital',orientation='h',labels={'charity_pct_of_costs':'Charity care as % of total costs','hospital':''},height=360,text='charity_pct_of_costs')
        fig.update_traces(marker_color='#b6383a',texttemplate='%{text:.1f}%',textposition='outside')
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Stroger, the county safety-net hospital, carries a far larger share of unpaid care than the other trauma centers.')
    st.caption('Source: CMS Hospital Cost Reports, FY2024. Costs are hospital-wide, not gunshot-specific.')

elif page == 'Tableau view':
    st.subheader('Neighborhood map (Tableau)')
    choice = area if area != 'All Chicago' else ''
    flt = f'<viz-filter field="{TABLEAU_YEAR_FIELD}" value="{year}"></viz-filter>'
    if choice:
        flt += f'<viz-filter field="{TABLEAU_AREA_FIELD}" value="{choice}"></viz-filter>'
    components.html(f"""
    <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
    <tableau-viz src="{TABLEAU_URL}" toolbar="bottom" hide-tabs width="100%" height="720">{flt}</tableau-viz>
    """, height=760)
    st.caption('Follows the Reporting year and Focus community area in the sidebar. Built in Tableau Public from the master dataset.')

else:
    st.subheader('Data sources and interpretation')
    st.markdown("""
| Topic | Source | Years |
|---|---|---|
| Shooting victims (fatal + nonfatal) | Chicago Data Portal, Violence Reduction victims (gumc-mgzr) | 2012-2025, live for current year |
| Gun deaths by manner | Cook County Medical Examiner case archive (cjeq-bs86) | 2015-2025 |
| Poverty, unemployment, income, insurance | US Census ACS 5-year estimates, tracts rolled up to community areas | 2012-2024 |
| Vacancy, household profile | CMAP Community Data Snapshots | 2020-24 |
| Social Vulnerability Index | CDC SVI via Chicago Health Atlas | 2022 |
| Medicaid enrollment | Illinois HFS, ZIP codes split to community areas by population | 2021-2025 |
| Eviction filings | Lawyers' Committee for Better Housing | 2010-2019 |
| Hospital finances | CMS Hospital Cost Reports | FY2024 |
| Trauma centers, drive time | Illinois DPH trauma list; estimated drive times | 2026 |
""")
    st.write('Counts are people shot, not incidents. Rates are per 10,000 residents. Drive times are estimates, not ambulance response times. Hospital costs are hospital-wide, not gunshot-specific.')
    st.write('The Risk factors page uses a negative binomial count model (77 areas x 2021-2025) with a population offset, year effects and errors clustered by area. Inputs: poverty, vacant housing, Medicaid enrollment, 2019 eviction filings and the income gap with the wealthiest neighboring area. It was the best-fitting of the models tested and predicted held-out neighborhoods with an average error of 15 victims (vs 38 for a city-average guess). It describes patterns between neighborhoods, not causes.')
    st.subheader('Download the selected shooting data')
    st.download_button('Download selected area/year CSV',selected.to_csv(index=False).encode(),file_name=f'chicago_shootings_{year}.csv',mime='text/csv')

st.divider()
st.caption('CCH Exploration of Gun Violence • Cook County Health • For research and planning')