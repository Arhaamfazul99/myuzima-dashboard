from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Myuzima | Chicago Dashboard', page_icon='🏙️', layout='wide')

st.markdown("""
<style>
/* Keep custom sidebar and metric cards readable in both themes */
[data-testid="stSidebar"] {
    background-color: #f7f9fc !important;
}

[data-testid="stSidebar"] * {
    color: #202938;
}

[data-testid="stMetric"] {
    background-color: #f5f8fc !important;
    border: 1px solid #e2e9f2 !important;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #202938 !important;
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
st.markdown('''<style>
.block-container {padding-top:1.4rem; max-width:1320px}
h1 {font-size:2.1rem !important; letter-spacing:-.035em}
[data-testid="stMetric"] {background:#f5f8fc; padding:16px; border:1px solid #e2e9f2; border-radius:12px}
[data-testid="stMetricLabel"] {font-weight:600}
[data-testid="stSidebar"] {background:#f7f9fc}
</style>''',unsafe_allow_html=True)

st.sidebar.title('MYUZIMA')
st.sidebar.caption('Chicago community health insights')
page=st.sidebar.radio('Explore', ['Overview','Neighborhoods','Healthcare access','Neighborhood insights','Hospital finances','Data & methods'])
years=sorted(ny.year.dropna().astype(int).unique())
year=st.sidebar.selectbox('Reporting year',years,index=len(years)-1)
areas=sorted(ny.ca_name.dropna().unique())
area=st.sidebar.selectbox('Focus community area',['All Chicago']+areas)
st.sidebar.caption('Study area: Chicago’s 77 community areas')

st.title('Chicago Gun Violence & Healthcare Dashboard')
st.caption('Myuzima • Chicago community areas • Descriptive data, not causal predictions')
current=ny[ny.year==year].copy()
selected=current if area=='All Chicago' else current[current.ca_name==area]
cityrow=cy[cy.year==year]

def fmt(v, decimals=0):
    return 'Not available' if pd.isna(v) else f'{v:,.{decimals}f}'
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

if page=='Overview':
    st.subheader(f'{area} · {year}')
    metrics()
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
        top=current.nlargest(10,'shot_total')[['ca_name','shot_total']].sort_values('shot_total')
        chart=px.bar(top,x='shot_total',y='ca_name',orientation='h',labels={'ca_name':'','shot_total':'Shooting victims'},height=490)
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
        b.metric('Listed hospitals', len(hospitals))
        b.caption('Includes facilities outside Chicago')
        c.metric(
            'Areas with drive-time estimates',
            filtered['trauma_drive_min'].notna().sum()
        )
    else:
        record = filtered.iloc[0]
        a.metric(
            'Estimated drive time',
            fmt(record['trauma_drive_min'], 1) + ' min'
            if pd.notna(record['trauma_drive_min'])
            else 'Not available'
        )
        b.metric(
            'Clinics (source estimate)',
            fmt(record['clinics'])
        )
        c.metric(
            'Nearest listed center',
            record['nearest_center']
            if pd.notna(record['nearest_center'])
            else 'Not available'
        )

    st.divider()
    st.subheader('Chicago Healthcare Access Map')

    map_measure = st.radio(
        'Color neighborhoods by',
        ['Estimated drive time', 'Shooting victims'],
        horizontal=True
    )

    if map_measure == 'Estimated drive time':
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
        'Some listed hospitals are outside Chicago.'
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
        'trauma_drive_min': 'Estimated drive (min)',
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

    st.info(
        'These are descriptive healthcare access estimates. '
        'They do not measure actual ambulance dispatch, response, '
        'or patient transport times. Verify hospital designations '
        'and location data before presenting to the client.'
    )

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
                'trauma_drive_min': 'Estimated drive time (minutes)',
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
            'trauma_drive_min': 'Estimated drive (min)',
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
    st.caption('Hospital-wide 2024 cost-report measures; NOT gunshot-specific treatment costs. Missing values remain missing.')
    chicago_names=['Stroger (Cook County)','Mount Sinai','Northwestern Memorial','Rush University Medical Center','Advocate Illinois Masonic','UChicago Medical Center']
    cityh=hospitals[hospitals.hospital.isin(chicago_names)].copy()
    st.info('Showing only the Chicago-located hospitals present in the starter hospital file. This is not a complete list of all Chicago hospitals.')
    fields=['hospital','total_costs','charity_care_cost','uncompensated_care_cost']
    st.dataframe(cityh[fields].rename(columns={'hospital':'Hospital','total_costs':'Total costs ($)','charity_care_cost':'Charity care ($)','uncompensated_care_cost':'Uncompensated care ($)'}),hide_index=True,use_container_width=True)
    plot=cityh.dropna(subset=['total_costs']).sort_values('total_costs')
    if not plot.empty:
        fig=px.bar(plot,x='total_costs',y='hospital',orientation='h',labels={'total_costs':'Hospital-wide total costs ($)','hospital':''},height=410)
        fig.update_traces(marker_color='#245d93')
        st.plotly_chart(fig,use_container_width=True)
    st.caption('Hospital financial values and designations must be checked against original CMS filings before external publication.')

else:
    st.subheader('Data sources and interpretation')
    st.write('This dashboard uses the five data files bundled with the original Myuzima Streamlit starter: neighborhood_by_year.csv, citywide_by_year.csv, neighborhood_profiles.csv, dashboard_map.geojson, and trauma_hospitals.csv.')
    st.write('The dashboard is a descriptive visualization of supplied data. It does not independently verify the underlying source records, calculate real ambulance response times, or estimate gunshot-specific hospital costs.')
    st.write('Neighborhood rates use the population denominator in the supplied data. Review source years and missing values before comparing different indicators.')
    st.write('The year selector controls shooting metrics and the map; hospital financial information is a separate 2024 snapshot.')
    st.subheader('Download the selected shooting data')
    st.download_button('Download selected area/year CSV',selected.to_csv(index=False).encode(),file_name=f'chicago_shootings_{year}.csv',mime='text/csv')

st.divider()
st.caption('Myuzima | Chicago community-area analysis • For research and planning; not emergency response guidance')
