import json

import pandas as pd
import plotly.express as px
import streamlit as st
from unidecode import unidecode

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="COVID 19 en la Provincia de Santa Fe",
    layout="wide",
)

# =====================================
# FUNCIONES
# =====================================


def normalizar(texto):
    return unidecode(str(texto).upper().strip())


@st.cache_data
def cargar_datos():
    df = pd.read_parquet("Covid_Final.parquet")

    df = df[df["residencia_provincia_nombre"].str.upper().eq("SANTA FE")].copy()

    df = df[
        df["residencia_departamento_nombre"].str.upper().ne("SIN ESPECIFICAR")
    ].copy()

    return df


@st.cache_data
def cargar_geojson():
    with open("departamentos-santa_fe.json", encoding="utf-8") as f:
        return json.load(f)


# =====================================
# CARGA
# =====================================

df = cargar_datos()
geojson = cargar_geojson()

# =====================================
# TITULO
# =====================================

st.title("COVID-19 en la Provincia de Santa Fe")

# st.caption("Análisis de los casos registrados en la Provincia de Santa Fe.")

# =====================================
# KPIs
# =====================================

fallecidos_total = (
    df["fallecido"].astype(str).str.upper().isin(["SI", "SÍ", "TRUE"]).sum()
)

internados_total = int(df["internacion"].sum())

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Casos", f"{len(df):,}")

col2.metric(
    "Casos Confirmados",
    f"{len(df[df['clasificacion_resumen'].str.upper().isin(['CONFIRMADO'])]):,}",
)

col3.metric("Fallecidos", f"{fallecidos_total:,}")

col4.metric("Internados", f"{internados_total:,}")


col5.metric(
    "Edad Promedio",
    round(
        df[df["clasificacion_resumen"].str.upper().isin(["CONFIRMADO"])]["edad"].mean(),
        1,
    ),
)

st.divider()

# =====================================
# MAPA + RANKING
# =====================================

modo_mapa = st.radio("Visualizar", ["Casos", "Fallecidos"], horizontal=True)

df_aux = df.copy()

df_aux["departamento"] = df_aux["residencia_departamento_nombre"].apply(normalizar)

if modo_mapa == "Casos":
    mapa_df = df_aux.groupby("departamento").size().reset_index(name="valor")

    titulo_mapa = "Casos por Departamento"

else:
    mapa_df = (
        df_aux[df_aux["fallecido"].astype(str).str.upper().isin(["SI", "SÍ", "TRUE"])]
        .groupby("departamento")
        .size()
        .reset_index(name="valor")
    )

    titulo_mapa = "Fallecidos por Departamento"

ranking = (
    df.groupby("residencia_departamento_nombre")
    .agg(
        Casos=("id_evento_caso", "count"),
        Fallecidos=(
            "fallecido",
            lambda x: x.astype(str).str.upper().isin(["SI", "SÍ", "TRUE"]).sum(),
        ),
    )
    .sort_values("Casos", ascending=False)
    .reset_index()
)

ranking.columns = ["Departamento", "Casos", "Fallecidos"]
col_mapa, col_rank = st.columns([3, 1])

with col_mapa:
    escala_colores = "Blues" if modo_mapa == "Casos" else "Reds"

    fig_mapa = px.choropleth(
        mapa_df,
        geojson=geojson,
        locations="departamento",
        featureidkey="properties.departamento",
        color="valor",
        hover_name="departamento",
        hover_data={"valor": ":,.0f"},
        color_continuous_scale=escala_colores,
    )

    fig_mapa.update_geos(fitbounds="locations", visible=False)

    titulo_barra = "Casos" if modo_mapa == "Casos" else "Fallecidos"

    fig_mapa.update_layout(
        title=titulo_mapa,
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar_title=titulo_barra,
    )
    st.plotly_chart(fig_mapa, use_container_width=True)

with col_rank:
    st.subheader("Top por Departamento")

    st.dataframe(ranking.head(10), hide_index=True, use_container_width=True)

# =====================================
# EVOLUCION TEMPORAL
# =====================================

casos_por_dia = df.groupby("fecha_apertura").size().reset_index(name="casos")

casos_por_dia["media_7d"] = casos_por_dia["casos"].rolling(window=7).mean()

fig_linea = px.line(casos_por_dia, x="fecha_apertura", y="media_7d")

fig_linea.update_layout(title="Evolución Temporal de Casos Registrados", height=450)

fig_linea.update_traces(line=dict(width=3))

st.plotly_chart(fig_linea, use_container_width=True)

# =====================================
# ANALISIS DEMOGRAFICO
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    fig_edad = px.histogram(
        df,
        x="edad",
        nbins=40,
    )

    fig_edad.update_layout(title="Distribución de Edad")

    st.plotly_chart(fig_edad, use_container_width=True)

with col3:
    financiamiento = df["origen_financiamiento"].value_counts().reset_index()

    financiamiento.columns = ["Origen", "Casos"]

    fig_financiamiento = px.pie(
        financiamiento, names="Origen", values="Casos", hole=0.55
    )

    fig_financiamiento.update_layout(
        title="Financiamiento de la Atención", height=450, showlegend=True
    )

    fig_financiamiento.update_traces(textinfo="percent+label")

    st.plotly_chart(fig_financiamiento, use_container_width=True)

with col2:
    # -----------------------------
    # Casos por sexo
    # -----------------------------

    sexo = df["sexo"].astype(str).value_counts().reset_index()

    sexo.columns = ["Sexo", "Casos"]

    fig_sexo = px.bar(
        sexo,
        x="Sexo",
        y="Casos",
        text="Casos",
    )

    fig_sexo.update_layout(title="Distribución de Casos por Sexo", height=450)

    st.plotly_chart(fig_sexo, use_container_width=True)


# =====================================
# ANALISIS EPIDEMIOLOGICO
# =====================================

st.divider()

st.header("Análisis Epidemiológico")

# -----------------------------
# Preparación de datos
# -----------------------------

df_mortalidad = df.copy()

df_mortalidad["grupo_edad"] = pd.cut(
    df_mortalidad["edad"],
    bins=[0, 18, 30, 40, 50, 60, 70, 80, 120],
    labels=[
        "0-18",
        "19-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "80+",
    ],
)

mortalidad = (
    df_mortalidad.groupby("grupo_edad", observed=False)["fallecido"]
    .apply(lambda x: x.astype(str).str.upper().isin(["SI", "SÍ", "TRUE"]).mean() * 100)
    .reset_index(name="porcentaje")
)

internados = df[df["internacion"] == True].copy()

internados["fallecio"] = (
    internados["fallecido"].astype(str).str.upper().isin(["SI", "SÍ", "TRUE"])
)

# -----------------------------
# Mortalidad + Boxplot
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    fig_mortalidad = px.bar(
        mortalidad,
        x="grupo_edad",
        y="porcentaje",
        text_auto=".1f",
    )

    fig_mortalidad.update_layout(
        title="Tasa de Mortalidad por Grupo Etario (%)",
        yaxis_title="% Fallecidos",
        xaxis_title="Grupo Etario",
        height=450,
    )

    st.plotly_chart(
        fig_mortalidad,
        use_container_width=True,
    )

with col2:
    fig_box = px.box(
        internados,
        x="fallecio",
        y="edad",
        points="outliers",
    )

    fig_box.update_layout(
        title="Edad de Pacientes Internados según Fallecimiento",
        xaxis_title="Falleció",
        yaxis_title="Edad",
        height=450,
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True,
    )
# =====================================
# TABLA
# =====================================

with st.expander("Explorar datos"):
    st.dataframe(df.sample(min(500, len(df))), use_container_width=True)
    st.divider()

st.markdown(
    """

        **COVID-19 en la Provincia de Santa Fe**

        Carrera: Licenciatura en Ciencia de Datos

        Materia: Procesamiento, Limpieza y Visualización de Datos"

        Grupo: 7

        **Integrantes**
        - Axel Exequiel Luna
        - Tomás Lottersberger
        - Valentín Hernán Mansilla
        - Sofia Marenoni
        """
)
