import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard · Riesgo Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paleta de colores ───────────────────────────────────────────────────────
color_neutro          = "#F4F1EC"
color_neutro_oscuro   = "#EAE6DF"
color_superficie      = "#FFFFFF"
texto_principal       = "#263238"
texto_secundario      = "#6F6A64"
azul_claro            = "#B8D8F0"
azul_medio            = "#7BB8DC"
azul_principal        = "#3D86B8"
color_positivo        = "#58BFA3"
color_alerta          = "#E3B04B"
color_alerta_oscuro   = "#C48E28"

paleta_riesgo = {
    "Low Risk": azul_claro,
    "Moderate Risk": azul_medio,
    "High Risk": azul_principal
}
orden_riesgo = ["Low Risk", "Moderate Risk", "High Risk"]

# ─── Estilos CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #F4F1EC; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #EAE6DF; }
    
    /* Tarjetas KPI */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #EAE6DF;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #6F6A64;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #263238;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #A09B94;
        margin-top: 4px;
    }
    
    /* Título de sección */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #263238;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .section-desc {
        font-size: 0.85rem;
        color: #6F6A64;
        margin-bottom: 20px;
    }
    
    /* Chart card */
    .chart-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #EAE6DF;
    }

    /* Insight box */
    .insight-box {
        background: #FFFFFF;
        border-left: 4px solid #3D86B8;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 0.85rem;
        color: #263238;
    }
    
    /* Header */
    .dashboard-header {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 28px;
        border: 1px solid #EAE6DF;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Carga y preparación de datos ────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("hybrid_student_performance_1200.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    columnas_cat = df.select_dtypes(include="object").columns
    df[columnas_cat] = df[columnas_cat].fillna("Sin dato")

    df["is_high_risk"] = np.where(df["performance_risk_level"] == "High Risk", 1, 0)
    mapa_riesgo = {"Low Risk": 1, "Moderate Risk": 2, "High Risk": 3}
    df["risk_score"] = df["performance_risk_level"].map(mapa_riesgo)

    mapa_horas_estudio = {"Less than 1 hour": 0.5, "1–2 hours": 1.5, "1-2 hours": 1.5, "More than 2 hours": 2.5}
    mapa_horas_sueno   = {"4–5 hours": 4.5, "4-5 hours": 4.5, "6–7 hours": 6.5, "6-7 hours": 6.5, "More than 8 hours": 8.5, "Sin dato": np.nan}
    mapa_asistencia    = {"Less than 50%": 45, "50% – 65%": 57.5, "50% - 65%": 57.5, "66% – 75%": 70.5, "66% - 75%": 70.5, "76% – 85%": 80.5, "76% - 85%": 80.5, "Above 85%": 90}
    mapa_cgpa          = {"5.0 – 6.9": 5.95, "5.0 - 6.9": 5.95, "7.0 – 8.4": 7.70, "7.0 - 8.4": 7.70, "8.5 – 9.4": 8.95, "8.5 - 9.4": 8.95, "9.5 – 10.0": 9.75, "9.5 - 10.0": 9.75}

    df["study_hours_midpoint"] = df["study_hours_daily"].map(mapa_horas_estudio)
    df["sleep_hours_midpoint"] = df["sleep_hours"].map(mapa_horas_sueno)
    df["attendance_midpoint"]  = df["attendance_percentage"].map(mapa_asistencia)
    df["cgpa_midpoint"]        = df["cgpa_category"].map(mapa_cgpa)

    return df


def calcular_kpis(data):
    total = data["student_id"].nunique()
    alto_riesgo = data["is_high_risk"].sum()
    mod_alto = data[data["performance_risk_level"].isin(["Moderate Risk", "High Risk"])]["student_id"].nunique()
    baja_consist = data[data["study_consistency"] == "Rarely"]["student_id"].nunique()
    menos_hora = data[data["study_hours_daily"] == "Less than 1 hour"]["student_id"].nunique()
    estres = data["stress_level"].mean()
    asistencia = data["attendance_midpoint"].mean()

    return {
        "total": total,
        "pct_alto": round(alto_riesgo / total * 100, 1),
        "pct_mod_alto": round(mod_alto / total * 100, 1),
        "pct_baja_consist": round(baja_consist / total * 100, 1),
        "pct_menos_hora": round(menos_hora / total * 100, 1),
        "estres": round(estres, 2),
        "asistencia": round(asistencia, 1),
    }


def plot_config(fig, height=420):
    fig.update_layout(
        height=height,
        plot_bgcolor=color_superficie,
        paper_bgcolor=color_superficie,
        title_x=0.02,
        title_font=dict(color=texto_principal, size=17),
        font=dict(color=texto_principal, family="sans-serif"),
        margin=dict(l=40, r=50, t=65, b=50),
    )
    return fig


# ─── Carga de datos ──────────────────────────────────────────────────────────
df_raw = cargar_datos()


# ─── Sidebar con filtros ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Dashboard Académico")
    st.markdown("---")
    st.markdown("**Filtros globales**")

    programas_disponibles = sorted(df_raw["program_stream"].unique())
    programa_sel = st.multiselect(
        "Programa académico",
        options=programas_disponibles,
        default=programas_disponibles,
        placeholder="Seleccionar programas…"
    )

    años_disponibles = sorted(df_raw["year_class"].unique())
    anio_sel = st.multiselect(
        "Año académico",
        options=años_disponibles,
        default=años_disponibles,
        placeholder="Seleccionar año…"
    )

    generos_disponibles = sorted(df_raw["gender"].unique())
    genero_sel = st.multiselect(
        "Género",
        options=generos_disponibles,
        default=generos_disponibles,
        placeholder="Seleccionar género…"
    )

    st.markdown("---")
    st.markdown("**Navegación**")
    seccion = st.radio(
        "",
        options=[
            "📊 Panorama general",
            "📚 Hábitos de estudio",
            "🌿 Bienestar y satisfacción",
            "🏫 Programas y años",
            "🎯 Acciones prioritarias",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Datos: hybrid student performance · 1200 estudiantes")


# ─── Filtrar datos ────────────────────────────────────────────────────────────
df = df_raw[
    df_raw["program_stream"].isin(programa_sel if programa_sel else programas_disponibles) &
    df_raw["year_class"].isin(anio_sel if anio_sel else años_disponibles) &
    df_raw["gender"].isin(genero_sel if genero_sel else generos_disponibles)
].copy()

if df.empty:
    st.warning("No hay datos con los filtros seleccionados. Ajusta los filtros en el panel izquierdo.")
    st.stop()

kpis = calcular_kpis(df)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — PANORAMA GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
if seccion == "📊 Panorama general":

    st.markdown('<div class="section-title">Panorama general del riesgo académico</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Visión global de los indicadores clave en la población filtrada.</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_items = [
        (c1, "Estudiantes", f"{kpis['total']:,}", "en la selección"),
        (c2, "Alto riesgo", f"{kpis['pct_alto']}%", "de los estudiantes"),
        (c3, "Riesgo mod. o alto", f"{kpis['pct_mod_alto']}%", "requieren atención"),
        (c4, "Baja consistencia", f"{kpis['pct_baja_consist']}%", "estudian raramente"),
        (c5, "< 1h de estudio", f"{kpis['pct_menos_hora']}%", "estudian < 1 hora/día"),
        (c6, "Estrés promedio", f"{kpis['estres']}", "escala 1–10"),
    ]
    for col, label, value, sub in kpi_items:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        riesgo_df = (
            df["performance_risk_level"]
            .value_counts()
            .reindex(orden_riesgo)
            .reset_index()
        )
        riesgo_df.columns = ["performance_risk_level", "estudiantes"]

        fig = px.pie(
            riesgo_df,
            names="performance_risk_level",
            values="estudiantes",
            hole=0.48,
            color="performance_risk_level",
            color_discrete_map=paleta_riesgo,
            title="Distribución por nivel de riesgo",
        )
        fig.update_traces(textinfo="percent+label", sort=False)
        fig = plot_config(fig, height=400)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        riesgo_prog = (
            df.groupby("program_stream")
            .agg(total=("student_id", "count"), high_risk=("is_high_risk", "sum"))
            .reset_index()
        )
        riesgo_prog["pct_high"] = (riesgo_prog["high_risk"] / riesgo_prog["total"] * 100).round(1)
        riesgo_prog = riesgo_prog.sort_values("pct_high", ascending=True)

        fig2 = px.bar(
            riesgo_prog,
            x="pct_high",
            y="program_stream",
            orientation="h",
            text=riesgo_prog["pct_high"].astype(str) + "%",
            title="% Alto riesgo por programa académico",
            labels={"pct_high": "% alto riesgo", "program_stream": "Programa"},
            color="pct_high",
            color_continuous_scale=[[0, azul_claro], [0.5, azul_medio], [1, azul_principal]],
        )
        fig2.update_traces(textposition="outside", marker_line_width=0)
        fig2 = plot_config(fig2, height=400)
        fig2.update_layout(coloraxis_showscale=False)
        fig2.update_xaxes(ticksuffix="%", gridcolor=color_neutro_oscuro)
        fig2.update_yaxes(showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        💡 <b>Lectura rápida:</b> El <b>{kpis['pct_alto']}%</b> de los estudiantes está en alto riesgo académico y el 
        <b>{kpis['pct_mod_alto']}%</b> presenta riesgo moderado o alto. Esto significa que más de la mitad de la población 
        requiere algún nivel de acompañamiento. El estrés promedio de <b>{kpis['estres']}/10</b> es otro indicador de alerta.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — HÁBITOS DE ESTUDIO
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "📚 Hábitos de estudio":

    st.markdown('<div class="section-title">Hábitos de estudio y riesgo académico</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">¿Qué hábitos de estudio se asocian más con el alto riesgo académico?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        consist_df = (
            df.groupby("study_consistency")
            .agg(estudiantes=("student_id", "count"), pct_high=("is_high_risk", "mean"))
            .reset_index()
        )
        consist_df["pct_high"] = (consist_df["pct_high"] * 100).round(1)
        orden_consist = ["Mostly consistent", "Sometimes", "Rarely"]
        consist_df["study_consistency"] = pd.Categorical(consist_df["study_consistency"], categories=orden_consist, ordered=True)
        consist_df = consist_df.sort_values("study_consistency")

        fig = px.line(
            consist_df,
            x="study_consistency",
            y="pct_high",
            markers=True,
            text=consist_df["pct_high"].astype(str) + "%",
            title="Alto riesgo según consistencia de estudio",
            labels={"study_consistency": "Consistencia de estudio", "pct_high": "% en alto riesgo"},
        )
        fig.update_traces(
            line=dict(color=azul_principal, width=3),
            marker=dict(size=12, color=color_alerta, line=dict(color=color_alerta_oscuro, width=1)),
            textposition="top center",
        )
        fig = plot_config(fig)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(range=[0, 90], ticksuffix="%", gridcolor=color_neutro_oscuro)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        horas_df = (
            df.groupby("study_hours_daily")
            .agg(estudiantes=("student_id", "count"), pct_high=("is_high_risk", "mean"))
            .reset_index()
        )
        horas_df["pct_high"] = (horas_df["pct_high"] * 100).round(1)
        orden_horas = ["More than 2 hours", "1–2 hours", "Less than 1 hour"]
        horas_df["study_hours_daily"] = pd.Categorical(horas_df["study_hours_daily"], categories=orden_horas, ordered=True)
        horas_df = horas_df.dropna(subset=["study_hours_daily"]).sort_values("study_hours_daily")

        fig2 = px.bar(
            horas_df,
            x="pct_high",
            y="study_hours_daily",
            orientation="h",
            text=horas_df["pct_high"].astype(str) + "%",
            title="Alto riesgo según horas de estudio diario",
            labels={"study_hours_daily": "Horas de estudio", "pct_high": "% en alto riesgo"},
            color_discrete_sequence=[azul_principal],
        )
        fig2.update_traces(textposition="outside", marker_line_width=0)
        fig2 = plot_config(fig2)
        fig2.update_xaxes(range=[0, 65], ticksuffix="%", gridcolor=color_neutro_oscuro)
        fig2.update_yaxes(showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        focus_df = (
            df.groupby("focus_duration")
            .agg(estudiantes=("student_id", "count"), pct_high=("is_high_risk", "mean"))
            .reset_index()
        )
        focus_df["pct_high"] = (focus_df["pct_high"] * 100).round(1)
        focus_df = focus_df.sort_values("pct_high", ascending=True)

        fig3 = px.bar(
            focus_df,
            x="pct_high",
            y="focus_duration",
            orientation="h",
            text=focus_df["pct_high"].astype(str) + "%",
            title="Alto riesgo según duración del foco de estudio",
            labels={"focus_duration": "Duración de foco", "pct_high": "% en alto riesgo"},
            color_discrete_sequence=[azul_medio],
        )
        fig3.update_traces(textposition="outside", marker_line_width=0)
        fig3 = plot_config(fig3)
        fig3.update_xaxes(ticksuffix="%", gridcolor=color_neutro_oscuro)
        fig3.update_yaxes(showgrid=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        screen_df = (
            df.groupby("screen_time_non_study")
            .agg(estudiantes=("student_id", "count"), pct_high=("is_high_risk", "mean"))
            .reset_index()
        )
        screen_df["pct_high"] = (screen_df["pct_high"] * 100).round(1)
        screen_df = screen_df.sort_values("pct_high", ascending=False)

        fig4 = px.bar(
            screen_df,
            x="screen_time_non_study",
            y="pct_high",
            text=screen_df["pct_high"].astype(str) + "%",
            title="Alto riesgo según tiempo de pantalla (no estudio)",
            labels={"screen_time_non_study": "Tiempo de pantalla", "pct_high": "% en alto riesgo"},
            color_discrete_sequence=[azul_principal],
        )
        fig4.update_traces(textposition="outside", marker_line_width=0)
        fig4 = plot_config(fig4)
        fig4.update_xaxes(showgrid=False)
        fig4.update_yaxes(ticksuffix="%", gridcolor=color_neutro_oscuro)
        st.plotly_chart(fig4, use_container_width=True)

    # Distractor principal
    distractores = (
        df[df["main_distractor"] != "Sin dato"]
        .groupby("main_distractor")
        .agg(estudiantes=("student_id", "count"), pct_high=("is_high_risk", "mean"))
        .reset_index()
    )
    distractores["pct_high"] = (distractores["pct_high"] * 100).round(1)
    distractores = distractores.sort_values("pct_high", ascending=True)

    fig5 = px.bar(
        distractores,
        x="pct_high",
        y="main_distractor",
        orientation="h",
        text=distractores["pct_high"].astype(str) + "%",
        title="Alto riesgo según principal distractor",
        labels={"main_distractor": "Distractor principal", "pct_high": "% en alto riesgo"},
        color="pct_high",
        color_continuous_scale=[[0, azul_claro], [0.5, azul_medio], [1, azul_principal]],
    )
    fig5.update_traces(textposition="outside", marker_line_width=0)
    fig5 = plot_config(fig5, height=380)
    fig5.update_layout(coloraxis_showscale=False)
    fig5.update_xaxes(ticksuffix="%", gridcolor=color_neutro_oscuro)
    fig5.update_yaxes(showgrid=False)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <b>Hallazgo clave:</b> Los estudiantes que estudian <b>raramente de forma consistente</b> y 
        <b>menos de 1 hora diaria</b> muestran los porcentajes más altos de riesgo académico. 
        La consistencia del hábito parece ser más determinante que la cantidad de horas aisladas.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — BIENESTAR Y SATISFACCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "🌿 Bienestar y satisfacción":

    st.markdown('<div class="section-title">Bienestar y satisfacción académica</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">¿Qué factores de bienestar y satisfacción se relacionan con el nivel de riesgo?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sat_df = (
            df[df["academic_satisfaction"] != "Sin dato"]
            .groupby("academic_satisfaction")
            .agg(estudiantes=("student_id", "count"), pct_high=("is_high_risk", "mean"))
            .reset_index()
        )
        sat_df["pct_high"] = (sat_df["pct_high"] * 100).round(1)
        orden_sat = ["Very satisfied", "Satisfied", "Neutral", "Unsatisfied", "Very unsatisfied"]
        sat_df["academic_satisfaction"] = pd.Categorical(sat_df["academic_satisfaction"], categories=orden_sat, ordered=True)
        sat_df = sat_df.sort_values("academic_satisfaction")

        fig = px.line(
            sat_df,
            x="academic_satisfaction",
            y="pct_high",
            markers=True,
            text=sat_df["pct_high"].astype(str) + "%",
            title="Alto riesgo según satisfacción académica",
            labels={"academic_satisfaction": "Satisfacción académica", "pct_high": "% en alto riesgo"},
        )
        fig.update_traces(
            line=dict(color=azul_principal, width=3),
            marker=dict(size=12, color=color_alerta, line=dict(color=color_alerta_oscuro, width=1)),
            textposition="top center",
        )
        fig = plot_config(fig)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(range=[0, 80], ticksuffix="%", gridcolor=color_neutro_oscuro)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sueno_df = (
            df[df["sleep_hours"].isin(["4–5 hours", "4-5 hours", "6–7 hours", "6-7 hours", "More than 8 hours"])]
            .groupby("sleep_hours")
            .agg(estudiantes=("student_id", "count"), pct_high=("is_high_risk", "mean"))
            .reset_index()
        )
        sueno_df["pct_high"] = (sueno_df["pct_high"] * 100).round(1)
        # Normalizar categorías
        sueno_df["sleep_hours"] = sueno_df["sleep_hours"].replace({"4-5 hours": "4–5 hours", "6-7 hours": "6–7 hours"})
        sueno_df = sueno_df.groupby("sleep_hours").mean(numeric_only=True).reset_index()
        orden_sueno = ["More than 8 hours", "6–7 hours", "4–5 hours"]
        sueno_df["sleep_hours"] = pd.Categorical(sueno_df["sleep_hours"], categories=orden_sueno, ordered=True)
        sueno_df = sueno_df.sort_values("sleep_hours")

        fig2 = px.bar(
            sueno_df,
            x="sleep_hours",
            y="pct_high",
            text=sueno_df["pct_high"].round(1).astype(str) + "%",
            title="Alto riesgo según horas de sueño",
            labels={"sleep_hours": "Horas de sueño", "pct_high": "% en alto riesgo"},
            color_discrete_sequence=[azul_principal],
        )
        fig2.update_traces(textposition="outside", marker_line_width=0)
        fig2 = plot_config(fig2)
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(range=[0, 65], ticksuffix="%", gridcolor=color_neutro_oscuro)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(
            df,
            x="performance_risk_level",
            y="stress_level",
            color="performance_risk_level",
            category_orders={"performance_risk_level": orden_riesgo},
            color_discrete_map=paleta_riesgo,
            title="Distribución del estrés según nivel de riesgo",
            labels={"performance_risk_level": "Nivel de riesgo", "stress_level": "Nivel de estrés"},
            points="outliers",
        )
        fig3 = plot_config(fig3)
        fig3.update_layout(showlegend=False)
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(gridcolor=color_neutro_oscuro, dtick=1)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        energia_df = (
            df.groupby("performance_risk_level")
            .agg(energia_prom=("energy_level", "mean"), estres_prom=("stress_level", "mean"))
            .reset_index()
        )
        energia_df["performance_risk_level"] = pd.Categorical(energia_df["performance_risk_level"], categories=orden_riesgo, ordered=True)
        energia_df = energia_df.sort_values("performance_risk_level")

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            name="Energía promedio",
            x=energia_df["performance_risk_level"],
            y=energia_df["energia_prom"].round(2),
            marker_color=color_positivo,
            text=energia_df["energia_prom"].round(2),
            textposition="outside",
        ))
        fig4.add_trace(go.Bar(
            name="Estrés promedio",
            x=energia_df["performance_risk_level"],
            y=energia_df["estres_prom"].round(2),
            marker_color=azul_principal,
            text=energia_df["estres_prom"].round(2),
            textposition="outside",
        ))
        fig4.update_layout(
            title="Energía vs. estrés por nivel de riesgo",
            barmode="group",
            height=420,
            plot_bgcolor=color_superficie,
            paper_bgcolor=color_superficie,
            title_x=0.02,
            title_font=dict(color=texto_principal, size=17),
            font=dict(color=texto_principal),
            margin=dict(l=40, r=40, t=65, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig4.update_xaxes(showgrid=False, title="Nivel de riesgo")
        fig4.update_yaxes(range=[0, 8], gridcolor=color_neutro_oscuro, title="Puntaje promedio (1-10)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <b>Hallazgo clave:</b> La <b>insatisfacción académica</b> y el <b>sueño insuficiente (4–5 horas)</b> 
        se asocian con los mayores porcentajes de alto riesgo. Los estudiantes en alto riesgo presentan 
        mayor estrés y menor energía, lo que sugiere la importancia de intervenciones de bienestar integral.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — PROGRAMAS Y AÑOS
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "🏫 Programas y años":

    st.markdown('<div class="section-title">Concentración de riesgo por programa y año</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">¿En qué programas o años académicos se concentra más el alto riesgo?</div>', unsafe_allow_html=True)

    heatmap_data = (
        df.groupby(["program_stream", "year_class"])
        .agg(estudiantes=("student_id", "count"), pct_high=("is_high_risk", "mean"))
        .reset_index()
    )
    heatmap_data["pct_high"] = (heatmap_data["pct_high"] * 100).round(1)
    heatmap_pivot = heatmap_data.pivot(index="program_stream", columns="year_class", values="pct_high")

    escala_heatmap = [[0.0, "#EAF4FB"], [0.35, azul_claro], [0.70, azul_medio], [1.0, "#1F5F8B"]]

    fig = px.imshow(
        heatmap_pivot,
        text_auto=".1f",
        color_continuous_scale=escala_heatmap,
        title="% de alto riesgo por programa y año académico",
        labels=dict(x="Año académico", y="Programa", color="% alto riesgo"),
        aspect="auto",
    )
    fig.update_traces(texttemplate="%{z:.1f}%", textfont=dict(size=12))
    fig = plot_config(fig, height=520)
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        prog_total = (
            df.groupby("program_stream")
            .agg(
                total=("student_id", "count"),
                high_risk=("is_high_risk", "sum"),
                cgpa_prom=("cgpa_midpoint", "mean"),
                estres_prom=("stress_level", "mean"),
            )
            .reset_index()
        )
        prog_total["pct_high"] = (prog_total["high_risk"] / prog_total["total"] * 100).round(1)
        prog_total = prog_total.sort_values("pct_high", ascending=False)

        fig2 = px.bar(
            prog_total,
            x="program_stream",
            y="pct_high",
            text=prog_total["pct_high"].astype(str) + "%",
            color="pct_high",
            color_continuous_scale=[[0, azul_claro], [0.5, azul_medio], [1, azul_principal]],
            title="% Alto riesgo por programa",
            labels={"program_stream": "Programa", "pct_high": "% alto riesgo"},
        )
        fig2.update_traces(textposition="outside", marker_line_width=0)
        fig2 = plot_config(fig2)
        fig2.update_layout(coloraxis_showscale=False)
        fig2.update_xaxes(showgrid=False, tickangle=20)
        fig2.update_yaxes(ticksuffix="%", gridcolor=color_neutro_oscuro, range=[0, prog_total["pct_high"].max() + 10])
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        anio_total = (
            df.groupby("year_class")
            .agg(total=("student_id", "count"), high_risk=("is_high_risk", "sum"))
            .reset_index()
        )
        anio_total["pct_high"] = (anio_total["high_risk"] / anio_total["total"] * 100).round(1)
        anio_total = anio_total.sort_values("year_class")

        fig3 = px.line(
            anio_total,
            x="year_class",
            y="pct_high",
            markers=True,
            text=anio_total["pct_high"].astype(str) + "%",
            title="Tendencia del alto riesgo por año académico",
            labels={"year_class": "Año académico", "pct_high": "% en alto riesgo"},
        )
        fig3.update_traces(
            line=dict(color=azul_principal, width=3),
            marker=dict(size=12, color=color_alerta, line=dict(color=color_alerta_oscuro, width=1)),
            textposition="top center",
        )
        fig3 = plot_config(fig3)
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(ticksuffix="%", gridcolor=color_neutro_oscuro)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <b>Hallazgo clave:</b> El mapa de calor permite identificar combinaciones críticas de 
        programa + año con concentración de alto riesgo. Usa los filtros de la barra lateral para 
        aislar programas específicos y comparar su comportamiento.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — ACCIONES PRIORITARIAS
# ═══════════════════════════════════════════════════════════════════════════════
elif seccion == "🎯 Acciones prioritarias":

    st.markdown('<div class="section-title">Acciones de acompañamiento prioritarias</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">¿Qué variables tienen mayor impacto en el riesgo y qué acciones deberían priorizarse?</div>', unsafe_allow_html=True)

    # Ranking de variables por diferencia en % high risk
    variables_para_ranking = [
        "academic_satisfaction", "study_hours_daily", "focus_duration",
        "screen_time_non_study", "main_distractor", "study_consistency", "sleep_hours"
    ]

    ranking = []
    for col in variables_para_ranking:
        sub = (
            df[df[col] != "Sin dato"]
            .groupby(col)["is_high_risk"]
            .mean() * 100
        )
        if len(sub) < 2:
            continue
        ranking.append({
            "variable": col,
            "high_risk_max": sub.max(),
            "high_risk_min": sub.min(),
            "diferencia": round(sub.max() - sub.min(), 1),
            "categoria_mas_riesgosa": sub.idxmax(),
            "categoria_menos_riesgosa": sub.idxmin(),
        })

    ranking_df = pd.DataFrame(ranking).sort_values("diferencia", ascending=False)
    ranking_plot = ranking_df.sort_values("diferencia", ascending=True)

    fig = px.bar(
        ranking_plot,
        x="diferencia",
        y="variable",
        orientation="h",
        text=ranking_plot["diferencia"].astype(str) + " pp",
        title="Factores con mayor diferencia en el porcentaje de alto riesgo",
        labels={"diferencia": "Diferencia (puntos porcentuales)", "variable": "Factor analizado"},
        color="diferencia",
        color_continuous_scale=[[0, azul_claro], [0.5, azul_medio], [1, azul_principal]],
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig = plot_config(fig, height=440)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(gridcolor=color_neutro_oscuro)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Detalle por factor")
    ranking_display = ranking_df[["variable", "diferencia", "categoria_mas_riesgosa", "categoria_menos_riesgosa", "high_risk_max", "high_risk_min"]].copy()
    ranking_display.columns = ["Factor", "Diferencia (pp)", "Categoría más riesgosa", "Categoría menos riesgosa", "% Máx. riesgo", "% Mín. riesgo"]
    ranking_display["% Máx. riesgo"] = ranking_display["% Máx. riesgo"].round(1)
    ranking_display["% Mín. riesgo"] = ranking_display["% Mín. riesgo"].round(1)
    st.dataframe(ranking_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🚀 Acciones recomendadas según hallazgos")

    acciones = [
        ("🔴 Alta prioridad", azul_principal, [
            "Intervención inmediata con estudiantes que estudian menos de 1h/día o raramente.",
            "Tutoría personalizada para estudiantes muy insatisfechos académicamente.",
            "Programa de higiene del sueño para quienes duermen 4–5 horas.",
        ]),
        ("🟡 Media prioridad", color_alerta, [
            "Talleres de gestión del tiempo y hábitos de estudio consistentes.",
            "Estrategias para reducir distractores (redes sociales, pantallas).",
            "Seguimiento de estudiantes con estrés elevado (≥ 7/10).",
        ]),
        ("🟢 Monitoreo continuo", color_positivo, [
            "Rastreo mensual de los KPIs de riesgo por programa y año.",
            "Encuesta periódica de satisfacción académica y bienestar.",
            "Identificación temprana de programas con tendencia creciente de alto riesgo.",
        ]),
    ]

    for titulo, color, items in acciones:
        st.markdown(f"""
        <div style="background:#FFFFFF; border-left: 4px solid {color}; border-radius: 0 8px 8px 0; 
                    padding: 14px 18px; margin-bottom: 12px;">
            <b style="color:{color}">{titulo}</b>
            <ul style="margin: 8px 0 0 0; padding-left: 18px; color: {texto_principal}; font-size: 0.87rem;">
                {"".join(f"<li style='margin-bottom:4px'>{item}</li>" for item in items)}
            </ul>
        </div>
        """, unsafe_allow_html=True)