git init
git add .
git commit -m "Versão inicial da Calculadora AVLT"
git branch -M main
git remote add origin https://github.com/SEU_UTILIZADOR/NOME_DO_REPOSITORIO.git
git push -u origin main
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="Calculadora AVLT (Cavaco et al., 2015)",
    page_icon="🧠",
    layout="wide"
)

# Título e Cabeçalho
st.title("🧠 Calculadora Neuropsicológica do AVLT")
st.markdown("""
Esta aplicação calcula as pontuações transformadas do **Auditory Verbal Learning Test (AVLT)** ajustadas para **sexo, idade e anos de escolaridade**, de acordo com as normas portuguesas publicadas por **Cavaco et al. (2015)** (*Applied Neuropsychology: Adult*).
""")

# --- FUNÇÕES DE CÁLCULO NORMATIVO ---

def calc_avlt(sex_val, age, edu, e1, e2, e3, e4, e5, ed, rec):
    """
    Algoritmos de ajuste do AVLT segundo Cavaco et al. (2015).
    sex_val: 0 para Mulheres, 1 para Homens.
    """
    # Medidas Derivadas Brutas
    tl_raw = e1 + e2 + e3 + e4 + e5
    lot_raw = tl_raw - (5 * e1)
    ltpr_raw = (ed / e5 * 100) if e5 > 0 else 0.0

    # Algoritmos de Regressão Linear (Tabela 3 - Cavaco et al., 2015)
    z_e1 = (e1 - 6.069 + (sex_val * 0.328) + (age * 0.0185) + (age**2 * 0.000161) - (edu * 0.190) + (edu**2 * 0.00617)) / 1.63
    z_e2 = (e2 - 7.35 + (sex_val * 0.425) - (age * 0.00559) + (age**2 * 0.000413) - (edu * 0.239) + (edu**2 * 0.00672)) / 1.848
    z_e3 = (e3 - 8.81 + (sex_val * 0.487) - (age * 0.0258) + (age**2 * 0.000720) - (edu * 0.206) + (edu**2 * 0.00481)) / 2.12
    z_e4 = (e4 - 9.43 + (sex_val * 0.456) - (age * 0.0520) + (age**2 * 0.00104) - (edu * 0.229) + (edu**2 * 0.00663)) / 2.20
    z_e5 = (e5 - 9.63 + (sex_val * 0.439) - (age * 0.0570) + (age**2 * 0.00107) - (edu * 0.278) + (edu**2 * 0.00808)) / 2.242
    z_ed = (ed - 9.06 + (sex_val * 0.749) - (age * 0.0405) + (age**2 * 0.00106) - (edu * 0.297) + (edu**2 * 0.00928)) / 2.502
    
    # Medidas Derivadas Ajustadas
    z_tl = (tl_raw - 41.3 + (sex_val * 2.14) - (age * 0.122) + (age**2 * 0.00341) - (edu * 1.143) + (edu**2 * 0.0324)) / 8.19
    z_lot = (lot_raw - 10.9 + (sex_val * 0.497) - (age * 0.214) + (age**2 * 0.00260) - (edu * 0.192) + (edu**2 * 0.00158)) / 7.10
    z_ltpr = (ltpr_raw - 81.7 + (sex_val * 4.11) - (age * 0.335) + (age**2 * 0.00536) - (edu * 1.10) + (edu**2 * 0.0466)) / 19.9

    # Algoritmos de Regressão Quantílica para Reconhecimento (Tabela 5 - Cavaco et al., 2015)
    p5_rec = 28.3 - (sex_val * 0.307) - (age * 0.269) + (age**2 * 0.00668) - (age**3 * 0.000058) + (edu * 0.591) - (edu**2 * 0.0224)
    p10_rec = 28.1 - (sex_val * 0.477) - (age * 0.160) + (age**2 * 0.00406) - (age**3 * 0.000037) + (edu * 0.457) - (edu**2 * 0.0169)
    p18_rec = 29.2 - (sex_val * 0.582) - (age * 0.172) + (age**2 * 0.00395) - (age**3 * 0.000033) + (edu * 0.391) - (edu**2 * 0.0133)

    return {
        "Ensaio 1": (e1, z_e1),
        "Ensaio 2": (e2, z_e2),
        "Ensaio 3": (e3, z_e3),
        "Ensaio 4": (e4, z_e4),
        "Ensaio 5": (e5, z_e5),
        "Evocação Diferida": (ed, z_ed),
        "Total Learning (TL)": (tl_raw, z_tl),
        "Learning Over Trials (LOT)": (lot_raw, z_lot),
        "Long-Term % Retention (LTPR)": (ltpr_raw, z_ltpr),
    }, {
        "rec_raw": rec,
        "p5": p5_rec,
        "p10": p10_rec,
        "p18": p18_rec
    }

def get_classification(z):
    if z >= 2.0:
        return "Muito Superior / Excecional"
    elif z >= 1.33:
        return "Superior"
    elif z >= 0.67:
        return "Média-Alta"
    elif z >= -0.67:
        return "Médio / Normal"
    elif z >= -1.33:
        return "Média-Baixa"
    elif z >= -2.0:
        return "Limítrofe / Rebaixado"
    else:
        return "Deficitário / Muito Rebaixado"

# --- SIDEBAR: DADOS DO PACIENTE ---
st.sidebar.header("📋 Dados Demográficos")
nome_paciente = st.sidebar.text_input("Identificador/Nome do Paciente", "Ex: Paciente 01")
sexo = st.sidebar.selectbox("Sexo", ["Feminino", "Masculino"])
idade = st.sidebar.number_input("Idade (anos)", min_value=18, max_value=100, value=45, step=1)
escolaridade = st.sidebar.number_input("Escolaridade (anos concluídos)", min_value=0, max_value=25, value=12, step=1)

sex_val = 0 if sexo == "Feminino" else 1

# --- PAINEL PRINCIPAL: ENTRADA DE RESULTADOS ---
st.header("📥 Pontuações Brutas do AVLT")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Ensaios Imediatos (0-15)")
    e1 = st.number_input("Ensaio 1", 0, 15, 5)
    e2 = st.number_input("Ensaio 2", 0, 15, 7)
    e3 = st.number_input("Ensaio 3", 0, 15, 9)
    e4 = st.number_input("Ensaio 4", 0, 15, 10)
    e5 = st.number_input("Ensaio 5", 0, 15, 11)

with col2:
    st.subheader("Evocação e Reconhecimento")
    ed = st.number_input("Evocação Diferida (0-15)", 0, 15, 9)
    rec = st.number_input("Reconhecimento Diferido (0-30)", 0, 30, 28)

with col3:
    st.subheader("Índices Derivados (Calculados)")
    tl_calc = e1 + e2 + e3 + e4 + e5
    lot_calc = tl_calc - (5 * e1)
    ltpr_calc = (ed / e5 * 100) if e5 > 0 else 0.0

    st.metric("Total Learning (TL)", f"{tl_calc} / 75")
    st.metric("Learning Over Trials (LOT)", f"{lot_calc}")
    st.metric("Long-Term Percent Retention (LTPR)", f"{ltpr_calc:.1f}%")

st.markdown("---")

# --- PROCESSAMENTO E EXIBIÇÃO DE RESULTADOS ---
results_recall, results_rec = calc_avlt(sex_val, idade, escolaridade, e1, e2, e3, e4, e5, ed, rec)

st.header("📊 Resultados Transformados")

table_data = []
for test_name, (raw_val, z_val) in results_recall.items():
    p_val = norm.cdf(z_val) * 100
    scaled_score = max(1, min(19, int(round(10 + 3 * z_val))))
    class_str = get_classification(z_val)
    
    table_data.append({
        "Medida AVLT": test_name,
        "Pontuação Bruta": f"{raw_val:.1f}" if isinstance(raw_val, float) else str(raw_val),
        "Z-Score (Ajustado)": f"{z_val:.2f}",
        "Percentil Est. (%)": f"{p_val:.1f}%",
        "Nota Escalar (M=10, SD=3)": scaled_score,
        "Classificação Neuropsicológica": class_str
    })

df_results = pd.DataFrame(table_data)
st.dataframe(df_results, use_container_width=True, hide_index=True)

# --- ANÁLISE DO RECONHECIMENTO ---
st.subheader("🎯 Teste de Reconhecimento Diferido (Análise Quantílica)")
p5, p10, p18 = results_rec["p5"], results_rec["p10"], results_rec["p18"]
rec_score = results_rec["rec_raw"]

col_rec1, col_rec2 = st.columns([1, 2])

with col_rec1:
    st.markdown(f"**Pontuação de Reconhecimento do Paciente:** `{rec_score}` / 30")
    st.markdown(f"* Cutoff Percentil 5 (P5): **{p5:.1f}**")
    st.markdown(f"* Cutoff Percentil 10 (P10): **{p10:.1f}**")
    st.markdown(f"* Cutoff Percentil 18 (P18): **{p18:.1f}**")

with col_rec2:
    if rec_score < p5:
        st.error("⚠️ **Desempenho Deficitário (< Percentil 5)**: A pontuação situa-se abaixo do limite do 5º percentil para a idade, sexo e escolaridade do doente.")
    elif rec_score < p10:
        st.warning("⚡ **Desempenho Rebaixado / Limítrofe (Percentil 5-10)**: A pontuação situa-se no intervalo do 5º ao 10º percentil.")
    elif rec_score < p18:
        st.info("ℹ️ **Desempenho Média-Baixa (Percentil 10-18)**: A pontuação situa-se entre o 10º e o 18º percentil.")
    else:
        st.success("✅ **Desempenho Normal (≥ Percentil 18)**: A pontuação de reconhecimento encontra-se dentro dos limites normativos esperados.")

st.markdown("---")

# --- GRÁFICOS VISUAIS ---
st.header("📈 Perfil Neuropsicológico e Curva de Aprendizagem")

tab1, tab2 = st.tabs(["Curva de Aprendizagem (E1 a E5 e Evocação)", "Perfil de Z-Scores"])

with tab1:
    fig_learning = go.Figure()
    trials_x = ["Ensaio 1", "Ensaio 2", "Ensaio 3", "Ensaio 4", "Ensaio 5", "Evocação Diferida"]
    raw_y = [e1, e2, e3, e4, e5, ed]
    
    fig_learning.add_trace(go.Scatter(
        x=trials_x, y=raw_y,
        mode="lines+markers+text",
        name="Paciente",
        text=raw_y,
        textposition="top center",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=10)
    ))
    
    fig_learning.update_layout(
        title="Curva de Aprendizagem e Retenção Diferida (Pontuações Brutas)",
        yaxis=dict(title="Nº de Palavras Evocadas", range=[0, 16]),
        xaxis=dict(title="Ensaios"),
        template="plotly_white"
    )
    st.plotly_chart(fig_learning, use_container_width=True)

with tab2:
    fig_z = go.Figure()
    z_measures = ["Ensaio 1", "Ensaio 2", "Ensaio 3", "Ensaio 4", "Ensaio 5", "Evocação Diferida", "Total Learning (TL)", "Learning Over Trials (LOT)", "Long-Term % Retention (LTPR)"]
    z_values = [results_recall[m][1] for m in z_measures]

    # Faixas normativas
    fig_z.add_hrect(y0=-1.0, y1=1.0, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Faixa Média (-1 a +1 DP)")
    fig_z.add_hrect(y0=-2.0, y1=-1.0, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="Rebaixado (-1 a -2 DP)")
    fig_z.add_hrect(y0=-4.0, y1=-2.0, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Deficitário (< -2 DP)")

    fig_z.add_trace(go.Scatter(
        x=z_measures, y=z_values,
        mode="lines+markers+text",
        name="Z-Score",
        text=[f"{z:.2f}" for z in z_values],
        textposition="top center",
        line=dict(color="#d62728", width=3),
        marker=dict(size=10)
    ))

    fig_z.update_layout(
        title="Perfil Ajustado de Z-Scores por Medida do AVLT",
        yaxis=dict(title="Pontuação Z (Ajustada)", range=[-4, 3]),
        template="plotly_white"
    )
    st.plotly_chart(fig_z, use_container_width=True)

# --- EXPORTAÇÃO DOS DADOS ---
st.subheader("💾 Exportar Resultados")
csv = df_results.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descarregar Tabela em CSV",
    data=csv,
    file_name=f"AVLT_{nome_paciente.replace(' ', '_')}.csv",
    mime="text/csv"
)