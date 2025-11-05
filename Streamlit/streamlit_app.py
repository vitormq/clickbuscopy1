import os
import gc
import traceback
from pathlib import Path
import datetime

import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from faker import Faker
from matplotlib import pyplot as plt

# -------------------------------
# Config e caminhos
# -------------------------------
st.set_page_config(page_title="Previsão de Próxima Compra", layout="wide")

# base = raiz do repositório (Streamlit/ está um nível abaixo)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Dados"
MODEL_DIR = BASE_DIR / "Modelos"

REQUIRED_FILES = {
    "parquet": [
        DATA_DIR / "dataframe.parquet",
        DATA_DIR / "cb_previsao_data.parquet",
        DATA_DIR / "cb_previsao_trecho.parquet",
        DATA_DIR / "classes.parquet",
    ],
    "models": [
        MODEL_DIR / "xgboost_model_dia_exato.json",
        MODEL_DIR / "xgboost_model_trecho.json",
    ],
}

def _exists_all(paths):
    return all(p.is_file() for p in paths)

def _human_size(bytes_):
    for unit in ["B","KB","MB","GB","TB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} PB"

def _total_size(paths):
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except Exception:
            pass
    return total

# -------------------------------
# Cache de dados e modelos
# -------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_data():
    """Carrega os 4 Parquet (lento e pesado)."""
    df_compras = pd.read_parquet(DATA_DIR / "dataframe.parquet", engine="pyarrow")
    features_dia = pd.read_parquet(DATA_DIR / "cb_previsao_data.parquet", engine="pyarrow")
    features_trecho = pd.read_parquet(DATA_DIR / "cb_previsao_trecho.parquet", engine="pyarrow")
    classes = pd.read_parquet(DATA_DIR / "classes.parquet", engine="pyarrow")
    return df_compras, features_dia, features_trecho, classes

@st.cache_resource(show_spinner=False)
def load_models():
    """Carrega os 2 modelos XGBoost como Booster."""
    modelo_dia = xgb.Booster()
    modelo_destino = xgb.Booster()
    modelo_dia.load_model(str(MODEL_DIR / "xgboost_model_dia_exato.json"))
    modelo_destino.load_model(str(MODEL_DIR / "xgboost_model_trecho.json"))
    return modelo_dia, modelo_destino

# -------------------------------
# UI
# -------------------------------
st.title("Previsão de Próxima Compra por Cliente")

with st.expander("🔧 Diagnóstico rápido (arquivos esperados)", expanded=False):
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Parquet**")
        for p in REQUIRED_FILES["parquet"]:
            st.write(("✅" if p.is_file() else "❌"), p.relative_to(BASE_DIR))
        st.write("Tamanho total:", _human_size(_total_size(REQUIRED_FILES["parquet"])))
    with cols[1]:
        st.markdown("**Modelos**")
        for p in REQUIRED_FILES["models"]:
            st.write(("✅" if p.is_file() else "❌"), p.relative_to(BASE_DIR))
        st.write("Tamanho total:", _human_size(_total_size(REQUIRED_FILES["models"])))

# Botão explícito para evitar OOM no health-check
carregar = st.button("🚀 Carregar dados e modelos")

if not carregar:
    st.info("Clique em **Carregar dados e modelos** para iniciar (carregamento preguiçoso para evitar estouro de memória no deploy).")
    st.stop()

# Verificações antes de carregar
missing = [p for p in REQUIRED_FILES["parquet"] + REQUIRED_FILES["models"] if not p.is_file()]
if missing:
    st.error("Arquivos ausentes:\n" + "\n".join(f"- {m.relative_to(BASE_DIR)}" for m in missing))
    st.stop()

# Carregar
try:
    with st.spinner("🔄 Carregando modelos..."):
        modelo_dia, modelo_destino = load_models()
    with st.spinner("🔄 Carregando dados (parquet)..."):
        df_compras_cliente, features_dia, features_trecho, classes = load_data()
    st.success("✅ Tudo carregado.")
except Exception as e:
    st.error(f"Falha ao carregar dados/modelos: {e}")
    st.exception(e)
    print("TRACEBACK:\n", traceback.format_exc())
    st.stop()

# -------------------------------
# Lógica principal
# -------------------------------
# Nomes fake
Faker.seed(42)
fake = Faker("pt_BR")

unique_ids = features_trecho["id_cliente"].dropna().unique().tolist()
if len(unique_ids) == 0:
    st.warning("Nenhum id_cliente encontrado em cb_previsao_trecho.parquet.")
    st.stop()

fake_names = [fake.name() for _ in unique_ids]
id_to_name = dict(zip(unique_ids, fake_names))
name_to_id = dict(zip(fake_names, unique_ids))

selected_fake_name = st.selectbox("Selecione o cliente", fake_names)
id_cliente = name_to_id[selected_fake_name]

# Previsão de dia
input_dia = features_dia[features_dia["id_cliente"] == id_cliente].drop(columns=["id_cliente"], errors="ignore")
if input_dia.empty:
    st.error("Cliente não possui features de dia em cb_previsao_data.parquet.")
    st.stop()

input_dia_dmatrix = xgb.DMatrix(input_dia)
data_prevista = float(modelo_dia.predict(input_dia_dmatrix)[0])

# Previsão de destino
input_trecho = features_trecho[features_trecho["id_cliente"] == id_cliente].drop(columns=["id_cliente"], errors="ignore")
if input_trecho.empty:
    st.error("Cliente não possui features de trecho em cb_previsao_trecho.parquet.")
    st.stop()

input_trecho_dmatrix = xgb.DMatrix(input_trecho)
probs = modelo_destino.predict(input_trecho_dmatrix)[0]
destino_pred = int(np.argmax(probs))

# Classes/trechos fake
df_compras_cliente["Trechos"] = df_compras_cliente["origem_ida"].astype(str) + "_" + df_compras_cliente["destino_ida"].astype(str)

todos_ids = set()
for item in df_compras_cliente["Trechos"]:
    if isinstance(item, str) and "_" in item:
        origem, destino = item.split("_", 1)
        todos_ids.update([origem, destino])

Faker.seed(42)
def gerar_cidade_fake(id_unico: str) -> str:
    return fake.city()

id_para_cidade = {id_: gerar_cidade_fake(id_) for id_ in todos_ids}

def mapear_para_cidades(par: str) -> str:
    if not isinstance(par, str) or "_" not in par:
        return par
    origem, destino = par.split("_", 1)
    return f"{id_para_cidade.get(origem, origem)} -> {id_para_cidade.get(destino, destino)}"

classes = classes.copy()
classes["trecho_fake"] = classes["Trechos"].apply(mapear_para_cidades)

cliente_data = df_compras_cliente[df_compras_cliente["id_cliente"] == id_cliente].copy()

data_final = datetime.date.today() + datetime.timedelta(days=int(round(data_prevista)))
st.write(f"📅 Data provável da próxima compra: **{data_final.strftime('%Y-%m-%d')}**")
st.write(f"🚌 Trecho provável da próxima compra: **{classes.iloc[destino_pred]['trecho_fake']}**")

col1, col2, col3 = st.columns(3)
try:
    col1.metric("🛒 Compras (total)", int(cliente_data["qtd_total_compras"].iloc[0]))
    col2.metric("📊 Intervalo médio (dias)", int(cliente_data["intervalo_medio_dias"].iloc[0]))
    col3.metric("💳 Ticket médio (R$)", float(cliente_data["vl_medio_compra"].iloc[0]))
except Exception:
    pass

st.metric("Cluster", str(cliente_data.get("cluster_name", pd.Series(["?"])).iloc[0]))

# Histórico
st.subheader("🛒 Histórico de compras do cliente")
cliente_data["trecho_fake"] = cliente_data["Trechos"].apply(mapear_para_cidades)
cliente_data = cliente_data.sort_values("data_compra", ascending=False)
if "data_compra" in cliente_data.columns and not pd.api.types.is_string_dtype(cliente_data["data_compra"]):
    try:
        cliente_data["data_compra"] = pd.to_datetime(cliente_data["data_compra"]).dt.strftime("%Y-%m-%d")
    except Exception:
        pass
ren = {
    "data_compra": "Data",
    "trecho_fake": "Trecho",
    "qnt_passageiros": "Quantidade de Passageiros",
    "vl_total_compra": "Valor do Ticket (R$)",
}
st.dataframe(
    cliente_data.rename(columns=ren)[["Data","Trecho","Quantidade de Passageiros","Valor do Ticket (R$)"]]
      .dropna(axis=1, how="all"),
    use_container_width=True,
)

# SHAP (lazy e protegido)
try:
    st.subheader("🔍 Explicação da previsão da data (SHAP)")
    exp_dia = shap.Explainer(modelo_dia)
    sv_dia = exp_dia(input_dia)
    fig1, _ = plt.subplots()
    shap.plots.waterfall(sv_dia[0], show=False)
    st.pyplot(fig1)

    st.subheader("🔍 Explicação da previsão do trecho (SHAP)")
    exp_dest = shap.Explainer(modelo_destino)
    sv_dest = exp_dest(input_trecho)
    sv_classe = sv_dest[0, :, destino_pred]
    fig2, _ = plt.subplots()
    shap.plots.waterfall(sv_classe, show=False)
    st.pyplot(fig2)
except Exception as e:
    st.warning(f"Não foi possível gerar gráficos SHAP: {e}")
    print("SHAP TRACEBACK:\n", traceback.format_exc())

# Libera memoria
gc.collect()
