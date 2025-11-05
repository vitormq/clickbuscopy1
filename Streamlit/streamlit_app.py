# Streamlit/streamlit_app.py
# -----------------------------------------------------------------------------
# App de previsão de próxima compra (carrega dados/modelos fora da pasta Streamlit)
# Pastas esperadas na RAIZ do repositório:
#   Dados/    -> *.parquet
#   Modelos/  -> *.json  (modelos XGBoost)
# -----------------------------------------------------------------------------

from pathlib import Path
import datetime
import random

import numpy as np
import pandas as pd
import streamlit as st
from faker import Faker
import xgboost as xgb

# -----------------------------------------------------------------------------
# Config da página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Previsão de Próxima Compra",
    page_icon="🧭",
    layout="wide",
)

st.title("Previsão de Próxima Compra por Cliente")
st.caption("Carrega arquivos de **Dados/** e **Modelos/** na raiz do repositório.")

# -----------------------------------------------------------------------------
# Caminhos robustos (independem do diretório corrente)
#   Este arquivo está em:   <repo>/Streamlit/streamlit_app.py
#   Precisamos chegar na raiz: parents[1]
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]  # sobe da pasta Streamlit/ para a RAIZ

DADOS_DIR = REPO_ROOT / "Dados"
MODELOS_DIR = REPO_ROOT / "Modelos"

ARQUIVOS_DADOS = {
    "dataframe": DADOS_DIR / "dataframe.parquet",
    "features_dia": DADOS_DIR / "cb_previsao_data.parquet",
    "features_trecho": DADOS_DIR / "cb_previsao_trecho.parquet",
    "classes": DADOS_DIR / "classes.parquet",
}

ARQUIVOS_MODELOS = {
    "dia_exato": MODELOS_DIR / "xgboost_model_dia_exato.json",
    "trecho": MODELOS_DIR / "xgboost_model_trecho.json",
}

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _assert_files_exist(arquivos: dict) -> list[str]:
    """Retorna lista com caminhos que não existem."""
    faltando = []
    for _, path in arquivos.items():
        if not Path(path).exists():
            faltando.append(str(path))
    return faltando


@st.cache_data(show_spinner=False, ttl=60 * 60)
def carregar_parquets():
    """Carrega dataframes .parquet."""
    # Validação de existência
    faltando = _assert_files_exist(ARQUIVOS_DADOS)
    if faltando:
        raise FileNotFoundError(
            "Alguns arquivos de **Dados/** não foram encontrados:\n- "
            + "\n- ".join(faltando)
        )

    df_compras = pd.read_parquet(ARQUIVOS_DADOS["dataframe"], engine="pyarrow")
    features_dia = pd.read_parquet(ARQUIVOS_DADOS["features_dia"], engine="pyarrow")
    features_trecho = pd.read_parquet(ARQUIVOS_DADOS["features_trecho"], engine="pyarrow")
    classes = pd.read_parquet(ARQUIVOS_DADOS["classes"], engine="pyarrow")
    return df_compras, features_dia, features_trecho, classes


@st.cache_resource(show_spinner=False)
def carregar_modelos():
    """Carrega modelos XGBoost a partir de arquivos .json."""
    faltando = _assert_files_exist(ARQUIVOS_MODELOS)
    if faltando:
        raise FileNotFoundError(
            "Alguns arquivos de **Modelos/** não foram encontrados:\n- "
            + "\n- ".join(faltando)
        )

    modelo_dia = xgb.Booster()
    modelo_dia.load_model(str(ARQUIVOS_MODELOS["dia_exato"]))

    modelo_destino = xgb.Booster()
    modelo_destino.load_model(str(ARQUIVOS_MODELOS["trecho"]))

    return modelo_dia, modelo_destino


def nomes_fakes_por_cliente(features_trecho: pd.DataFrame) -> tuple[list[str], dict, dict]:
    Faker.seed(42)
    fake = Faker("pt_BR")
    unique_ids = features_trecho["id_cliente"].unique()
    fake_names = [fake.name() for _ in unique_ids]
    id_to_name = dict(zip(unique_ids, fake_names))
    name_to_id = dict(zip(fake_names, unique_ids))
    return fake_names, id_to_name, name_to_id


def make_trechos_fake(df_compras: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Gera nomes de cidades fictícias mas estáveis, e adiciona nas tabelas."""
    Faker.seed(42)
    fake = Faker("pt_BR")

    # Construir conjunto de ids de origem/destino
    df_temp = df_compras.copy()
    df_temp["Trechos"] = df_temp["origem_ida"] + "_" + df_temp["destino_ida"]

    todos_ids = set()
    for item in df_temp["Trechos"]:
        o, d = item.split("_")
        todos_ids.update([o, d])

    def gerar_cidade_fake(id_unico: str) -> str:
        random.seed(hash(id_unico))
        return fake.city()

    id_para_cidade = {id_: gerar_cidade_fake(id_) for id_ in todos_ids}

    def mapear_para_cidades(par: str) -> str:
        o, d = par.split("_")
        return f"{id_para_cidade[o]} -> {id_para_cidade[d]}"

    classes = classes.copy()
    classes["trecho_fake"] = classes["Trechos"].apply(mapear_para_cidades)

    df_compras = df_compras.copy()
    df_compras["Trechos"] = df_compras["origem_ida"] + "_" + df_compras["destino_ida"]
    df_compras["trecho_fake"] = df_compras["Trechos"].apply(mapear_para_cidades)

    return df_compras, classes


# -----------------------------------------------------------------------------
# Carregamento automático (sem botão)
# -----------------------------------------------------------------------------
with st.spinner("🔄 Carregando dados e modelos…"):
    try:
        df_compras, features_dia, features_trecho, classes = carregar_parquets()
        modelo_dia, modelo_destino = carregar_modelos()
    except FileNotFoundError as e:
        st.error(f"🚫 {e}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

# Enriquecimentos de exibição
df_compras, classes = make_trechos_fake(df_compras, classes)

# -----------------------------------------------------------------------------
# UI – Seleção de cliente
# -----------------------------------------------------------------------------
fake_names, id_to_name, name_to_id = nomes_fakes_por_cliente(features_trecho)
selected_fake_name = st.selectbox("👤 Selecione o cliente", fake_names, index=0)
id_cliente = name_to_id[selected_fake_name]

# -----------------------------------------------------------------------------
# Predições
# -----------------------------------------------------------------------------
# Previsão de dia
input_dia = features_dia[features_dia["id_cliente"] == id_cliente].drop(columns=["id_cliente"])
dmatrix_dia = xgb.DMatrix(input_dia)
prev_dias = float(modelo_dia.predict(dmatrix_dia)[0])
data_prevista = datetime.date.today() + datetime.timedelta(days=int(round(prev_dias)))

# Previsão de destino (classe)
input_trecho = features_trecho[features_trecho["id_cliente"] == id_cliente].drop(columns=["id_cliente"])
dmatrix_trecho = xgb.DMatrix(input_trecho)
probs = modelo_destino.predict(dmatrix_trecho)[0]
destino_pred_idx = int(np.argmax(probs))

# -----------------------------------------------------------------------------
# Saídas
# -----------------------------------------------------------------------------
st.success(f"📅 **Data provável da próxima compra:** {data_prevista.strftime('%Y-%m-%d')}")
st.info(f"✈️ **Trecho provável:** {classes.iloc[destino_pred_idx]['trecho_fake']}")

# Métricas
cliente_data = df_compras[df_compras["id_cliente"] == id_cliente].copy()
col1, col2, col3 = st.columns(3)
col1.metric("🛒 Total de compras", int(cliente_data["qtd_total_compras"].iloc[0]))
col2.metric("📊 Intervalo médio (dias)", int(cliente_data["intervalo_medio_dias"].iloc[0]))
col3.metric("💳 Ticket médio (R$)", int(cliente_data["vl_medio_compra"].iloc[0]))
st.metric("Cluster", str(cliente_data["cluster_name"].iloc[0]))

# Tabela – histórico
st.subheader("🛍️ Histórico de compras do cliente")
cliente_hist = (
    cliente_data.sort_values("data_compra", ascending=False)
    .assign(data_compra=lambda d: d["data_compra"].dt.strftime("%Y-%m-%d"))
    .rename(
        columns={
            "data_compra": "Data",
            "trecho_fake": "Trecho",
            "qnt_passageiros": "Quantidade de Passageiros",
            "vl_total_compra": "Valor do Ticket (R$)",
        }
    )[["Data", "Trecho", "Quantidade de Passageiros", "Valor do Ticket (R$)"]]
)
st.dataframe(cliente_hist, use_container_width=True, height=380)

# -----------------------------------------------------------------------------
# SHAP (opcional – só calcula se o usuário pedir, para economizar memória)
# -----------------------------------------------------------------------------
with st.expander("🔍 Explicações (SHAP) – clique para calcular"):
    calc_shap = st.checkbox("Calcular SHAP para as predições", value=False)
    if calc_shap:
        try:
            import shap
            import matplotlib.pyplot as plt

            st.caption("Isso pode levar alguns segundos.")
            # Dia
            expl_dia = shap.Explainer(modelo_dia)
            shap_values_dia = expl_dia(input_dia)

            st.write("**Impacto das variáveis para a data prevista**")
            fig1 = plt.figure()
            shap.plots.waterfall(shap_values_dia[0], show=False)
            st.pyplot(fig1, clear_figure=True)

            # Trecho (classe predita)
            expl_tr = shap.Explainer(modelo_destino)
            shap_values_tr = expl_tr(input_trecho)
            shap_value_classe = shap_values_tr[0, :, destino_pred_idx]

            st.write("**Impacto das variáveis para o trecho previsto**")
            fig2 = plt.figure()
            shap.plots.waterfall(shap_value_classe, show=False)
            st.pyplot(fig2, clear_figure=True)

        except Exception as e:
            st.warning(
                "Não foi possível calcular/exibir SHAP nesta instância (pode faltar memória "
                "ou dependências). Detalhes no erro abaixo."
            )
            st.exception(e)

# Rodapé
st.caption(
    "Arquivos lidos de **Dados/** e **Modelos/** (na raiz). "
    "Se aparecer erro de caminho, confirme se os arquivos foram enviados para o repositório."
)
