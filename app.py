import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Keras / AI Model Helpers ──────────────────────────────────────────────────
@st.cache_resource
def load_keras_model():
    os.environ["KERAS_BACKEND"] = "torch"
    try:
        import keras
        with open('config.json', 'r') as f:
            config = json.load(f)
        model = keras.models.model_from_json(json.dumps(config))
        model.load_weights('model.weights.h5')
        return model, None
    except Exception as e:
        return None, str(e)

def predict_virtual_temp(model, history_window, min_val, max_val):
    denom = (max_val - min_val) if max_val != min_val else 1.0
    norm_window = [(x - min_val) / denom for x in history_window]
    input_data = np.array(norm_window, dtype=np.float32).reshape(1, 12, 1)
    pred_norm = model.predict(input_data, verbose=0)[0][0]
    pred_val = pred_norm * denom + min_val
    return float(pred_val)

def forecast_12_steps(model, history_window, min_val, max_val):
    future = []
    window = list(history_window)
    for _ in range(12):
        pred = predict_virtual_temp(
            model,
            window,
            min_val,
            max_val
        )
        future.append(pred)
        window.pop(0)
        window.append(pred)
    return future

st.set_page_config(page_title="ESP32 · Monitor de Temperatura", page_icon="🌡️", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0f1729 60%, #0a1628 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.03); border-right: 1px solid rgba(255,255,255,0.07); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #ffffff !important; }
.hero { background: linear-gradient(120deg, #1a3a5c, #0d2137); border: 1px solid rgba(56,189,248,0.2);
        border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(56,189,248,0.1); }
.hero h1 { font-size: 1.9rem; font-weight: 700; color: #fff; margin: 0; }
.hero p  { color: rgba(255,255,255,0.65); margin: 0.3rem 0 0; font-size: 0.95rem; }
.kpi { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
       border-radius: 14px; padding: 1.2rem 1.4rem; text-align: center; }
.kpi-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: .08em; color: #64748b; margin-bottom: .4rem; }
.kpi-value { font-size: 2.4rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.kpi-unit  { font-size: 1rem; color: #94a3b8; }
.kpi-sub   { font-size: 0.75rem; color: #475569; margin-top: .3rem; }
.alert-box { border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1rem; font-weight: 600; }
.alert-hot  { background: rgba(239,68,68,.15); border: 1px solid rgba(239,68,68,.4); color: #fca5a5; }
.alert-ok   { background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.3); color: #86efac; }
.badge { display:inline-block; border-radius:999px; padding:.15rem .7rem; font-size:.75rem; font-weight:600; }
.badge-on  { background:rgba(34,197,94,.15); color:#4ade80; border:1px solid rgba(34,197,94,.3); }
.badge-off { background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.3); }
.section-title { font-size:.95rem; font-weight:600; color:#38bdf8;
                 border-left:3px solid #38bdf8; padding-left:.6rem; margin:1.2rem 0 .8rem; }
.stButton>button { background:linear-gradient(135deg,#0ea5e9,#2563eb)!important;
                   color:#fff!important; border:none!important; border-radius:8px!important;
                   font-weight:600!important; box-shadow:0 4px 12px rgba(14,165,233,.35)!important; }
#MainMenu,footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Firebase helpers ──────────────────────────────────────────────────────────
def init_firebase(cred_dict, db_url):
    if firebase_admin._apps:
        try: firebase_admin.delete_app(firebase_admin.get_app())
        except: pass
    cred = credentials.Certificate(cred_dict) if cred_dict else credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {"databaseURL": db_url})

def fetch_path(path):
    try:
        return db.reference(path).get(), None
    except Exception as e:
        return None, str(e)

# ── Parse helpers ─────────────────────────────────────────────────────────────
def extract_current(data, k1, k2):
    t1 = t2 = ts = None
    if isinstance(data, dict):
        if "real_time" in data and isinstance(data["real_time"], dict):
            t1, t2, ts = extract_current(data["real_time"], k1, k2)
            if t1 is not None or t2 is not None:
                return t1, t2, ts

        t1 = data.get(k1) or data.get("temperatura1") or data.get("temp1")
        t2 = data.get(k2) or data.get("temperatura2") or data.get("temp2")
        ts = data.get("timestamp") or data.get("ts") or data.get("time")

        if t1 is None and t2 is None:
            sample = next(iter(data.values()), None)
            if isinstance(sample, dict) and (k1 in sample or k2 in sample):
                last_val = data[list(data.keys())[-1]]
                t1 = last_val.get(k1) or last_val.get("temperatura1") or last_val.get("temp1")
                t2 = last_val.get(k2) or last_val.get("temperatura2") or last_val.get("temp2")
                ts = last_val.get("timestamp") or last_val.get("ts") or last_val.get("time")

        if t1 is None and k1 in data and isinstance(data[k1], dict):
            t1 = data[k1].get("temperatura") or data[k1].get("temp") or data[k1].get("value")
        if t2 is None and k2 in data and isinstance(data[k2], dict):
            t2 = data[k2].get("temperatura") or data[k2].get("temp") or data[k2].get("value")
    return t1, t2, ts

def extract_setpoint(data):
    if not isinstance(data, dict):
        return None
    
    # 1. Se houver real_time, tenta de lá primeiro
    if "real_time" in data and isinstance(data["real_time"], dict):
        sp = extract_setpoint(data["real_time"])
        if sp is not None:
            return sp
            
    # 2. Tenta obter "setpoint" na raiz (ignorando maiúsculas/minúsculas)
    for k in ["setpoint", "Setpoint", "setPoint", "set_point"]:
        if k in data and not isinstance(data[k], dict):
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
                
    # 3. Tenta obter da última chave (se for histórico/lista)
    try:
        sample = next(iter(data.values()), None)
        if isinstance(sample, dict):
            last_key = list(data.keys())[-1]
            last_val = data[last_key]
            for k in ["setpoint", "Setpoint", "setPoint", "set_point"]:
                if k in last_val and not isinstance(last_val[k], dict):
                    try:
                        return float(last_val[k])
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
        
    # 4. Caso o setpoint esteja em um dicionário (ex: {"setpoint": {"value": 50}})
    for k in ["setpoint", "Setpoint", "setPoint", "set_point"]:
        if k in data and isinstance(data[k], dict):
            val = data[k].get("value") or data[k].get("valor") or data[k].get("val")
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
                    
    return None

def build_history(data, k1, k2):
    rows = []
    if not isinstance(data, dict):
        return pd.DataFrame()
    
    fuso_br = timezone(timedelta(hours=-3))
    
    for v in data.values():
        if not isinstance(v, dict):
            continue
        t1 = v.get(k1) or v.get("temperatura1") or v.get("temp1")
        t2 = v.get(k2) or v.get("temperatura2") or v.get("temp2")
        ts = v.get("timestamp") or v.get("ts") or v.get("time")
        if t1 is not None and t2 is not None:
            dt = datetime.fromtimestamp(ts, fuso_br) if ts and ts > 1e8 else None
            rows.append({"datetime": dt, "sensor1": float(t1), "sensor2": float(t2)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("datetime").tail(200)
    return df

def gauge_fig(value, label, max_temp, alert_temp, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value if value is not None else 0,
        number={"suffix": " °C", "font": {"size": 36, "color": "#f1f5f9"}},
        delta={"reference": alert_temp, "suffix": "°C",
               "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
        title={"text": label, "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, max_temp], "tickcolor": "#475569", "tickwidth": 1,
                     "tickfont": {"color": "#475569", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, alert_temp * 0.7], "color": "rgba(34,197,94,0.1)"},
                {"range": [alert_temp * 0.7, alert_temp], "color": "rgba(234,179,8,0.1)"},
                {"range": [alert_temp, max_temp], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 2}, "thickness": 0.8,
                          "value": alert_temp},
        }
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Modo de Operação")
    modo_simulacao = st.checkbox("Modo Simulação Local", value=False, key="modo_simulacao",
                                 help="Simula dados de temperatura localmente para testar a IA sem precisar de conexão com o Firebase.")

    st.markdown("---")
    st.markdown("## 🔥 Configuração Firebase")
    
    projeto_selecionado = st.selectbox("Projeto Firebase", ["Digital Twin ESP32", "Cetel Site"])
    prefixo_secret = "digital_twin" if projeto_selecionado == "Digital Twin ESP32" else "cetel_site"
    default_url_placeholder = "https://digital-twin-esp32-default-rtdb.firebaseio.com/" if projeto_selecionado == "Digital Twin ESP32" else "https://cetel-site.firebaseio.com/"

    cred_method = st.selectbox("Autenticação", ["Streamlit Secrets", "Upload JSON", "Colar JSON", "App Default"])
    cred_dict = None

    if cred_method == "Streamlit Secrets":
        try:
            if prefixo_secret in st.secrets and "firebase" in st.secrets[prefixo_secret]:
                cred_dict = dict(st.secrets[prefixo_secret]["firebase"])
                st.markdown(f'<span class="badge badge-on">✔ Secrets carregados ({projeto_selecionado})</span>', unsafe_allow_html=True)
            elif "firebase" in st.secrets:
                cred_dict = dict(st.secrets["firebase"])
                st.markdown('<span class="badge badge-on">✔ Secrets (padrão) carregados</span>', unsafe_allow_html=True)
            elif "FIREBASE_KEY" in st.secrets:
                try:
                    cred_dict = json.loads(st.secrets["FIREBASE_KEY"])
                    if "private_key" in cred_dict:
                        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
                    st.markdown('<span class="badge badge-on">✔ Secrets (JSON) carregados</span>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown('<span class="badge badge-off">✖ Erro no JSON do Secrets</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-off">✖ Secret não encontrado</span>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<span class="badge badge-off">✖ Nenhum secrets.toml configurado</span>', unsafe_allow_html=True)
    elif cred_method == "Upload JSON":
        f = st.file_uploader("serviceAccountKey.json", type=["json"])
        if f:
            cred_dict = json.load(f)
            st.markdown('<span class="badge badge-on">✔ Arquivo OK</span>', unsafe_allow_html=True)
    elif cred_method == "Colar JSON":
        raw = st.text_area("Cole o JSON", height=160, placeholder='{"type":"service_account",...}')
        if raw.strip():
            try:
                cred_dict = json.loads(raw)
                st.markdown('<span class="badge badge-on">✔ JSON válido</span>', unsafe_allow_html=True)
            except:
                st.markdown('<span class="badge badge-off">✖ JSON inválido</span>', unsafe_allow_html=True)

    st.markdown("---")
    default_db_url = ""
    try:
        if prefixo_secret in st.secrets and "DATABASE_URL" in st.secrets[prefixo_secret]:
            default_db_url = st.secrets[prefixo_secret]["DATABASE_URL"]
        elif "FIREBASE_DATABASE_URL" in st.secrets:
            default_db_url = st.secrets["FIREBASE_DATABASE_URL"]
        else:
            default_db_url = os.getenv("FIREBASE_DATABASE_URL", "")
    except Exception:
        default_db_url = os.getenv("FIREBASE_DATABASE_URL", "")

    db_url  = st.text_input("Database URL", value=default_db_url, placeholder=default_url_placeholder)
    
    if projeto_selecionado == "Digital Twin ESP32":
        opcoes_caminho = ["/long_time", "/real_time", "Completo"]
    else:
        opcoes_caminho = ["/dispositivos/-OsSLxXdlG3w3AhCdDIQ", "Completo"]
        
    db_path_selection = st.selectbox("Caminho dos dados", opcoes_caminho, index=0)
    db_path = "/" if db_path_selection == "Completo" else db_path_selection

    st.markdown("---")
    st.markdown("**Nomes dos campos no JSON**")
    key1 = st.text_input("Campo Sensor 1", value="temp_ambiente")
    key2 = st.text_input("Campo Sensor 2", value="temp_resistor")

    st.markdown("---")
    st.markdown("## 🤖 Termômetro Virtual (IA)")
    ia_enabled = st.checkbox("Habilitar IA", value=True, key="ia_enabled")
    
    ia_pontos_historico = st.slider(
        "Pontos de histórico (IA e Estatísticas)", 
        min_value=5, 
        max_value=200, 
        value=50, 
        step=5,
        help="Controla a extensão da linha da IA no gráfico e a janela de cálculo das Estatísticas abaixo."
    )
    
    ia_input_sensor = st.selectbox(
        "Sensor de entrada", 
        ["Sensor 1 (Ambiente)", "Sensor 2 (Resistor)"], 
        index=1, 
        key="ia_input_sensor"
    )
    ia_min_temp = st.number_input("Temp. Mínima de Normalização (°C)", value=20.0, step=1.0, key="ia_min_temp")
    ia_max_temp = st.number_input("Temp. Máxima de Normalização (°C)", value=100.0, step=1.0, key="ia_max_temp")

    st.markdown("---")
    st.markdown("**Configurações de exibição**")
    alert_t    = st.slider("Temperatura de alerta (°C)", 30, 150, 80)
    max_t      = st.slider("Escala máx. do gauge (°C)", 50, 300, 150)
    refresh_s  = st.selectbox("Auto-refresh", [5, 10, 15, 30, 60], index=1, format_func=lambda x: f"{x}s")
    history_on = st.checkbox("Mostrar histórico", value=True, help="Apenas se o caminho contiver push keys com registros históricos")

    connect_btn = st.button("🔌 Conectar", use_container_width=True)

    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "cred_dict_saved" not in st.session_state:
        st.session_state.cred_dict_saved = None

    if connect_btn:
        if not db_url:
            st.error("Informe a Database URL.")
        else:
            try:
                saved = st.session_state.cred_dict_saved
                init_firebase(cred_dict or saved, db_url)
                st.session_state.connected = True
                st.session_state.cred_dict_saved = cred_dict or saved
                st.success("Conectado!")
            except Exception as e:
                st.session_state.connected = False
                st.error(f"Erro: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌡️ Monitor de Temperatura · ESP32</h1>
  <p>Leitura em tempo real de dois sensores de temperatura no resistor via Firebase</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.connected and not st.session_state.get("modo_simulacao", False):
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "1️⃣", "Credenciais Firebase", "Faça upload do `serviceAccountKey.json` gerado no Firebase Console → Contas de serviço."),
        (c2, "2️⃣", "Caminho & Sensores", "Informe o caminho no banco (ex: `/temperatura`) e os nomes dos campos do JSON."),
        (c3, "3️⃣", "Conectar", "Clique em **Conectar**. O dashboard atualiza automaticamente no intervalo escolhido."),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi" style="text-align:left;padding:1.4rem">
              <div style="font-size:2rem">{icon}</div>
              <div style="font-weight:700;color:#38bdf8;margin:.5rem 0 .4rem">{title}</div>
              <div style="font-size:.85rem;color:#64748b">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ── Dashboard em tempo real ───────────────────────────────────────────────────
@st.fragment(run_every=refresh_s)
def dashboard():
    if st.session_state.get("modo_simulacao", False):
        if "mock_history" not in st.session_state:
            now = time.time()
            history = []
            for i in range(60):
                t = now - (60 - i) * refresh_s
                s1 = 24.5 + 0.6 * np.sin(i / 6.0) + np.random.uniform(-0.15, 0.15)
                s2 = 25.0 + (55.0 / (1.0 + np.exp(-(i - 15) / 8.0))) + np.random.uniform(-0.25, 0.25)
                history.append({"timestamp": t, "temp_ambiente": s1, "temp_resistor": s2, "setpoint": 50.0})
            st.session_state.mock_history = history
        else:
            now = time.time()
            i = len(st.session_state.mock_history)
            s1 = 24.5 + 0.6 * np.sin(i / 6.0) + np.random.uniform(-0.15, 0.15)
            s2 = 25.0 + (55.0 / (1.0 + np.exp(-(i - 15) / 8.0))) + np.random.uniform(-0.25, 0.25)
            st.session_state.mock_history.append({"timestamp": now, "temp_ambiente": s1, "temp_resistor": s2, "setpoint": 50.0})
            st.session_state.mock_history = st.session_state.mock_history[-200:]
            
        data = {}
        for idx, entry in enumerate(st.session_state.mock_history):
            data[f"mock_key_{idx}"] = {
                key1: entry["temp_ambiente"],
                key2: entry["temp_resistor"],
                "timestamp": entry["timestamp"],
                "setpoint": entry.get("setpoint", 50.0)
            }
        err = None
    else:
        data, err = fetch_path(db_path)
    
    now_str = datetime.now().strftime("%H:%M:%S")

    if err:
        st.error(f"❌ Erro ao ler Firebase: {err}")
        return

    if data is None:
        st.warning(f"⚠️ Nenhum dado encontrado em `{db_path}`")
        return

    t1, t2, ts = extract_current(data, key1, key2)
    setpoint_val = extract_setpoint(data)

    col_st, col_time = st.columns([1, 3])
    with col_st:
        badge = '<span class="badge badge-on">● ONLINE</span>' if (t1 or t2) else '<span class="badge badge-off">● SEM DADOS</span>'
        if st.session_state.get("modo_simulacao", False):
            badge = '<span class="badge badge-on" style="background:rgba(56,189,248,0.15);color:#38bdf8;border:1px solid rgba(56,189,248,0.3);">● SIMULAÇÃO</span>'
        st.markdown(badge, unsafe_allow_html=True)
    with col_time:
        fuso_br = timezone(timedelta(hours=-3))
        ts_str = datetime.fromtimestamp(ts, fuso_br).strftime("%d/%m/%Y %H:%M:%S") if ts and ts > 1e8 else "—"
        st.markdown(f'<span style="font-size:.8rem;color:#475569">Última medição ESP32: {ts_str} &nbsp;|&nbsp; Atualizado: {now_str}</span>', unsafe_allow_html=True)

    hist_data = None
    if isinstance(data, dict):
        for hk in ["historico", "history", "medicoes", "readings", "logs", "long_time"]:
            if hk in data and isinstance(data[hk], dict):
                hist_data = data[hk]
                break
        if hist_data is None:
            sample = next(iter(data.values()), None)
            if isinstance(sample, dict) and (key1 in sample or key2 in sample):
                hist_data = data

    df = build_history(hist_data, key1, key2) if hist_data else pd.DataFrame()

    pred_val = None
    future_df = None
    future_predictions = None
    ia_status_msg = ""
    
    if ia_enabled:
        model, model_err = load_keras_model()
        if model_err:
            ia_status_msg = f"Erro ao carregar modelo IA: {model_err}"
        elif df.empty or len(df) < 12:
            ia_status_msg = f"IA: Aguardando dados históricos suficientes (mínimo de 12 leituras, atualmente {len(df)})"
        else:
            try:
                selected_col = "sensor1" if ia_input_sensor == "Sensor 1 (Ambiente)" else "sensor2"
                values = df[selected_col].values
                
                future_predictions = forecast_12_steps(
                    model,
                    values[-12:],
                    ia_min_temp,
                    ia_max_temp
                )
                pred_val = future_predictions[-1]
                ia_status_msg = "OK"
                
                future_df = pd.DataFrame({
                    "datetime": pd.date_range(
                        start=df["datetime"].iloc[-1],
                        periods=13,
                        freq="min"
                    )[1:],
                    "forecast": future_predictions
                })
            except Exception as e:
                ia_status_msg = f"Erro na inferência da IA: {e}"
                future_df = None
                future_predictions = None

    st.markdown('<div class="section-title">Leitura Atual</div>', unsafe_allow_html=True)
    
    if ia_enabled and ia_status_msg != "OK" and ia_status_msg != "":
        st.info(f"🤖 {ia_status_msg}")

    show_ia_card = (ia_enabled and pred_val is not None)
    cols_kpi = st.columns(5) if show_ia_card else st.columns(4)
    
    if show_ia_card:
        k1c, k2c, kia, kdiff, kstat = cols_kpi
    else:
        k1c, k2c, kdiff, kstat = cols_kpi

    def kpi(col, label, val, unit="°C", sub=""):
        v_str = f"{val:.1f}" if val is not None else "—"
        col.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{v_str}<span class="kpi-unit"> {unit if val is not None else ''}</span></div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    hot = (t1 is not None and t1 >= alert_t) or (t2 is not None and t2 >= alert_t) or (pred_val is not None and pred_val >= alert_t)

    kpi(k1c, "Sensor 1", t1, sub=f"Campo: `{key1}`")
    kpi(k2c, "Sensor 2", t2, sub=f"Campo: `{key2}`")
    
    # ✨ ATUALIZAÇÃO DO CARTÃO DE DIFERENÇA
    if show_ia_card:
        kpi(kia, "Termômetro Virtual (IA)", pred_val, sub=f"Previsão (In: {'S1' if ia_input_sensor == 'Sensor 1 (Ambiente)' else 'S2'})")
        
        diff_val = round(t2 - pred_val, 2) if (t2 is not None and pred_val is not None) else None
        kpi(kdiff, "Diferença S2 - IA", diff_val, sub="Resistor vs Previsão")
    else:
        diff_val = round(t1 - t2, 2) if (t1 is not None and t2 is not None) else None
        kpi(kdiff, "Diferença S1 - S2", diff_val, sub="Entre sensores")
    
    kpi(kstat, "Setpoint", setpoint_val, sub="Firebase")

    if hot:
        victims = []
        if t1 is not None and t1 >= alert_t: victims.append(f"Sensor 1: {t1:.1f} °C")
        if t2 is not None and t2 >= alert_t: victims.append(f"Sensor 2: {t2:.1f} °C")
        if pred_val is not None and pred_val >= alert_t: victims.append(f"Termômetro Virtual (IA): {pred_val:.1f} °C")
        st.markdown(f'<div class="alert-box alert-hot">🔥 TEMPERATURA ACIMA DO LIMIAR ({alert_t} °C) — {" | ".join(victims)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box alert-ok">✅ Temperaturas dentro do limite normal (abaixo de {alert_t} °C)</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Gauges em Tempo Real</div>', unsafe_allow_html=True)
    
    if show_ia_card:
        g1, g2, g3 = st.columns(3)
    else:
        g1, g2 = st.columns(2)
        
    with g1:
        st.plotly_chart(gauge_fig(t1, f"Sensor 1 · {key1}", max_t, alert_t, "#38bdf8"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_fig(t2, f"Sensor 2 · {key2}", max_t, alert_t, "#a78bfa"), use_container_width=True)
        
    if show_ia_card:
        with g3:
            st.plotly_chart(gauge_fig(pred_val, "Termômetro Virtual (IA)", max_t, alert_t, "#10b981"), use_container_width=True)

    if history_on:
        if not df.empty and "datetime" in df.columns and df["datetime"].notna().any():
            st.markdown('<div class="section-title">Histórico de Temperatura</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["datetime"], y=df["sensor1"], name=f"Sensor 1 ({key1})",
                                     line=dict(color="#38bdf8", width=2), fill="tozeroy",
                                     fillcolor="rgba(56,189,248,0.07)"))
            fig.add_trace(go.Scatter(x=df["datetime"], y=df["sensor2"], name=f"Sensor 2 ({key2})",
                                     line=dict(color="#a78bfa", width=2), fill="tozeroy",
                                     fillcolor="rgba(167,139,250,0.07)"))
            
            if ia_enabled and future_df is not None:
                fig.add_trace(go.Scatter(x=future_df["datetime"], y=future_df["forecast"], name="Previsão IA (+12 passos)",
                                         line=dict(color="#10b981", width=3, dash="dash")))
                
            fig.add_hline(y=alert_t, line_dash="dash", line_color="rgba(239,68,68,0.6)",
                          annotation_text=f"Limiar {alert_t}°C", annotation_font_color="#ef4444")
            fig.update_layout(
                height=320, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#FFFFFF", size=12)),
                margin=dict(t=30, b=20, l=0, r=0),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="°C"),
            )
            st.plotly_chart(fig, use_container_width=True)

            pontos_exibicao = ia_pontos_historico if ia_enabled else len(df)
            df_stats = df.tail(pontos_exibicao)
            
            st.markdown(f'<div class="section-title">Estatísticas (últimos {pontos_exibicao} pontos)</div>', unsafe_allow_html=True)
            
            stat_cols_count = 9 if (ia_enabled and future_predictions is not None) else 6
            sc = st.columns(stat_cols_count)
            
            sc[0].markdown(f'<div class="kpi"><div class="kpi-label">S1 Mín</div><div class="kpi-value" style="font-size:1.5rem;color:#38bdf8">{df_stats["sensor1"].min():.1f}°C</div></div>', unsafe_allow_html=True)
            sc[1].markdown(f'<div class="kpi"><div class="kpi-label">S1 Méd</div><div class="kpi-value" style="font-size:1.5rem;color:#38bdf8">{df_stats["sensor1"].mean():.1f}°C</div></div>', unsafe_allow_html=True)
            sc[2].markdown(f'<div class="kpi"><div class="kpi-label">S1 Máx</div><div class="kpi-value" style="font-size:1.5rem;color:#38bdf8">{df_stats["sensor1"].max():.1f}°C</div></div>', unsafe_allow_html=True)
            
            sc[3].markdown(f'<div class="kpi"><div class="kpi-label">S2 Mín</div><div class="kpi-value" style="font-size:1.5rem;color:#a78bfa">{df_stats["sensor2"].min():.1f}°C</div></div>', unsafe_allow_html=True)
            sc[4].markdown(f'<div class="kpi"><div class="kpi-label">S2 Méd</div><div class="kpi-value" style="font-size:1.5rem;color:#a78bfa">{df_stats["sensor2"].mean():.1f}°C</div></div>', unsafe_allow_html=True)
            sc[5].markdown(f'<div class="kpi"><div class="kpi-label">S2 Máx</div><div class="kpi-value" style="font-size:1.5rem;color:#a78bfa">{df_stats["sensor2"].max():.1f}°C</div></div>', unsafe_allow_html=True)
            
            if ia_enabled and future_predictions is not None:
                ia_valid = pd.Series(future_predictions)

                sc[6].markdown(
                    f'<div class="kpi"><div class="kpi-label">IA Mín</div>'
                    f'<div class="kpi-value" style="font-size:1.5rem;color:#10b981">'
                    f'{ia_valid.min():.1f}°C</div></div>',
                    unsafe_allow_html=True
                )

                sc[7].markdown(
                    f'<div class="kpi"><div class="kpi-label">IA Méd</div>'
                    f'<div class="kpi-value" style="font-size:1.5rem;color:#10b981">'
                    f'{ia_valid.mean():.1f}°C</div></div>',
                    unsafe_allow_html=True
                )

                sc[8].markdown(
                    f'<div class="kpi"><div class="kpi-label">IA Máx</div>'
                    f'<div class="kpi-value" style="font-size:1.5rem;color:#10b981">'
                    f'{ia_valid.max():.1f}°C</div></div>',
                    unsafe_allow_html=True
                )

    with st.expander("🧩 JSON bruto recebido do Firebase"):
        st.json(data)

dashboard()
