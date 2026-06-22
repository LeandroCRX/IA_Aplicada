import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, time
import joblib
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

@st.cache_resource
def load_scaler():
    """Carrega o normalizador automático oficial treinado (MinMaxScaler)"""
    try:
        scaler = joblib.load('temperature_scaler.joblib')
        return scaler, None
    except Exception as e:
        return None, str(e)

def predict_virtual_temp(model, scaler, history_window):
    # Molda os dados para o formato do Scikit-Learn (1 coluna)
    window_2d = np.array(history_window, dtype=np.float32).reshape(-1, 1)
    # Normaliza usando a escala exata do treinamento
    norm_window = scaler.transform(window_2d).flatten()
    # Molda para o formato tridimensional do Keras (batch, timesteps, feature)
    input_data = np.array(norm_window, dtype=np.float32).reshape(1, 12, 1)
    pred_norm = model.predict(input_data, verbose=0)[0][0]
    # Converte de volta para a escala real em °C
    pred_val = scaler.inverse_transform(np.array([[pred_norm]], dtype=np.float32))[0][0]
    return float(pred_val)

def forecast_6_steps(model, scaler, history_window):
    future = []
    window = list(history_window)
    for _ in range(6):
        pred = predict_virtual_temp(model, scaler, window)
        future.append(pred)
        window.pop(0)
        window.append(pred)
    return future

st.set_page_config(page_title="ESP32 · Monitor de Temperatura", page_icon="🌡️", layout="wide")

# ── CSS Customizado ───────────────────────────────────────────────────────────
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

# ── Firebase e UI Helpers ─────────────────────────────────────────────────────
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
    if "real_time" in data and isinstance(data["real_time"], dict):
        sp = extract_setpoint(data["real_time"])
        if sp is not None: return sp
            
    for k in ["setpoint", "Setpoint", "setPoint", "set_point"]:
        if k in data and not isinstance(data[k], dict):
            try: return float(data[k])
            except (ValueError, TypeError): pass
                
    try:
        sample = next(iter(data.values()), None)
        if isinstance(sample, dict):
            last_key = list(data.keys())[-1]
            last_val = data[last_key]
            for k in ["setpoint", "Setpoint", "setPoint", "set_point"]:
                if k in last_val and not isinstance(last_val[k], dict):
                    try: return float(last_val[k])
                    except (ValueError, TypeError): pass
    except Exception: pass
        
    for k in ["setpoint", "Setpoint", "setPoint", "set_point"]:
        if k in data and isinstance(data[k], dict):
            val = data[k].get("value") or data[k].get("valor") or data[k].get("val")
            if val is not None:
                try: return float(val)
                except (ValueError, TypeError): pass
    return None

def build_history(data, k1, k2):
    rows = []
    if not isinstance(data, dict):
        return pd.DataFrame()
    fuso_br = timezone(timedelta(hours=-3))
    for v in data.values():
        if not isinstance(v, dict): continue
        t1 = v.get(k1) or v.get("temperatura1") or v.get("temp1")
        t2 = v.get(k2) or v.get("temperatura2") or v.get("temp2")
        ts = v.get("timestamp") or v.get("ts") or v.get("time")
        sp = None
        for spk in ["setpoint", "Setpoint", "setPoint", "set_point"]:
            if spk in v and not isinstance(v[spk], dict):
                try:
                    sp = float(v[spk])
                    break
                except (ValueError, TypeError): pass
        if t1 is not None and t2 is not None:
            dt = datetime.fromtimestamp(ts, fuso_br) if ts and ts > 1e8 else None
            rows.append({"datetime": dt, "sensor1": float(t1), "sensor2": float(t2), "setpoint": sp})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("datetime").tail(200)
    return df

def kpi(col, label, val, unit="°C", sub="", color="#f1f5f9"):
    v_str = f"{val:.1f}" if val is not None else "—"
    col.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{v_str}<span class="kpi-unit"> {unit if val is not None else ''}</span></div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

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

# ── Barra Lateral (Sidebar) ───────────────────────────────────────────────────
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
                st.markdown(f'<span class="badge badge-on">✔ Secrets carregados</span>', unsafe_allow_html=True)
            elif "firebase" in st.secrets:
                cred_dict = dict(st.secrets["firebase"])
                st.markdown('<span class="badge badge-on">✔ Secrets (padrão) carregados</span>', unsafe_allow_html=True)
            elif "FIREBASE_KEY" in st.secrets:
                try:
                    cred_dict = json.loads(st.secrets["FIREBASE_KEY"])
                    if "private_key" in cred_dict:
                        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
                    st.markdown('<span class="badge badge-on">✔ Secrets carregados</span>', unsafe_allow_html=True)
                except Exception:
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
    except Exception:
        pass

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
    
    ia_pontos_historico = st.slider("Pontos de histórico (Estatísticas)", 5, 200, 50, 5)
    
    ia_input_sensor = st.selectbox(
        "Sensor de entrada", 
        ["Sensor 1 (Ambiente)", "Sensor 2 (Resistor)"], 
        index=1, 
        key="ia_input_sensor_selection"
    )

    st.markdown("---")
    st.markdown("**Configurações de exibição**")
    alert_t    = st.slider("Temperatura de alerta (°C)", 30, 150, 80)
    max_t      = st.slider("Escala máx. do gauge (°C)", 50, 300, 150)
    refresh_s  = st.selectbox("Auto-refresh", [5, 10, 15, 30, 60], index=1, format_func=lambda x: f"{x}s")
    history_on = st.checkbox("Mostrar histórico", value=True)

    connect_btn = st.button("🔌 Conectar", use_container_width=True)

    if "connected" not in st.session_state: st.session_state.connected = False
    if "cred_dict_saved" not in st.session_state: st.session_state.cred_dict_saved = None

    if connect_btn:
        if not db_url: st.error("Informe a Database URL.")
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

# ── Dashboard principal em tempo real ─────────────────────────────────────────
@st.fragment(run_every=refresh_s)
def dashboard():
    ## ── NOVO: Inicialização do armazenamento estável de predições passadas ──
    if "ia_history" not in st.session_state:
        st.session_state.ia_history = {}

    # 1. Captura ou simulação de dados
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
                key1: entry["temp_ambiente"], key2: entry["temp_resistor"],
                "timestamp": entry["timestamp"], "setpoint": entry.get("setpoint", 50.0)
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

    # Status de Conexão Superior
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

    hist_data_node = None
    if isinstance(data, dict):
        for hk in ["historico", "history", "medicoes", "readings", "logs", "long_time"]:
            if hk in data and isinstance(data[hk], dict):
                hist_data_node = data[hk]
                break
        if hist_data_node is None:
            sample = next(iter(data.values()), None)
            if isinstance(sample, dict) and (key1 in sample or key2 in sample):
                hist_data_node = data

    df = build_history(hist_data_node, key1, key2) if hist_data_node else pd.DataFrame()

    pred_val = None
    future_df = None
    ia_status_msg = ""
    
    # ── PROCESSAMENTO DA INTELIGÊNCIA ARTIFICIAL ──────────────────────────────
    if ia_enabled:
        model, model_err = load_keras_model()
        scaler, scaler_err = load_scaler()
        
        if model_err: ia_status_msg = f"Erro no modelo: {model_err}"
        elif scaler_err: ia_status_msg = f"Erro no Normalizador (.joblib): {scaler_err}"
        elif df.empty or len(df) < 12:
            ia_status_msg = f"Aguardando dados históricos (mínimo 12, atualmente {len(df)})"
        else:
            try:
                selected_col = "sensor1" if ia_input_sensor == "Sensor 1 (Ambiente)" else "sensor2"
                raw_values = df[selected_col].values
                
                # Suavização de ruído via Média Móvel de 3 pontos
                smoothed_values = pd.Series(raw_values).rolling(window=3, min_periods=1).mean().values
                
                # Executa a previsão oficial de 6 passos futuros
                future_predictions = forecast_6_steps(model, scaler, smoothed_values[-12:])
                pred_val = future_predictions[-1]
                ia_status_msg = "OK"
                
                # Ajuste do eixo X do tempo baseado na cadência real das mensagens
                last_dt = df["datetime"].iloc[-1]
                if pd.isnull(last_dt): last_dt = datetime.now(timezone(timedelta(hours=-3)))
                
                intervalo_seg = refresh_s
                if len(df) > 1:
                    calc_diff = (df["datetime"].iloc[-1] - df["datetime"].iloc[-2]).total_seconds()
                    if calc_diff > 0: intervalo_seg = calc_diff

                future_times = [last_dt + timedelta(seconds=intervalo_seg * i) for i in range(7)]
                
                # Conexão visual da ponta do gráfico histórico com o início do futuro
                last_real_value = smoothed_values[-1]
                plot_predictions = [last_real_value] + future_predictions

                future_df = pd.DataFrame({
                    "datetime": future_times,
                    "forecast": plot_predictions
                })

                ## ── NOVO: Registrar a predição para manter o histórico ──
                # O primeiro item de future_predictions é a previsão exata para o próximo passo (+1)
                # Associamos ela ao seu tempo correto no futuro (future_times[1])
                if len(future_times) > 1:
                    tempo_alvo = future_times[1]
                    st.session_state.ia_history[tempo_alvo] = future_predictions[0]

                # Controle de memória: remove registros mais antigos se passar de 200 pontos
                if len(st.session_state.ia_history) > 200:
                    tempos_ordenados = sorted(st.session_state.ia_history.keys())
                    for tempo_antigo in tempos_ordenados[:-200]:
                        st.session_state.ia_history.pop(tempo_antigo, None)

                # Converte a memória acumulada em um DataFrame para plotar
                df_ia_hist = pd.DataFrame(
                    list(st.session_state.ia_history.items()), 
                    columns=["datetime", "ia_past_pred"]
                ).sort_values("datetime")

            except Exception as e:
                ia_status_msg = f"Erro na inferência: {e}"
                future_df = None

    # Exibição dos KPIs Superiores
    st.markdown('<div class="section-title">Leitura Atual</div>', unsafe_allow_html=True)
    if ia_enabled and ia_status_msg != "OK" and ia_status_msg != "":
        st.info(f"🤖 Status IA: {ia_status_msg}")

    show_ia_card = (ia_enabled and pred_val is not None)
    cols_kpi = st.columns(5) if show_ia_card else st.columns(4)
    k1c, k2c, *mid_cols, kstat = cols_kpi
    
    kpi(k1c, "Sensor 1", t1, sub=f"Campo: `{key1}`")
    kpi(k2c, "Sensor 2", t2, sub=f"Campo: `{key2}`")
    
    if show_ia_card:
        kia, kdiff = mid_cols
        kpi(kia, "Previsão +6 Passos", pred_val, sub="MinMaxScaler Joblib", color="#10b981")
        diff_val = round(t2 - pred_val, 2) if (t2 is not None and pred_val is not None) else None
        kpi(kdiff, "Diferença S2 - IA", diff_val, sub="Resistor vs Previsão")
    else:
        kdiff = mid_cols[0]
        diff_val = round(t1 - t2, 2) if (t1 is not None and t2 is not None) else None
        kpi(kdiff, "Diferença S1 - S2", diff_val, sub="Entre sensores")
        
    kpi(kstat, "Setpoint", setpoint_val, sub="Firebase")

    # Alertas de segurança Térmica
    hot = (t1 is not None and t1 >= alert_t) or (t2 is not None and t2 >= alert_t) or (pred_val is not None and pred_val >= alert_t)
    if hot:
        st.markdown(f'<div class="alert-box alert-hot">🔥 TEMPERATURA CRÍTICA (Limiar: {alert_t} °C)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box alert-ok">✅ Operação Normal (Abaixo de {alert_t} °C)</div>', unsafe_allow_html=True)

    # Relógios (Gauges)
    st.markdown('<div class="section-title">Gauges em Tempo Real</div>', unsafe_allow_html=True)
    g_cols = st.columns(3) if show_ia_card else st.columns(2)
    with g_cols[0]: st.plotly_chart(gauge_fig(t1, f"Sensor 1 · {key1}", max_t, alert_t, "#38bdf8"), use_container_width=True)
    with g_cols[1]: st.plotly_chart(gauge_fig(t2, f"Sensor 2 · {key2}", max_t, alert_t, "#a78bfa"), use_container_width=True)
    if show_ia_card:
        with g_cols[2]: st.plotly_chart(gauge_fig(pred_val, "Previsão IA (6 Passos)", max_t, alert_t, "#10b981"), use_container_width=True)

    # ── GRÁFICO HISTÓRICO COMPLETO ────────────────────────────────────────────
    if history_on and not df.empty and "datetime" in df.columns and df["datetime"].notna().any():
        st.markdown('<div class="section-title">Histórico de Temperatura e Tendência IA</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["sensor1"], name="Sensor 1", mode="lines", line=dict(color="#38bdf8", width=2)))
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["sensor2"], name="Sensor 2", mode="lines", line=dict(color="#a78bfa", width=2)))
        
        ## ── NOVO: Adiciona a linha do histórico acumulado de predições passadas ──
        if ia_enabled and 'df_ia_hist' in locals() and not df_ia_hist.empty:
            fig.add_trace(go.Scatter(
                x=df_ia_hist["datetime"], 
                y=df_ia_hist["ia_past_pred"], 
                name="Histórico IA (Passado)",
                mode="lines", 
                line=dict(color="#f43f5e", width=2, dash="dot") # Linha pontilhada rosa/coral para destaque
            ))

        # Inserção explícita da linha de predição futura (6 passos à frente)
        if ia_enabled and future_df is not None:
            fig.add_trace(go.Scatter(
                x=future_df["datetime"], 
                y=future_df["forecast"], 
                name="Projeção IA (+6 passos)",
                mode="lines+markers", 
                marker=dict(size=7, symbol="circle"),
                line=dict(color="#10b981", width=3, dash="dash")
            ))
            
        if setpoint_val is not None:
            end_date = future_df["datetime"].iloc[-1] if future_df is not None else df["datetime"].iloc[-1]
            fig.add_trace(go.Scatter(x=[df["datetime"].iloc[0], end_date], y=[setpoint_val, setpoint_val], name="Setpoint", line=dict(color="#f59e0b", width=1.5, dash="dot"), mode="lines"))
            
        fig.add_hline(y=alert_t, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        
        fig.update_layout(
            height=350, 
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            yaxis=dict(autorange=True),
            legend=dict(font=dict(color="white"))
        )
        st.plotly_chart(fig, use_container_width=True)

        # Painel Inferior de Estatísticas
        df_stats = df.tail(ia_pontos_historico)
        st.markdown(f'<div class="section-title">Estatísticas Rápidas (Últimos {len(df_stats)} pontos)</div>', unsafe_allow_html=True)
        sc = st.columns(6)
        sc[0].markdown(f'<div class="kpi"><div class="kpi-label">S1 Mín</div><div class="kpi-value" style="font-size:1.4rem;color:#38bdf8">{df_stats["sensor1"].min():.1f}°C</div></div>', unsafe_allow_html=True)
        sc[1].markdown(f'<div class="kpi"><div class="kpi-label">S1 Méd</div><div class="kpi-value" style="font-size:1.4rem;color:#38bdf8">{df_stats["sensor1"].mean():.1f}°C</div></div>', unsafe_allow_html=True)
        sc[2].markdown(f'<div class="kpi"><div class="kpi-label">S1 Máx</div><div class="kpi-value" style="font-size:1.4rem;color:#38bdf8">{df_stats["sensor1"].max():.1f}°C</div></div>', unsafe_allow_html=True)
        sc[3].markdown(f'<div class="kpi"><div class="kpi-label">S2 Mín</div><div class="kpi-value" style="font-size:1.4rem;color:#a78bfa">{df_stats["sensor2"].min():.1f}°C</div></div>', unsafe_allow_html=True)
        sc[4].markdown(f'<div class="kpi"><div class="kpi-label">S2 Méd</div><div class="kpi-value" style="font-size:1.4rem;color:#a78bfa">{df_stats["sensor2"].mean():.1f}°C</div></div>', unsafe_allow_html=True)
        sc[5].markdown(f'<div class="kpi"><div class="kpi-label">S2 Máx</div><div class="kpi-value" style="font-size:1.4rem;color:#a78bfa">{df_stats["sensor2"].max():.1f}°C</div></div>', unsafe_allow_html=True)

    with st.expander("🧩 JSON bruto recebido do Firebase"):
        st.json(data)

# Executa o fragmento do painel principal
dashboard()
