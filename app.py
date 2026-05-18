import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import plotly.graph_objects as go
import json, os, time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ESP32 · Monitor de Temperatura", page_icon="🌡️", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0f1729 60%, #0a1628 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.03); border-right: 1px solid rgba(255,255,255,0.07); }
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
    """Extrai temperaturas atuais independente da estrutura do JSON."""
    t1 = t2 = ts = None
    if isinstance(data, dict):
        if "real_time" in data and isinstance(data["real_time"], dict):
            t1, t2, ts = extract_current(data["real_time"], k1, k2)
            if t1 is not None or t2 is not None:
                return t1, t2, ts

        # Estrutura plana: {sensor1: 45.2, sensor2: 47.8, timestamp: ...}
        t1 = data.get(k1) or data.get("temperatura1") or data.get("temp1")
        t2 = data.get(k2) or data.get("temperatura2") or data.get("temp2")
        ts = data.get("timestamp") or data.get("ts") or data.get("time")

        # Se for um dicionário de push keys, pega a última chave (mais recente)
        if t1 is None and t2 is None:
            sample = next(iter(data.values()), None)
            if isinstance(sample, dict) and (k1 in sample or k2 in sample):
                last_val = data[list(data.keys())[-1]]
                t1 = last_val.get(k1) or last_val.get("temperatura1") or last_val.get("temp1")
                t2 = last_val.get(k2) or last_val.get("temperatura2") or last_val.get("temp2")
                ts = last_val.get("timestamp") or last_val.get("ts") or last_val.get("time")

        # Estrutura aninhada: {sensor1: {temperatura: 45.2}, sensor2: {...}}
        if t1 is None and k1 in data and isinstance(data[k1], dict):
            t1 = data[k1].get("temperatura") or data[k1].get("temp") or data[k1].get("value")
        if t2 is None and k2 in data and isinstance(data[k2], dict):
            t2 = data[k2].get("temperatura") or data[k2].get("temp") or data[k2].get("value")
    return t1, t2, ts

def build_history(data, k1, k2):
    """Tenta montar DataFrame histórico a partir de dados com push keys do Firebase."""
    rows = []
    if not isinstance(data, dict):
        return pd.DataFrame()
    for v in data.values():
        if not isinstance(v, dict):
            continue
        t1 = v.get(k1) or v.get("temperatura1") or v.get("temp1")
        t2 = v.get(k2) or v.get("temperatura2") or v.get("temp2")
        ts = v.get("timestamp") or v.get("ts") or v.get("time")
        if t1 is not None and t2 is not None:
            dt = datetime.fromtimestamp(ts) if ts and ts > 1e8 else None
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
    st.markdown("## 🔥 Configuração Firebase")
    cred_method = st.selectbox("Autenticação", ["Streamlit Secrets", "Upload JSON", "Colar JSON", "App Default"])
    cred_dict = None

    if cred_method == "Streamlit Secrets":
        if "firebase" in st.secrets:
            cred_dict = dict(st.secrets["firebase"])
            st.markdown('<span class="badge badge-on">✔ Secrets (TOML) carregados</span>', unsafe_allow_html=True)
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
    # Pega a URL dos Secrets se existir, senao variavel de ambiente
    default_db_url = ""
    if "FIREBASE_DATABASE_URL" in st.secrets:
        default_db_url = st.secrets["FIREBASE_DATABASE_URL"]
    else:
        default_db_url = os.getenv("FIREBASE_DATABASE_URL", "")

    db_url  = st.text_input("Database URL", value=default_db_url,
                             placeholder="https://digital-twin-esp32-default-rtdb.firebaseio.com/")
    db_path_selection = st.selectbox("Caminho dos dados", ["/long_time", "/real_time", "Completo"], index=0)
    db_path = "/" if db_path_selection == "Completo" else db_path_selection

    st.markdown("---")
    st.markdown("**Nomes dos campos no JSON**")
    key1 = st.text_input("Campo Sensor 1", value="temp_ambiente")
    key2 = st.text_input("Campo Sensor 2", value="temp_resistor")

    st.markdown("---")
    st.markdown("**Configurações de exibição**")
    alert_t    = st.slider("Temperatura de alerta (°C)", 30, 150, 80)
    max_t      = st.slider("Escala máx. do gauge (°C)", 50, 300, 150)
    refresh_s  = st.selectbox("Auto-refresh", [5, 10, 15, 30, 60], index=1, format_func=lambda x: f"{x}s")
    history_on = st.checkbox("Mostrar histórico", value=True,
                              help="Apenas se o caminho contiver push keys com registros históricos")

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

if not st.session_state.connected:
    # Tela de boas-vindas
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

    st.markdown("---")
    with st.expander("📟 Código de exemplo para ESP32 (Arduino)"):
        st.code("""
#include <Arduino.h>
#include <WiFi.h>
#include <FirebaseESP32.h>      // mobizt/Firebase ESP32 Client
#include <OneWire.h>
#include <DallasTemperature.h>  // Sensores DS18B20

#define WIFI_SSID     "SUA_REDE"
#define WIFI_PASSWORD "SUA_SENHA"
#define FIREBASE_HOST "seu-projeto-default-rtdb.firebaseio.com"
#define FIREBASE_AUTH "SEU_DATABASE_SECRET"

#define ONE_WIRE_BUS 4   // pino DATA dos sensores
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

FirebaseData   fbdo;
FirebaseConfig config;
FirebaseAuth   auth;

void setup() {
  Serial.begin(115200);
  sensors.begin();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println("\\nWiFi conectado");

  config.host = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

void loop() {
  sensors.requestTemperatures();
  float t1 = sensors.getTempCByIndex(0);
  float t2 = sensors.getTempCByIndex(1);
  long  ts  = millis() / 1000;   // ou use NTP para timestamp UNIX real

  // Escreve leitura atual
  Firebase.setFloat(fbdo, "/temperatura/sensor1", t1);
  Firebase.setFloat(fbdo, "/temperatura/sensor2", t2);
  Firebase.setInt  (fbdo, "/temperatura/timestamp", ts);

  // Push para histórico
  FirebaseJson json;
  json.set("sensor1", t1);
  json.set("sensor2", t2);
  json.set("timestamp", ts);
  Firebase.pushJSON(fbdo, "/temperatura/historico", json);

  Serial.printf("S1: %.2f°C  S2: %.2f°C\\n", t1, t2);
  delay(10000);   // envia a cada 10 segundos
}
""", language="cpp")
    st.stop()

# ── Dashboard em tempo real ───────────────────────────────────────────────────
@st.fragment(run_every=refresh_s)
def dashboard():
    data, err = fetch_path(db_path)
    now_str = datetime.now().strftime("%H:%M:%S")

    if err:
        st.error(f"❌ Erro ao ler Firebase: {err}")
        return

    if data is None:
        st.warning(f"⚠️ Nenhum dado encontrado em `{db_path}`")
        return

    t1, t2, ts = extract_current(data, key1, key2)

    # Status bar
    col_st, col_time = st.columns([1, 3])
    with col_st:
        badge = '<span class="badge badge-on">● ONLINE</span>' if (t1 or t2) else '<span class="badge badge-off">● SEM DADOS</span>'
        st.markdown(badge, unsafe_allow_html=True)
    with col_time:
        ts_str = datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M:%S") if ts and ts > 1e8 else "—"
        st.markdown(f'<span style="font-size:.8rem;color:#475569">Última medição ESP32: {ts_str} &nbsp;|&nbsp; Atualizado: {now_str}</span>', unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Leitura Atual</div>', unsafe_allow_html=True)
    k1c, k2c, kdiff, kstat = st.columns(4)

    def kpi(col, label, val, unit="°C", sub=""):
        v_str = f"{val:.1f}" if val is not None else "—"
        col.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{v_str}<span class="kpi-unit"> {unit if val is not None else ''}</span></div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    diff = round(t1 - t2, 2) if (t1 is not None and t2 is not None) else None
    hot  = (t1 is not None and t1 >= alert_t) or (t2 is not None and t2 >= alert_t)

    kpi(k1c,  "Sensor 1", t1, sub=f"Campo: `{key1}`")
    kpi(k2c,  "Sensor 2", t2, sub=f"Campo: `{key2}`")
    kpi(kdiff,"Diferença S1-S2", diff, sub="Entre sensores")
    with kstat:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">Status Térmico</div>
          <div class="kpi-value" style="font-size:1.4rem;padding-top:.3rem">
            {'🔴 ALERTA' if hot else '🟢 NORMAL'}
          </div>
          <div class="kpi-sub">Limiar: {alert_t} °C</div>
        </div>""", unsafe_allow_html=True)

    # ── Alerta ───────────────────────────────────────────────────────────────
    if hot:
        victims = []
        if t1 is not None and t1 >= alert_t: victims.append(f"Sensor 1: {t1:.1f} °C")
        if t2 is not None and t2 >= alert_t: victims.append(f"Sensor 2: {t2:.1f} °C")
        st.markdown(f'<div class="alert-box alert-hot">🔥 TEMPERATURA ACIMA DO LIMIAR ({alert_t} °C) — {" | ".join(victims)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box alert-ok">✅ Temperaturas dentro do limite normal (abaixo de {alert_t} °C)</div>', unsafe_allow_html=True)

    # ── Gauges ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Gauges em Tempo Real</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge_fig(t1, f"Sensor 1 · {key1}", max_t, alert_t, "#38bdf8"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_fig(t2, f"Sensor 2 · {key2}", max_t, alert_t, "#a78bfa"), use_container_width=True)

    # ── Histórico ─────────────────────────────────────────────────────────────
    if history_on:
        # Tenta encontrar histórico dentro do mesmo nó ou em sub-nó
        hist_data = None
        if isinstance(data, dict):
            # Verifica se tem sub-nó "historico" ou "history"
            for hk in ["historico", "history", "medicoes", "readings", "logs", "long_time"]:
                if hk in data and isinstance(data[hk], dict):
                    hist_data = data[hk]
                    break
            # Se não, tenta o próprio nó (push keys)
            if hist_data is None:
                sample = next(iter(data.values()), None)
                if isinstance(sample, dict) and (key1 in sample or key2 in sample):
                    hist_data = data

        df = build_history(hist_data, key1, key2) if hist_data else pd.DataFrame()

        if not df.empty and "datetime" in df.columns and df["datetime"].notna().any():
            st.markdown('<div class="section-title">Histórico de Temperatura</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["datetime"], y=df["sensor1"], name=f"Sensor 1 ({key1})",
                                     line=dict(color="#38bdf8", width=2), fill="tozeroy",
                                     fillcolor="rgba(56,189,248,0.07)"))
            fig.add_trace(go.Scatter(x=df["datetime"], y=df["sensor2"], name=f"Sensor 2 ({key2})",
                                     line=dict(color="#a78bfa", width=2), fill="tozeroy",
                                     fillcolor="rgba(167,139,250,0.07)"))
            fig.add_hline(y=alert_t, line_dash="dash", line_color="rgba(239,68,68,0.6)",
                          annotation_text=f"Limiar {alert_t}°C", annotation_font_color="#ef4444")
            fig.update_layout(
                height=320, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=30, b=20, l=0, r=0),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="°C"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Estatísticas do histórico
            st.markdown('<div class="section-title">Estatísticas (histórico)</div>', unsafe_allow_html=True)
            sc = st.columns(6)
            for i, (sensor, col_color) in enumerate([(df["sensor1"], "#38bdf8"), (df["sensor2"], "#a78bfa")]):
                sc[i*3].markdown(f'<div class="kpi"><div class="kpi-label">S{i+1} Mín</div><div class="kpi-value" style="font-size:1.5rem;color:{col_color}">{sensor.min():.1f}°C</div></div>', unsafe_allow_html=True)
                sc[i*3+1].markdown(f'<div class="kpi"><div class="kpi-label">S{i+1} Méd</div><div class="kpi-value" style="font-size:1.5rem;color:{col_color}">{sensor.mean():.1f}°C</div></div>', unsafe_allow_html=True)
                sc[i*3+2].markdown(f'<div class="kpi"><div class="kpi-label">S{i+1} Máx</div><div class="kpi-value" style="font-size:1.5rem;color:{col_color}">{sensor.max():.1f}°C</div></div>', unsafe_allow_html=True)

    # ── JSON Bruto ────────────────────────────────────────────────────────────
    with st.expander("🧩 JSON bruto recebido do Firebase"):
        st.code(json.dumps(data, ensure_ascii=False, indent=2, default=str), language="json")

dashboard()
