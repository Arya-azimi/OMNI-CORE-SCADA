import os
import json
import asyncio
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn


# --- Section 1: PyTorch Extractor ---
class DeepFeatureExtractor(nn.Module):
    def __init__(self):
        super(DeepFeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, x):
        return self.network(x) + x


# --- Section 2: Environment & AI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_virtual_core.json')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'virtual_core_scaler.pkl')

xgb_model = xgb.XGBRegressor()
try:
    xgb_model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_ready = True
    print(">>> SYSTEM: AI Models Active.")
except Exception:
    print(">>> WARNING: AI Models Offline. Dummy Mode Engaged.")
    model_ready = False
    scaler = None

torch_model = DeepFeatureExtractor()
torch_model.eval()

# --- Section 3: Simulation Data ---
depths = np.arange(5000, 6000, 0.5)
n = len(depths)
nphi = 0.15 + 0.1 * np.sin(depths * 0.08) + np.random.normal(0, 0.02, n)
rhob = 2.65 - 1.5 * nphi + np.random.normal(0, 0.05, n)
gr = 60 + 40 * np.sin(depths * 0.05) + np.random.normal(0, 5, n)
resd = 10 ** (1 + 1 * np.cos(depths * 0.03) + np.random.normal(0, 0.2, n))
dt = 90 - 40 * nphi + np.random.normal(0, 2, n)

t = (depths - 5000) / 1000
x = np.sin(t * 2 * np.pi) * 150
y = np.cos(t * 2 * np.pi) * 150

df_stream = pd.DataFrame({'DEPT': depths, 'GR': gr, 'RHOB': rhob, 'NPHI': nphi, 'DT': dt, 'RESD': resd, 'X': x, 'Y': y})

# --- Section 4: FastAPI Engine ---
app = FastAPI(title="Magnora OMNI-CORE")

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MAGNORA | SCADA</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #050505; color: #00ffcc; font-family: 'Orbitron', sans-serif;
            margin: 0; padding: 15px; overflow-x: hidden;
        }
        h1 { text-align: center; text-shadow: 0 0 10px rgba(0,255,204,0.5); letter-spacing: 2px; margin-bottom: 30px; }
        .grid-container { display: flex; flex-direction: column; gap: 15px; max-width: 1800px; margin: auto; }
        .kpi-row { display: flex; justify-content: space-between; gap: 15px; }
        .kpi-card {
            background: linear-gradient(135deg, rgba(15,15,15,0.9), rgba(5,5,5,0.95));
            border: 1px solid rgba(0,255,204,0.3); border-radius: 8px; padding: 15px;
            flex: 1; text-align: center; box-shadow: 0 0 15px rgba(0,255,204,0.1);
        }
        .kpi-title { font-size: 13px; color: #888; margin-bottom: 10px; letter-spacing: 1px;}
        .kpi-value { font-size: 32px; font-weight: bold; }
        .charts-row { display: grid; grid-template-columns: 1fr 2fr 1fr; gap: 15px; height: 65vh; }
        .chart-box {
            background: rgba(10,10,10,0.8); border: 1px solid rgba(0,255,204,0.2);
            border-radius: 8px; padding: 10px; position: relative;
        }
        .right-col { display: flex; flex-direction: column; gap: 15px; }
        .right-col .chart-box { flex: 1; }
        .controls { display: flex; justify-content: center; gap: 20px; margin-top: 15px; }
        button {
            background: #000; color: #00ffcc; border: 1px solid #00ffcc;
            padding: 12px 35px; font-family: 'Orbitron', sans-serif; font-size: 16px; font-weight: bold;
            cursor: pointer; transition: 0.3s; border-radius: 4px;
        }
        button:hover { background: #00ffcc; color: #000; box-shadow: 0 0 20px #00ffcc; }
        .btn-danger { border-color: #ff0055; color: #ff0055; }
        .btn-danger:hover { background: #ff0055; color: #000; box-shadow: 0 0 20px #ff0055; }
        .status-ready { color: #00ffcc; }
        .status-running { color: #ff0055; text-shadow: 0 0 10px #ff0055; }
    </style>
</head>
<body>

    <h1>MAGNORA // OMNI-CORE SCADA SYSTEM</h1>

    <div class="grid-container">
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">BIT DEPTH (FT)</div>
                <div class="kpi-value" id="val-depth">5000.0</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">AI POROSITY PREDICTION (%)</div>
                <div class="kpi-value" id="val-por" style="color: #ff0055;">0.00</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">RESERVOIR DENSITY (G/CC)</div>
                <div class="kpi-value" id="val-rhob" style="color: #ffea00;">0.00</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">DRILLING STATUS</div>
                <div class="kpi-value status-ready" id="val-status">READY</div>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-box" id="3d-chart"></div>
            <div class="chart-box" id="log-chart"></div>
            <div class="right-col">
                <div class="chart-box" id="cross-chart"></div>
                <div class="chart-box" id="hist-chart"></div>
            </div>
        </div>

        <div class="controls">
            <button onclick="sendCommand('play')">▶ START OPERATIONS</button>
            <button onclick="sendCommand('pause')">⏸ PAUSE</button>
            <button onclick="sendCommand('reset')" class="btn-danger">⏹ SYSTEM RESET</button>
        </div>
    </div>

    <script>
        var ws = new WebSocket("ws://" + location.host + "/ws");

        function sendCommand(action) {
            ws.send(JSON.stringify({action: action}));
        }

        var layout_dark = {
            plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
            font: {color: '#a0a0a0', family: 'Orbitron'},
            margin: {l: 40, r: 20, t: 40, b: 20}, showlegend: false
        };

        // --- HOLOGRAPHIC 3D TRAJECTORY SETUP ---
        var trace_payzone = {
            type: 'mesh3d',
            x: [-200, 200, 200, -200], y: [-200, -200, 200, 200], z: [5800, 5800, 5800, 5800],
            opacity: 0.15, color: '#ffea00', hoverinfo: 'skip', name: 'TARGET ZONE'
        };

        var layout_3d = Object.assign({}, layout_dark, {
            title: {text: '3D HOLOGRAPHIC RADAR', font: {color: '#00ffcc', size: 14}},
            scene: {
                camera: {eye: {x: 1.8, y: 1.8, z: 0.5}},
                xaxis: {showgrid: true, gridcolor: 'rgba(0,255,204,0.1)', zeroline: false, showticklabels: false, title: ''},
                yaxis: {showgrid: true, gridcolor: 'rgba(0,255,204,0.1)', zeroline: false, showticklabels: false, title: ''},
                zaxis: {autorange: 'reversed', gridcolor: 'rgba(0,255,204,0.3)', title: 'DEPTH (FT)', showbackground: true, backgroundcolor: 'rgba(5,5,5,0.8)'}
            }
        });

        Plotly.newPlot('3d-chart', [
            trace_payzone,
            {x: [], y: [], z: [], type: 'scatter3d', mode: 'lines', line: {color: 'rgba(0,255,204,0.1)', width: 10}},
            {x: [], y: [], z: [], type: 'scatter3d', mode: 'lines', line: {color: '#00ffcc', width: 3}, projection: {z: {show: true, opacity: 0.2}}},
            {x: [], y: [], z: [], type: 'scatter3d', mode: 'markers', marker: {color: 'rgba(255,0,85,0.3)', size: 18}},
            {x: [], y: [], z: [], type: 'scatter3d', mode: 'markers', marker: {color: '#ff0055', size: 6, symbol: 'diamond', line: {color: '#fff', width: 1}}},
            {x: [0], y: [150], z: [5000], type: 'scatter3d', mode: 'markers', marker: {color: '#ffea00', size: 6, symbol: 'square-open', line: {width: 2}}, hoverinfo: 'skip'}
        ], layout_3d);

        // --- MULTI-TRACK LOG SETUP ---
        var log_layout = Object.assign({}, layout_dark, {
            title: {text: 'LIVE PETROPHYSICS LOG (AI)', font: {color: '#fff', size: 14}},
            grid: {rows: 1, columns: 3, pattern: 'independent'},
            yaxis: {autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.05)', title: 'DEPTH (FT)'},
            yaxis2: {autorange: 'reversed', showticklabels: false, matches: 'y'},
            yaxis3: {autorange: 'reversed', showticklabels: false, matches: 'y'},
            xaxis: {gridcolor: 'rgba(255,255,255,0.05)', side: 'top', title: 'GAMMA (GR)', range: [0, 150]},
            xaxis2: {gridcolor: 'rgba(255,255,255,0.05)', side: 'top', title: 'DENSITY/NEUTRON', range: [1.95, 2.95]},
            xaxis3: {gridcolor: 'rgba(255,255,255,0.05)', side: 'top', title: 'SMART POROSITY', range: [0, 30]}
        });
        Plotly.newPlot('log-chart', [
            {x: [], y: [], type: 'scattergl', mode: 'lines', line: {color: '#00ffcc'}, xaxis: 'x', yaxis: 'y'},
            {x: [], y: [], type: 'scattergl', mode: 'lines', line: {color: '#ff3333'}, xaxis: 'x2', yaxis: 'y2'},
            {x: [], y: [], type: 'scattergl', mode: 'lines', line: {color: '#33aaff', dash: 'dot'}, xaxis: 'x2', yaxis: 'y2'},
            {x: [], y: [], type: 'scattergl', mode: 'lines', line: {color: '#ff0055', width: 2}, fill: 'tozerox', fillcolor: 'rgba(255,0,85,0.1)', xaxis: 'x3', yaxis: 'y3'}
        ], log_layout);

        // --- CROSSPLOT SETUP ---
        var cross_layout = Object.assign({}, layout_dark, {
            title: {text: 'LITHOLOGY CROSSPLOT', font: {color: '#fff', size: 14}},
            xaxis: {title: 'NEUTRON (NPHI)', autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.05)'},
            yaxis: {title: 'DENSITY (RHOB)', autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.05)'}
        });
        Plotly.newPlot('cross-chart', [{x: [], y: [], type: 'scattergl', mode: 'markers', marker: {color: [], colorscale: 'Viridis'}}], cross_layout);

        // --- HISTOGRAM SETUP ---
        var hist_layout = Object.assign({}, layout_dark, {
            title: {text: 'POROSITY DISTRIBUTION', font: {color: '#fff', size: 14}},
            xaxis: {title: 'POROSITY (%)', gridcolor: 'rgba(255,255,255,0.05)'},
            yaxis: {title: 'FREQUENCY', gridcolor: 'rgba(255,255,255,0.05)'},
            bargap: 0.1
        });
        Plotly.newPlot('hist-chart', [{x: [], type: 'histogram', marker: {color: 'rgba(255,0,85,0.7)', line: {color: '#ff0055', width: 1}}}], hist_layout);

        // --- WEBSOCKET EVENT LISTENER ---
        ws.onmessage = function(event) {
            var data = JSON.parse(event.data);

            document.getElementById('val-depth').innerText = data.curr_depth.toFixed(1);
            document.getElementById('val-por').innerText = data.curr_por.toFixed(2);
            document.getElementById('val-rhob').innerText = data.curr_rhob.toFixed(2);

            var statusEl = document.getElementById('val-status');
            if(data.is_running) {
                statusEl.innerText = "DRILLING & SCANNING...";
                statusEl.className = "kpi-value status-running";
            } else {
                statusEl.innerText = "PAUSED / READY";
                statusEl.className = "kpi-value status-ready";
            }

            Plotly.react('3d-chart', [
                trace_payzone,
                {x: data.x_arr, y: data.y_arr, z: data.dept_arr, type: 'scatter3d', mode: 'lines', line: {color: 'rgba(0,255,204,0.1)', width: 10}},
                {x: data.x_arr, y: data.y_arr, z: data.dept_arr, type: 'scatter3d', mode: 'lines', line: {color: '#00ffcc', width: 3}, projection: {z: {show: true, opacity: 0.2}}},
                {x: [data.x_arr[data.x_arr.length-1]], y: [data.y_arr[data.y_arr.length-1]], z: [data.dept_arr[data.dept_arr.length-1]], type: 'scatter3d', mode: 'markers', marker: {color: 'rgba(255,0,85,0.3)', size: 18}},
                {x: [data.x_arr[data.x_arr.length-1]], y: [data.y_arr[data.y_arr.length-1]], z: [data.dept_arr[data.dept_arr.length-1]], type: 'scatter3d', mode: 'markers', marker: {color: '#ff0055', size: 6, symbol: 'diamond', line: {color: '#fff', width: 1}}},
                {x: [0], y: [150], z: [5000], type: 'scatter3d', mode: 'markers', marker: {color: '#ffea00', size: 6, symbol: 'square-open', line: {width: 2}}, hoverinfo: 'skip'}
            ], layout_3d);

            Plotly.react('log-chart', [
                {x: data.gr_arr, y: data.dept_arr, type: 'scattergl', mode: 'lines', line: {color: '#00ffcc'}, xaxis: 'x', yaxis: 'y'},
                {x: data.rhob_arr, y: data.dept_arr, type: 'scattergl', mode: 'lines', line: {color: '#ff3333'}, xaxis: 'x2', yaxis: 'y2'},
                {x: data.nphi_arr, y: data.dept_arr, type: 'scattergl', mode: 'lines', line: {color: '#33aaff', dash: 'dot'}, xaxis: 'x2', yaxis: 'y2'},
                {x: data.por_arr, y: data.dept_arr, type: 'scattergl', mode: 'lines', line: {color: '#ff0055', width: 2}, fill: 'tozerox', fillcolor: 'rgba(255,0,85,0.1)', xaxis: 'x3', yaxis: 'y3'}
            ], log_layout);

            Plotly.react('cross-chart', [{
                x: data.nphi_arr, y: data.rhob_arr, type: 'scattergl', mode: 'markers',
                marker: {color: data.gr_arr, colorscale: 'Viridis', size: 5, showscale: true, colorbar: {title: 'GR', thickness: 10, len: 0.8}}
            }], cross_layout);

            Plotly.react('hist-chart', [{
                x: data.por_arr, type: 'histogram', marker: {color: 'rgba(255,0,85,0.7)', line: {color: '#ff0055', width: 1}}
            }], hist_layout);
        };
    </script>
</body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html_content)


# --- Section 5: WebSocket Loop ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    current_idx = 0
    is_running = False
    speed = 5

    async def listen_for_commands():
        nonlocal current_idx, is_running, speed
        try:
            while True:
                data = await websocket.receive_text()
                cmd = json.loads(data)
                if cmd['action'] == 'play':
                    is_running = True
                elif cmd['action'] == 'pause':
                    is_running = False
                elif cmd['action'] == 'reset':
                    current_idx = 0
                    is_running = False
        except WebSocketDisconnect:
            pass

    listener_task = asyncio.create_task(listen_for_commands())

    try:
        while True:
            if is_running:
                current_idx += speed
                if current_idx >= len(df_stream):
                    current_idx = len(df_stream) - 1
                    is_running = False

            window_size = 250
            start_idx = max(0, current_idx - window_size)
            current_df = df_stream.iloc[start_idx:current_idx + 1].copy()

            if model_ready:
                # حذف .values برای حفظ نام ستون‌ها و جلوگیری از خطای اسکیلر
                features_raw = current_df[['GR', 'RHOB', 'NPHI', 'DT', 'RESD']]
                features_scaled = scaler.transform(features_raw)

                with torch.no_grad():
                    tensor_x = torch.FloatTensor(features_scaled)
                    pytorch_features = torch_model(tensor_x).numpy()

                current_df['AI_POROSITY'] = xgb_model.predict(pytorch_features)
            else:
                current_df['AI_POROSITY'] = np.random.uniform(10, 20, len(current_df))

            payload = {
                'is_running': is_running,
                'curr_depth': float(current_df['DEPT'].iloc[-1]),
                'curr_por': float(current_df['AI_POROSITY'].iloc[-1]),
                'curr_rhob': float(current_df['RHOB'].iloc[-1]),
                'dept_arr': current_df['DEPT'].tolist(),
                'gr_arr': current_df['GR'].tolist(),
                'rhob_arr': current_df['RHOB'].tolist(),
                'nphi_arr': current_df['NPHI'].tolist(),
                'por_arr': current_df['AI_POROSITY'].tolist(),
                'x_arr': current_df['X'].tolist(),
                'y_arr': current_df['Y'].tolist()
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        listener_task.cancel()


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8050)