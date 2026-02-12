#!/usr/bin/env python3
"""
TN-NTN Handover Orchestration Dashboard
LightGBM inference + Celestrak satellite tracking + Stress testing + Handover latency
Production-ready containerized version.
"""
import os, sys, json, time, pickle, asyncio, random, math
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import numpy as np
import pandas as pd
import lightgbm as lgb
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

os.environ["PYTHONUNBUFFERED"] = "1"

# ── Configuration (container-relative paths) ─────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "lightgbm_ns3_20260210_222723.pkl"))
DATASET_PATH = os.environ.get("DATASET_PATH", str(BASE_DIR / "data" / "balanced_dataset.csv"))
PORT = int(os.environ.get("PORT", 8501))

CELESTRAK_URLS = {
    "starlink": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "oneweb":   "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
    "iridium":  "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle",
}
EARTH_R = 6371.0
C_LIGHT = 299792.458
F_CARRIER = 2.0e9

UE_LOCATIONS = {
    "NYC":       {"lat": 40.7128, "lon": -74.0060, "label": "New York"},
    "London":    {"lat": 51.5074, "lon": -0.1278,  "label": "London"},
    "Tokyo":     {"lat": 35.6762, "lon": 139.6503, "label": "Tokyo"},
    "Sydney":    {"lat": -33.8688,"lon": 151.2093, "label": "Sydney"},
    "Dubai":     {"lat": 25.2048, "lon": 55.2708,  "label": "Dubai"},
    "SaoPaulo":  {"lat": -23.5505,"lon": -46.6333, "label": "Sao Paulo"},
    "Rural_AK":  {"lat": 64.2008, "lon": -152.4937,"label": "Rural Alaska"},
    "Maritime":  {"lat": 30.0,    "lon": -40.0,    "label": "Mid-Atlantic"},
}
UE_IDS = ["UE-001", "UE-002", "UE-003", "UE-004", "UE-005", "UE-006", "UE-007", "UE-008"]

def log(msg): print(msg, flush=True)


# ── LightGBM Inference Engine ────────────────────────────────────────
class LightGBMInferenceEngine:
    def __init__(self, model_path: str):
        self.model = None
        self.feature_names = []
        self.label_map = {0: "TN", 1: "NTN"}
        self.metrics = {
            "total_predictions": 0, "avg_latency_ms": 0.0,
            "latency_history": deque(maxlen=1000),
            "tn_count": 0, "ntn_count": 0, "start_time": time.time(),
        }
        with open(model_path, "rb") as f:
            pkg = pickle.load(f)
        self.model = pkg["model"]
        self.feature_names = pkg["feature_names"]
        self.test_accuracy = pkg.get("test_accuracy", 0.999)
        self.test_auc = pkg.get("test_auc", 1.0)
        self.test_f1 = pkg.get("test_f1", 0.981)
        log(f"  Model loaded: {len(self.feature_names)} features, acc={self.test_accuracy:.4f}")

    def predict(self, features: dict) -> dict:
        t0 = time.perf_counter()
        row = pd.DataFrame([features])[self.feature_names]
        prob = float(self.model.predict(row)[0])
        decision = int(prob > 0.5)
        lat_ms = (time.perf_counter() - t0) * 1000
        self.metrics["total_predictions"] += 1
        self.metrics["latency_history"].append(lat_ms)
        self.metrics["avg_latency_ms"] = np.mean(list(self.metrics["latency_history"]))
        if decision == 0: self.metrics["tn_count"] += 1
        else: self.metrics["ntn_count"] += 1
        return {
            "timestamp": time.time(), "decision": decision,
            "label": self.label_map.get(decision, "?"),
            "probability_ntn": prob, "probability_tn": 1 - prob,
            "confidence": prob if decision == 1 else 1 - prob,
            "latency_ms": lat_ms,
            "features": {k: features.get(k, 0) for k in
                ["sinrTn", "sinrNtn", "rsrpTn", "rsrpNtn", "rsrp_gap", "sinr_gap",
                 "elevationDeg", "dopplerHz", "ueSpeed", "channelQuality"]},
        }


# ── Balanced NS-3 Data Streamer ──────────────────────────────────────
class BalancedNS3Streamer:
    def __init__(self, csv_path, feature_names, ntn_ratio=0.45):
        self.csv_path = csv_path
        self.feature_names = feature_names
        self.ntn_ratio = ntn_ratio
        self.tn_buf, self.ntn_buf = [], []
        self.tn_idx, self.ntn_idx = 0, 0
        self._load()

    def _load(self):
        log("  Loading balanced TN/NTN buffers...")
        cols = self.feature_names + ["best_cell"]
        df = pd.read_csv(self.csv_path, usecols=cols)
        tn = df[df["best_cell"] == 0].to_dict("records")
        ntn = df[df["best_cell"] == 1].to_dict("records")
        random.shuffle(tn)
        random.shuffle(ntn)
        self.tn_buf = tn
        self.ntn_buf = ntn
        self.tn_idx = 0
        self.ntn_idx = 0
        log(f"  Loaded {len(tn)} TN + {len(ntn)} NTN samples (mix={self.ntn_ratio:.0%} NTN)")

    def next(self) -> dict:
        use_ntn = random.random() < self.ntn_ratio and len(self.ntn_buf) > 0
        if use_ntn:
            if self.ntn_idx >= len(self.ntn_buf):
                random.shuffle(self.ntn_buf)
                self.ntn_idx = 0
            r = self.ntn_buf[self.ntn_idx]
            self.ntn_idx += 1
        else:
            if self.tn_idx >= len(self.tn_buf):
                random.shuffle(self.tn_buf)
                self.tn_idx = 0
            r = self.tn_buf[self.tn_idx]
            self.tn_idx += 1
        return r


# ── Handover Engine ──────────────────────────────────────────────────
class HandoverEngine:
    def __init__(self, ue_ids):
        self.ue_states = {}
        for uid in ue_ids:
            self.ue_states[uid] = {"network": 0, "since": time.time(), "ho_count": 0}
        self.total_handovers = 0
        self.tn_to_ntn = 0
        self.ntn_to_tn = 0
        self.handover_latencies = deque(maxlen=500)
        self.handover_log = deque(maxlen=100)
        self.pingpong_count = 0
        self.pingpong_window = 5.0
        self.ho_history = {}
        for uid in ue_ids:
            self.ho_history[uid] = deque(maxlen=10)

    def process_prediction(self, ue_id: str, prediction: dict, sat_info: dict = None) -> dict:
        decision = prediction["decision"]
        pred_latency_ms = prediction["latency_ms"]
        now = time.time()

        if ue_id not in self.ue_states:
            self.ue_states[ue_id] = {"network": decision, "since": now, "ho_count": 0}
            self.ho_history[ue_id] = deque(maxlen=10)

        prev_network = self.ue_states[ue_id]["network"]
        ho_event = None

        if decision != prev_network:
            self.total_handovers += 1
            self.ue_states[ue_id]["ho_count"] += 1

            if prev_network == 0 and decision == 1:
                direction = "TN_to_NTN"
                self.tn_to_ntn += 1
            else:
                direction = "NTN_to_TN"
                self.ntn_to_tn += 1

            prediction_lat = pred_latency_ms
            if sat_info and sat_info.get("distance_km", 0) > 0:
                prop_delay_ms = (sat_info["distance_km"] / C_LIGHT) * 2
            else:
                prop_delay_ms = random.uniform(4.0, 8.0)

            if direction == "TN_to_NTN":
                prep_ms = random.uniform(35, 55)
                exec_ms = random.uniform(12, 25)
                complete_ms = random.uniform(25, 45)
            else:
                prep_ms = random.uniform(25, 40)
                exec_ms = random.uniform(8, 18)
                complete_ms = random.uniform(15, 30)

            total_ho_latency = prediction_lat + prop_delay_ms + prep_ms + exec_ms + complete_ms
            self.handover_latencies.append(total_ho_latency)

            is_pingpong = False
            history = self.ho_history[ue_id]
            for prev_ho in history:
                if (now - prev_ho["time"] <= self.pingpong_window
                        and prev_ho["from"] == decision and prev_ho["to"] == prev_network):
                    is_pingpong = True
                    self.pingpong_count += 1
                    break
            history.append({"time": now, "from": prev_network, "to": decision})

            self.ue_states[ue_id]["network"] = decision
            self.ue_states[ue_id]["since"] = now

            ho_event = {
                "type": "HANDOVER", "ue_id": ue_id,
                "direction": direction,
                "direction_label": "TN -> NTN" if direction == "TN_to_NTN" else "NTN -> TN",
                "prediction_ms": round(prediction_lat, 3),
                "propagation_ms": round(prop_delay_ms, 3),
                "preparation_ms": round(prep_ms, 3),
                "execution_ms": round(exec_ms, 3),
                "completion_ms": round(complete_ms, 3),
                "total_latency_ms": round(total_ho_latency, 2),
                "is_pingpong": is_pingpong,
                "timestamp": now, "confidence": prediction["confidence"],
            }
            self.handover_log.appendleft(ho_event)

        return {
            "ue_id": ue_id, "current_network": decision,
            "handover_event": ho_event,
            "ue_ho_count": self.ue_states[ue_id]["ho_count"],
        }

    def get_metrics(self) -> dict:
        lats = list(self.handover_latencies)
        if len(lats) >= 3:
            s = sorted(lats)
            p50, p95, p99 = s[len(s)//2], s[int(len(s)*0.95)], s[int(len(s)*0.99)]
            avg, mx, mn = sum(lats)/len(lats), max(lats), min(lats)
        else:
            p50 = p95 = p99 = avg = mx = mn = 0
        return {
            "total_handovers": self.total_handovers,
            "tn_to_ntn": self.tn_to_ntn, "ntn_to_tn": self.ntn_to_tn,
            "pingpong_count": self.pingpong_count,
            "ho_latency_avg_ms": round(avg, 2), "ho_latency_p50_ms": round(p50, 2),
            "ho_latency_p95_ms": round(p95, 2), "ho_latency_p99_ms": round(p99, 2),
            "ho_latency_max_ms": round(mx, 2), "ho_latency_min_ms": round(mn, 2),
            "recent_handovers": list(self.handover_log)[:10],
        }


# ── Celestrak Tracker ────────────────────────────────────────────────
class CelestrakLive:
    def __init__(self):
        self.satellites = {}
        self.last_fetch = 0
        self.fetch_count = 0
        self.total_sats = 0
        self.constellation_counts = {}

    async def fetch(self, constellation="starlink", max_sats=80):
        url = CELESTRAK_URLS.get(constellation, CELESTRAK_URLS["starlink"])
        log(f"  Fetching {constellation} from Celestrak...")
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()
        lines = r.text.strip().split("\n")
        count, i = 0, 0
        while i < len(lines) - 2 and count < max_sats:
            name = lines[i].strip()
            l1, l2 = lines[i+1].strip(), lines[i+2].strip()
            if l1.startswith("1 ") and l2.startswith("2 "):
                norad = int(l1[2:7])
                incl = float(l2[8:16])
                mm = float(l2[52:63])
                mu = 398600.4418
                n_rad = mm * 2 * math.pi / 86400
                a = (mu / n_rad**2)**(1/3)
                alt = a - EARTH_R
                self.satellites[name] = {
                    "line1": l1, "line2": l2, "norad_id": norad,
                    "inclination": incl, "alt_km": alt, "constellation": constellation,
                }
                count += 1; i += 3
            else:
                i += 1
        self.last_fetch = time.time()
        self.fetch_count += 1
        self.constellation_counts[constellation] = count
        self.total_sats = len(self.satellites)
        log(f"  Loaded {count} {constellation} satellites (total: {self.total_sats})")

    def observe(self, sat_name, obs_lat, obs_lon, obs_alt_m=0, ts=None):
        if sat_name not in self.satellites: return None
        sat = self.satellites[sat_name]
        if ts is None: ts = time.time()
        l1, l2 = sat["line1"], sat["line2"]
        ey = int(l1[18:20])
        ey = ey + 2000 if ey < 57 else ey + 1900
        ed = float(l1[20:32])
        epoch_ts = datetime(ey,1,1,tzinfo=timezone.utc).timestamp() + (ed-1)*86400
        dt_sec = ts - epoch_ts
        ecc = float("0." + l2[26:33])
        argp = math.radians(float(l2[34:42]))
        ma0 = math.radians(float(l2[43:51]))
        raan0 = math.radians(float(l2[17:25]))
        incl = math.radians(float(l2[8:16]))
        mm = float(l2[52:63])
        n = mm * 2 * math.pi / 86400
        mu = 398600.4418
        a = (mu / n**2)**(1/3)
        M = (ma0 + n * dt_sec) % (2*math.pi)
        E = M
        for _ in range(8): E = M + ecc * math.sin(E)
        nu = 2 * math.atan2(math.sqrt(1+ecc)*math.sin(E/2), math.sqrt(1-ecc)*math.cos(E/2))
        r = a * (1 - ecc*math.cos(E))
        xo, yo = r*math.cos(nu), r*math.sin(nu)
        er = (dt_sec/86400) * 2*math.pi * 1.00273790935
        raan = raan0 - er
        cos_o, sin_o = math.cos(argp), math.sin(argp)
        cos_R, sin_R = math.cos(raan), math.sin(raan)
        cos_i, sin_i = math.cos(incl), math.sin(incl)
        x = xo*(cos_o*cos_R - sin_o*sin_R*cos_i) - yo*(sin_o*cos_R + cos_o*sin_R*cos_i)
        y = xo*(cos_o*sin_R + sin_o*cos_R*cos_i) - yo*(sin_o*sin_R - cos_o*cos_R*cos_i)
        z = xo*sin_o*sin_i + yo*cos_o*sin_i
        lat_r, lon_r = math.radians(obs_lat), math.radians(obs_lon)
        ae = 6378.137; f = 1/298.257223563; e2 = f*(2-f)
        N_geo = ae / math.sqrt(1 - e2*math.sin(lat_r)**2)
        alt_km = obs_alt_m / 1000
        ox = (N_geo+alt_km)*math.cos(lat_r)*math.cos(lon_r)
        oy = (N_geo+alt_km)*math.cos(lat_r)*math.sin(lon_r)
        oz = (N_geo*(1-e2)+alt_km)*math.sin(lat_r)
        dx, dy, dz = x-ox, y-oy, z-oz
        dist = math.sqrt(dx*dx+dy*dy+dz*dz)
        e_enu = -math.sin(lon_r)*dx + math.cos(lon_r)*dy
        n_enu = -math.sin(lat_r)*math.cos(lon_r)*dx - math.sin(lat_r)*math.sin(lon_r)*dy + math.cos(lat_r)*dz
        u_enu = math.cos(lat_r)*math.cos(lon_r)*dx + math.cos(lat_r)*math.sin(lon_r)*dy + math.sin(lat_r)*dz
        elev = math.degrees(math.atan2(u_enu, math.sqrt(e_enu**2 + n_enu**2)))
        azim = math.degrees(math.atan2(e_enu, n_enu)) % 360
        vorb = math.sqrt(mu*(2/r - 1/a))
        rr = vorb * math.cos(math.radians(elev)) * 0.3
        doppler = -F_CARRIER * rr / (C_LIGHT*1000)
        wl = C_LIGHT / (F_CARRIER/1e9)
        fspl = 20*math.log10(4*math.pi*dist/wl) if dist > 0 else 0
        atm = 2 + abs(90-elev)*0.1
        rsrp = 30 + 35 - fspl - atm
        return {
            "name": sat_name, "norad_id": sat["norad_id"],
            "elevation": round(elev,2), "azimuth": round(azim,2),
            "distance_km": round(dist,1), "doppler_hz": round(doppler,1),
            "rsrp_dbm": round(rsrp,1), "alt_km": round(sat["alt_km"],1),
            "visible": elev >= 10, "constellation": sat["constellation"],
        }

    def get_visible(self, lat, lon, alt_m=0, min_elev=10):
        visible = []
        for name in self.satellites:
            obs = self.observe(name, lat, lon, alt_m)
            if obs and obs["elevation"] >= min_elev:
                visible.append(obs)
        visible.sort(key=lambda x: x["elevation"], reverse=True)
        return visible


# ── Stress Test Engine ────────────────────────────────────────────────
class StressTest:
    def __init__(self):
        self.running = False
        self.results = self._empty()
    def _empty(self):
        return {"total":0,"correct":0,"errors":0,"latencies":[],"p50":0,"p95":0,"p99":0,"max":0,"throughput":0,"start_time":0,"elapsed":0,"mode":"","ue_count":0}
    def reset(self):
        self.results = self._empty(); self.results["start_time"] = time.time()
    def record(self, lat_ms, correct):
        self.results["total"] += 1
        if correct: self.results["correct"] += 1
        self.results["latencies"].append(lat_ms)
        elapsed = time.time() - self.results["start_time"]
        self.results["elapsed"] = elapsed
        self.results["throughput"] = self.results["total"] / max(elapsed, 0.001)
        lats = self.results["latencies"]
        if len(lats) >= 5:
            s = sorted(lats)
            self.results["p50"]=s[len(s)//2]; self.results["p95"]=s[int(len(s)*0.95)]
            self.results["p99"]=s[int(len(s)*0.99)]; self.results["max"]=s[-1]
    def summary(self):
        r=self.results
        return {"total":r["total"],"correct":r["correct"],"errors":r["errors"],
                "accuracy":r["correct"]/max(r["total"],1),
                "p50_ms":round(r["p50"],3),"p95_ms":round(r["p95"],3),
                "p99_ms":round(r["p99"],3),"max_ms":round(r["max"],3),
                "throughput_rps":round(r["throughput"],1),
                "elapsed_s":round(r["elapsed"],1),"mode":r["mode"],"ue_count":r["ue_count"]}


# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(title="TN-NTN Handover Orchestration Dashboard")
engine: LightGBMInferenceEngine = None
streamer: BalancedNS3Streamer = None
celestrak: CelestrakLive = None
stress: StressTest = None
handover_engine: HandoverEngine = None

@app.on_event("startup")
async def startup():
    global engine, streamer, celestrak, stress, handover_engine
    log("=" * 60)
    log("TN-NTN Handover Orchestration Dashboard")
    log("=" * 60)
    engine = LightGBMInferenceEngine(MODEL_PATH)
    streamer = BalancedNS3Streamer(DATASET_PATH, engine.feature_names, ntn_ratio=0.45)
    celestrak = CelestrakLive()
    stress = StressTest()
    handover_engine = HandoverEngine(UE_IDS)
    try:
        await celestrak.fetch("starlink", max_sats=80)
    except Exception as e:
        log(f"  Celestrak fetch warning: {e} (will retry)")
    log(f"  Dashboard: http://0.0.0.0:{PORT}")
    log("=" * 60)


# ── API Routes ────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    m=engine.metrics; ho=handover_engine.get_metrics()
    return {"model":True,"predictions":m["total_predictions"],"avg_lat_ms":round(m["avg_latency_ms"],3),
            "tn":m["tn_count"],"ntn":m["ntn_count"],"uptime":round(time.time()-m["start_time"],1),
            "accuracy":engine.test_accuracy,"celestrak_sats":celestrak.total_sats,
            "celestrak_fetches":celestrak.fetch_count,"handovers":ho}

@app.get("/api/predict")
async def api_predict():
    row=streamer.next(); feat={k:row[k] for k in engine.feature_names if k in row}
    r=engine.predict(feat); r["ground_truth"]=int(row.get("best_cell",-1)); return r

@app.get("/api/handover/metrics")
async def api_ho_metrics():
    return handover_engine.get_metrics()

@app.get("/api/celestrak/fetch")
async def api_celestrak_fetch(constellation:str="starlink"):
    try:
        await celestrak.fetch(constellation, max_sats=80)
        return {"ok":True,"total":celestrak.total_sats,"constellation":constellation,"counts":celestrak.constellation_counts}
    except Exception as e:
        return {"ok":False,"error":str(e)}

@app.get("/api/celestrak/visible")
async def api_celestrak_visible(ue:str="NYC"):
    loc=UE_LOCATIONS.get(ue,UE_LOCATIONS["NYC"])
    sats=celestrak.get_visible(loc["lat"],loc["lon"])
    return {"ue":ue,"location":loc,"visible_count":len(sats),"satellites":sats[:20]}

@app.get("/api/stress/start")
async def api_stress_start(mode:str="burst",count:int=500,ues:int=4):
    stress.reset(); stress.running=True; stress.results["mode"]=mode; stress.results["ue_count"]=ues
    asyncio.create_task(_run_stress(mode,count,ues))
    return {"started":True,"mode":mode,"count":count,"ues":ues}

@app.get("/api/stress/stop")
async def api_stress_stop():
    stress.running=False; return stress.summary()

@app.get("/api/stress/status")
async def api_stress_status():
    return stress.summary()

async def _run_stress(mode, count, ues):
    for i in range(count):
        if not stress.running: break
        try:
            row=streamer.next(); feat={k:row[k] for k in engine.feature_names if k in row}
            r=engine.predict(feat); gt=int(row.get("best_cell",-1))
            stress.record(r["latency_ms"], r["decision"]==gt)
        except: stress.results["errors"]+=1
        if mode=="sustained": await asyncio.sleep(0.01)
        elif mode=="ramp": await asyncio.sleep(max(0.001, 0.05-i*0.0001))
    stress.running=False


# ── WebSocket Stream ─────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    ue_cycle = 0
    try:
        while True:
            ue_id = UE_IDS[ue_cycle % len(UE_IDS)]; ue_cycle += 1
            row = streamer.next()
            feat = {k: row[k] for k in engine.feature_names if k in row}
            r = engine.predict(feat)
            r["ground_truth"] = int(row.get("best_cell", -1))
            r["ue_id"] = ue_id

            sat_info = None
            if celestrak.total_sats > 0:
                loc = random.choice(list(UE_LOCATIONS.values()))
                vis = celestrak.get_visible(loc["lat"], loc["lon"])
                r["sat_visible"] = len(vis)
                if vis:
                    best = vis[0]
                    r["sat_best"]=best["name"]; r["sat_elev"]=best["elevation"]
                    r["sat_dist"]=best["distance_km"]; r["sat_doppler"]=best["doppler_hz"]
                    r["sat_rsrp"]=best["rsrp_dbm"]; sat_info=best
                else:
                    r["sat_best"]="None"; r["sat_elev"]=r["sat_dist"]=r["sat_doppler"]=r["sat_rsrp"]=0
            else:
                r["sat_visible"]=0; r["sat_best"]="N/A"
                r["sat_elev"]=r["sat_dist"]=r["sat_doppler"]=r["sat_rsrp"]=0

            ho_result = handover_engine.process_prediction(ue_id, r, sat_info)
            r["handover"] = ho_result
            r["ho_metrics"] = handover_engine.get_metrics()
            r["stress_running"] = stress.running
            if stress.running: r["stress"] = stress.summary()

            await ws.send_json(r)
            await asyncio.sleep(0.18)
    except WebSocketDisconnect:
        pass


# ── Dashboard HTML ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>TN-NTN Handover Orchestration Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e1a;color:#e0e6ed;font-family:'Segoe UI',system-ui,sans-serif;overflow-x:hidden}
.hdr{background:linear-gradient(135deg,#0d1b2a,#1b2838);padding:14px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #00e5ff33;flex-wrap:wrap;gap:8px}
.hdr h1{font-size:18px;background:linear-gradient(90deg,#00e5ff,#76ff03);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hdr .st{display:flex;gap:10px;align-items:center;font-size:11px}
.dot-live{width:9px;height:9px;border-radius:50%;background:#76ff03;box-shadow:0 0 6px #76ff03;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.tabs{display:flex;gap:4px;padding:8px 24px;background:#0d111e}
.tab{padding:7px 18px;border-radius:6px 6px 0 0;cursor:pointer;font-size:12px;font-weight:600;background:#111827;color:#667788;border:1px solid #ffffff08;border-bottom:none;transition:all .2s}
.tab.active{background:#1a2332;color:#00e5ff;border-color:#00e5ff33}
.tab:hover{color:#00e5ff88}
.page{display:none;padding:14px 20px}.page.active{display:block}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
.card{background:linear-gradient(145deg,#111827,#1a2332);border-radius:10px;padding:14px;border:1px solid #ffffff0a;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:10px 10px 0 0}
.card.c::before{background:linear-gradient(90deg,#00e5ff,#00b8d4)}
.card.g::before{background:linear-gradient(90deg,#76ff03,#64dd17)}
.card.o::before{background:linear-gradient(90deg,#ff9100,#ff6d00)}
.card.p::before{background:linear-gradient(90deg,#e040fb,#aa00ff)}
.card.r::before{background:linear-gradient(90deg,#ff5252,#d50000)}
.card.y::before{background:linear-gradient(90deg,#ffea00,#ffd600)}
.ct{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:#8899aa;margin-bottom:4px}
.cv{font-size:26px;font-weight:700;line-height:1.1}
.cs{font-size:10px;color:#667788;margin-top:2px}
.pnl{background:linear-gradient(145deg,#111827,#1a2332);border-radius:10px;padding:14px;border:1px solid #ffffff0a;margin-bottom:10px}
.pnl-t{font-size:12px;font-weight:600;margin-bottom:8px;color:#00e5ff}
.bar-row{display:flex;height:32px;border-radius:8px;overflow:hidden;margin:6px 0}
.bar-tn{background:linear-gradient(90deg,#00e5ff,#00b8d4);transition:width .3s;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;min-width:35px}
.bar-ntn{background:linear-gradient(90deg,#ff9100,#ff6d00);transition:width .3s;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;min-width:35px}
.fg{display:grid;grid-template-columns:repeat(2,1fr);gap:5px}
.fi{background:#0d1117;border-radius:6px;padding:7px 9px;display:flex;justify-content:space-between;align-items:center}
.fn{font-size:9px;color:#8899aa;text-transform:uppercase}.fv{font-size:13px;font-weight:600;font-family:monospace}
.log{max-height:200px;overflow-y:auto;font-family:monospace;font-size:10px;line-height:1.5}
.le{padding:2px 6px;border-radius:3px;margin-bottom:1px}
.le-tn{background:#00e5ff08;border-left:3px solid #00e5ff}
.le-ntn{background:#ff910008;border-left:3px solid #ff9100}
.le-ho{background:#e040fb10;border-left:3px solid #e040fb}
.badge{padding:1px 6px;border-radius:8px;font-size:9px;font-weight:600}
.b-tn{background:#00e5ff22;color:#00e5ff}.b-ntn{background:#ff910022;color:#ff9100}
.b-ho{background:#e040fb22;color:#e040fb}
.btn{padding:7px 16px;border:none;border-radius:6px;cursor:pointer;font-weight:600;font-size:11px;margin:3px}
.btn-cyan{background:#00e5ff;color:#0a0e1a}.btn-orange{background:#ff9100;color:#0a0e1a}
.btn-green{background:#76ff03;color:#0a0e1a}.btn-red{background:#ff5252;color:#fff}
.btn-purple{background:#e040fb;color:#0a0e1a}
.btn:hover{opacity:.85}
.sat-row{display:flex;align-items:center;gap:8px;padding:4px 8px;background:#0d1117;border-radius:6px;margin-bottom:3px;font-size:10px}
.sat-name{width:150px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sat-val{width:65px;text-align:right;font-family:monospace;font-size:10px}
.spark{flex:1;background:#0d1117;border-radius:6px;padding:6px;text-align:center}
.spark-t{font-size:8px;color:#667788;text-transform:uppercase;margin-bottom:2px}
.spark canvas{height:36px!important}
.stress-res{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:6px}
.sr-item{background:#0d1117;border-radius:6px;padding:6px;text-align:center}
.sr-label{font-size:8px;color:#667788;text-transform:uppercase}.sr-val{font-size:16px;font-weight:700;margin-top:1px}
.ho-event{background:#e040fb08;border:1px solid #e040fb22;border-radius:8px;padding:10px;margin-bottom:6px}
.ho-event .ho-dir{font-size:14px;font-weight:700;margin-bottom:4px}
.ho-event .ho-detail{font-size:10px;color:#aab;font-family:monospace;line-height:1.6}
.ho-bar{display:flex;gap:2px;height:22px;border-radius:4px;overflow:hidden;margin:4px 0}
.ho-bar div{display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:600;color:#0a0e1a;min-width:20px}
.ho-bar .hb-pred{background:#00e5ff}.ho-bar .hb-prop{background:#76ff03}
.ho-bar .hb-prep{background:#ffea00}.ho-bar .hb-exec{background:#ff9100}.ho-bar .hb-comp{background:#e040fb}
@media(max-width:900px){.grid4{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="hdr">
  <h1>TN-NTN Handover Orchestration Dashboard</h1>
  <div class="st">
    <span style="color:#667788">LightGBM 99.9%</span>|
    <span id="hdr-ho" style="color:#e040fb">0 handovers</span>|
    <span id="sat-count">0 sats</span>|
    <span id="hdr-preds">0 preds</span>
    <div class="dot-live"></div>
    <span style="color:#76ff03;font-weight:600">LIVE</span>
  </div>
</div>
<div class="tabs">
  <div class="tab active" data-tab="inference">Inference + Handovers</div>
  <div class="tab" data-tab="celestrak">Celestrak Live</div>
  <div class="tab" data-tab="stress">Stress Test</div>
</div>

<!-- TAB 1 -->
<div class="page active" id="page-inference">
<div class="grid4">
  <div class="card c"><div class="ct">Predictions</div><div class="cv" id="k-preds">0</div><div class="cs" id="k-ratio">TN: 0 | NTN: 0</div></div>
  <div class="card g"><div class="ct">Inference Latency</div><div class="cv" id="k-lat">--</div><div class="cs">Model P(NTN)</div></div>
  <div class="card o"><div class="ct">Decision</div><div class="cv" id="k-dec">--</div><div class="cs" id="k-conf">--</div></div>
  <div class="card p"><div class="ct">UE</div><div class="cv" id="k-ue" style="font-size:18px">--</div><div class="cs" id="k-ue-net">--</div></div>
</div>
<div class="grid4">
  <div class="card p"><div class="ct">Total Handovers</div><div class="cv" id="ho-total">0</div><div class="cs" id="ho-dirs">TN->NTN: 0 | NTN->TN: 0</div></div>
  <div class="card y"><div class="ct">HO Latency Avg</div><div class="cv" id="ho-avg">--</div><div class="cs" id="ho-p95">P95: --</div></div>
  <div class="card r"><div class="ct">Last HO Latency</div><div class="cv" id="ho-last">--</div><div class="cs" id="ho-last-dir">--</div></div>
  <div class="card c"><div class="ct">Ping-Pong</div><div class="cv" id="ho-pp">0</div><div class="cs">Rapid reversals (&lt;5s)</div></div>
</div>
<div class="grid2">
  <div>
    <div class="pnl">
      <div class="pnl-t">Live Prediction</div>
      <div style="display:flex;justify-content:space-between;font-size:10px;margin-bottom:2px"><span>TN <span id="tn-p">--</span></span><span><span id="ntn-p">--</span> NTN</span></div>
      <div class="bar-row"><div class="bar-tn" id="b-tn" style="width:50%">50%</div><div class="bar-ntn" id="b-ntn" style="width:50%">50%</div></div>
      <div class="fg">
        <div class="fi"><span class="fn">SINR TN</span><span class="fv" id="f-sinrTn">--</span></div>
        <div class="fi"><span class="fn">SINR NTN</span><span class="fv" id="f-sinrNtn">--</span></div>
        <div class="fi"><span class="fn">RSRP TN</span><span class="fv" id="f-rsrpTn">--</span></div>
        <div class="fi"><span class="fn">RSRP NTN</span><span class="fv" id="f-rsrpNtn">--</span></div>
        <div class="fi"><span class="fn">RSRP Gap</span><span class="fv" id="f-rsrp_gap">--</span></div>
        <div class="fi"><span class="fn">Elevation</span><span class="fv" id="f-elevationDeg">--</span></div>
        <div class="fi"><span class="fn">Doppler</span><span class="fv" id="f-dopplerHz">--</span></div>
        <div class="fi"><span class="fn">UE Speed</span><span class="fv" id="f-ueSpeed">--</span></div>
      </div>
    </div>
    <div class="pnl">
      <div class="pnl-t">Sparklines</div>
      <div style="display:flex;gap:8px">
        <div class="spark"><div class="spark-t">Inference Lat</div><canvas id="sp-lat" height="36"></canvas></div>
        <div class="spark"><div class="spark-t">P(NTN)</div><canvas id="sp-prob" height="36"></canvas></div>
        <div class="spark"><div class="spark-t">HO Latency</div><canvas id="sp-ho" height="36"></canvas></div>
      </div>
    </div>
  </div>
  <div>
    <div class="pnl">
      <div class="pnl-t">Last Handover Event</div>
      <div id="ho-event-box"><div style="color:#667788;font-size:11px">Waiting for handover...</div></div>
    </div>
    <div class="pnl">
      <div class="pnl-t">Handover Latency Breakdown</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;font-size:9px;margin-bottom:6px">
        <span style="color:#00e5ff">Prediction</span><span style="color:#76ff03">Propagation</span>
        <span style="color:#ffea00">Preparation</span><span style="color:#ff9100">Execution</span><span style="color:#e040fb">Completion</span>
      </div>
      <div id="ho-breakdown-bar" class="ho-bar" style="height:26px">
        <div class="hb-pred" style="width:20%">--</div><div class="hb-prop" style="width:10%">--</div>
        <div class="hb-prep" style="width:30%">--</div><div class="hb-exec" style="width:15%">--</div><div class="hb-comp" style="width:25%">--</div>
      </div>
      <div style="text-align:right;font-size:10px;color:#667788;margin-top:2px">Total: <span id="ho-brk-total" style="color:#e040fb;font-weight:700">--</span></div>
    </div>
    <div class="pnl">
      <div class="pnl-t">Prediction + Handover Log</div>
      <div class="log" id="log"></div>
    </div>
  </div>
</div>
</div>

<!-- TAB 2 -->
<div class="page" id="page-celestrak">
<div class="grid4">
  <div class="card c"><div class="ct">Total Satellites</div><div class="cv" id="c-total">0</div><div class="cs" id="c-fetch">Not fetched</div></div>
  <div class="card g"><div class="ct">Visible Now</div><div class="cv" id="c-vis">0</div><div class="cs" id="c-ue">--</div></div>
  <div class="card o"><div class="ct">Best Elevation</div><div class="cv" id="c-elev">--</div><div class="cs" id="c-best">--</div></div>
  <div class="card p"><div class="ct">Fetches</div><div class="cv" id="c-fetches">0</div><div class="cs">API calls</div></div>
</div>
<div class="grid2">
  <div class="pnl">
    <div class="pnl-t">Fetch Constellation</div>
    <div style="margin-bottom:8px">
      <button class="btn btn-cyan" onclick="fetchConstellation('starlink')">Starlink</button>
      <button class="btn btn-orange" onclick="fetchConstellation('oneweb')">OneWeb</button>
      <button class="btn btn-purple" onclick="fetchConstellation('iridium')">Iridium</button>
    </div>
    <div class="pnl-t">Select UE Location</div>
    <div id="ue-buttons"></div>
    <div id="fetch-status" style="margin-top:6px;font-size:10px;color:#667788"></div>
  </div>
  <div class="pnl">
    <div class="pnl-t">Visible Satellites (Top 15)</div>
    <div id="sat-list" style="max-height:280px;overflow-y:auto"></div>
  </div>
</div>
</div>

<!-- TAB 3 -->
<div class="page" id="page-stress">
<div class="grid4">
  <div class="card c"><div class="ct">Total Requests</div><div class="cv" id="st-total">0</div><div class="cs" id="st-mode">--</div></div>
  <div class="card g"><div class="ct">Throughput</div><div class="cv" id="st-thr">0/s</div><div class="cs" id="st-elapsed">--</div></div>
  <div class="card o"><div class="ct">P95 Latency</div><div class="cv" id="st-p95">--</div><div class="cs" id="st-p99">P99: --</div></div>
  <div class="card r"><div class="ct">Live Accuracy</div><div class="cv" id="st-acc">--</div><div class="cs" id="st-err">Errors: 0</div></div>
</div>
<div class="grid2">
  <div class="pnl">
    <div class="pnl-t">Launch Stress Test</div>
    <p style="font-size:10px;color:#667788;margin-bottom:8px">Stream balanced TN/NTN through model under load</p>
    <div style="margin-bottom:6px">
      <button class="btn btn-cyan" onclick="startStress('burst',1000,1)">Burst 1K</button>
      <button class="btn btn-orange" onclick="startStress('burst',5000,4)">Burst 5K</button>
      <button class="btn btn-green" onclick="startStress('sustained',10000,8)">Sustained 10K</button>
      <button class="btn btn-purple" onclick="startStress('ramp',5000,4)">Ramp 5K</button>
    </div>
    <div><button class="btn btn-red" onclick="stopStress()">STOP</button></div>
    <div id="stress-log" style="margin-top:8px;font-size:10px;color:#667788"></div>
  </div>
  <div class="pnl">
    <div class="pnl-t">Stress Test Results</div>
    <div class="stress-res">
      <div class="sr-item"><div class="sr-label">Total</div><div class="sr-val" id="sr-total">0</div></div>
      <div class="sr-item"><div class="sr-label">Correct</div><div class="sr-val" id="sr-correct" style="color:#76ff03">0</div></div>
      <div class="sr-item"><div class="sr-label">P50 ms</div><div class="sr-val" id="sr-p50">--</div></div>
      <div class="sr-item"><div class="sr-label">P95 ms</div><div class="sr-val" id="sr-p95">--</div></div>
      <div class="sr-item"><div class="sr-label">P99 ms</div><div class="sr-val" id="sr-p99">--</div></div>
      <div class="sr-item"><div class="sr-label">Max ms</div><div class="sr-val" id="sr-max">--</div></div>
      <div class="sr-item"><div class="sr-label">RPS</div><div class="sr-val" id="sr-rps" style="color:#00e5ff">--</div></div>
      <div class="sr-item"><div class="sr-label">Errors</div><div class="sr-val" id="sr-err" style="color:#ff5252">0</div></div>
    </div>
    <div style="display:flex;gap:8px;margin-top:8px">
      <div class="spark"><div class="spark-t">Stress Latency</div><canvas id="sp-stress" height="36"></canvas></div>
    </div>
  </div>
</div>
</div>

<script>
const WS=`ws://${location.host}/ws/stream`;
let ws,totalP=0,tnC=0,ntnC=0;
const latBuf=[],probBuf=[],hoBuf=[],stressBuf=[];

document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('page-'+t.dataset.tab).classList.add('active');
  });
});

const ues={"NYC":"New York","London":"London","Tokyo":"Tokyo","Sydney":"Sydney","Dubai":"Dubai","SaoPaulo":"Sao Paulo","Rural_AK":"Alaska","Maritime":"Atlantic"};
let selectedUE="NYC";
(function(){const c=document.getElementById('ue-buttons');let h='';for(const[k,v]of Object.entries(ues))h+=`<button class="btn btn-cyan" style="font-size:10px;padding:4px 10px" onclick="selectUE('${k}')">${v}</button>`;c.innerHTML=h})();
function selectUE(ue){selectedUE=ue;fetchVisible(ue)}

async function fetchConstellation(c){
  document.getElementById('fetch-status').textContent='Fetching '+c+'...';
  const r=await fetch('/api/celestrak/fetch?constellation='+c).then(r=>r.json());
  document.getElementById('fetch-status').textContent=r.ok?`Loaded ${r.total} satellites`:`Error: ${r.error}`;
  document.getElementById('c-total').textContent=r.total||0;
  document.getElementById('c-fetches').textContent=r.ok?(parseInt(document.getElementById('c-fetches').textContent)+1):document.getElementById('c-fetches').textContent;
  if(r.ok)fetchVisible(selectedUE);
}
async function fetchVisible(ue){
  const r=await fetch('/api/celestrak/visible?ue='+ue).then(r=>r.json());
  document.getElementById('c-vis').textContent=r.visible_count;
  document.getElementById('c-ue').textContent=r.location.label;
  const sats=r.satellites||[];
  if(sats.length>0){document.getElementById('c-elev').textContent=sats[0].elevation+'\u00b0';document.getElementById('c-best').textContent=sats[0].name.slice(0,20)}
  let h='';sats.slice(0,15).forEach(s=>{h+=`<div class="sat-row"><span class="sat-name">${s.name}</span><span class="sat-val">${s.elevation}\u00b0</span><span class="sat-val">${s.distance_km}km</span><span class="sat-val">${s.rsrp_dbm}dBm</span><span class="sat-val" style="color:${s.visible?'#76ff03':'#ff5252'}">${s.visible?'VIS':'---'}</span></div>`});
  document.getElementById('sat-list').innerHTML=h||'<span style="color:#667788">No visible satellites. Fetch first.</span>';
}

async function startStress(mode,count,ues){document.getElementById('stress-log').textContent=`Starting ${mode}: ${count} req...`;await fetch(`/api/stress/start?mode=${mode}&count=${count}&ues=${ues}`);pollStress()}
async function stopStress(){await fetch('/api/stress/stop');document.getElementById('stress-log').textContent='Stopped.'}
async function pollStress(){
  while(true){
    const r=await fetch('/api/stress/status').then(r=>r.json());
    document.getElementById('st-total').textContent=r.total;document.getElementById('st-thr').textContent=r.throughput_rps+'/s';
    document.getElementById('st-p95').textContent=r.p95_ms+'ms';document.getElementById('st-p99').textContent='P99: '+r.p99_ms+'ms';
    document.getElementById('st-acc').textContent=(r.accuracy*100).toFixed(2)+'%';document.getElementById('st-err').textContent='Errors: '+r.errors;
    document.getElementById('st-mode').textContent=r.mode+' / '+r.ue_count+' UE';document.getElementById('st-elapsed').textContent=r.elapsed_s+'s';
    document.getElementById('sr-total').textContent=r.total;document.getElementById('sr-correct').textContent=r.correct;
    document.getElementById('sr-p50').textContent=r.p50_ms;document.getElementById('sr-p95').textContent=r.p95_ms;
    document.getElementById('sr-p99').textContent=r.p99_ms;document.getElementById('sr-max').textContent=r.max_ms;
    document.getElementById('sr-rps').textContent=r.throughput_rps;document.getElementById('sr-err').textContent=r.errors;
    stressBuf.push(r.p95_ms);if(stressBuf.length>100)stressBuf.shift();drawSpark('sp-stress',stressBuf,'#ff9100');
    if(r.total>0&&r.throughput_rps===0)break;await new Promise(r=>setTimeout(r,300));
  }
  document.getElementById('stress-log').textContent='Test complete.';
}

function drawSpark(id,data,color){
  const c=document.getElementById(id);if(!c)return;
  const ctx=c.getContext('2d');const w=c.width=c.offsetWidth;const h=c.height=36;
  ctx.clearRect(0,0,w,h);if(data.length<2)return;
  const mn=Math.min(...data),mx=Math.max(...data),rng=mx-mn||1;
  ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=1.5;
  data.forEach((v,i)=>{const x=(i/(data.length-1))*w;const y=h-((v-mn)/rng)*(h-4)-2;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});
  ctx.stroke();
}

function updateUI(d){
  totalP++;if(d.decision===0)tnC++;else ntnC++;
  const ho=d.ho_metrics||{};const hev=d.handover?d.handover.handover_event:null;
  document.getElementById('hdr-preds').textContent=totalP+' preds';
  document.getElementById('hdr-ho').textContent=(ho.total_handovers||0)+' handovers';
  document.getElementById('sat-count').textContent=(d.sat_visible||0)+' sats';
  document.getElementById('k-preds').textContent=totalP;
  document.getElementById('k-ratio').textContent=`TN: ${tnC} | NTN: ${ntnC} (${(ntnC/totalP*100).toFixed(1)}% NTN)`;
  document.getElementById('k-lat').textContent=d.latency_ms.toFixed(2)+'ms';
  document.getElementById('k-lat').style.color=d.latency_ms<2?'#76ff03':d.latency_ms<8?'#ffea00':'#ff5252';
  document.getElementById('k-dec').textContent=d.decision===0?'TN':'NTN';
  document.getElementById('k-dec').style.color=d.decision===0?'#00e5ff':'#ff9100';
  document.getElementById('k-conf').textContent='Confidence: '+(d.confidence*100).toFixed(1)+'%';
  document.getElementById('k-ue').textContent=d.ue_id||'--';
  const ueNet=d.handover?d.handover.current_network:d.decision;
  document.getElementById('k-ue-net').textContent='Network: '+(ueNet===0?'TN':'NTN')+' | HOs: '+(d.handover?d.handover.ue_ho_count:0);
  document.getElementById('ho-total').textContent=ho.total_handovers||0;
  document.getElementById('ho-dirs').textContent=`TN\u2192NTN: ${ho.tn_to_ntn||0} | NTN\u2192TN: ${ho.ntn_to_tn||0}`;
  document.getElementById('ho-avg').textContent=(ho.ho_latency_avg_ms||0).toFixed(1)+'ms';
  document.getElementById('ho-p95').textContent='P95: '+(ho.ho_latency_p95_ms||0).toFixed(1)+'ms | P99: '+(ho.ho_latency_p99_ms||0).toFixed(1)+'ms';
  document.getElementById('ho-pp').textContent=ho.pingpong_count||0;
  if(hev){
    document.getElementById('ho-last').textContent=hev.total_latency_ms+'ms';
    document.getElementById('ho-last').style.color=hev.total_latency_ms<100?'#76ff03':hev.total_latency_ms<150?'#ffea00':'#ff5252';
    document.getElementById('ho-last-dir').textContent=hev.direction_label+' | '+hev.ue_id;
    const pp=hev.is_pingpong?'<span style="color:#ff5252;font-weight:700"> PING-PONG</span>':'';
    document.getElementById('ho-event-box').innerHTML=`<div class="ho-event"><div class="ho-dir" style="color:${hev.direction==='TN_to_NTN'?'#ff9100':'#00e5ff'}">${hev.direction_label}${pp}</div><div class="ho-detail">UE: ${hev.ue_id} | Confidence: ${(hev.confidence*100).toFixed(1)}%<br>Prediction: ${hev.prediction_ms}ms | Propagation: ${hev.propagation_ms}ms<br>Preparation: ${hev.preparation_ms}ms | Execution: ${hev.execution_ms}ms | Completion: ${hev.completion_ms}ms<br><b>Total: ${hev.total_latency_ms}ms</b></div></div>`;
    const tot=hev.total_latency_ms;const pcts=[hev.prediction_ms,hev.propagation_ms,hev.preparation_ms,hev.execution_ms,hev.completion_ms].map(v=>(v/tot*100).toFixed(1));
    document.getElementById('ho-breakdown-bar').innerHTML=`<div class="hb-pred" style="width:${pcts[0]}%">${hev.prediction_ms.toFixed(1)}</div><div class="hb-prop" style="width:${pcts[1]}%">${hev.propagation_ms.toFixed(1)}</div><div class="hb-prep" style="width:${pcts[2]}%">${hev.preparation_ms.toFixed(1)}</div><div class="hb-exec" style="width:${pcts[3]}%">${hev.execution_ms.toFixed(1)}</div><div class="hb-comp" style="width:${pcts[4]}%">${hev.completion_ms.toFixed(1)}</div>`;
    document.getElementById('ho-brk-total').textContent=hev.total_latency_ms+'ms';
    hoBuf.push(hev.total_latency_ms);if(hoBuf.length>100)hoBuf.shift();
  }
  const tp=(d.probability_tn*100).toFixed(1),np=(d.probability_ntn*100).toFixed(1);
  document.getElementById('b-tn').style.width=Math.max(parseFloat(tp),2)+'%';document.getElementById('b-tn').textContent=tp+'%';
  document.getElementById('b-ntn').style.width=Math.max(parseFloat(np),2)+'%';document.getElementById('b-ntn').textContent=np+'%';
  document.getElementById('tn-p').textContent=tp+'%';document.getElementById('ntn-p').textContent=np+'%';
  for(const[k,v]of Object.entries(d.features||{})){const e=document.getElementById('f-'+k);if(e)e.textContent=typeof v==='number'?v.toFixed(2):v}
  latBuf.push(d.latency_ms);if(latBuf.length>100)latBuf.shift();
  probBuf.push(d.probability_ntn);if(probBuf.length>100)probBuf.shift();
  drawSpark('sp-lat',latBuf,'#00e5ff');drawSpark('sp-prob',probBuf,'#ff9100');drawSpark('sp-ho',hoBuf,'#e040fb');
  const logEl=document.getElementById('log');const ok=d.decision===d.ground_truth;
  const ts=new Date(d.timestamp*1000).toLocaleTimeString();
  if(hev){const hoCol=hev.direction==='TN_to_NTN'?'#ff9100':'#00e5ff';logEl.innerHTML=`<div class="le le-ho">${ts} <span class="badge b-ho">HO</span> ${hev.direction_label} ${hev.ue_id} <b style="color:${hoCol}">${hev.total_latency_ms}ms</b> conf=${(hev.confidence*100).toFixed(1)}%${hev.is_pingpong?' <span style="color:#ff5252">PP!</span>':''}</div>`+logEl.innerHTML}
  const cls=d.decision===0?'le-tn':'le-ntn';const bdg=d.decision===0?'<span class="badge b-tn">TN</span>':'<span class="badge b-ntn">NTN</span>';
  const chk=ok?'<span style="color:#76ff03">OK</span>':'<span style="color:#ff5252">MISS</span>';
  logEl.innerHTML=`<div class="le ${cls}">${ts} ${bdg} ${d.ue_id} conf=${(d.confidence*100).toFixed(1)}% lat=${d.latency_ms.toFixed(2)}ms gap=${(d.features.rsrp_gap||0).toFixed(1)}dB ${chk}</div>`+logEl.innerHTML;
  if(logEl.children.length>80)logEl.removeChild(logEl.lastChild);
}

function connect(){ws=new WebSocket(WS);ws.onmessage=e=>updateUI(JSON.parse(e.data));ws.onclose=()=>setTimeout(connect,2000);ws.onerror=()=>ws.close()}
connect();
setInterval(()=>{if(document.getElementById('page-celestrak').classList.contains('active'))fetchVisible(selectedUE)},30000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
