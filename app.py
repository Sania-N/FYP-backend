import os
import time

# 🔴 FORCE FFMPEG FOR PYDUB (WINDOWS)
FFMPEG_PATH = r"C:\Program Files\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

from pydub import AudioSegment
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")

import tempfile
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, HttpUrl
import librosa
import numpy as np
import tensorflow as tf
import requests
import io
import pandas as pd
import joblib
import random
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.middleware("http")
async def log_requests(request, call_next):
    started_at = time.time()
    print(f"[REQ] {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        elapsed_ms = round((time.time() - started_at) * 1000, 1)
        print(f"[RES] {request.method} {request.url.path} -> {response.status_code} ({elapsed_ms} ms)")
        return response
    except Exception as exc:
        elapsed_ms = round((time.time() - started_at) * 1000, 1)
        print(f"[ERR] {request.method} {request.url.path} failed after {elapsed_ms} ms: {exc}")
        raise

# Allow frontend (Expo / phone / emulator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- EMOTION DETECTION MODEL ---------------- #

try:
    model = tf.keras.models.load_model("cnn.h5")
    print("Emotion detection model loaded")
except Exception as e:
    print("Error loading emotion model:", e)
    raise e

EMOTION_LABELS = [
    'angry', 'calm', 'disgust', 'fearful',
    'happy', 'neutral', 'sad', 'surprised'
]

PANIC_EMOTIONS = ['angry', 'fearful', 'sad']

# ---------------- SAFE ROUTE ML MODEL ---------------- #

try:
    crime_df = pd.read_csv("crime_dataset.csv")

    crime_coords = np.radians(crime_df[['latitude', 'longitude']].values)

    knn = NearestNeighbors(
        n_neighbors=20,
        metric='haversine'
    )

    knn.fit(crime_coords)

    print("Crime safety model loaded")

except Exception as e:
    print("Error loading crime dataset:", e)

try:
    multimodal_model = joblib.load("multimodal_fusion_model.pkl")
    print("Multimodal model loaded")
except Exception as e:
    print("Error loading multimodal model:", e)

# ---------------- REQUEST MODELS ---------------- #

class AudioRequest(BaseModel):
    audio_url: HttpUrl


class Coordinate(BaseModel):
    lat: float
    lng: float


class Route(BaseModel):
    route_id: int
    coordinates: List[Coordinate]


class RouteRequest(BaseModel):
    routes: List[Route]
    timestamp: Optional[str] = None


class MultimodalRequest(BaseModel):
    emotion_confidence: float
    motion: Optional[float] = None
    heart_rate: Optional[float] = None


class RealtimeRequest(BaseModel):
    audio_url: HttpUrl
    motion: float
    heart_rate: float
    trigger_reason: Optional[str] = None


# ---------------- AUDIO FEATURE EXTRACTION ---------------- #

def extract_feature(data, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])


# ---------------- EMOTION PREDICTION API ---------------- #

@app.post("/predict")
def predict(req: AudioRequest, request: Request = None):
    try:
        # Determine source and log
        source = "manual"
        if request is not None:
            hdr = request.headers.get("x-trigger-source") or request.headers.get("x-realtime")
            if hdr and hdr.lower() in ("realtime", "trigger"):
                source = "realtime"

        # Download audio
        print(f"[predict] source={source} downloading: {req.audio_url}")
        response = requests.get(str(req.audio_url), timeout=20)
        if response.status_code != 200 or len(response.content) == 0:
            raise HTTPException(status_code=400, detail="Failed to download audio")

        audio_bytes = io.BytesIO(response.content)

        # Convert audio to WAV - try auto-detection then common fallbacks
        audio = None
        try:
            audio_bytes.seek(0)
            audio = AudioSegment.from_file(audio_bytes)
        except Exception:
            # try common formats
            for fmt in ("m4a", "mp3", "wav", "ogg", "flac", "aac"):
                try:
                    audio_bytes.seek(0)
                    audio = AudioSegment.from_file(audio_bytes, format=fmt)
                    break
                except Exception:
                    continue

        if audio is None:
            raise HTTPException(status_code=400, detail="Pydub could not decode audio (tried auto + common formats)")

        # Save temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio.export(temp_wav.name, format="wav")
            wav_path = temp_wav.name

        if not os.path.exists(wav_path):
            raise HTTPException(status_code=500, detail="Temp WAV file not created")

        # Load audio
        y, sr = librosa.load(wav_path, sr=None)

        if y is None or len(y) == 0:
            raise HTTPException(status_code=500, detail="Librosa could not load audio")

        print("Audio loaded:", y.shape, "Sample rate:", sr)

        # Extract features
        features = extract_feature(y, sr)

        # Prepare model input
        X = np.expand_dims(features, axis=0)
        X = np.expand_dims(X, axis=2)

        preds = model.predict(X)[0]

        idx = np.argmax(preds)

        emotion = EMOTION_LABELS[idx]

        return {
            "emotion": emotion,
            "panic": emotion in PANIC_EMOTIONS,
            "confidence": float(preds[idx])
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        print("Unexpected error:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # cleanup temp wav if created
        try:
            if 'wav_path' in locals() and wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as _err:
            print("Warning: failed to remove temp wav:", _err)


@app.get("/health")
def health():
    return {"ok": True}


# ---------------- SAFE ROUTE API ---------------- #
EARTH_RADIUS = 6371000  # meters
MAX_DISTANCE = 350      # meters
MAX_SEVERITY = 7.5      # max adjusted severity (5 * 1.5)
MIN_CRIME_THRESHOLD = 2 # minimum crimes before area becomes unsafe

CRIME_TYPE_WEIGHT = {
    'assault': 1.5,
    'robbery': 1.4,
    'snatching': 1.3,
    'harassment': 1.1,
    'vehicle theft': 1.0,
    'theft': 0.9,
    'vandalism': 0.7,
}

@app.post("/safe-route")
def safe_route(data: RouteRequest):

    if data.timestamp:
        request_hour = datetime.fromisoformat(
            data.timestamp.replace("Z", "+00:00")
        ).hour
    else:
        request_hour = datetime.now().hour

    results = []

    for route in data.routes:

        weighted_severity_total = 0
        total_weight = 0
        crime_count = 0

        for point in route.coordinates:

            coord = np.radians([[point.lat, point.lng]])

            distances, indices = knn.kneighbors(coord)

            distances_m = distances[0] * EARTH_RADIUS

            for i, d in zip(indices[0], distances_m):

                # ignore crimes too far away
                if d > MAX_DISTANCE:
                    continue

                severity = crime_df.iloc[i]["severity"]

                # ignore minor crimes
                if severity <= 2:
                    continue

                crime_hour = crime_df.iloc[i]["hour"]
                hour_diff = min(
                    abs(crime_hour - request_hour),
                    24 - abs(crime_hour - request_hour)
                )
                if hour_diff <= 3:
                    time_weight = 1.0
                elif hour_diff <= 6:
                    time_weight = 0.6
                else:
                    time_weight = 0.2

                crime_type = crime_df.iloc[i]["crime_type"]
                type_multiplier = CRIME_TYPE_WEIGHT.get(crime_type, 1.0)
                adjusted_severity = severity * type_multiplier

                crime_count += 1

                # closer crimes matter more
                weight = (1 / (d + 1)) * time_weight

                weighted_severity_total += adjusted_severity * weight
                total_weight += weight

        # if very few crimes nearby → safe area
        if crime_count < MIN_CRIME_THRESHOLD:
            safety_score = 95

        elif total_weight == 0:
            safety_score = 100

        else:
            route_risk = weighted_severity_total / total_weight
            risk_normalized = (route_risk - 2) / (MAX_SEVERITY - 2)
            safety_score = max(0.0, min(100.0, (1 - risk_normalized) * 100))

        results.append({
            "route_id": route.route_id,
            "safety_score": float(round(safety_score, 2))
        })

    return results


@app.get("/danger-zones")
def danger_zones(min_severity: float = 3.0, radius_meters: int = 250, limit: int = 200):
    if 'crime_df' not in globals() or crime_df is None:
        raise HTTPException(status_code=500, detail="crime dataset not loaded")

    filtered = crime_df[crime_df["severity"] >= min_severity].copy().head(limit)
    zones = []
    for idx, row in filtered.iterrows():
        zones.append({
            "id": int(idx),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "radius_meters": radius_meters,
            "title": str(row.get("crime_type", "High-risk area")),
            "severity": float(row["severity"]),
        })
    return {"danger_zones": zones}


@app.post("/multimodal-predict")
def multimodal_predict(data: MultimodalRequest, request: Request = None):
    try:
        # Determine source and log
        source = "manual"
        if request is not None:
            hdr = request.headers.get("x-trigger-source") or request.headers.get("x-realtime")
            if hdr and hdr.lower() in ("realtime", "trigger"):
                source = "realtime"

        print(f"[multimodal-predict] source={source} emotion_confidence={data.emotion_confidence} motion={data.motion} heart_rate={data.heart_rate}")

        # Require real sensor values (do not synthesize)
        if data.motion is None or data.heart_rate is None:
            raise HTTPException(status_code=400, detail="motion and heart_rate are required for multimodal prediction")

        features = np.array([[
            data.emotion_confidence,
            float(data.motion),
            float(data.heart_rate)
        ]])

        prediction = multimodal_model.predict(features)[0]

        return {
            "risk_level": str(prediction),
            "motion_used": float(data.motion),
            "heart_rate_used": float(data.heart_rate)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/realtime-threat")
def realtime_threat(req: RealtimeRequest, request: Request = None):
    try:
        print(f"[realtime-threat] received trigger_reason={req.trigger_reason} audio_url={req.audio_url}")

        # Call emotion predictor internally (pass through header info)
        pred = predict(AudioRequest(audio_url=req.audio_url), request=request)

        # Build multimodal input using real sensor values provided
        mm_req = MultimodalRequest(
            emotion_confidence=float(pred.get("confidence", 0.0)),
            motion=float(req.motion),
            heart_rate=float(req.heart_rate)
        )

        mm = multimodal_predict(mm_req, request=request)

        timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "emotion": pred.get("emotion"),
            "confidence": pred.get("confidence"),
            "panic": pred.get("panic"),
            "risk_level": mm.get("risk_level"),
            "motion_used": mm.get("motion_used"),
            "heart_rate_used": mm.get("heart_rate_used"),
            "trigger_reason": req.trigger_reason,
            "timestamp": timestamp
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print("Unexpected error in realtime-threat:", e)
        raise HTTPException(status_code=500, detail=str(e))