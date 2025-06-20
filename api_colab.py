from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from time import strftime
import torch
import shutil
import sys
from src.utils.init_path import init_path
import requests
import os

from src.utils.preprocess import CropAndExtract
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
import logging

from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, status
import numpy as np
import soundfile as sf

load_dotenv()

# AWS variables (no las usamos pero las mantenemos por compatibilidad)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_S3_REGION = os.getenv("AWS_S3_REGION")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

def download_file(file_url, save_dir):
    """Descarga un archivo desde una URL y lo guarda en el directorio especificado"""
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raise an exception if the request was not successful
        
        # Crear el directorio si no existe
        os.makedirs(save_dir, exist_ok=True)
        
        # Extraer el nombre del archivo desde la URL
        filename = file_url.split('/')[-1]
        
        # Si no tiene extensión, intentar detectarla del content-type
        if '.' not in filename:
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                if 'jpeg' in content_type or 'jpg' in content_type:
                    filename += '.jpg'
                elif 'png' in content_type:
                    filename += '.png'
                elif 'gif' in content_type:
                    filename += '.gif'
            elif 'audio' in content_type:
                if 'wav' in content_type:
                    filename += '.wav'
                elif 'mp3' in content_type:
                    filename += '.mp3'
        
        file_path = os.path.join(save_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        print(f"{filename} downloaded and saved successfully.")
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        raise

app = FastAPI()

from fastapi.staticfiles import StaticFiles

# Monta la carpeta 'results' en la ruta '/results'
app.mount("/results", StaticFiles(directory="results"), name="results")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # ← aquí permitimos todo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request

def make_silence(path: str, duration_s: float = 2.0, sr: int = 16000) -> str:
    """
    Crea un archivo WAV de silencio de `duration_s` segundos a `sr` Hz
    y lo guarda en `path`. Devuelve la ruta `path`.
    """
    samples = np.zeros(int(sr * duration_s), dtype="float32")
    sf.write(path, samples, sr)
    return path

# ——————————————— Preflight OPTIONS ———————————————
@app.options("/generate/")
async def options_generate(request: Request):
    # FastAPI/CORSMiddleware añadirá los headers CORS
    return JSONResponse(content={}, status_code=status.HTTP_200_OK)
# ————————————————————————————————————————————————————————

class Item(BaseModel):
    image_link: str
    audio_link: str
    s3_object_path: str = 'uploads/avatar/'

# Define a POST endpoint to create new items
@app.post("/generate/")
async def sadtalker_create(item: Item):
    try:
        # 1) Crea carpeta de resultados
        RESULT_DIR = "./results"
        save_dir = os.path.join(RESULT_DIR, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        # 2) Descarga la imagen
        print(f"Descargando imagen desde: {item.image_link}")
        PIC_PATH = download_file(item.image_link, save_dir)
        print(f"Imagen guardada en: {PIC_PATH}")

        # 3) Genera o descarga el audio
        if item.audio_link.endswith("silence.wav"):
            AUDIO_PATH = make_silence(
                os.path.join(save_dir, "silence.wav"),
                duration_s=2.0,
                sr=16000
            )
            print(f"Audio de silencio creado en: {AUDIO_PATH}")
        else:
            print(f"Descargando audio desde: {item.audio_link}")
            AUDIO_PATH = download_file(item.audio_link, save_dir)
            print(f"Audio guardado en: {AUDIO_PATH}")

        # 4) Inicializa tu pipeline SadTalker
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        POSE_STYLE = 0
        BATCH_SIZE = 2
        INPUT_YAW_LIST = None
        INPUT_PITCH_LIST = None
        INPUT_ROLL_LIST = None
        REF_EYEBLINK = None
        REF_POSE = None
        PREPROCESS = "full"
        SIZE = 256
        ENHANCER = None
        BACKGROUND_ENHANCER = None
        FACE3DVIS = False
        VERBOSE = False

        print("Inicializando modelos SadTalker...")
        # Inicializar modelos
        sadtalker_paths = init_path("./checkpoints", "./src/config", SIZE, False, PREPROCESS)
        preprocess_model = CropAndExtract(sadtalker_paths, DEVICE)
        audio_to_coeff = Audio2Coeff(sadtalker_paths, DEVICE)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, DEVICE)

        # 5) Extracción de coeficientes 3DMM
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)

        # debug
        print("Archivos en save_dir:", os.listdir(save_dir))
        print("PIC_PATH:", PIC_PATH)
        print("AUDIO_PATH:", AUDIO_PATH)
        print("==== SadTalker API Debug ====")
        print("save_dir:", save_dir)
        # debug

        print("Procesando imagen...")
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            PIC_PATH, first_frame_dir, PREPROCESS, source_image_flag=True, pic_size=SIZE
        )
        if first_coeff_path is None:
            raise RuntimeError("No se obtuvieron coeficientes del input")

        # 6) Audio2Coeff + FaceRenderer
        print("Procesando audio...")
        batch = get_data(first_coeff_path, AUDIO_PATH, DEVICE, REF_EYEBLINK, still=True)
        coeff_path = audio_to_coeff.generate(batch, save_dir, POSE_STYLE, REF_POSE)
        
        print("Generando video...")
        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path, AUDIO_PATH,
            BATCH_SIZE, INPUT_YAW_LIST, INPUT_PITCH_LIST, INPUT_ROLL_LIST,
            expression_scale=1.0, still_mode=True, preprocess=PREPROCESS, size=SIZE
        )
        result = animate_from_coeff.generate(
            data, save_dir, PIC_PATH, crop_info,
            enhancer=ENHANCER, background_enhancer=BACKGROUND_ENHANCER, img_size=SIZE
        )

        # 7) Mueve el vídeo final
        out_file = save_dir + ".mp4"
        shutil.move(result, out_file)
        
        print(f"Video generado: {out_file}")

        # 8) Devuelve la URL local (ajustar según tu configuración de ngrok)
        video_filename = os.path.basename(out_file)
        video_url = f"https://bd73-34-125-172-251.ngrok-free.app/results/{video_filename}" #conexion grok actual
        
        return JSONResponse({"video_url": video_url, "message": "Video generado exitosamente"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": "Internal processing error", "detail": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
