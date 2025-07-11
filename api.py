from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from time import  strftime
import torch
import shutil
import sys
from src.utils.init_path import init_path
import requests
import os
import boto3
from botocore.exceptions import ClientError

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

AWS_ACCESS_KEY= os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY= os.getenv("AWS_SECRET_KEY")
AWS_S3_REGION= os.getenv("AWS_S3_REGION")
AWS_S3_BUCKET_NAME= os.getenv("AWS_S3_BUCKET_NAME")

def download_file(image_url, save_dir):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception if the request was not successful
        with open(save_dir + "/" + image_url.split('/')[-1], "wb") as f:
            f.write(response.content)
        print(image_url.split('/')[-1], " downloaded and saved successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")

def upload_file_aws(file_name, object_name):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """


    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY,
                    region_name=AWS_S3_REGION)
    
    extra_args = {
        'ContentType': 'video/mp4',
        'ACL': 'public-read',  # Add ACL information here
    }

    try:
        response = s3_client.upload_file(file_name, AWS_S3_BUCKET_NAME, object_name, ExtraArgs=extra_args)
        print(f'File "{object_name}" successfully uploaded to S3 bucket "{AWS_S3_BUCKET_NAME}" with object key "{object_name}"')
    except ClientError as e:
        logging.error(e)
        return False
    return True

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
# https://s3.us-west-1.amazonaws.com/dev.talktent/uploads/audio/d07VqU6UPC.mp3
# In-memory list to store the items (for demonstration purposes)

# curl -X POST "http://localhost:8000/generate /" -H "Content-Type: application/json" -d '{
#   "image_link": "https://zmp3-photo-fbcrawler.zadn.vn/avatars/3/a/6/d/3a6de9f068f10fcee2c50cdf9772ebaa.jpg",
#   "audio_link": "https://s3.us-west-1.amazonaws.com/dev.talktent/uploads/audio/d07VqU6UPC.mp3"
# }'

# Define a POST endpoint to create new items
@app.post("/generate/")
async def sadtalker_create(item: Item):
    try:
        # 1) Crea carpeta de resultados
        RESULT_DIR = "./results"
        save_dir   = os.path.join(RESULT_DIR, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        # 2) Descarga la imagen
        download_file(item.image_link, save_dir)
        PIC_PATH = os.path.join(save_dir, os.path.basename(item.image_link))

        # 3) Genera o descarga el audio
        if item.audio_link.endswith("silence.wav"):
            AUDIO_PATH = make_silence(
                os.path.join(save_dir, "silence.wav"),
                duration_s=2.0,
                sr=16000
            )
        else:
            download_file(item.audio_link, save_dir)
            AUDIO_PATH = os.path.join(save_dir, os.path.basename(item.audio_link))

        # 4) Inicializa tu pipeline SadTalker
        DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
        POSE_STYLE    = 0
        BATCH_SIZE    = 2
        INPUT_YAW_LIST   = None
        INPUT_PITCH_LIST = None
        INPUT_ROLL_LIST  = None
        REF_EYEBLINK     = None
        REF_POSE         = None
        PREPROCESS        = "full"
        SIZE              = 256
        ENHANCER          = None
        BACKGROUND_ENHANCER = None
        FACE3DVIS         = False
        VERBOSE           = False

        # Inicializar modelos
        sadtalker_paths    = init_path("./checkpoints", "./src/config", SIZE, False, PREPROCESS)
        preprocess_model   = CropAndExtract(sadtalker_paths, DEVICE)
        audio_to_coeff     = Audio2Coeff(sadtalker_paths, DEVICE)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, DEVICE)

        # 5) Extracción de coeficientes 3DMM
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            PIC_PATH, first_frame_dir, PREPROCESS, source_image_flag=True, pic_size=SIZE
        )
        if first_coeff_path is None:
            raise RuntimeError("No se obtuvieron coeficientes del input")

        # 6) Audio2Coeff + FaceRenderer
        batch      = get_data(first_coeff_path, AUDIO_PATH, DEVICE, REF_EYEBLINK, still=True)
        coeff_path = audio_to_coeff.generate(batch, save_dir, POSE_STYLE, REF_POSE)
        data       = get_facerender_data(
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

        # 8) Devuelve la URL local
        video_url = f"http://localhost:8000/results/{os.path.basename(out_file)}"
        return JSONResponse({"video_url": video_url})

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(
            {"error": "Internal processing error", "detail": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
