import asyncio
import aiohttp
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import cv2
from transformers import pipeline
import torch
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import scipy.ndimage
import uuid
import os

app = FastAPI()

# Configura las plantillas y archivos estáticos
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Carga el modelo usando pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# Función para procesar una sola imagen
async def process_image(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGBA")
        
        # Realiza la segmentación
        result = pipe(image)
        
        # Convierte la máscara a numpy array
        mask = np.array(result.convert("L"))
        
        # Aplica un umbral y suaviza la máscara
        threshold = 2
        binary_mask = (mask > threshold).astype(np.float32)
        smoothed_mask = gaussian_filter(binary_mask, sigma=0.5)
        
        # Aplica la máscara al canal alfa
        image_array = np.array(image)
        image_array[:,:,3] = (smoothed_mask * 255).astype(np.uint8)
        
        # Convierte de vuelta a imagen PIL
        result_image = Image.fromarray(image_array, mode="RGBA")
        
        # Genera un nombre de archivo único
        unique_filename = f"{uuid.uuid4()}.png"
        result_path = f"static/{unique_filename}"
        result_image.save(result_path, quality=95)
        
        return {
            "original_filename": file.filename,
            "processed_filename": unique_filename,
            "width": image.width,
            "height": image.height
        }
    except Exception as e:
        print(f"Error processing {file.filename}: {str(e)}")
        return {
            "original_filename": file.filename,
            "error": str(e)
        }

@app.post("/remove-background")
async def remove_background(files: list[UploadFile] = File(...)):
    tasks = [process_image(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    # Filtrar resultados duplicados
    unique_results = []
    seen = set()
    for result in results:
        if result['original_filename'] not in seen:
            unique_results.append(result)
            seen.add(result['original_filename'])
    
    return JSONResponse(content={"results": unique_results})

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)