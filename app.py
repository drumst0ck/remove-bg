from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse
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

app = FastAPI()

# Configura las plantillas y archivos estáticos
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Carga el modelo usando pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    try:
        # Lee la imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGBA")
        
        # Aumenta la resolución para el procesamiento
        width, height = image.size
        image = image.resize((width*3, height*3), Image.LANCZOS)
        
        # Realiza la segmentación
        result = pipe(image)
        
        # Convierte la máscara a numpy array
        mask = np.array(result.convert("L"))
        
        # Mejora la detección de bordes
        edges = cv2.Canny(mask, 30, 150)
        mask = cv2.addWeighted(mask, 1, edges, 0.7, 0)
        
        # Aplica un umbral y suaviza la máscara
        threshold = 2  # Umbral aún más bajo para capturar detalles más finos
        binary_mask = (mask > threshold).astype(np.float32)
        smoothed_mask = gaussian_filter(binary_mask, sigma=0.3)
        
        # Aplica operaciones morfológicas para refinar la máscara
        kernel = np.ones((3,3), np.uint8)
        refined_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Suaviza los bordes de la máscara
        refined_mask = gaussian_filter(refined_mask, sigma=0.5)
        
        # Crea una imagen RGBA
        image_array = np.array(image)
        
        # Aplica la máscara refinada al canal alfa
        image_array[:,:,3] = (refined_mask * 255).astype(np.uint8)
        
        # Elimina píxeles aislados de forma más selectiva
        def remove_isolated_pixels(alpha_channel, size, threshold):
            mask = alpha_channel > 0
            labeled, num_features = scipy.ndimage.label(mask)
            sizes = scipy.ndimage.sum(mask, labeled, range(1, num_features + 1))
            small_objects = sizes < size
            remove_pixel = small_objects[labeled - 1]
            remove_pixel[alpha_channel < threshold] = 0
            alpha_channel[remove_pixel] = 0
            return alpha_channel

        image_array[:,:,3] = remove_isolated_pixels(image_array[:,:,3], size=10, threshold=50)
        
        # Aplica un matting más suave para el cabello
        alpha = refined_mask[:,:,np.newaxis]
        fg = image_array[:,:,:3] * alpha + (1 - alpha) * 255
        image_array[:,:,:3] = fg.astype(np.uint8)
        
        # Convierte de vuelta a imagen PIL
        result_image = Image.fromarray(image_array, mode="RGBA")
        
        # Mejora el contraste y la nitidez de manera más precisa
        enhancer = ImageEnhance.Contrast(result_image)
        result_image = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Sharpness(result_image)
        result_image = enhancer.enhance(1.2)
        
        # Redimensiona la imagen a su tamaño original
        result_image = result_image.resize((width, height), Image.LANCZOS)
        
        # Elimina pequeños artefactos
        result_array = np.array(result_image)
        result_array[:,:,3] = cv2.medianBlur(result_array[:,:,3], 3)
        result_image = Image.fromarray(result_array, mode="RGBA")
        
        # Guarda la imagen resultante
        result_path = "static/result.png"
        result_image.save(result_path, quality=95)
        
        return FileResponse(result_path)
    except Exception as e:
        print(f"Error completo: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)