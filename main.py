from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.predict import predict_disease

app = FastAPI(
    title="Plant Disease Detection API",
    description="Upload a plant leaf image to detect disease",
    version="1.0.0"
)

# ✅ Streamlit ko allow karo connect hone ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health Check
@app.get("/")
def home():
    return {"status": "API is running ✅"}


# ✅ Main Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # Sirf image files allow karo
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Sirf JPG ya PNG image upload karo!"
        )
    
    try:
        # Image bytes parho
        img_bytes = await file.read()
        
        # Prediction karo
        result = predict_disease(img_bytes)
        
        return {
            "filename": file.filename,
            "disease": result["disease"],
            "confidence": result["confidence"],
            "is_healthy": result["is_healthy"],
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Classes List
@app.get("/classes")
def get_classes():
    from utils.predict import CLASS_NAMES
    return {"total": len(CLASS_NAMES), "classes": CLASS_NAMES}