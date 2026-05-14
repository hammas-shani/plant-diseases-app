import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# ✅ 15 Classes — tumhare dataset ke mutabiq
CLASS_NAMES = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus"
]

# ✅ Model ek baar load hoga
model = load_model("model/plant_disease.keras")

def predict_disease(img_bytes: bytes):
    # Bytes ko image mein convert karo
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 64x64 resize karo (tumhari training size)
    img = img.resize((64, 64))
    
    # Array mein convert karo
    img_array = np.array(img) / 255.0
    
    # Batch dimension add karo
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction karo
    predictions = model.predict(img_array)
    
    # Sabse zyada confidence wali class
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions[0]))
    
    # Healthy hai ya nahi
    is_healthy = "healthy" in predicted_class.lower()
    
    return {
        "disease": predicted_class,
        "confidence": round(confidence * 100, 2),
        "is_healthy": is_healthy
    }