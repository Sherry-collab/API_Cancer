from flask import Flask, request, jsonify
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import torch

# Initialize Flask
app = Flask(__name__)

# Load model + image processor
MODEL_NAME = "ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20"
image_processor = BeitImageProcessor.from_pretrained(MODEL_NAME)
model = BeitForImageClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    use_safetensors=True
)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        # Load and preprocess image
        file = request.files['file']
        image = Image.open(file).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[pred_idx]
        confidence = torch.softmax(logits, dim=-1)[0, pred_idx].item()

        # Return result
        return jsonify({
            "class_code": str(pred_idx),
            "predicted_class": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check route
@app.route("/")
def index():
    return "Skin Lesion Detection API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
