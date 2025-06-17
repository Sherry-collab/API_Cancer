from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import torch

# ... other imports ...

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    methods="*"
)
# # Allow CORS for all routes (safe for dev; restrict in prod if needed)

# Load model and processor
MODEL_NAME = "ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20"
image_processor = BeitImageProcessor.from_pretrained(MODEL_NAME)
model = BeitForImageClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    use_safetensors=True
)

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Optional: Check allowed file types
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        # Load and preprocess image
        image = Image.open(file).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")

        # Run prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_idx = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1)[0, pred_idx].item()

        # Get class label
        label = model.config.id2label[pred_idx]

        # Return JSON response
        return jsonify({
            "class_code": str(pred_idx),
            "predicted_class": label,
            "confidence": confidence
        })

    except Exception as e:
        print("Prediction error:", str(e))  # Print for logs (not in response)
        return jsonify({"error": "Internal server error"}), 500

# Health check route
@app.route("/")
def index():
    return "Skin Lesion Detection API is running."

# Entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
