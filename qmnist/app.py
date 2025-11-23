from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = torch.load('qmnist_mobile.pt')
model.eval()  # Set model to evaluation mode

# Define transformations (adjust as per your training pipeline)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel (if needed)
    transforms.Resize((28, 28)),  # QMNIST image size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalization based on your dataset
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image file from the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))
        
        # Preprocess the image
        img = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
        
        # Return the prediction as a response
        return jsonify({"prediction": predicted.item()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
