import os
import json
import base64
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from openai import OpenAI
from model import AdAgency, Department, Role  # Assuming these classes are defined in model.py
from controller import Controller  # Assuming this is defined in controller.py
import logging
from logging.handlers import RotatingFileHandler
from io import BytesIO
import traceback
from firebase_admin import storage
import requests
from flask_cors import CORS

# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
    handlers=[
        RotatingFileHandler('aime.log', maxBytes=10000, backupCount=3),
        logging.StreamHandler()
    ]
)

# Define a path for image storage relative to the current file
image_storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')

# Make sure the directory exists
os.makedirs(image_storage_path, exist_ok=True)


app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize the AdAgency and Controller
agency = AdAgency("AiMe")
controller = Controller(agency, None)  # The view component is replaced by Flask


def upload_to_firebase(file_stream, filename):
    # Get the bucket reference
    bucket = storage.bucket('aime-5b083.appspot.com')

    # Create a blob in the bucket and upload the file
    blob = bucket.blob(filename)
    blob.upload_from_string(
        file_stream.read(),
        content_type='image/jpeg'  # This is assuming the file is a JPEG
    )

    # Make the blob publicly viewable
    # Note: Adjust this if you need more privacy
    blob.make_public()

    # Return the public URL
    return blob.public_url

@app.route('/')
def index():
    # Welcome message or API info
    return jsonify({
        "message": "Welcome to the AiMe API. Use our endpoints for process brand advertisement campaigns in choosen channels."
    })

@app.route('/process', methods=['POST'])
def process_ad():
    response_data = {}
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    try:
        # Extract form data and files from the request
        form_data = request.form.to_dict()
        image_file = request.files.get('image')
        public_url = None

        # Validate expected fields including 'channels'
        expected_fields = {
            'brand_name': 'Brand name is required.',
            'product_service': 'Product/Service details are required.',
            'goals': 'Campaign goals are required.',
            'target_audience': 'Target audience details are required.',
            'audience_insight': 'Audience insight is required.',
            'message': 'Key message is required.',
            'city': 'Campaign location (city) is required.',
            'time': 'Campaign timeline is required.',
            'budget': 'Campaign budget is required.',
            'big_idea': 'The big idea of the campaign is required.',
            'channels': 'At least one channel selection is required.'
        }

        missing_fields = [field for field in expected_fields if field not in form_data or not form_data[field]]
        if missing_fields:
            return jsonify({"status": "error", "message": "Missing form data", "errors": {field: expected_fields[field] for field in missing_fields}}), 400

        # Handling image upload and analysis
        if image_file:
            # Assuming a function upload_to_firebase exists and works as intended
            public_url = upload_to_firebase(image_file, image_file.filename)  # This function needs to be defined as shown earlier

            # Image analysis with OpenAI (assuming the image is relevant to the prompt)
            vision_response = client.create_image(
                model="gpt-4-vision-preview",
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": public_url}]}],
                max_tokens=500
            )
            vision_analysis = vision_response.choices[0].message.content  # Simplified for illustration

        # Text analysis and strategy generation for each channel
        selected_channels = form_data['channels'].split(',')
        channel_strategies = {}
        for channel in selected_channels:
            channel_trimmed = channel.strip()
            prompt = f"{form_data['message']} [Strategy for: {channel_trimmed}]"

            text_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-3.5-turbo",  # Ensure you're using the correct model
                    "messages": [
                        {"role": "system", "content": "You are a Brand Advertising Expert who tasked with tailoring brand messaging strategies across different media-platforms."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )

            if text_response.status_code == 200:
                generated_text = text_response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                channel_strategies[channel_trimmed] = generated_text
            else:
                channel_strategies[channel_trimmed] = "Unable to generate response for the channel."

        # Construct successful response data
        response_data = {
            "status": "success",
            "data": {
                "channel_strategies": channel_strategies
            }
        }

        # Error response structure
        response_data = {
            "status": "error",
            "message": "An unexpected error occurred.",
            "errors": detailed_errors_if_any
        }

    except Exception as e:
        response_data = {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }
        return jsonify(response_data), 500

    return jsonify(response_data)





# Additional routes for image processing and other functionalities...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81, debug=False)
