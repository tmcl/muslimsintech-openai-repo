import os
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from werkzeug.utils import secure_filename
import requests
import json
import base64
import PyPDF2

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def allowed_image_file(filename):
    allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

def allowed_document_file(filename):
    allowed = {'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_current_weather(location, unit):
    if unit is None:
        unit = "celsius"

    api_key = os.environ.get("WEATHER_API_KEY") #Retrieve the API key from environment variables
   
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"




    response = requests.get(url)      
    data = response.json()


    if unit is not None and unit.lower() == "celsius":
        temperature = data["current"]["temp_c"]
    elif unit is not None and unit.lower() == "fahrenheit":
        temperature = data["current"]["temp_f"]
    else:
        temperature = data["current"]["temp_c"]
   
    formatted_data = {
        "location": data["location"]["name"],
        "temperature": str(int(temperature)),  # Convert to int and then string as requested
        "unit": unit  # Ensure unit is lowercase
    }


    return json.dumps(formatted_data)





app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if request.content_type.startswith("multipart/form-data"):
        user_message = request.form.get('message', '')
        messages = [{"role": "user", "content": []}]
        if user_message:
            messages[0]["content"].append({"type": "text", "text": user_message})

        # Process image file (from the 'image' input)
        image_file = request.files.get('image')
        if image_file and allowed_image_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join('uploads', filename)
            image_file.save(filepath)
            base64_image = encode_image(filepath)
            ext = filename.rsplit('.', 1)[1].lower()
            mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
            })

        # Process PDF document (from the 'file' input)
        document_file = request.files.get('file')
        if document_file and allowed_document_file(document_file.filename):
            filename = secure_filename(document_file.filename)
            filepath = os.path.join('uploads', filename)
            document_file.save(filepath)
            pdf_text = ""
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"
            max_chars = 3000
            if len(pdf_text) > max_chars:
                pdf_text = pdf_text[:max_chars] + "..."
            messages[0]["content"].append({
                "type": "text",
                "text": f"Attached PDF contents:\n{pdf_text}"
            })

        # (Proceed with tool calling or response processing as before)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        final_response_text = response.choices[0].message.content
        return jsonify({"response": final_response_text})
    else:
        # Fallback for JSON-only requests
        data = request.json
        user_message = data.get('message', '')
        messages = [{"role": "user", "content": [{"type": "text", "text": user_message}]}]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        final_response_text = response.choices[0].message.content
        return jsonify({"response": final_response_text})



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
       
        with open(filepath, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return jsonify({"transcript": transcript.text})
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
