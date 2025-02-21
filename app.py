import os
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from werkzeug.utils import secure_filename
import requests
import json

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_current_weather(location, unit):
    """
    Retrieves weather information for a given city and returns it in a specified format.

    Args:
        location (str): The name of the city.
        unit (str): The unit of temperature, either "celsius" or "fahrenheit".

    Returns:
        dict: A dictionary containing the location, temperature, and unit, or None if an error occurs.
    """

    api_key = os.environ.get("WEATHER_API_KEY") #Retrieve the API key from environment variables
    if not api_key:
        return "Error: API Key not found. Set WEATHER_API_KEY environment variable."


    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if unit.lower() == "celsius":
            temperature = data["current"]["temp_c"]
        elif unit.lower() == "fahrenheit":
            temperature = data["current"]["temp_f"]
        else:
            return "Error: Invalid unit.  Choose 'celsius' or 'fahrenheit'."

        formatted_data = {
            "location": data["location"]["name"],
            "temperature": str(int(temperature)),  # Convert to int and then string as requested
            "unit": unit.lower()  # Ensure unit is lowercase
        }

        return formatted_data

    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {e}"
    except KeyError:
        return "Error: Could not parse weather data. Check city name and API key."
    except Exception as e:
        return f"Error: An unexpected error occurred - {e}"





app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if request.is_json:
            data = request.json
            user_message = data.get('message')
    else:
        user_message = request.form.get('message')

    tools = [
               {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }

    ]

    messages = [
            {"role": "system", "content": "You are a helpful assistant based in Melbourne."},
            {"role": "user", "content": user_message}
        ]

    response = client.chat.completions.create(
        model="gpt-4o",
        tools=tools, tool_choice='auto',
        messages=messages
    )


    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {"get_current_weather": get_current_weather}
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_response = available_functions[function_name](
                location=function_args.get("location"),
                unit=function_args.get("unit")
            )
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            })
        second_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
    else:
        second_response = response


    response_text = second_response.choices[0].message.content
   
    speech_file_path = Path("static/speech.mp3")
    tts_response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=response_text
    )
    tts_response.stream_to_file(speech_file_path)


    return jsonify({
        "response": response_text,
        "audio_url": "/static/speech.mp3"
    })



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
