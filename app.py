import os
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path


load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


app = Flask(__name__)


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


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant based in Melbourne."},
            {"role": "user", "content": user_message}
        ]
    )


    response_text = response.choices[0].message.content
   
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



if __name__ == '__main__':
    app.run(debug=True, port=5001)
