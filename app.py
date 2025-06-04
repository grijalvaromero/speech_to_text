from flask import Flask, request, jsonify, render_template
import os
from datetime import datetime
from main import SpeechText
import requests
from bs4 import BeautifulSoup
app = Flask(__name__)
sp= SpeechText()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/url")
def url():
    return render_template("fromurl.html")

@app.route("/free")
def free():
    return render_template("free.html")

@app.route("/wiki", methods=["POST"])
def wiki():
    url = request.form.get("url")
    print(url)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            html_content = response.text
            #print(html_content)
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator='\n', strip=True)
            return jsonify({"data": text})
        else:
            return jsonify({"data": "No se pudo extraer el texto"}), 400

    except Exception as e:
        print("Error:", e)
        return jsonify({"data": "No se pudo extraer el texto"}), 500
   


@app.route("/onlytext", methods=["POST"])
def onlytext():
    text = request.form.get("text")
    
    #print(text)
    speaker = request.form.get("speaker", "female-en-5")
    res = sp.processText(text, speaker)
    return jsonify(res)
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No se recibió el archivo"}), 400

    audio = request.files["audio"]
    speaker = request.form.get("speaker", "female-en-5")

    # Guardar audio con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audio_{timestamp}.wav"
    input_path = os.path.join("static", "audios", "input")
    os.makedirs(input_path, exist_ok=True)
    filepath = os.path.join(input_path, filename)
    audio.save(filepath)

    # Procesar audio
    res = sp.processData(filepath, speaker)

    return jsonify(res)


@app.route("/speakers", methods=['GET'])
def speakers():
    speakers = ['female-en-5', 'female-en-5', 'female-pt-4', 'male-en-2', 'male-en-2', 'male-pt-3']
    return jsonify({"data":speakers}), 200



if __name__ == "__main__":
    app.run(debug=True)
