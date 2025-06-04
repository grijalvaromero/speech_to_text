import whisper
import spacy
from TTS.api import TTS
from transformers import MarianMTModel, MarianTokenizer
from datetime import datetime
import os

class SpeechText:
    def __init__(self):
        self.model_whisper = whisper.load_model("small")
        self.nlp = spacy.load("es_core_news_md")
        model_name = "Helsinki-NLP/opus-mt-es-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model_translate = MarianMTModel.from_pretrained(model_name)
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)

    def processData(self, input_file, speaker):
    
        text = self.transcribe(input_file)
        print(f"TEXTO: {text}")
        doc = self.getEntities(text)
        print("ENTIDADES:")
        for ent in doc.ents:
            print(f"{ent.text} -> {ent.label_}")

        translation = self.translate(text)
        audio_url = self.generateAudio(translation, speaker)
        print(doc)
        return {
            "text": text,
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "translation": translation,
            "url": audio_url
        }
    def processText(self, text, speaker):
    
        #text = self.transcribe(input_file)
        print(f"TEXTO: {text}")
        doc = self.getEntities(text)
        print("ENTIDADES:")
        for ent in doc.ents:
            print(f"{ent.text} -> {ent.label_}")

        translation = self.translate(text)
        audio_url = self.generateAudio(translation, speaker)
        print(doc)
        return {
            "text": text,
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "translation": translation,
            "url": audio_url
        }
    def transcribe(self, file_path):
        result = self.model_whisper.transcribe(file_path, language="Spanish")
        return result["text"]

    def getEntities(self, text):
        doc = self.nlp(text)
        return doc

    def translate(self, text):
        tokens = self.tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
        translated = self.model_translate.generate(**tokens)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

    def generateAudio(self, text, speaker):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("static", "audios", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"output_{timestamp}.wav"
        output_path = os.path.join(output_dir, output_file)

        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker=speaker,
            language="en"
        )


        return f"/static/audios/output/{output_file}"