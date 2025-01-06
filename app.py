import pandas as pd 
from transformers import pipeline 
from tqdm.notebook import tqdm
import time
import re
from pydub import AudioSegment
import json
import os
import torch
import numpy as np
from faster_whisper import WhisperModel
import shutil
import rapidfuzz
import google.generativeai as genai
import typing_extensions as typing
import gradio as gr

pd.set_option('display.max_colwidth', None)
pd.set_option('display.min_rows', 500)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def remove_tashkeel(text):
    # Arabic Tashkeel Unicode range
    tashkeel_pattern = r'[\u0617-\u061A\u064B-\u0652]'
    # Remove Tashkeel using re.sub
    return re.sub(tashkeel_pattern, '', text)

def remove_consecutive_duplicates(input_list):
    if not input_list:
        return []
    result = [input_list[0]]  # Start with the first element
    for item in input_list[1:]:
        if item != result[-1]:  # Add only if it's not a duplicate of the last added element
            result.append(item)
    return result



# model = whisper.load_model("base")
# result = model.transcribe(audio_path, language="ar")

# Returns transcriptions dictionary
def split_into_chunks(audio_path):
    fast_model = WhisperModel("base")
    segments, info = fast_model.transcribe(audio_path, beam_size=5, language="ar")
    
    
    audio = AudioSegment.from_file(audio_path)
    
    output_folder = "output_chunks"
    os.makedirs(output_folder, exist_ok=True)
    transcriptions = []
    
    
    # Extract and save each segment based on Whisper's timestamps
    for i, segment in tqdm(enumerate(segments)):
        start_time = segment.start * 1000  # Convert seconds to milliseconds
        end_time = segment.end  * 1000      # Convert seconds to milliseconds
    
        # Extract the segment
        chunk = audio[start_time:end_time]
        
        # Save the chunk as a WAV file
        chunk.export(f"{output_folder}/chunk_{i + 1}.wav", format="wav")
    
        # Optional: Print transcription for each chunk
        # print(f"Chunk {i + 1}: Start {segment.start }s - End {segment.end }s | Text: {segment.text}")
    
        transcriptions.append({
            "Start Time (s)": segment.start ,
            "End Time (s)": segment.end  ,
            "Transcription": ""
        })
        
    # print(f"All chunks saved in '{output_folder}' folder.")

    return transcriptions


#Tarteel - base

def transcribe_tarteel(transcriptions):
    MODEL_NAME = 'tarteel-ai/whisper-base-ar-quran'
    lang = 'ar'
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=40,
        device=device,
    )
    
    # pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
    
    # Process each chunk
    for i in tqdm(range(len(transcriptions))):
    
        # Export chunk to temporary file
        chunk_file = f"output_chunks/chunk_{i+1}.wav"
    
        # Perform transcription
        tarteel_result = pipe(chunk_file , generate_kwargs={"input_features": None} )
        text = tarteel_result["text"]
    
        transcriptions[i]["Transcription"] = text
    
    
    # Delete the folder and all its contents
    if os.path.exists("output_chunks"):
        shutil.rmtree("output_chunks")
        print(f"Cleaned up temporary directory: output_chunks")

    return transcriptions



def ask_gemini(surah_name, ayat):
    class Chunks(typing.TypedDict):
        ayat_list : list[str]

    
    # model = genai.GenerativeModel("gemini-1.0-pro-latest") 
    model = genai.GenerativeModel("gemini-2.0-flash-exp") 

    ayat_chunks = "\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(ayat)])
    result = model.generate_content(
    f'''From Surat {surah_name}, 
        You are given a list of ayat. Each item in the list may represent one or more complete ayahs combined together. 
        Your task is to:
        - Split each item into a list of individual ayahs.
        - Do not add any tashkeel (diacritical marks) to the output.
        
    The list of ayat chunks is:
    {ayat_chunks}    
    ''',
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=list[Chunks]
        ),
    )
    
    json_text = result.candidates[0].content.parts[0].text
    
    parsed_data = json.loads(json_text)
    
    ayahs = [item["ayat_list"] for item in parsed_data]

    return ayahs


def match_ayat(surah_name, transcriptions):
    df = pd.DataFrame(transcriptions)
     
    surah_texts_clean = [value for value in clean_quran[str(surah_mapping[surah_name])].values()]
    surah_texts_uthmani = [value for value in uthmani_quran[str(surah_mapping[surah_name])].values()]
    
    surah_numbers = {surah_texts_clean[i]: i+1 for i in range(len(surah_texts_clean))}
    
    for j in range(0, len(df), 7):
    
        if (j%70==0 and j!=0):
            print("Waiting for 45 seconds")
            time.sleep(45)
        
        chunk = df.iloc[j:j+7]  # Select a slice of up to 7 rows
        seven_ayat = ask_gemini(surah_name, chunk['Transcription'].tolist())
    
        # print("SEVEN", seven_ayat)
        for i, chunk_ayat in enumerate(seven_ayat):
            
            chunk_ayat_numbers = []
            for ayah in chunk_ayat:
                
                matched_ayat = rapidfuzz.process.extract(remove_tashkeel(ayah), surah_texts_clean, scorer=rapidfuzz.fuzz.partial_ratio, limit=5)
                
                # Sort by score and then by length
                # matched_ayat = sorted(matched_ayat, key=lambda x: (-x[1], -len(x[0])))
    
    
                # print("Ayah:", ayah)
                top_ayah_number = matched_ayat[0][2] + 1  # Get ayah number
                # print(matched_ayat[0][0])
                chunk_ayat_numbers.append(top_ayah_number)
        
                            
                # print("Base Text", remove_tashkeel(ayah))
                # print("matched ayat", matched_ayat)
                df.at[j+i, 'Transcription'] =  " ".join(remove_consecutive_duplicates([surah_texts_clean[i-1] for i in chunk_ayat_numbers]))
                # print(df.at[j+i, 'Transcription'])
                # print()
                df.at[j+i, 'Uthamni'] =  " ".join(remove_consecutive_duplicates([surah_texts_uthmani[i-1] for i in chunk_ayat_numbers]))  
    
    return df   


# Convert to SRT

def format_time(seconds):
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02}:{mins:02}:{secs:02},{millis:03}"

def convert_to_srt(df, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, row in df.iterrows():
            start_time = format_time(row['Start Time (s)'])
            end_time = format_time(row['End Time (s)'])
            transcription = row['Uthamni']
            
            # Write subtitle block
            f.write(f"{i + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{transcription}\n\n")
    return output_file


def run_app(file, surah_name, gemini_api_key):

    audio_path = file.name  # Temporary path to the uploaded file
    
    folder_name, audio_name = os.path.split(audio_path)
    
    genai.configure(api_key=gemini_api_key)
    
    transcriptions = split_into_chunks(audio_path)
    transcriptions = transcribe_tarteel(transcriptions)
    
    df = match_ayat(surah_name, transcriptions)
    
    subtitles_file = convert_to_srt(df, f"{audio_name}.srt")

    return df, subtitles_file

with open("quran-simple-clean.json", "r") as file:
        clean_quran = json.load(file)
    
with open("quran-uthmani.json", "r") as file:
    uthmani_quran = json.load(file)

with open("surah-mapping.json", "r") as file:
    surah_mapping = json.load(file)


iface = gr.Interface(
    fn=run_app,
    inputs=[
        gr.File(label="Upload Audio File"),
        gr.Dropdown(label="Surah Name", choices=list(surah_mapping.keys()), type="value"),
        gr.Textbox(label="Gemini API Key", type="password")
    ],
    outputs=[gr.DataFrame(label="Transcriptions"), gr.File(label="Download Subtitles")],
     allow_flagging="never" 
)

iface.launch()   