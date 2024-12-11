import time
import threading
import pyaudio
import webrtcvad
import numpy as np

from queue import  Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from src.utils.common import *
from src.config.app_config import Speech2TxtConfig as sc
from src.module.processing.text_processing import TextProcessing
from src.module.wav2vec2.wav2vec2_inference import Wav2vec2Inference

class Speech2Txt:
    exit_event = threading.Event()
    def __init__(self, device_name='default'):
        self.device_name = device_name
        self.silence_limit_seconds = sc.silence_limit_seconds
        with ThreadPoolExecutor() as executor:
           future_wav2vec2 = executor.submit(Wav2vec2Inference)
           future_txt_processing = executor.submit(TextProcessing)

           self.wav2vec2 = future_wav2vec2.result()
           self.txt_processing = future_txt_processing.result()


    def stop(self):
        logging.info("stop the asr process")
        Speech2Txt.exit_event.set()
        self.asr_input_queue.put("close")
        self.asr_output_queue.put("close")
        self.asr_output_queue.put(None)

        # Wait for threads to finish
        self.vad_process.join()
        self.asr_process.join()
        self.spelling_correction_process.join()
        logging.info("Speech to text process stopped")


    def start(self):
        logging.info("Start the speech to text process")
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.corrected_output_queue = Queue()

        self.asr_process = threading.Thread(
            target=Speech2Txt.asr_process, 
            args=(self.asr_input_queue, self.asr_output_queue, self.wav2vec2))
        self.asr_process.start()

        self.vad_process = threading.Thread(
            target=Speech2Txt.vad_process, 
            args=(self.device_name, self.asr_input_queue))
        self.vad_process.start()

        self.spelling_correction_process = threading.Thread(
            target=Speech2Txt.spelling_correction_process, 
            args=(self.asr_output_queue, self.corrected_output_queue, self.txt_processing))
        self.spelling_correction_process.start()
  

    @staticmethod
    def vad_process(device_name, asr_input_queue):
        logging.info('Start voice activity detection process ...')
        vad = webrtcvad.Vad()
        vad.set_mode(sc.vad_mode)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHUNK = int(sc.rate * sc.frame_duration / 1000)

        microphones = Speech2Txt.list_microphones(audio)
        selected_input_device_id = Speech2Txt.get_input_device_id(
            device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=FORMAT,
                            channels=sc.channels,
                            rate=sc.rate,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''
        while True:
            if Speech2Txt.exit_event.is_set():
                break
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, sc.rate)
            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    asr_input_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    
    @staticmethod
    def asr_process(in_queue, output_queue, wave2vec_asr):
        logging.info("\n--------------------------Listening to your voice--------------------------\n")

        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            text = wave2vec_asr.speech_recognition(float64_buffer)
            text = text.lower()
            if text != "":
                output_queue.put(text)
                logging.info(f'Raw text: {text}')
    

    @staticmethod
    def spelling_correction_process(input_queue, output_queue, text_processing):
        while True:
            text = input_queue.get()
            if text == "close":
                break

            corrected_text = text_processing.text_post_processing(text)
            output_queue.put(corrected_text)


    @staticmethod
    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]


    @staticmethod
    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result


    def get_last_text(self):
        logging.info("returns the text, sample length and inference time in seconds.")
        return self.corrected_output_queue.get(timeout=self.silence_limit_seconds)
    

    def run(self, live_record: bool):
        if live_record:
            logging.info('Real time inferencing Wav2vec2 ...')
            self.start()
            final_text = ''

            try:
                while True:
                    try:
                        text = self.get_last_text()
                        if text is None:  
                            break
                        
                        elif text:
                            logging.info(f"Current text: {text}")
                            final_text += text + ' '
                    
                    except Empty:
                        logging.info("No voice detected for the timeout duration. Exiting...")
                        break

                logging.info(f"Final Text: {final_text}")

            except KeyboardInterrupt:
                logging.info("\nInterrupted by user.")
            finally:
                self.stop()
            
            return final_text
    

