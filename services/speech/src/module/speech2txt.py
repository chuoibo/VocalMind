import threading
import pyaudio
import webrtcvad
import wave
import time
import numpy as np

from queue import  Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from src.utils.common import *
from src.config.app_config import Speech2TxtConfig as sc
from src.module.processing.text_processing import TextProcessing
from src.module.wav2vec2.wav2vec2_inference import Wav2vec2Inference

WAV2VEC2_INIT = None
TEXT_PROCESSING_INIT = None

class VADProcessor:
    """Handles Voice Activity Detection."""
    def __init__(self, device_name='default'):
        self.device_name = device_name
        self.rate = sc.rate
        self.frame_duration = sc.frame_duration
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.chunk = int(sc.rate * sc.frame_duration / 1000)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(sc.vad_mode)
    

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


    def process_stream(self, asr_input_queue):
        """Processes audio from a stream and detects speech."""
        logging.info('Processing audio stream...')
        microphones = VADProcessor.list_microphones(self.audio)

        selected_input_device_id = VADProcessor.get_input_device_id(
            self.device_name, microphones)

        stream = self.audio.open(input_device_index=selected_input_device_id,
                                 format=self.format,
                                 channels=sc.channels,
                                 rate=sc.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk)
        
        frames = b''

        while True:
            if Speech2Txt.exit_event.is_set():
                break
            
            frame = stream.read(self.chunk, exception_on_overflow=False)
            is_speech = self.vad.is_speech(frame, self.rate)

            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    asr_input_queue.put(frames)
                frames = b''


    def process_file(self, audio_file_path, asr_input_queue):
        """Processes audio from a file."""
        logging.info('Processing audio file...')

        with wave.open(audio_file_path, 'rb') as wf:
            if wf.getframerate() != self.rate:
                raise ValueError(f"Audio sample rate mismatch. Expected: {self.rate}, Got: {wf.getframerate()}")
            
            frames = b''

            while True:
                data = wf.readframes(self.chunk)
                if not data:
                    break

                is_speech = self.vad.is_speech(data, self.rate)
                if is_speech:
                    frames += data
                else:
                    if len(frames) > 1:
                        asr_input_queue.put(frames)
                    frames = b''

        asr_input_queue.put("close")


class ASRProcessor:
    """Handles Automatic Speech Recognition."""
    def __init__(self, model):
        self.model = model

    def process_audio(self, in_queue, out_queue):
        """Processes audio frames and performs ASR."""
        logging.info("Processing audio for ASR...")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(audio_frames, dtype=np.int16) / 32767
            text = self.model.speech_recognition(float64_buffer).lower()

            if text:
                out_queue.put(text)
                logging.info(f"Recognized Text: {text}")


class TextProcessor:
    """Handles Text Processing (e.g., Spelling Correction)."""
    def __init__(self, processor):
        self.processor = processor

    def correct_text(self, input_queue, output_queue):
        """Processes text for corrections."""
        logging.info("Processing text for corrections...")
        while True:
            text = input_queue.get()
            if text == "close":
                break

            corrected_text = self.processor.text_post_processing(text)
            output_queue.put(corrected_text)


class Speech2Txt:
    """Main Speech-to-Text Pipeline."""
    exit_event = threading.Event()

    def __init__(self, live_record, input_audio_file_path):
        global WAV2VEC2_INIT, TEXT_PROCESSING_INIT

        self.live_record = live_record
        self.input_audio_file_path = input_audio_file_path
        self.device_name = sc.device_name
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.corrected_output_queue = Queue()

        if (WAV2VEC2_INIT is None) or (TEXT_PROCESSING_INIT is None):
            with ThreadPoolExecutor() as executor:
                future_wav2vec2 = executor.submit(Wav2vec2Inference)
                future_txt_processing = executor.submit(TextProcessing)

                if WAV2VEC2_INIT is None:
                    WAV2VEC2_INIT = future_wav2vec2.result()
                
                if TEXT_PROCESSING_INIT is None:
                    TEXT_PROCESSING_INIT = future_txt_processing.result()

        self.wav2vec2 = WAV2VEC2_INIT
        self.txt_processing = TEXT_PROCESSING_INIT

        self.vad_processor = VADProcessor()
        self.asr_processor = ASRProcessor(self.wav2vec2)
        self.text_processor = TextProcessor(self.txt_processing)


    def start(self):
        """Start the Speech-to-Text process."""
        logging.info("Starting Speech-to-Text process...")
        if self.live_record:
            self.vad_thread = threading.Thread(
                target=self.vad_processor.process_stream,
                args=(self.device_name, self.asr_input_queue),
            )
        else:
            self.vad_thread = threading.Thread(
                target=self.vad_processor.process_file,
                args=(self.input_audio_file_path, self.asr_input_queue),
            )
        self.vad_thread.start()

        self.asr_thread = threading.Thread(
            target=self.asr_processor.process_audio,
            args=(self.asr_input_queue, self.asr_output_queue),
        )
        self.asr_thread.start()

        self.text_thread = threading.Thread(
            target=self.text_processor.correct_text,
            args=(self.asr_output_queue, self.corrected_output_queue),
        )
        self.text_thread.start()


    def stop(self):
        """Stop the Speech-to-Text process."""
        logging.info("Stopping Speech-to-Text process...")
        Speech2Txt.exit_event.set()
        self.asr_input_queue.put("close")
        self.asr_output_queue.put("close")
        self.corrected_output_queue.put(None)

        self.vad_thread.join()
        self.asr_thread.join()
        self.text_thread.join()


    def run(self):
        """Run the pipeline."""
        start_time = time.time()
        self.start()
        final_text = ""

        try:
            while True:
                try:
                    text = self.corrected_output_queue.get(timeout=sc.silence_limit_seconds)
                    if text:
                        logging.info(f"Corrected Text: {text}")
                        final_text += text + " "
                except Empty:
                    logging.info("No more input detected. Stopping.")
                    break
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            self.stop()
        end_time = time.time()

        logging.info(f"Final Text: {final_text} with inference time: {end_time-start_time}s")
        return final_text
    

