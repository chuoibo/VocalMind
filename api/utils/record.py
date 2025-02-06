import os
import pyaudio
import webrtcvad
import wave
import numpy as np

from config.app_config import RecordingConfig as rcf
from utils.api_logger import logging


class Recording:
    def __init__(self):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(rcf.VAD_MODE)
        self.device_name = rcf.DEVICE_NAME
        self.format = pyaudio.paInt16  
        self.audio = pyaudio.PyAudio()
        self.channels = rcf.CHANNELS    
        self.rate = rcf.RATE  
        self.frame_duration = rcf.FRAME_DURATION  
        self.frame_size = int(self.rate * self.frame_duration / 1000)  

        self.silence_threshold = rcf.SILENCE_THRESHOLD   
        self.silence_frame = int(self.silence_threshold * self.rate / self.frame_size)

        if not os.path.exists(rcf.SAVE_DIR):
            os.makedirs(rcf.SAVE_DIR, exist_ok=True)
        
        self.output_file_path = rcf.SAVE_PATH


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


    def get_record(self):
        microphones = Recording.list_microphones(self.audio) 

        selected_input_device_id = Recording.get_input_device_id(
            self.device_name, microphones
            )

        stream = self.audio.open(input_device_index=selected_input_device_id,
                                 format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.frame_size)

        frames = []
        silence_counter = 0
        is_recording = False

        logging.info("Recording... Speak now!")

        try:
            while True:
                # Read audio frame
                frame = stream.read(self.frame_size, exception_on_overflow=False)
                frames.append(frame)

                # Convert frame to numpy array for VAD
                audio_data = np.frombuffer(frame, dtype=np.int16)

                # Check if the frame contains speech
                if self.vad.is_speech(audio_data.tobytes(), self.rate):
                    is_recording = True
                    silence_counter = 0
                else:
                    if is_recording:
                        silence_counter += 1

                # Stop recording if silence exceeds the threshold
                if is_recording and silence_counter >= self.silence_frame:
                    logging.info("Silence detected. Stopping recording.")
                    break

        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

            with wave.open(self.output_file_path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b"".join(frames))

            logging.info("Recording saved to output.wav")