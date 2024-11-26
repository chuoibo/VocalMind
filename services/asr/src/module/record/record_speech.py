import numpy as np
import webrtcvad
import pyaudio
import time

from config.app_config import Speech2TxtConfig as sc
import logging
class Record:
    def __init__(self):
        self.vad_mode = sc.vad_mode
        self.silence_limit_seconds = sc.silence_limit_seconds
        self.rate = sc.rate
        self.frame_duration = sc.frame_duration
        self.max_pause = sc.max_pause
        self.min_pause = sc.min_pause
        logging.info('Initialize speech recording module ...')


    def record_audio(self):
        vad = webrtcvad.Vad()
        vad.set_mode(self.vad_mode)

        FORMAT = pyaudio.paInt16  
        CHANNELS = 1  
        FRAME_SIZE = int(self.rate * self.frame_duration / 1000)

        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=FRAME_SIZE)

        audio_buffer = []
        silence_start_time = None
        pause_detected = False
        pause_markers = {} 

        def is_speech(frame):
            """Check if the frame contains speech using VAD."""
            return vad.is_speech(frame, self.rate)

        logging.info("Recording audio...")

        try:
            while True:
                audio_data = stream.read(FRAME_SIZE, exception_on_overflow=False)

                if is_speech(audio_data):
                    if pause_detected:
                        pause_duration = time.time() - silence_start_time
                        logging.info("Pause ended, noting PAUSE marker position.")
                        pause_markers[len(audio_buffer)] = pause_duration
                        pause_detected = False
                    logging.info("Voice detected")
                    audio_buffer.append(audio_data)
                    silence_start_time = None  
                else:
                    logging.info("No voice detected")
                    if silence_start_time is None:
                        silence_start_time = time.time()  
                    elif time.time() - silence_start_time > self.min_pause and time.time() - silence_start_time <= self.max_pause:
                        logging.info("PAUSE detected")
                        pause_detected = True
                    elif time.time() - silence_start_time > self.silence_limit_seconds:
                        logging.info(f"Stopping recording after {self.silence_limit_seconds} seconds of silence...")
                        break

        except KeyboardInterrupt:
            logging.info("Stopped recording manually.")

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        audio_data_combined = b''.join(audio_buffer)
        audio_array = np.frombuffer(audio_data_combined, dtype=np.int16).astype(np.float32) / 32768.0

        return audio_array, pause_markers