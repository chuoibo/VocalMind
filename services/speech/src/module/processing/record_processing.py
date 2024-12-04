import numpy as np

import logging

from src.config.app_config import (Speech2TxtConfig as sc)


class RecordProcessing:
    def __init__(self, pause_markers: dict):
        self.pause_markers = pause_markers
        self.pre_processing_multiplier = sc.pre_processing_multiplier
        self.frame_duration = sc.frame_duration
        self.frame_rate = sc.rate
        self.frame_size = (self.frame_rate * self.frame_duration) / 1000
        logging.info('Initialize pre-processing module')
    

    def calculate_dynamic_threshold(self, time_diffs, multiplier=1.2):
        mean_time_diff = np.mean(time_diffs)
        std_time_diff = np.std(time_diffs)
        dynamic_threshold = mean_time_diff + multiplier * std_time_diff
        return dynamic_threshold


    def find_nearest_previous_punctuation(self, current_pause, classified_pauses, punctuation_type=None):
        logging.info('Finding the nearest previous punctuation based on its type')
        if punctuation_type == 'full_stop':
            previous_punctuations = [k for k, v in classified_pauses.items() if v == 'full_stop' and k < current_pause]
        else:
            previous_punctuations = [k for k in classified_pauses.keys() if k < current_pause]

        if previous_punctuations:
            return max(previous_punctuations) 
        
        return None


    def process_pause_markers(self):        
        pause_markers = {int(k): v for k, v in self.pause_markers.items() if int(k) >= 10}

        if len(pause_markers) == 0:
            return {}

        elif len(pause_markers) == 1:
            single_key = list(pause_markers.keys())[0]
            pause_markers[single_key] = 'full_stop'
            return pause_markers

        else:
            classified_pauses = {}

            pause_durations = np.array(list(pause_markers.values()))
            q25 = np.percentile(pause_durations, 25)  
            q75 = np.percentile(pause_durations, 75) 

            for index, duration in pause_markers.items():
                if duration < q25:  
                    classified_pauses[index] = 'comma'
                elif duration > q75:  #
                    classified_pauses[index] = 'full_stop'
            unclassified_markers = {k: v for k, v in pause_markers.items() if k not in classified_pauses}

            if len(unclassified_markers) == 0:
                return classified_pauses

            unclassified_keys = list(unclassified_markers.keys())

            all_pauses = sorted(pause_markers.keys())  
            time_diffs = []
            for i in range(1, len(all_pauses)):
                time_diff = (all_pauses[i] - all_pauses[i - 1]) * self.frame_duration
                time_diffs.append(time_diff)

            dynamic_threshold = self.calculate_dynamic_threshold(time_diffs)
            logging.info(f'Dynamic distance threshold is set at: {dynamic_threshold} ms')

            for current_pause in unclassified_keys:
                last_full_stop_index = self.find_nearest_previous_punctuation(current_pause, classified_pauses, 'full_stop')

                if last_full_stop_index is None:
                    time_diff_from_start = current_pause * self.frame_duration
                    classified_pauses[current_pause] = 'comma' if time_diff_from_start <= dynamic_threshold else 'full_stop'
                else:
                    time_diff_from_last_full_stop = (current_pause - last_full_stop_index) * self.frame_duration
                    logging.info(f'Distance difference between the current pause and the last full stops: {time_diff_from_last_full_stop} ms')
                    
                    if time_diff_from_last_full_stop > dynamic_threshold:
                        classified_pauses[current_pause] = 'full_stop'
                    else:
                        classified_pauses[current_pause] = 'comma'

            logging.info('Finished classifying pause markers.')
            return classified_pauses
    

    def mapping_punctuations(self, word_timings: dict, pause_markers: dict):
        logging.info('Start to map all punctuations to the transcription')
        for buffer_len, punctuation in pause_markers.items():
            pause_time = buffer_len * self.frame_size / self.frame_rate
            
            for idx in range(1, len(word_timings)):
                prev_word_end = word_timings[idx - 1]['end_time']
                current_word_start = word_timings[idx]['start_time']
                
                if (current_word_start <= pause_time <= word_timings[idx]['end_time']) or \
                    (prev_word_end <= pause_time <= current_word_start):
                    
                    if 'punctuation' in word_timings[idx]:
                        word_timings[idx - 1]['punctuation'] += f" {punctuation}"
                    else:
                        word_timings[idx - 1]['punctuation'] = punctuation
                    break

        sentence = ""
        capitalize_next = False

        for idx, word_data in word_timings.items():
            word_text = word_data['word']
            
            if capitalize_next:
                word_text = word_text.capitalize()
                capitalize_next = False

            if 'punctuation' in word_data:
                if word_data['punctuation'] == 'comma':
                    word_text += ","
                elif word_data['punctuation'] == 'full_stop':
                    word_text += "."
                    capitalize_next = True
                    
            sentence += word_text + " "

        sentence = sentence.strip()
        logging.info('Finish mapping all punctuations to the transcription')

        return sentence