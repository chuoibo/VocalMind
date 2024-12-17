
def audio_stream(audio_path):
    with open(audio_path, "rb") as audio_file:
        yield from audio_file