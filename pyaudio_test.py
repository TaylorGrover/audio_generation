import pyaudio
import wave

CHUNK = 1024

with wave.open("audio/mess_hall.wav", "rb") as wf:
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth())
                    , channels=wf.getnchannels()
                    , rate=wf.getframerate()
                    , output=True)

    while len(data := wf.readframes(CHUNK)): 
        stream.write(data)

    stream.close()

    p.terminate()
