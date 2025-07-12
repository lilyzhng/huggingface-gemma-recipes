# takes your local Python code and turns it into a production-ready cloud API that anyone can call via HTTP/
# cloud API: https://USERNAME--chatterbox-api-example-chatterbox-generate.modal.run
# Method: POST
# Input: prompt
# Output: audio/wav
# the api can be called in below ways
"""
import requests

response = requests.post("https://USERNAME--chatterbox-api-example-chatterbox-generate.modal.run", data={"prompt": "I will be working on a new project on Voice AI"})

with open("/tmp/chatterbox_tts/output.wav", "wb") as f:
    f.write(response.content)
"""

import io
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "chatterbox-tts==0.1.1", "fastapi[standard]"
)
app = modal.App("chatterbox-api-example", image=image)

with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from fastapi.responses import StreamingResponse

# acceleration with A10G

@app.cls(gpu="a10g", scaledown_window=60 * 5, enable_memory_snapshot=True)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    @modal.enter()
    def load(self):
        # ChatterboxTTS is using a 0.5B (500 million) parameter Llama backbone architecture
        # Trained on 500,000 hours (0.5M hours) of cleaned audio data
        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def generate(self, prompt: str):
        # Generate audio waveform from the input text
        wav = self.model.generate(prompt)

        # Create an in-memory buffer to store the WAV file
        buffer = io.BytesIO()

        # Save the generated audio to the buffer in WAV format
        # Uses the model's sample rate and WAV format
        ta.save(buffer, wav, self.model.sr, format="wav")

        # Reset buffer position to the beginning for reading
        buffer.seek(0)

        # Return the audio as a streaming response with appropriate MIME type.
        # This allows for browsers to playback audio directly.
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="audio/wav",
        )


