# Launch command
1. deploy the model
```
modal deploy chatterbox_tts.py
```

2. add the end point
```
mkdir -p /tmp/chatterbox_tts

curl -X POST --get "https://USERNAME--chatterbox-api-example-chatterbox-generate.modal.run" \
    --data-urlencode "prompt=Chatterbox running on Modal" 
    --output /tmp/chatterbox_tts/output.wav
```

# Workflow 
1. build docker image with requirement packages

2. allocate GPU with acceleration, use a pretrained model

3. generate waveform from input text

4. save the generated wav format 

5. return the audio via streaming response