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