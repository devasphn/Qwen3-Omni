import os
import asyncio
import json
import time
import torch
import logging
from typing import List, Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))

class QwenOmniHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.connections = []
        self.load_model()
    
    def load_model(self):
        """Load Qwen3-Omni model optimized for speech-to-speech"""
        try:
            logger.info("Loading Qwen3-Omni model...")
            
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    async def process_speech(self, audio_bytes: bytes, websocket: WebSocket):
        """Process speech input and stream response"""
        try:
            # Convert audio bytes to float32 array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Create conversation message
            messages = [{\n                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": "Please respond to what you heard in a conversational manner."}
                ]
            }]
            
            # Process with model
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
            
            inputs = self.processor(
                text=text,
                audio=audios,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            text_ids, audio_output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                speaker="Ethan",
                use_audio_in_video=False
            )
            
            # Get text response
            response_text = self.processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # Send text response first
            await websocket.send_json({
                "type": "text_response",
                "data": response_text,
                "timestamp": time.time()
            })
            
            # Stream audio response if available
            if audio_output is not None:
                await self.stream_audio(audio_output, websocket)
                
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
            await websocket.send_json({
                "type": "error", 
                "data": str(e)
            })
    
    async def stream_audio(self, audio_tensor, websocket):
        """Stream audio response in chunks"""
        try:
            audio_data = audio_tensor.reshape(-1).float().cpu().numpy()
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Send in chunks
            for i in range(0, len(audio_data), CHUNK_SIZE):
                chunk = audio_data[i:i + CHUNK_SIZE]
                
                await websocket.send_json({
                    "type": "audio_chunk",
                    "data": chunk.tobytes().hex(),
                    "sample_rate": 24000,
                    "is_final": i + CHUNK_SIZE >= len(audio_data)
                })
                
                await asyncio.sleep(0.01)  # Prevent overwhelming
                
        except Exception as e:
            logger.error(f"Audio streaming error: {e}")

# Initialize handler
handler = QwenOmniHandler()
app = FastAPI(title="Qwen3-Omni RunPod Speech-to-Speech")

@app.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for speech communication"""
    await websocket.accept()
    handler.connections.append(websocket)
    
    try:
        while True:
            message = await websocket.receive_json()
            
            if message.get("type") == "audio_data":
                audio_hex = message.get("data", "")
                if audio_hex:
                    audio_bytes = bytes.fromhex(audio_hex)
                    await handler.process_speech(audio_bytes, websocket)
                    
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        if websocket in handler.connections:
            handler.connections.remove(websocket)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": handler.model is not None}

@app.get("/", response_class=HTMLResponse)
async def main():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qwen3-Omni Speech-to-Speech</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .controls { text-align: center; margin: 20px 0; }
            button { padding: 15px 30px; margin: 10px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
            .start { background: #4CAF50; color: white; }
            .stop { background: #f44336; color: white; }
            .clear { background: #008CBA; color: white; }
            .status { text-align: center; padding: 10px; margin: 10px 0; border-radius: 5px; background: #f0f0f0; }
            .response { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">🎤 Qwen3-Omni Speech-to-Speech Demo</h1>
        <p style="text-align: center;">Real-time AI speech conversation on RunPod</p>
        
        <div class="controls">
            <button id="startBtn" class="start">🎤 Start Recording</button>
            <button id="stopBtn" class="stop" disabled>⏹️ Stop Recording</button>
            <button id="clearBtn" class="clear">🗑️ Clear</button>
        </div>
        
        <div class="status" id="status">Ready to record. Click "Start Recording" to begin.</div>
        
        <div class="response">
            <h3>💬 Text Response:</h3>
            <div id="textResponse"></div>
        </div>
        
        <div class="response">
            <h3>🔊 Audio Response:</h3>
            <audio id="audioResponse" controls style="width: 100%;"></audio>
        </div>

        <script>
            let mediaRecorder, audioChunks = [], websocket, isRecording = false;
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const textResponse = document.getElementById('textResponse');
            const audioResponse = document.getElementById('audioResponse');

            function connectWebSocket() {
                const wsUrl = `ws://${window.location.host}/ws/speech`;
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = () => {
                    status.textContent = 'Connected. Ready to record.';
                    status.style.backgroundColor = '#d4edda';
                };
                
                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'text_response') {
                        textResponse.innerHTML = `<p>${data.data}</p>`;
                    } else if (data.type === 'audio_chunk') {
                        handleAudioChunk(data);
                    }
                };
                
                websocket.onclose = () => {
                    status.textContent = 'Disconnected. Reconnecting...';
                    setTimeout(connectWebSocket, 2000);
                };
            }

            let audioBuffer = [];
            function handleAudioChunk(data) {
                const bytes = new Uint8Array(data.data.match(/.{1,2}/g).map(b => parseInt(b, 16)));
                const audio = new Int16Array(bytes.buffer);
                audioBuffer.push(audio);
                
                if (data.is_final) {
                    const totalLength = audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
                    const combined = new Int16Array(totalLength);
                    let offset = 0;
                    audioBuffer.forEach(chunk => {
                        combined.set(chunk, offset);
                        offset += chunk.length;
                    });
                    
                    const wavBlob = createWavBlob(combined, data.sample_rate);
                    audioResponse.src = URL.createObjectURL(wavBlob);
                    audioBuffer = [];
                }
            }

            function createWavBlob(audioData, sampleRate) {
                const buffer = new ArrayBuffer(44 + audioData.length * 2);
                const view = new DataView(buffer);
                
                view.setUint32(0, 0x52494646); // "RIFF"
                view.setUint32(4, 36 + audioData.length * 2, true);
                view.setUint32(8, 0x57415645); // "WAVE"
                view.setUint32(12, 0x666d7420); // "fmt "
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true);
                view.setUint16(22, 1, true);
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true);
                view.setUint16(32, 2, true);
                view.setUint16(34, 16, true);
                view.setUint32(36, 0x64617461); // "data"
                view.setUint32(40, audioData.length * 2, true);
                
                for (let i = 0; i < audioData.length; i++) {
                    view.setInt16(44 + i * 2, audioData[i], true);
                }
                
                return new Blob([buffer], {type: 'audio/wav'});
            }

            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = event => {
                        if (event.data.size > 0) audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, {type: 'audio/webm'});
                        await processAudio(audioBlob);
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    status.textContent = 'Recording... Speak now!';
                    status.style.backgroundColor = '#fff3cd';
                } catch (error) {
                    status.textContent = 'Microphone access denied';
                    status.style.backgroundColor = '#f8d7da';
                }
            }

            function stopRecording() {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    status.textContent = 'Processing...';
                    status.style.backgroundColor = '#cce5ff';
                }
            }

            async function processAudio(audioBlob) {
                try {
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioContext = new AudioContext();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    const pcmData = audioBuffer.getChannelData(0);
                    const int16Data = new Int16Array(pcmData.length);
                    
                    for (let i = 0; i < pcmData.length; i++) {
                        int16Data[i] = Math.max(-32768, Math.min(32767, pcmData[i] * 32768));
                    }
                    
                    const hexString = Array.from(new Uint8Array(int16Data.buffer))
                        .map(b => b.toString(16).padStart(2, '0')).join('');
                    
                    websocket.send(JSON.stringify({
                        type: 'audio_data',
                        data: hexString
                    }));
                } catch (error) {
                    status.textContent = 'Error processing audio';
                    status.style.backgroundColor = '#f8d7da';
                }
            }

            function clearAll() {
                textResponse.innerHTML = '';
                audioResponse.src = '';
                status.textContent = 'Ready to record';
                status.style.backgroundColor = '#f0f0f0';
            }

            startBtn.onclick = startRecording;
            stopBtn.onclick = stopRecording;
            document.getElementById('clearBtn').onclick = clearAll;

            connectWebSocket();
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
