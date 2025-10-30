import io
import os
import asyncio
import json
import time
import torch
import logging
import traceback
from typing import Optional, Dict, Any, List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))

class QwenOmniRunPod:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.active_connections: List[WebSocket] = []
        self._load_model()
        
    def _load_model(self):
        """Load the Qwen3-Omni model and processor"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Try transformers approach optimized for RunPod
            from transformers import Qwen3OmniMoeForConditionalGeneration
            
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype='auto',
                device_map="auto",
                attn_implementation='flash_attention_2'
            )
            logger.info("Successfully loaded model with transformers")
                
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
            logger.info("Model and processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def process_audio_streaming(self, audio_data: bytes, websocket: WebSocket, 
                                    temperature: float = 0.7, top_p: float = 0.8) -> None:
        """Process audio input and stream response back via WebSocket"""
        try:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Create message format for processing
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": "Please respond to what you heard."}
                ]
            }]
            
            # Process with transformers
            await self._process_with_transformers(messages, websocket, temperature, top_p)
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await websocket.send_json({
                "type": "error",
                "data": f"Processing error: {str(e)}"
            })
    
    async def _process_with_transformers(self, messages: List[Dict], websocket: WebSocket,
                                       temperature: float, top_p: float) -> None:
        """Process with transformers backend and stream audio response"""
        try:
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=None, 
                videos=None, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            # Generate with streaming capability
            text_ids, audio = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=1024,
                thinker_do_sample=True,
                thinker_temperature=temperature,
                thinker_top_p=top_p,
                speaker="Ethan",
                use_audio_in_video=False
            )
            
            # Process text response
            response_text = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Send text response
            await websocket.send_json({
                "type": "text_response", 
                "data": response_text,
                "timestamp": time.time()
            })
            
            # Stream audio response in chunks if available
            if audio is not None:
                await self._stream_audio_response(audio, websocket)
                
        except Exception as e:
            logger.error(f"Transformers processing error: {e}")
            raise
    
    async def _stream_audio_response(self, audio_tensor: torch.Tensor, websocket: WebSocket) -> None:
        """Stream audio response in chunks"""
        try:
            # Convert tensor to numpy array
            audio_data = audio_tensor.reshape(-1).float().detach().cpu().numpy()
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Stream in chunks
            total_samples = len(audio_data)
            chunk_samples = CHUNK_SIZE
            
            for i in range(0, total_samples, chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                
                # Convert chunk to bytes
                chunk_bytes = chunk.tobytes()
                
                # Send audio chunk
                await websocket.send_json({
                    "type": "audio_chunk",
                    "data": chunk_bytes.hex(),
                    "sample_rate": SAMPLE_RATE,
                    "chunk_index": i // chunk_samples,
                    "is_final": i + chunk_samples >= total_samples,
                    "timestamp": time.time()
                })
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)
                
            logger.info(f"Streamed {total_samples} audio samples in {(total_samples // chunk_samples) + 1} chunks")
            
        except Exception as e:
            logger.error(f"Audio streaming error: {e}")
            raise

# Initialize the model
qwen_omni = QwenOmniRunPod(MODEL_PATH)

# FastAPI app
app = FastAPI(title="Qwen3-Omni RunPod Speech-to-Speech", version="1.0.0")

@app.websocket("/ws/speech")
async def websocket_speech_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time speech-to-speech communication"""
    await websocket.accept()
    qwen_omni.active_connections.append(websocket)
    
    try:
        logger.info("WebSocket connection established")
        
        while True:
            # Receive message from client
            message = await websocket.receive_json()
            
            if message.get("type") == "audio_data":
                # Process audio data
                audio_hex = message.get("data", "")
                if audio_hex:
                    try:
                        audio_bytes = bytes.fromhex(audio_hex)
                        temperature = message.get("temperature", 0.7)
                        top_p = message.get("top_p", 0.8)
                        
                        await qwen_omni.process_audio_streaming(
                            audio_bytes, websocket, temperature, top_p
                        )
                        
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "data": f"Audio processing failed: {str(e)}"
                        })
                        
            elif message.get("type") == "ping":
                # Respond to ping with pong for connection health check
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if websocket in qwen_omni.active_connections:
            qwen_omni.active_connections.remove(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": qwen_omni.model is not None,
        "active_connections": len(qwen_omni.active_connections),
        "timestamp": time.time()
    }

@app.get("/", response_class=HTMLResponse)
async def get_demo_page():
    """Serve the demo HTML page"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qwen3-Omni Speech-to-Speech Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .controls { text-align: center; margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            .status { text-align: center; margin: 10px 0; padding: 10px; border-radius: 5px; }
            .response { margin: 20px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🎤 Qwen3-Omni Speech-to-Speech Demo</h1>
        <div class="controls">
            <button id="startBtn">Start Recording</button>
            <button id="stopBtn" disabled>Stop Recording</button>
            <button id="clearBtn">Clear</button>
        </div>
        <div class="status" id="status">Ready to record</div>
        <div class="response">
            <h3>Text Response:</h3>
            <div id="textResponse"></div>
        </div>
        <div class="response">
            <h3>Audio Response:</h3>
            <audio id="audioResponse" controls style="width: 100%;"></audio>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let websocket;
            let isRecording = false;

            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const textResponse = document.getElementById('textResponse');
            const audioResponse = document.getElementById('audioResponse');

            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/speech`;
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
                    status.style.backgroundColor = '#f8d7da';
                    setTimeout(connectWebSocket, 2000);
                };
            }

            let audioChunkBuffer = [];
            function handleAudioChunk(data) {
                const hexString = data.data;
                const audioBytes = new Uint8Array(hexString.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
                const audioData = new Int16Array(audioBytes.buffer);
                audioChunkBuffer.push(audioData);
                
                if (data.is_final) {
                    combineAndPlayAudio();
                }
            }

            function combineAndPlayAudio() {
                if (audioChunkBuffer.length === 0) return;
                
                const totalLength = audioChunkBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
                const combinedAudio = new Int16Array(totalLength);
                let offset = 0;
                
                for (const chunk of audioChunkBuffer) {
                    combinedAudio.set(chunk, offset);
                    offset += chunk.length;
                }
                
                const floatAudio = new Float32Array(combinedAudio.length);
                for (let i = 0; i < combinedAudio.length; i++) {
                    floatAudio[i] = combinedAudio[i] / 32768.0;
                }
                
                const wavBlob = createWavBlob(floatAudio, 24000);
                const audioUrl = URL.createObjectURL(wavBlob);
                audioResponse.src = audioUrl;
                
                audioChunkBuffer = [];
            }

            function createWavBlob(audioData, sampleRate) {
                const buffer = new ArrayBuffer(44 + audioData.length * 2);
                const view = new DataView(buffer);
                
                const writeString = (offset, string) => {
                    for (let i = 0; i < string.length; i++) {
                        view.setUint8(offset + i, string.charCodeAt(i));
                    }
                };
                
                writeString(0, 'RIFF');
                view.setUint32(4, 36 + audioData.length * 2, true);
                writeString(8, 'WAVE');
                writeString(12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true);
                view.setUint16(22, 1, true);
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true);
                view.setUint16(32, 2, true);
                view.setUint16(34, 16, true);
                writeString(36, 'data');
                view.setUint32(40, audioData.length * 2, true);
                
                let offset = 44;
                for (let i = 0; i < audioData.length; i++) {
                    const sample = Math.max(-1, Math.min(1, audioData[i]));
                    view.setInt16(offset, sample * 0x7FFF, true);
                    offset += 2;
                }
                
                return new Blob([buffer], { type: 'audio/wav' });
            }

            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                        }
                    };
                    
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        await processAudioBlob(audioBlob);
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
                    status.textContent = 'Processing your speech...';
                    status.style.backgroundColor = '#cce5ff';
                }
            }

            async function processAudioBlob(audioBlob) {
                try {
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    const pcmData = audioBuffer.getChannelData(0);
                    const int16Array = new Int16Array(pcmData.length);
                    for (let i = 0; i < pcmData.length; i++) {
                        int16Array[i] = Math.max(-32768, Math.min(32767, pcmData[i] * 32768));
                    }
                    
                    const uint8Array = new Uint8Array(int16Array.buffer);
                    const hexString = Array.from(uint8Array).map(b => b.toString(16).padStart(2, '0')).join('');
                    
                    websocket.send(JSON.stringify({
                        type: 'audio_data',
                        data: hexString,
                        temperature: 0.7,
                        top_p: 0.8
                    }));
                    
                } catch (error) {
                    status.textContent = 'Error processing audio';
                    status.style.backgroundColor = '#f8d7da';
                }
            }

            function clearResponses() {
                textResponse.innerHTML = '';
                audioResponse.src = '';
                status.textContent = 'Ready to record';
                status.style.backgroundColor = '#d4edda';
            }

            startBtn.addEventListener('click', startRecording);
            stopBtn.addEventListener('click', stopRecording);
            document.getElementById('clearBtn').addEventListener('click', clearResponses);

            connectWebSocket();
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info",
        access_log=True
    )
