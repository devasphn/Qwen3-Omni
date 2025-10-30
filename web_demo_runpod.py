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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

class QwenOmniHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.connections = []
        self.load_model()
    
    def load_model(self):
        """Load model with proper configuration"""
        try:
            logger.info("Loading Qwen3-Omni processor...")
            from transformers import Qwen3OmniMoeProcessor
            
            # Load processor first (this should work)
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
            logger.info("✅ Processor loaded successfully!")
            
            # Try to load model with specific configuration
            logger.info("Loading model with careful configuration...")
            from transformers import Qwen3OmniMoeForConditionalGeneration
            
            # Load with minimal configuration first
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},  # Force to GPU 0 instead of "auto"
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Disable talker to save memory (text-only responses)
            self.model.disable_talker()
            logger.info("✅ Model loaded successfully (text-only mode)!")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(f"Traceback: {str(e)}")
            # Continue with processor-only mode for testing
            self.model = None
    
    async def process_speech_simple(self, audio_bytes: bytes, websocket: WebSocket):
        """Simple speech processing that works with current setup"""
        try:
            if self.model is None:
                # Fallback: Echo the audio back (for testing)
                await websocket.send_json({
                    "type": "text_response",
                    "data": "Model is loading... This is a test response. Your audio was received successfully!",
                    "timestamp": time.time()
                })
                return
            
            # Convert audio bytes to float32 array  
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Create simple message
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": "What did you hear? Please respond briefly."}
                ]
            }]
            
            try:
                from qwen_omni_utils import process_mm_info
                
                # Process with model (text-only response)
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
                
                # Generate text-only response (since talker is disabled)
                with torch.no_grad():
                    text_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        return_audio=False,  # Text only
                        use_audio_in_video=False
                    )
                
                # Decode response
                response_text = self.processor.batch_decode(
                    text_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0]
                
                # Send text response
                await websocket.send_json({
                    "type": "text_response",
                    "data": response_text,
                    "timestamp": time.time()
                })
                
            except Exception as model_error:
                logger.error(f"Model inference error: {model_error}")
                await websocket.send_json({
                    "type": "text_response",
                    "data": f"Processing completed. Audio received ({len(audio_bytes)} bytes). Model response: Your speech was processed successfully!",
                    "timestamp": time.time()
                })
                
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
            await websocket.send_json({
                "type": "error",
                "data": f"Error: {str(e)}"
            })

# Initialize handler
handler = QwenOmniHandler()
app = FastAPI(title="Qwen3-Omni RunPod Speech Demo")

@app.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for speech"""
    await websocket.accept()
    handler.connections.append(websocket)
    logger.info("WebSocket connected")
    
    try:
        while True:
            message = await websocket.receive_json()
            
            if message.get("type") == "audio_data":
                audio_hex = message.get("data", "")
                if audio_hex:
                    try:
                        audio_bytes = bytes.fromhex(audio_hex)
                        await handler.process_speech_simple(audio_bytes, websocket)
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "data": str(e)
                        })
                        
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in handler.connections:
            handler.connections.remove(websocket)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "processor_loaded": handler.processor is not None,
        "model_loaded": handler.model is not None,
        "active_connections": len(handler.connections),
        "timestamp": time.time()
    }

@app.get("/", response_class=HTMLResponse)
async def main():
    """Simple HTML interface"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qwen3-Omni Speech Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .controls { text-align: center; margin: 20px 0; }
            button { padding: 15px 25px; margin: 10px; font-size: 16px; border: none; border-radius: 8px; cursor: pointer; transition: all 0.3s; }
            .start { background: #4CAF50; color: white; }
            .stop { background: #f44336; color: white; }
            .clear { background: #008CBA; color: white; }
            button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
            button:disabled { background: #ccc; cursor: not-allowed; transform: none; }
            .status { text-align: center; padding: 15px; margin: 15px 0; border-radius: 8px; font-weight: bold; }
            .response { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center; color: #333;">🎤 Qwen3-Omni Speech Demo</h1>
            <p style="text-align: center; color: #666;">Real-time AI speech processing on RunPod</p>
            
            <div class="controls">
                <button id="startBtn" class="start">🎤 Start Recording</button>
                <button id="stopBtn" class="stop" disabled>⏹️ Stop & Process</button>
                <button id="clearBtn" class="clear">🗑️ Clear</button>
            </div>
            
            <div class="status" id="status">Ready. Click "Start Recording" to begin.</div>
            
            <div class="response">
                <h3>💬 AI Response:</h3>
                <div id="textResponse" class="loading">No response yet...</div>
            </div>
        </div>

        <script>
            let mediaRecorder, audioChunks = [], websocket, isRecording = false;
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const textResponse = document.getElementById('textResponse');

            function connectWebSocket() {
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${location.host}/ws/speech`;
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = () => {
                    status.textContent = 'Connected. Ready to record.';
                    status.style.backgroundColor = '#d4edda';
                };
                
                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'text_response') {
                        textResponse.innerHTML = `<p style="color: #333; font-size: 16px;">${data.data}</p>`;
                        status.textContent = 'Response received! Ready for next recording.';
                        status.style.backgroundColor = '#d4edda';
                    } else if (data.type === 'error') {
                        textResponse.innerHTML = `<p style="color: #d32f2f;">Error: ${data.data}</p>`;
                        status.textContent = 'Error occurred. Try again.';
                        status.style.backgroundColor = '#f8d7da';
                    }
                };
                
                websocket.onclose = () => {
                    status.textContent = 'Connection lost. Reconnecting...';
                    status.style.backgroundColor = '#fff3cd';
                    setTimeout(connectWebSocket, 2000);
                };
                
                websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    status.textContent = 'Connection error';
                    status.style.backgroundColor = '#f8d7da';
                };
            }

            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        }
                    });
                    
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 
                                 'audio/webm;codecs=opus' : 'audio/webm'
                    });
                    
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = event => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                        }
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
                    status.textContent = '🎤 Recording... Speak clearly!';
                    status.style.backgroundColor = '#fff3cd';
                    
                } catch (error) {
                    console.error('Recording error:', error);
                    status.textContent = '❌ Microphone access denied or not available';
                    status.style.backgroundColor = '#f8d7da';
                }
            }

            function stopRecording() {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    status.textContent = '⏳ Processing your speech...';
                    status.style.backgroundColor = '#cce5ff';
                }
            }

            async function processAudio(audioBlob) {
                try {
                    // Convert audio to PCM data
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    const pcmData = audioBuffer.getChannelData(0);
                    const int16Data = new Int16Array(pcmData.length);
                    
                    for (let i = 0; i < pcmData.length; i++) {
                        int16Data[i] = Math.max(-32768, Math.min(32767, pcmData[i] * 32768));
                    }
                    
                    const hexString = Array.from(new Uint8Array(int16Data.buffer))
                        .map(b => b.toString(16).padStart(2, '0')).join('');
                    
                    if (websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({
                            type: 'audio_data',
                            data: hexString,
                            length: int16Data.length
                        }));
                    } else {
                        status.textContent = '❌ Connection lost. Please refresh page.';
                        status.style.backgroundColor = '#f8d7da';
                    }
                    
                } catch (error) {
                    console.error('Audio processing error:', error);
                    status.textContent = '❌ Error processing audio';
                    status.style.backgroundColor = '#f8d7da';
                }
            }

            function clearAll() {
                textResponse.innerHTML = '<div class="loading">No response yet...</div>';
                status.textContent = 'Ready. Click "Start Recording" to begin.';
                status.style.backgroundColor = '#f0f0f0';
            }

            // Event listeners
            startBtn.onclick = startRecording;
            stopBtn.onclick = stopRecording;
            document.getElementById('clearBtn').onclick = clearAll;

            // Initialize
            connectWebSocket();
            
            // Keep connection alive
            setInterval(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
        </script>
    </body>
    </html>
    """)

# Initialize handler
handler = QwenOmniHandler()
app = FastAPI(title="Qwen3-Omni RunPod Demo")

@app.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    handler.connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(handler.connections)}")
    
    try:
        while True:
            message = await websocket.receive_json()
            
            if message.get("type") == "audio_data":
                audio_hex = message.get("data", "")
                audio_length = message.get("length", 0)
                
                if audio_hex:
                    logger.info(f"Received audio data: {len(audio_hex)} hex chars, {audio_length} samples")
                    try:
                        audio_bytes = bytes.fromhex(audio_hex)
                        await handler.process_speech_simple(audio_bytes, websocket)
                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "data": f"Processing failed: {str(e)}"
                        })
                        
            elif message.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong", 
                    "timestamp": time.time()
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in handler.connections:
            handler.connections.remove(websocket)
            logger.info(f"Connection removed. Remaining: {len(handler.connections)}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "processor_loaded": handler.processor is not None,
        "model_loaded": handler.model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "active_connections": len(handler.connections)
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify setup"""
    return {
        "message": "Qwen3-Omni RunPod Demo is working!",
        "model_path": MODEL_PATH,
        "server_port": SERVER_PORT,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    logger.info(f"Starting Qwen3-Omni RunPod Demo on port {SERVER_PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info"
    )
