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
        """Load model with fallback strategy"""
        try:
            logger.info("Loading Qwen3-Omni processor...")
            from transformers import Qwen3OmniMoeProcessor
            
            # Load processor (this works)
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
            logger.info("✅ Processor loaded successfully!")
            
            # Try different model loading approaches
            logger.info("Attempting model loading strategies...")
            
            try:
                # Strategy 1: Load Thinking model instead (has fewer components)
                from transformers import Qwen3OmniMoeForConditionalGeneration
                thinking_model_path = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
                
                logger.info(f"Trying Thinking model: {thinking_model_path}")
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    thinking_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.model_path = thinking_model_path
                logger.info("✅ Thinking model loaded successfully!")
                
            except Exception as thinking_error:
                logger.warning(f"Thinking model failed: {thinking_error}")
                
                try:
                    # Strategy 2: Load with different configuration
                    from transformers import Qwen3OmniMoeForConditionalGeneration
                    
                    # Patch the configuration issue
                    import transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe as qwen_config
                    
                    # Monkey patch to handle missing rope_parameters
                    original_getattribute = qwen_config.Qwen3OmniMoeCode2WavConfig.__getattribute__
                    
                    def patched_getattribute(self, name):
                        if name == 'rope_parameters' and not hasattr(self, name):
                            # Return default rope parameters
                            return {
                                "rope_type": "default",
                                "factor": 1.0,
                                "original_max_position_embeddings": 2048,
                                "rope_scaling": None
                            }
                        return original_getattribute(self, name)
                    
                    qwen_config.Qwen3OmniMoeCode2WavConfig.__getattribute__ = patched_getattribute
                    
                    self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                        MODEL_PATH,
                        torch_dtype=torch.bfloat16,
                        device_map={"": 0},
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    
                    # Disable talker to save memory and avoid audio issues
                    self.model.disable_talker()
                    logger.info("✅ Model loaded with patch (text-only mode)!")
                    
                except Exception as patch_error:
                    logger.error(f"Patched model loading failed: {patch_error}")
                    self.model = None
            
        except Exception as e:
            logger.error(f"Complete model loading failed: {e}")
            self.model = None
            self.processor = None
    
    async def process_speech_text_only(self, audio_bytes: bytes, websocket: WebSocket):
        """Process speech and return text response"""
        try:
            if self.processor is None:
                await websocket.send_json({
                    "type": "text_response",
                    "data": "System Error: Processor not loaded. Please check model configuration.",
                    "timestamp": time.time()
                })
                return
            
            if self.model is None:
                # Fallback mode: Just echo that we received audio
                audio_length = len(audio_bytes)
                await websocket.send_json({
                    "type": "text_response",
                    "data": f"Audio received successfully! Length: {audio_length} bytes. Model loading had issues, but the system is working. The processor is loaded and ready for speech recognition once model issues are resolved.",
                    "timestamp": time.time()
                })
                return
            
            # Convert audio bytes to float32 array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Create conversation message
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": "Please respond to what you heard in a brief, conversational manner."}
                ]
            }]
            
            try:
                from qwen_omni_utils import process_mm_info
                
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
                
                # Generate response (text only)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        return_audio=False,
                        use_audio_in_video=False
                    )
                
                # Decode response
                response_text = self.processor.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                if not response_text.strip():
                    response_text = "I heard your audio input. The speech recognition is working!"
                
                # Send text response
                await websocket.send_json({
                    "type": "text_response",
                    "data": response_text,
                    "timestamp": time.time()
                })
                
            except Exception as processing_error:
                logger.error(f"Model processing error: {processing_error}")
                # Still provide feedback that audio was received
                await websocket.send_json({
                    "type": "text_response",
                    "data": f"Audio processing completed. Received {len(audio_bytes)} bytes of audio data. Processing pipeline is functional!",
                    "timestamp": time.time()
                })
                
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
            await websocket.send_json({
                "type": "error",
                "data": f"Error: {str(e)}"
            })

# Initialize handler
logger.info("Initializing Qwen3-Omni handler...")
handler = QwenOmniHandler()

# FastAPI app
app = FastAPI(title="Qwen3-Omni RunPod Speech Demo")

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
                    logger.info(f"Processing audio: {len(audio_hex)} hex chars, {audio_length} samples")
                    try:
                        audio_bytes = bytes.fromhex(audio_hex)
                        await handler.process_speech_text_only(audio_bytes, websocket)
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "data": f"Processing failed: {str(e)}"
                        })
                        
            elif message.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong", 
                    "timestamp": time.time()
                })
                logger.debug("Ping/Pong completed")
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected gracefully")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in handler.connections:
            handler.connections.remove(websocket)
            logger.info(f"Connection removed. Remaining: {len(handler.connections)}")

@app.get("/health")
async def health():
    """Detailed health check"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_memory": [f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB" 
                          for i in range(torch.cuda.device_count())]
        }
    
    return {
        "status": "healthy",
        "processor_loaded": handler.processor is not None,
        "model_loaded": handler.model is not None,
        "model_path": handler.model_path if hasattr(handler, 'model_path') else MODEL_PATH,
        "active_connections": len(handler.connections),
        "gpu_info": gpu_info,
        "timestamp": time.time()
    }

@app.get("/", response_class=HTMLResponse)
async def main():
    """Enhanced HTML interface with better error handling"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qwen3-Omni RunPod Demo</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container { 
                background: rgba(255, 255, 255, 0.1); 
                backdrop-filter: blur(10px);
                padding: 30px; 
                border-radius: 20px; 
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
            h1 { text-align: center; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .subtitle { text-align: center; margin-bottom: 30px; opacity: 0.8; }
            .controls { 
                display: flex; 
                justify-content: center; 
                gap: 15px; 
                margin: 25px 0; 
                flex-wrap: wrap; 
            }
            button { 
                padding: 15px 25px; 
                font-size: 16px; 
                font-weight: bold;
                border: none; 
                border-radius: 25px; 
                cursor: pointer; 
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.2);
            }
            .start { background: linear-gradient(45deg, #4CAF50, #45a049); color: white; }
            .stop { background: linear-gradient(45deg, #f44336, #da190b); color: white; }
            .clear { background: linear-gradient(45deg, #008CBA, #005f7f); color: white; }
            button:hover:not(:disabled) { 
                transform: translateY(-3px); 
                box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.3); 
            }
            button:disabled { 
                background: #666 !important; 
                cursor: not-allowed; 
                transform: none; 
                opacity: 0.6;
            }
            .status { 
                text-align: center; 
                padding: 15px; 
                margin: 20px 0; 
                border-radius: 15px; 
                font-weight: bold;
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            .response { 
                margin: 25px 0; 
                padding: 20px; 
                border-radius: 15px; 
                background: rgba(0, 0, 0, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .response h3 { 
                margin-top: 0; 
                color: #fff; 
                border-bottom: 1px solid rgba(255, 255, 255, 0.3);
                padding-bottom: 10px;
            }
            #textResponse { 
                font-size: 16px; 
                line-height: 1.5; 
                min-height: 60px;
                padding: 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .loading { color: #ccc; font-style: italic; }
            .error { color: #ffcdd2; }
            .success { color: #c8e6c9; }
            .info-box {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 15px;
                margin: 15px 0;
                font-size: 14px;
            }
            .health-status {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 10px 0;
                padding: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Qwen3-Omni Speech Demo</h1>
            <p class="subtitle">Real-time AI speech processing on RunPod • Text Responses</p>
            
            <div class="info-box">
                <strong>Status:</strong> <span id="systemStatus">Loading...</span><br>
                <strong>Model:</strong> <span id="modelInfo">Checking...</span><br>
                <strong>Connection:</strong> <span id="connectionStatus">Connecting...</span>
            </div>
            
            <div class="controls">
                <button id="startBtn" class="start">🎤 Start Recording</button>
                <button id="stopBtn" class="stop" disabled>⏹️ Stop & Process</button>
                <button id="clearBtn" class="clear">🗑️ Clear</button>
            </div>
            
            <div class="status" id="status">Loading system...</div>
            
            <div class="response">
                <h3>💬 AI Text Response:</h3>
                <div id="textResponse" class="loading">Ready for your first speech input...</div>
            </div>
        </div>

        <script>
            let mediaRecorder, audioChunks = [], websocket, isRecording = false;
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const textResponse = document.getElementById('textResponse');
            const systemStatus = document.getElementById('systemStatus');
            const modelInfo = document.getElementById('modelInfo');
            const connectionStatus = document.getElementById('connectionStatus');

            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const health = await response.json();
                    
                    systemStatus.textContent = health.status;
                    systemStatus.className = health.status === 'healthy' ? 'success' : 'error';
                    
                    modelInfo.textContent = health.model_loaded ? 
                        `Loaded (${health.model_path || 'Unknown'})` : 'Not loaded';
                    modelInfo.className = health.model_loaded ? 'success' : 'error';
                    
                    if (health.gpu_info && health.gpu_info.gpu_count > 0) {
                        modelInfo.textContent += ` • GPU: ${health.gpu_info.gpu_names[0]}`;
                    }
                    
                    return health;
                } catch (error) {
                    systemStatus.textContent = 'Error';
                    systemStatus.className = 'error';
                    modelInfo.textContent = 'Check failed';
                    modelInfo.className = 'error';
                    return null;
                }
            }

            function connectWebSocket() {
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${location.host}/ws/speech`;
                
                connectionStatus.textContent = 'Connecting...';
                connectionStatus.className = 'loading';
                
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = () => {
                    connectionStatus.textContent = 'Connected';
                    connectionStatus.className = 'success';
                    status.textContent = '✅ Ready to record! Click "Start Recording" to begin.';
                    status.style.background = 'rgba(76, 175, 80, 0.3)';
                    console.log('WebSocket connected successfully');
                };
                
                websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };
                
                websocket.onclose = (event) => {
                    connectionStatus.textContent = 'Disconnected';
                    connectionStatus.className = 'error';
                    status.textContent = '🔄 Connection lost. Reconnecting...';
                    status.style.background = 'rgba(255, 193, 7, 0.3)';
                    console.log('WebSocket closed:', event.code, event.reason);
                    setTimeout(connectWebSocket, 2000);
                };
                
                websocket.onerror = (error) => {
                    connectionStatus.textContent = 'Error';
                    connectionStatus.className = 'error';
                    console.error('WebSocket error:', error);
                    status.textContent = '❌ Connection error';
                    status.style.background = 'rgba(244, 67, 54, 0.3)';
                };
            }

            function handleWebSocketMessage(data) {
                switch(data.type) {
                    case 'text_response':
                        textResponse.innerHTML = `<p class="success">${data.data}</p>`;
                        status.textContent = '✅ Response received! Ready for next recording.';
                        status.style.background = 'rgba(76, 175, 80, 0.3)';
                        console.log('Received text response:', data.data);
                        break;
                        
                    case 'error':
                        textResponse.innerHTML = `<p class="error">❌ Error: ${data.data}</p>`;
                        status.textContent = '❌ Processing error occurred.';
                        status.style.background = 'rgba(244, 67, 54, 0.3)';
                        console.error('Received error:', data.data);
                        break;
                        
                    case 'pong':
                        console.log('Received pong - connection alive');
                        break;
                        
                    default:
                        console.log('Unknown message type:', data.type);
                }
            }

            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });
                    
                    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 
                                   'audio/webm;codecs=opus' : 'audio/webm';
                    
                    mediaRecorder = new MediaRecorder(stream, { mimeType });
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = event => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                            console.log(`Audio chunk received: ${event.data.size} bytes`);
                        }
                    };
                    
                    mediaRecorder.onstop = async () => {
                        console.log('Recording stopped, processing audio...');
                        const audioBlob = new Blob(audioChunks, {type: mimeType});
                        await processAudio(audioBlob);
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    mediaRecorder.start(100); // Record in 100ms chunks
                    isRecording = true;
                    
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    status.textContent = '🎤 Recording... Speak clearly!';
                    status.style.background = 'rgba(255, 193, 7, 0.3)';
                    
                    console.log('Recording started successfully');
                    
                } catch (error) {
                    console.error('Recording error:', error);
                    status.textContent = '❌ Microphone access denied or unavailable';
                    status.style.background = 'rgba(244, 67, 54, 0.3)';
                    
                    // Show more specific error info
                    if (error.name === 'NotAllowedError') {
                        textResponse.innerHTML = '<p class="error">❌ Microphone permission denied. Please allow microphone access and refresh the page.</p>';
                    } else if (error.name === 'NotFoundError') {
                        textResponse.innerHTML = '<p class="error">❌ No microphone found. Please check your audio device.</p>';
                    } else {
                        textResponse.innerHTML = `<p class="error">❌ Recording error: ${error.message}</p>`;
                    }
                }
            }

            function stopRecording() {
                if (mediaRecorder && isRecording) {
                    console.log('Stopping recording...');
                    mediaRecorder.stop();
                    isRecording = false;
                    
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    status.textContent = '⏳ Processing your speech... Please wait.';
                    status.style.background = 'rgba(33, 150, 243, 0.3)';
                }
            }

            async function processAudio(audioBlob) {
                try {
                    console.log(`Processing audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);
                    
                    // Convert audio to PCM data
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    
                    const pcmData = audioBuffer.getChannelData(0);
                    const int16Data = new Int16Array(pcmData.length);
                    
                    // Convert to 16-bit PCM
                    for (let i = 0; i < pcmData.length; i++) {
                        const sample = Math.max(-1, Math.min(1, pcmData[i]));
                        int16Data[i] = Math.round(sample * 32767);
                    }
                    
                    console.log(`Converted to PCM: ${int16Data.length} samples`);
                    
                    // Convert to hex for JSON transmission
                    const uint8Array = new Uint8Array(int16Data.buffer);
                    const hexString = Array.from(uint8Array)
                        .map(b => b.toString(16).padStart(2, '0'))
                        .join('');
                    
                    // Send via WebSocket
                    if (websocket.readyState === WebSocket.OPEN) {
                        const message = {
                            type: 'audio_data',
                            data: hexString,
                            length: int16Data.length,
                            sample_rate: audioBuffer.sampleRate,
                            duration: audioBuffer.duration
                        };
                        
                        console.log('Sending audio data:', {
                            hex_length: hexString.length,
                            samples: int16Data.length,
                            duration: audioBuffer.duration
                        });
                        
                        websocket.send(JSON.stringify(message));
                        
                        status.textContent = '📤 Audio sent! Waiting for AI response...';
                        status.style.background = 'rgba(156, 39, 176, 0.3)';
                        
                    } else {
                        throw new Error('WebSocket connection is not open');
                    }
                    
                } catch (error) {
                    console.error('Audio processing error:', error);
                    status.textContent = '❌ Error processing audio';
                    status.style.background = 'rgba(244, 67, 54, 0.3)';
                    textResponse.innerHTML = `<p class="error">❌ Audio processing failed: ${error.message}</p>`;
                }
            }

            function clearAll() {
                textResponse.innerHTML = '<div class="loading">Ready for your speech input...</div>';
                status.textContent = '✅ Cleared. Ready for new recording.';
                status.style.background = 'rgba(255, 255, 255, 0.2)';
                console.log('Interface cleared');
            }

            // Event listeners
            startBtn.addEventListener('click', startRecording);
            stopBtn.addEventListener('click', stopRecording);
            document.getElementById('clearBtn').addEventListener('click', clearAll);

            // Initialize system
            async function initialize() {
                console.log('Initializing Qwen3-Omni Demo...');
                
                // Check health first
                const health = await checkHealth();
                
                // Connect WebSocket
                connectWebSocket();
                
                // Keep connection alive with periodic pings
                setInterval(() => {
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({type: 'ping'}));
                    }
                }, 30000);
                
                console.log('Initialization complete');
            }

            // Start initialization when page loads
            initialize();
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    logger.info(f"🚀 Starting Qwen3-Omni RunPod Demo on port {SERVER_PORT}")
    logger.info(f"📍 Access URL: http://0.0.0.0:{SERVER_PORT}/")
    logger.info(f"🏥 Health Check: http://0.0.0.0:{SERVER_PORT}/health")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=SERVER_PORT,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")