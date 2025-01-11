# src/eduai_detector/interface/api.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
from ..core.detector import AITextDetector

app = FastAPI(
    title="EduAI Detector",
    description="AI text detection for education"
)

class TextRequest(BaseModel):
    text: str

detector = AITextDetector()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>EduAI Detector</title>
            <style>
                body { 
                    font-family: Arial; 
                    max-width: 800px; 
                    margin: 40px auto; 
                    padding: 20px;
                }
                textarea { 
                    width: 100%; 
                    height: 200px; 
                    margin: 10px 0;
                    padding: 10px;
                    font-family: Arial;
                }
                button {
                    padding: 10px 20px;
                    background: #4CAF50;
                    color: white;
                    border: none;
                    cursor: pointer;
                    font-size: 16px;
                }
                #result {
                    margin-top: 20px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    display: none;
                }
                .error {
                    color: #d32f2f;
                    background: #ffebee;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <h1>EduAI Detector</h1>
            <p>Paste student text below to check if it might be AI-generated:</p>
            
            <textarea id="text" placeholder="Paste text here..."></textarea>
            <button onclick="analyze()">Analyze Text</button>
            
            <div id="result"></div>

            <script>
                async function analyze() {
                    const text = document.getElementById('text').value;
                    const result = document.getElementById('result');
                    
                    if (!text.trim()) {
                        result.innerHTML = '<div class="error">Please enter some text to analyze.</div>';
                        result.style.display = 'block';
                        return;
                    }
                    
                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ text })
                        });
                        
                        if (!response.ok) {
                            throw new Error('Analysis failed. Please try again.');
                        }
                        
                        const data = await response.json();
                        
                        result.innerHTML = `
                            <h3>Analysis Result:</h3>
                            <p><strong>AI Generated:</strong> ${data.is_ai_generated ? 'Yes' : 'No'}</p>
                            <p><strong>Explanation:</strong> ${data.explanation}</p>
                            <h4>Confidence Metrics:</h4>
                            <pre>${JSON.stringify(data.confidence_metrics, null, 2)}</pre>
                        `;
                        result.style.display = 'block';
                    } catch (error) {
                        result.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                        result.style.display = 'block';
                    }
                }
            </script>
        </body>
    </html>
    """

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        is_ai, metrics, explanation = detector.detect(request.text)
        return {
            "is_ai_generated": is_ai,
            "confidence_metrics": metrics,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}