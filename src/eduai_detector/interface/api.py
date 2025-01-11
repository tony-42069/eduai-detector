# src/eduai_detector/interface/api.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict
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
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {
                    --primary-color: #2563eb;
                    --secondary-color: #1e40af;
                    --success-color: #059669;
                    --danger-color: #dc2626;
                    --background-color: #f3f4f6;
                }
                
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: var(--background-color);
                    color: #1f2937;
                    line-height: 1.5;
                }
                
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 2rem 1rem;
                }
                
                .header {
                    background: white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    padding: 1rem 0;
                    margin-bottom: 2rem;
                }
                
                .header h1 {
                    margin: 0;
                    color: var(--primary-color);
                    font-size: 1.8rem;
                    text-align: center;
                }
                
                .card {
                    background: white;
                    border-radius: 0.5rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                }
                
                textarea {
                    width: 100%;
                    min-height: 200px;
                    padding: 1rem;
                    border: 1px solid #e5e7eb;
                    border-radius: 0.375rem;
                    font-family: inherit;
                    font-size: 1rem;
                    margin: 1rem 0;
                    resize: vertical;
                }
                
                button {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 0.375rem;
                    font-size: 1rem;
                    cursor: pointer;
                    transition: background-color 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                button:hover {
                    background: var(--secondary-color);
                }
                
                button:disabled {
                    background: #9ca3af;
                    cursor: not-allowed;
                }
                
                #result {
                    display: none;
                }
                
                .result-card {
                    border-left: 4px solid;
                    padding: 1rem;
                    margin: 1rem 0;
                }
                
                .ai-generated {
                    border-color: var(--danger-color);
                    background: #fee2e2;
                }
                
                .human-written {
                    border-color: var(--success-color);
                    background: #d1fae5;
                }
                
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                    margin-top: 1rem;
                }
                
                .metric {
                    background: white;
                    padding: 1rem;
                    border-radius: 0.375rem;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                }
                
                .metric-title {
                    font-size: 0.875rem;
                    color: #4b5563;
                    margin-bottom: 0.5rem;
                }
                
                .metric-value {
                    font-size: 1.25rem;
                    font-weight: 600;
                }
                
                .loading {
                    display: none;
                    text-align: center;
                    padding: 1rem;
                }
                
                .loading i {
                    color: var(--primary-color);
                    font-size: 2rem;
                }
                
                .explanation {
                    margin-top: 1rem;
                    white-space: pre-line;
                }
                
                .confidence {
                    display: inline-block;
                    padding: 0.25rem 0.5rem;
                    border-radius: 0.25rem;
                    font-weight: 500;
                    font-size: 0.875rem;
                    margin-left: 0.5rem;
                }
                
                .confidence.high {
                    background: #dcfce7;
                    color: #166534;
                }
                
                .confidence.medium {
                    background: #fef3c7;
                    color: #92400e;
                }
                
                .confidence.low {
                    background: #fee2e2;
                    color: #991b1b;
                }
                
                @media (max-width: 640px) {
                    .container {
                        padding: 1rem;
                    }
                    
                    .metrics {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1><i class="fas fa-robot"></i> EduAI Detector</h1>
                </div>
            </div>
            
            <div class="container">
                <div class="card">
                    <h2><i class="fas fa-magnifying-glass"></i> Analyze Text</h2>
                    <p>Paste student text below to check if it might be AI-generated:</p>
                    <textarea id="text" placeholder="Paste text here (minimum 100 characters recommended for accurate analysis)"></textarea>
                    <button onclick="analyze()" id="analyzeBtn">
                        <i class="fas fa-search"></i> Analyze Text
                    </button>
                </div>
                
                <div class="loading" id="loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Analyzing text...</p>
                </div>
                
                <div id="result"></div>
            </div>

            <script>
                async function analyze() {
                    const text = document.getElementById('text').value;
                    const result = document.getElementById('result');
                    const loading = document.getElementById('loading');
                    const analyzeBtn = document.getElementById('analyzeBtn');
                    
                    if (!text.trim()) {
                        alert('Please enter some text to analyze.');
                        return;
                    }
                    
                    // Show loading, hide result
                    loading.style.display = 'block';
                    result.style.display = 'none';
                    analyzeBtn.disabled = true;
                    
                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ text })
                        });
                        
                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(errorData.detail || 'Analysis failed. Please try again.');
                        }
                        
                        const data = await response.json();
                        
                        // Format metrics for display
                        const formattedMetrics = Object.entries(data.confidence_metrics).map(([key, value]) => {
                            const formattedKey = key.replace(/_/g, ' ').split(' ').map(word => 
                                word.charAt(0).toUpperCase() + word.slice(1)
                            ).join(' ');
                            return `
                                <div class="metric">
                                    <div class="metric-title">${formattedKey}</div>
                                    <div class="metric-value">${value.toFixed(3)}</div>
                                </div>
                            `;
                        }).join('');
                        
                        // Determine result class and icon
                        const resultClass = data.is_ai_generated ? 'ai-generated' : 'human-written';
                        const resultIcon = data.is_ai_generated ? 'robot' : 'user';
                        
                        result.innerHTML = `
                            <div class="card ${resultClass}">
                                <h3>
                                    <i class="fas fa-${resultIcon}"></i>
                                    Analysis Result
                                </h3>
                                <p class="explanation">${data.explanation}</p>
                                <div class="metrics">
                                    ${formattedMetrics}
                                </div>
                            </div>
                        `;
                        
                        result.style.display = 'block';
                    } catch (error) {
                        result.innerHTML = `
                            <div class="card">
                                <div class="error">
                                    <i class="fas fa-exclamation-circle"></i>
                                    ${error.message}
                                </div>
                            </div>
                        `;
                        result.style.display = 'block';
                    } finally {
                        loading.style.display = 'none';
                        analyzeBtn.disabled = false;
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
        print(f"Received analysis request with text length: {len(request.text)}")
        is_ai, metrics, explanation, confidence_score = detector.detect(request.text)
        
        response_data = {
            "is_ai_generated": is_ai,
            "confidence_score": confidence_score,
            "confidence_metrics": metrics,
            "explanation": explanation
        }
        
        print(f"Analysis completed successfully. AI Generated: {is_ai} (Confidence: {confidence_score:.2%})")
        return response_data
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}