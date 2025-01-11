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
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {
                    --primary-color: #3b82f6;
                    --primary-dark: #2563eb;
                    --success-color: #10b981;
                    --warning-color: #f59e0b;
                    --danger-color: #ef4444;
                    --background-color: #f8fafc;
                    --card-background: #ffffff;
                    --text-primary: #1e293b;
                    --text-secondary: #64748b;
                }
                
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body { 
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background: var(--background-color);
                    color: var(--text-primary);
                    line-height: 1.6;
                }
                
                .container {
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 1rem;
                }
                
                .header {
                    background: var(--card-background);
                    border-bottom: 1px solid #e2e8f0;
                    padding: 1.5rem 0;
                    margin-bottom: 2rem;
                }
                
                .header h1 {
                    color: var(--primary-color);
                    font-size: 2rem;
                    font-weight: 700;
                    text-align: center;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.75rem;
                }
                
                .card {
                    background: var(--card-background);
                    border-radius: 1rem;
                    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
                    padding: 2rem;
                    margin-bottom: 2rem;
                    transition: transform 0.2s ease;
                }
                
                .card:hover {
                    transform: translateY(-2px);
                }
                
                h2 {
                    color: var(--text-primary);
                    font-size: 1.5rem;
                    margin-bottom: 1rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                p {
                    color: var(--text-secondary);
                    margin-bottom: 1rem;
                }
                
                textarea {
                    width: 100%;
                    min-height: 200px;
                    padding: 1rem;
                    border: 1px solid #e2e8f0;
                    border-radius: 0.75rem;
                    font-family: inherit;
                    font-size: 1rem;
                    margin: 1rem 0;
                    resize: vertical;
                    transition: border-color 0.2s ease;
                }
                
                textarea:focus {
                    outline: none;
                    border-color: var(--primary-color);
                    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
                }
                
                button {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 0.875rem 1.75rem;
                    border-radius: 0.75rem;
                    font-size: 1rem;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                button:hover {
                    background: var(--primary-dark);
                    transform: translateY(-1px);
                }
                
                button:disabled {
                    background: var(--text-secondary);
                    cursor: not-allowed;
                }
                
                .loading {
                    display: none;
                    text-align: center;
                    padding: 2rem;
                    color: var(--text-secondary);
                }
                
                .loading i {
                    font-size: 2rem;
                    color: var(--primary-color);
                    margin-bottom: 1rem;
                }
                
                #result {
                    display: none;
                }
                
                .result-card {
                    border-radius: 0.75rem;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    border: 1px solid;
                }
                
                .result-card.ai-generated {
                    background: #fef2f2;
                    border-color: var(--danger-color);
                }
                
                .result-card.human-written {
                    background: #f0fdf4;
                    border-color: var(--success-color);
                }
                
                .confidence {
                    display: inline-block;
                    padding: 0.25rem 0.75rem;
                    border-radius: 0.5rem;
                    font-weight: 500;
                    font-size: 0.875rem;
                    margin: 0.5rem 0;
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
                
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1rem;
                    margin-top: 1.5rem;
                }
                
                .metric {
                    background: white;
                    padding: 1.25rem;
                    border-radius: 0.75rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                }
                
                .metric-title {
                    font-size: 0.875rem;
                    color: var(--text-secondary);
                    margin-bottom: 0.5rem;
                    display: flex;
                    align-items: center;
                    gap: 0.25rem;
                }
                
                .metric-value {
                    font-size: 1.25rem;
                    font-weight: 600;
                    color: var(--text-primary);
                }
                
                .metric-info {
                    font-size: 0.75rem;
                    color: var(--text-secondary);
                    margin-top: 0.25rem;
                }
                
                .explanation {
                    margin-top: 1.5rem;
                    white-space: pre-line;
                    line-height: 1.7;
                }
                
                @media (max-width: 768px) {
                    .container {
                        padding: 1rem;
                    }
                    
                    .card {
                        padding: 1.5rem;
                    }
                    
                    .metrics {
                        grid-template-columns: 1fr;
                    }
                    
                    .header h1 {
                        font-size: 1.75rem;
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
                    <p>Paste student text below to check if it might be AI-generated. For best results, provide at least a few paragraphs of text.</p>
                    <textarea id="text" placeholder="Paste text here (minimum 100 characters recommended for accurate analysis)"></textarea>
                    <button onclick="analyze()" id="analyzeBtn">
                        <i class="fas fa-search"></i> Analyze Text
                    </button>
                </div>
                
                <div class="loading" id="loading">
                    <i class="fas fa-circle-notch fa-spin"></i>
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
                            
                            // Get threshold and weight info
                            const thresholdInfo = value.toFixed(3);
                            
                            return `
                                <div class="metric">
                                    <div class="metric-title">
                                        <i class="fas fa-chart-line"></i>
                                        ${formattedKey}
                                    </div>
                                    <div class="metric-value">${thresholdInfo}</div>
                                </div>
                            `;
                        }).join('');
                        
                        // Determine confidence class
                        let confidenceClass = 'low';
                        if (data.confidence_score >= 0.7) confidenceClass = 'high';
                        else if (data.confidence_score >= 0.4) confidenceClass = 'medium';
                        
                        // Determine result class and icon
                        const resultClass = data.is_ai_generated ? 'ai-generated' : 'human-written';
                        const resultIcon = data.is_ai_generated ? 'robot' : 'user';
                        const resultTitle = data.is_ai_generated ? 'AI-Generated Content Detected' : 'Likely Human-Written Content';
                        
                        result.innerHTML = `
                            <div class="card ${resultClass}">
                                <h2>
                                    <i class="fas fa-${resultIcon}"></i>
                                    ${resultTitle}
                                </h2>
                                <div class="confidence ${confidenceClass}">
                                    Confidence: ${(data.confidence_score * 100).toFixed(1)}%
                                </div>
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
    """Analyze text for AI-generated content."""
    if not request.text or len(request.text.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least 100 characters of text for accurate analysis."
        )
    
    # Get analysis results
    is_ai_generated, confidence_score, metrics, explanation = detector.detect(request.text)
    
    return {
        "is_ai_generated": is_ai_generated,
        "confidence_score": confidence_score,
        "confidence_metrics": metrics,
        "explanation": explanation
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}