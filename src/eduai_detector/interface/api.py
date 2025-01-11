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
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {
                    --primary-color: #4f46e5;
                    --primary-light: #818cf8;
                    --primary-dark: #4338ca;
                    --success-color: #10b981;
                    --warning-color: #f59e0b;
                    --danger-color: #ef4444;
                    --background-color: #f8faff;
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
                    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
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
                    background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
                    color: white;
                    padding: 2rem 0;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                }
                
                .header h1 {
                    font-size: 2.5rem;
                    font-weight: 700;
                    text-align: center;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.75rem;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
                }

                .header p {
                    text-align: center;
                    color: rgba(255, 255, 255, 0.9);
                    margin-top: 0.5rem;
                    font-size: 1.1rem;
                }
                
                .card {
                    background: var(--card-background);
                    border-radius: 1.5rem;
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                    padding: 2rem;
                    margin-bottom: 2rem;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }
                
                .card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                }
                
                h2 {
                    color: var(--primary-color);
                    font-size: 1.75rem;
                    margin-bottom: 1rem;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }
                
                p {
                    color: var(--text-secondary);
                    margin-bottom: 1.5rem;
                    font-size: 1.1rem;
                }
                
                textarea {
                    width: 100%;
                    min-height: 200px;
                    padding: 1.25rem;
                    border: 2px solid #e2e8f0;
                    border-radius: 1rem;
                    font-family: inherit;
                    font-size: 1.1rem;
                    margin: 1rem 0;
                    resize: vertical;
                    transition: all 0.3s ease;
                    background: #f8fafc;
                }
                
                textarea:focus {
                    outline: none;
                    border-color: var(--primary-light);
                    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
                    background: white;
                }
                
                button {
                    background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
                    color: white;
                    border: none;
                    padding: 1rem 2rem;
                    border-radius: 1rem;
                    font-size: 1.1rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                }
                
                button:hover {
                    background: linear-gradient(135deg, var(--primary-dark), var(--primary-color));
                    transform: translateY(-2px);
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                }
                
                button:disabled {
                    background: var(--text-secondary);
                    cursor: not-allowed;
                    transform: none;
                }
                
                .loading {
                    display: none;
                    text-align: center;
                    padding: 2rem;
                    color: var(--primary-color);
                }
                
                .loading i {
                    font-size: 3rem;
                    margin-bottom: 1rem;
                    color: var(--primary-color);
                }

                .loading p {
                    color: var(--primary-color);
                    font-weight: 500;
                }
                
                #result {
                    display: none;
                }
                
                .result-card {
                    border-radius: 1rem;
                    padding: 2rem;
                    margin: 1rem 0;
                    border: 2px solid;
                    transition: all 0.3s ease;
                }
                
                .result-card.ai-generated {
                    background: linear-gradient(135deg, #fef2f2, #fee2e2);
                    border-color: var(--danger-color);
                }
                
                .result-card.human-written {
                    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
                    border-color: var(--success-color);
                }
                
                .confidence {
                    display: inline-block;
                    padding: 0.5rem 1rem;
                    border-radius: 0.75rem;
                    font-weight: 600;
                    font-size: 1rem;
                    margin: 1rem 0;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                
                .confidence.high {
                    background: #dcfce7;
                    color: #166534;
                    border: 1px solid #bbf7d0;
                }
                
                .confidence.medium {
                    background: #fef3c7;
                    color: #92400e;
                    border: 1px solid #fde68a;
                }
                
                .confidence.low {
                    background: #fee2e2;
                    color: #991b1b;
                    border: 1px solid #fecaca;
                }
                
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 1.5rem;
                    margin-top: 2rem;
                }
                
                .metric {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                    border: 1px solid rgba(0, 0, 0, 0.05);
                    transition: transform 0.2s ease;
                }

                .metric:hover {
                    transform: translateY(-2px);
                }
                
                .metric-title {
                    font-size: 1rem;
                    color: var(--text-secondary);
                    margin-bottom: 0.75rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-weight: 500;
                }
                
                .metric-value {
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: var(--primary-color);
                }
                
                .explanation {
                    margin-top: 1.5rem;
                    white-space: pre-line;
                    line-height: 1.8;
                    font-size: 1.1rem;
                    color: var(--text-primary);
                }

                .tip {
                    background: linear-gradient(135deg, #ede9fe, #ddd6fe);
                    border-radius: 1rem;
                    padding: 1.5rem;
                    margin-top: 1.5rem;
                    border: 1px solid #c4b5fd;
                }

                .tip i {
                    color: #7c3aed;
                }

                .tip-title {
                    font-weight: 600;
                    color: #5b21b6;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    margin-bottom: 0.5rem;
                }

                .tip-content {
                    color: #6d28d9;
                    font-size: 0.95rem;
                    margin: 0;
                }
                
                @media (max-width: 768px) {
                    .container {
                        padding: 1rem;
                    }
                    
                    .header h1 {
                        font-size: 2rem;
                    }

                    .header p {
                        font-size: 1rem;
                        padding: 0 1rem;
                    }
                    
                    .card {
                        padding: 1.5rem;
                    }
                    
                    .metrics {
                        grid-template-columns: 1fr;
                    }
                    
                    h2 {
                        font-size: 1.5rem;
                    }

                    p {
                        font-size: 1rem;
                    }

                    textarea {
                        font-size: 1rem;
                        padding: 1rem;
                    }

                    button {
                        width: 100%;
                        justify-content: center;
                    }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1><i class="fas fa-graduation-cap"></i> EduAI Detector</h1>
                    <p>Your trusted assistant in maintaining academic integrity</p>
                </div>
            </div>
            
            <div class="container">
                <div class="card">
                    <h2><i class="fas fa-magnifying-glass"></i> Analyze Student Text</h2>
                    <p>Paste your student's text below to check if it might be AI-generated. Our tool analyzes writing patterns and provides detailed insights.</p>
                    <textarea id="text" placeholder="Paste student text here (minimum 100 characters recommended for accurate analysis)"></textarea>
                    <div class="tip">
                        <div class="tip-title"><i class="fas fa-lightbulb"></i> Teacher's Tip</div>
                        <p class="tip-content">For best results, provide at least a full paragraph of text. The more text you provide, the more accurate the analysis will be.</p>
                    </div>
                    <button onclick="analyze()" id="analyzeBtn">
                        <i class="fas fa-search"></i> Analyze Text
                    </button>
                </div>
                
                <div class="loading" id="loading">
                    <i class="fas fa-circle-notch fa-spin"></i>
                    <p>Analyzing text patterns...</p>
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
                            throw new Error('Analysis failed. Please try again with a longer text sample.');
                        }
                        
                        const data = await response.json();
                        
                        // Format metrics for display
                        const formattedMetrics = Object.entries(data.confidence_metrics).map(([key, value]) => {
                            const formattedKey = key.replace(/_/g, ' ').split(' ').map(word => 
                                word.charAt(0).toUpperCase() + word.slice(1)
                            ).join(' ');
                            
                            // Get icon based on metric
                            let icon = 'chart-line';
                            if (key.includes('repetition')) icon = 'repeat';
                            if (key.includes('entropy')) icon = 'random';
                            if (key.includes('complexity')) icon = 'layer-group';
                            if (key.includes('vocabulary')) icon = 'book';
                            if (key.includes('sentence')) icon = 'align-left';
                            if (key.includes('readability')) icon = 'glasses';
                            
                            return `
                                <div class="metric">
                                    <div class="metric-title">
                                        <i class="fas fa-${icon}"></i>
                                        ${formattedKey}
                                    </div>
                                    <div class="metric-value">${value.toFixed(3)}</div>
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
                                    <i class="fas fa-percentage"></i>
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
                                <div style="color: var(--danger-color); display: flex; align-items: center; gap: 0.5rem;">
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