"""
Institutional-Grade Investment Analysis Backend
File: backend_server.py

Save this file and run: python backend_server.py
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import json
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Document Processing
try:
    import PyPDF2
    import openpyxl
    from PIL import Image
    import pytesseract
    import pdfplumber
except ImportError:
    print("Install: pip install PyPDF2 openpyxl pillow pytesseract pdfplumber")

# Financial Data APIs
try:
    import yfinance as yf
    import requests
except ImportError:
    print("Install: pip install yfinance requests")

# NLP & Sentiment Analysis
try:
    from transformers import pipeline
    from textblob import TextBlob
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Install: pip install transformers torch textblob nltk")

# Database
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    import redis
except ImportError:
    print("Install: pip install sqlalchemy redis psycopg2-binary")

app = FastAPI(title="Investment Analysis API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./investment.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

config = Config()

# Database Setup
Base = declarative_base()

try:
    engine = create_engine(config.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
except:
    print("Warning: Database connection failed. Some features may not work.")
    redis_client = None

# Database Models
class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    positions = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

try:
    Base.metadata.create_all(engine)
except:
    pass

# Initialize NLP Models (optional)
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except:
    sentiment_analyzer = None
    sia = None
    print("Warning: NLP models not loaded. Sentiment analysis will use fallback.")

# ==================== DOCUMENT PROCESSING ====================

class DocumentProcessor:
    @staticmethod
    async def process_pdf(file_path: str) -> Dict:
        """Extract text from PDF"""
        content = {"text": "", "tables": []}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    content["text"] += page.extract_text() + "\n"
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            content["tables"].append(table)
        except Exception as e:
            content["error"] = str(e)
        
        return content
    
    @staticmethod
    async def process_excel(file_path: str) -> Dict:
        """Process Excel files"""
        try:
            df = pd.read_excel(file_path)
            return {"data": df.to_dict('records')}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    async def process_image_ocr(file_path: str) -> Dict:
        """OCR for portfolio screenshots"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            positions = DocumentProcessor._parse_portfolio_text(text)
            return {"raw_text": text, "positions": positions}
        except Exception as e:
            return {"error": str(e), "positions": []}
    
    @staticmethod
    def _parse_portfolio_text(text: str) -> List[Dict]:
        import re
        positions = []
        pattern = r'([A-Z]{1,5})\s+(\d+(?:,\d{3})*(?:\.\d{2})?)\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            ticker, shares, price = match.groups()
            positions.append({
                "ticker": ticker,
                "shares": float(shares.replace(',', '')),
                "price": float(price.replace(',', ''))
            })
        
        return positions

# ==================== FINANCIAL DATA APIS ====================

class FinancialDataService:
    @staticmethod
    async def get_real_time_data(ticker: str) -> Dict:
        """Get real-time stock data"""
        cache_key = f"realtime:{ticker}"
        
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except:
                pass
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            result = {
                "ticker": ticker,
                "current_price": info.get('currentPrice', 0),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "beta": info.get('beta', 1.0),
                "52_week_high": info.get('fiftyTwoWeekHigh', 0),
                "52_week_low": info.get('fiftyTwoWeekLow', 0),
            }
            
            if redis_client:
                try:
                    redis_client.setex(cache_key, 300, json.dumps(result))
                except:
                    pass
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ==================== PORTFOLIO ANALYTICS ====================

class PortfolioAnalyzer:
    @staticmethod
    async def calculate_portfolio_metrics(positions: List[Dict]) -> Dict:
        """Calculate beta, Sharpe ratio, etc."""
        
        market = yf.Ticker("^GSPC")
        market_hist = market.history(period="1y")
        market_returns = market_hist['Close'].pct_change().dropna()
        
        portfolio_value = 0
        portfolio_returns = []
        position_details = []
        
        for position in positions:
            ticker = position.get('ticker', '')
            shares = position.get('shares', 0)
            
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                
                if len(hist) == 0:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                position_value = shares * current_price
                portfolio_value += position_value
                
                returns = hist['Close'].pct_change().dropna()
                common_dates = returns.index.intersection(market_returns.index)
                aligned_returns = returns.loc[common_dates]
                aligned_market = market_returns.loc[common_dates]
                
                if len(aligned_returns) > 30:
                    covariance = np.cov(aligned_returns, aligned_market)[0][1]
                    market_variance = np.var(aligned_market)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    beta = 1.0
                
                position_details.append({
                    "ticker": ticker,
                    "shares": shares,
                    "current_price": float(current_price),
                    "position_value": float(position_value),
                    "beta": float(beta)
                })
                
                portfolio_returns.append(aligned_returns * position_value)
            except:
                continue
        
        if portfolio_returns and portfolio_value > 0:
            total_returns = sum(portfolio_returns) / portfolio_value
            
            common_dates = total_returns.index.intersection(market_returns.index)
            aligned_portfolio = total_returns.loc[common_dates]
            aligned_market = market_returns.loc[common_dates]
            
            portfolio_beta = np.cov(aligned_portfolio, aligned_market)[0][1] / np.var(aligned_market)
            
            risk_free_rate = 0.04 / 252
            excess_returns = aligned_portfolio - risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            volatility = np.std(aligned_portfolio) * np.sqrt(252) * 100
        else:
            portfolio_beta = 1.0
            sharpe_ratio = 0.0
            volatility = 0.0
        
        return {
            "totalValue": float(portfolio_value),
            "beta": float(portfolio_beta),
            "sharpeRatio": float(sharpe_ratio),
            "volatility": float(volatility),
            "positions": position_details,
            "calculated_at": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    async def compare_to_hedge_funds(metrics: Dict) -> Dict:
        """Compare to hedge fund strategies"""
        
        philosophies = {
            "buffett": {"ideal_beta": 0.8},
            "dalio": {"ideal_beta": 0.6},
            "ackman": {"ideal_beta": 1.2},
            "druckenmiller": {"ideal_beta": 1.0}
        }
        
        portfolio_beta = metrics.get('beta', 1.0)
        scores = {}
        
        for manager, criteria in philosophies.items():
            score = 100
            beta_diff = abs(portfolio_beta - criteria['ideal_beta'])
            score -= min(beta_diff * 30, 40)
            scores[manager] = max(score, 0)
        
        recommendations = []
        if portfolio_beta > 1.3:
            recommendations.append("High beta - consider defensive positions")
        if metrics.get('sharpeRatio', 0) < 1.0:
            recommendations.append("Low Sharpe ratio - review underperforming positions")
        
        return {
            "alignment_scores": scores,
            "recommendations": recommendations
        }

# ==================== API ENDPOINTS ====================

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents"""
    
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    if file.filename.endswith('.pdf'):
        result = await DocumentProcessor.process_pdf(file_path)
    elif file.filename.endswith(('.xlsx', '.xls')):
        result = await DocumentProcessor.process_excel(file_path)
    elif file.filename.endswith(('.jpg', '.png', '.jpeg')):
        result = await DocumentProcessor.process_image_ocr(file_path)
    else:
        result = {"error": "Unsupported file type"}
    
    try:
        os.remove(file_path)
    except:
        pass
    
    return {"status": "success", "data": result}

@app.post("/api/portfolio/analyze")
async def analyze_portfolio(positions: List[Dict]):
    """Analyze portfolio"""
    
    metrics = await PortfolioAnalyzer.calculate_portfolio_metrics(positions)
    comparison = await PortfolioAnalyzer.compare_to_hedge_funds(metrics)
    
    return {
        "metrics": metrics,
        "hedge_fund_comparison": comparison,
        "market_sentiment": {},
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/stock/{ticker}/realtime")
async def get_realtime_data(ticker: str):
    """Get real-time stock data"""
    return await FinancialDataService.get_real_time_data(ticker)

@app.post("/api/ai/chat")
async def ai_chat(message: str, context: Dict):
    """AI chat endpoint"""
    
    response = f"Analysis: {message[:100]}... [Perplexity AI integration required]"
    
    return {
        "analysis": response,
        "sources": [],
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Investment Analysis Backend...")
    print("API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
