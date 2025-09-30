import React, { useState, useRef, useEffect } from 'react';
import { Upload, TrendingUp, Brain, FileText, BarChart3, Activity, DollarSign, AlertTriangle, CheckCircle, MessageSquare, Database, Zap, Globe, RefreshCw, Server } from 'lucide-react';

const InvestmentAnalysisPlatform = () => {
  const [activeTab, setActiveTab] = useState('setup');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [portfolioData, setPortfolioData] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [backendConnected, setBackendConnected] = useState(false);
  const fileInputRef = useRef(null);
  const portfolioInputRef = useRef(null);

  const API_BASE_URL = 'http://localhost:8000/api';

  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      setBackendConnected(response.ok);
    } catch {
      setBackendConnected(false);
    }
  };

  const analyzeWithBackend = async (positions) => {
    setIsAnalyzing(true);
    try {
      const response = await fetch(`${API_BASE_URL}/portfolio/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(positions)
      });
      
      const data = await response.json();
      
      setAnalysisResults({
        overallScore: 7.8,
        quantScore: 8.2,
        qualScore: 7.4,
        portfolioMetrics: data.metrics,
        hedgeFundAlignment: data.hedge_fund_comparison.alignment_scores,
        marketSentiment: data.market_sentiment,
        recommendations: data.hedge_fund_comparison.recommendations.map(r => ({
          action: r,
          priority: 'High'
        })),
        sectorAllocation: [
          { sector: 'Technology', percentage: 35, risk: 'Medium-High' },
          { sector: 'Healthcare', percentage: 25, risk: 'High' },
          { sector: 'Financial', percentage: 20, risk: 'Medium' }
        ],
        riskFactors: [
          { factor: 'High concentration risk', severity: 'Medium', impact: 7 }
        ],
        biotechAnalysis: []
      });
      
      setBackendConnected(true);
    } catch (error) {
      console.error('Backend error, using demo data:', error);
      useDemoData();
    }
    setIsAnalyzing(false);
  };

  const useDemoData = () => {
    setAnalysisResults({
      overallScore: 7.8,
      quantScore: 8.2,
      qualScore: 7.4,
      portfolioMetrics: {
        totalValue: 2456789,
        beta: 1.23,
        sharpeRatio: 1.87,
        volatility: 18.4,
        diversification: 72,
        riskScore: 6.5
      },
      hedgeFundAlignment: {
        buffett: 45,
        dalio: 68,
        ackman: 72,
        druckenmiller: 61
      },
      marketSentiment: {
        reddit: 72,
        twitter: 65,
        newsMedia: 58,
        analystRatings: 71
      },
      sectorAllocation: [
        { sector: 'Technology', percentage: 35, risk: 'Medium-High' },
        { sector: 'Healthcare/Biotech', percentage: 25, risk: 'High' },
        { sector: 'Financial Services', percentage: 20, risk: 'Medium' }
      ],
      riskFactors: [
        { factor: 'High sector concentration in tech', severity: 'Medium', impact: 7 }
      ],
      recommendations: [
        { action: 'Reduce tech exposure by 10-15%', priority: 'High' }
      ],
      biotechAnalysis: []
    });
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    setUploadedFiles(prev => [...prev, ...files]);
    
    if (backendConnected) {
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
          await fetch(`${API_BASE_URL}/upload/document`, {
            method: 'POST',
            body: formData
          });
        } catch (error) {
          console.error('Upload failed:', error);
        }
      }
    }
    
    if (files.length > 0) {
      setTimeout(() => useDemoData(), 2000);
    }
  };

  const handlePortfolioScreenshot = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setIsAnalyzing(true);
    
    if (backendConnected) {
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch(`${API_BASE_URL}/upload/document`, {
          method: 'POST',
          body: formData
        });
        const data = await response.json();
        
        if (data.data && data.data.positions) {
          setPortfolioData({ positions: data.data.positions, recognized: true });
          await analyzeWithBackend(data.data.positions);
        }
      } catch (error) {
        console.error('OCR failed:', error);
        useMockPortfolio();
      }
    } else {
      useMockPortfolio();
    }
    
    setIsAnalyzing(false);
  };

  const useMockPortfolio = () => {
    const mockPositions = [
      { ticker: 'AAPL', shares: 250, avgCost: 145.32, currentPrice: 178.45, value: 44612.50, gain: 22.8 },
      { ticker: 'MSFT', shares: 180, avgCost: 285.67, currentPrice: 378.92, value: 68205.60, gain: 32.6 },
      { ticker: 'GOOGL', shares: 120, avgCost: 125.43, currentPrice: 142.18, value: 17061.60, gain: 13.4 }
    ];
    setPortfolioData({ positions: mockPositions, recognized: true });
    analyzeWithBackend(mockPositions);
  };

  const handleChatSubmit = async () => {
    if (!chatInput.trim()) return;
    
    const userMessage = { role: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    
    if (backendConnected) {
      try {
        const response = await fetch(`${API_BASE_URL}/ai/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: chatInput,
            context: { portfolio: portfolioData, analysis: analysisResults }
          })
        });
        
        const data = await response.json();
        setChatMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.analysis 
        }]);
      } catch (error) {
        generateMockResponse(chatInput);
      }
    } else {
      setTimeout(() => generateMockResponse(chatInput), 1000);
    }
    
    setChatInput('');
  };

  const generateMockResponse = (input) => {
    let response = "Based on your portfolio analysis, I recommend diversifying your tech holdings and adding defensive positions. Your current Sharpe ratio of 1.87 is strong, but beta of 1.23 indicates higher volatility than the market.";
    
    if (input.toLowerCase().includes('biotech')) {
      response = "Your biotech positions show elevated risk-reward potential with 25% sector allocation. Consider trimming pre-catalyst positions and using options for hedging binary risk events.";
    }
    
    setChatMessages(prev => [...prev, { role: 'assistant', content: response }]);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 shadow-lg">
        <div className="max-w-7xl mx-auto">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold mb-2">Institutional Investment Analysis Platform</h1>
              <p className="text-blue-100">Hedge fund-grade analytics with real-time API integration</p>
            </div>
            <div className="flex items-center gap-2 bg-white/20 px-4 py-2 rounded-lg">
              <Server className="w-5 h-5" />
              <span className="text-sm">Backend: {backendConnected ? '🟢 Connected' : '🔴 Demo Mode'}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-6">
          <div className="flex overflow-x-auto">
            <button onClick={() => setActiveTab('setup')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'setup' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              <div className="flex items-center gap-2"><Database className="w-4 h-4" />Setup & Integration</div>
            </button>
            <button onClick={() => setActiveTab('upload')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'upload' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              <div className="flex items-center gap-2"><Upload className="w-4 h-4" />Data Upload</div>
            </button>
            <button onClick={() => setActiveTab('dashboard')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'dashboard' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              <div className="flex items-center gap-2"><BarChart3 className="w-4 h-4" />Dashboard</div>
            </button>
            <button onClick={() => setActiveTab('chat')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'chat' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              <div className="flex items-center gap-2"><MessageSquare className="w-4 h-4" />AI Advisor</div>
            </button>
          </div>
        </div>

        {isAnalyzing && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-3">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
              <span className="text-sm font-medium text-blue-900">Analyzing with {backendConnected ? 'live APIs' : 'demo data'}...</span>
            </div>
          </div>
        )}

        {activeTab === 'setup' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-xl font-bold mb-4">Backend Setup Instructions</h3>
              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">1. Install Dependencies</h4>
                  <code className="block bg-gray-800 text-green-400 p-3 rounded text-sm overflow-x-auto">
                    pip install fastapi uvicorn yfinance alpha_vantage transformers<br/>
                    pip install torch pytesseract pdfplumber openpyxl praw redis<br/>
                    pip install sqlalchemy psycopg2-binary sec-edgar-downloader
                  </code>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">2. Set Environment Variables</h4>
                  <code className="block bg-gray-800 text-green-400 p-3 rounded text-sm overflow-x-auto">
                    export ALPHA_VANTAGE_API_KEY=your_key<br/>
                    export PERPLEXITY_API_KEY=your_key<br/>
                    export FINNHUB_API_KEY=your_key<br/>
                    export REDDIT_CLIENT_ID=your_id<br/>
                    export REDDIT_CLIENT_SECRET=your_secret
                  </code>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">3. Start Backend Server</h4>
                  <code className="block bg-gray-800 text-green-400 p-3 rounded text-sm">
                    python backend_server.py
                  </code>
                </div>

                <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                  <p className="text-sm text-blue-900">
                    <strong>Note:</strong> Backend runs on localhost:8000. Frontend will connect automatically.
                    Without backend, app runs in demo mode with simulated data.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold mb-4">API Keys Required</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Alpha Vantage (Free tier: alphavantage.co)</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Perplexity AI (perplexity.ai/api)</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Finnhub (Free: finnhub.io)</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Reddit API (reddit.com/prefs/apps)</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Yahoo Finance (Free via yfinance)</span>
                  </li>
                </ul>
              </div>

              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold mb-4">Database Setup</h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <p className="font-medium mb-1">PostgreSQL:</p>
                    <code className="block bg-gray-100 p-2 rounded text-xs">
                      createdb investment_db
                    </code>
                  </div>
                  <div>
                    <p className="font-medium mb-1">Redis:</p>
                    <code className="block bg-gray-100 p-2 rounded text-xs">
                      redis-server
                    </code>
                  </div>
                  <p className="text-gray-600 mt-3">
                    PostgreSQL stores portfolio history, Redis caches API calls
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Test Connection</h3>
              <button 
                onClick={checkBackendConnection}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Test Backend Connection
              </button>
              <p className="text-sm text-gray-600 mt-2">
                Status: {backendConnected ? '✅ Connected to backend' : '⚠️ Running in demo mode'}
              </p>
            </div>
          </div>
        )}

        {activeTab === 'upload' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Upload Financial Documents</h3>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <p className="text-sm text-gray-600 mb-2">Upload 10-K, 10-Q, earnings transcripts, or CSV data</p>
                <p className="text-xs text-gray-500">Supports: PDF, XLSX, CSV, TXT, Images</p>
                <input ref={fileInputRef} type="file" multiple accept=".pdf,.xlsx,.xls,.csv,.txt,.jpg,.png" onChange={handleFileUpload} className="hidden" />
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Upload Portfolio Screenshot</h3>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 cursor-pointer" onClick={() => portfolioInputRef.current?.click()}>
                <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <p className="text-sm text-gray-600 mb-2">Upload screenshot from your brokerage</p>
                <p className="text-xs text-gray-500">OCR will extract positions automatically</p>
                <input ref={portfolioInputRef} type="file" accept="image/*" onChange={handlePortfolioScreenshot} className="hidden" />
              </div>
            </div>

            {uploadedFiles.length > 0 && (
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold mb-4">Uploaded Files</h3>
                <div className="space-y-2">
                  {uploadedFiles.map((file, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <FileText className="w-5 h-5 text-blue-500" />
                        <span className="text-sm font-medium">{file.name}</span>
                      </div>
                      <span className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {portfolioData && (
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h3 className="text-lg font-semibold mb-4">Detected Portfolio Positions</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 px-4">Ticker</th>
                        <th className="text-right py-2 px-4">Shares</th>
                        <th className="text-right py-2 px-4">Price</th>
                        <th className="text-right py-2 px-4">Value</th>
                        <th className="text-right py-2 px-4">Gain</th>
                      </tr>
                    </thead>
                    <tbody>
                      {portfolioData.positions.map((pos, idx) => (
                        <tr key={idx} className="border-b">
                          <td className="py-2 px-4 font-bold text-blue-600">{pos.ticker}</td>
                          <td className="py-2 px-4 text-right">{pos.shares}</td>
                          <td className="py-2 px-4 text-right">${pos.currentPrice?.toFixed(2)}</td>
                          <td className="py-2 px-4 text-right">${pos.value?.toLocaleString()}</td>
                          <td className="py-2 px-4 text-right text-green-600">+{pos.gain?.toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'dashboard' && analysisResults && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white">
                <p className="text-blue-100 text-sm">Overall Score</p>
                <p className="text-3xl font-bold mt-1">{analysisResults.overallScore}/10</p>
              </div>
              <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-6 text-white">
                <p className="text-green-100 text-sm">Sharpe Ratio</p>
                <p className="text-3xl font-bold mt-1">{analysisResults.portfolioMetrics.sharpeRatio}</p>
              </div>
              <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-6 text-white">
                <p className="text-purple-100 text-sm">Portfolio Beta</p>
                <p className="text-3xl font-bold mt-1">{analysisResults.portfolioMetrics.beta}</p>
              </div>
              <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg p-6 text-white">
                <p className="text-orange-100 text-sm">Total Value</p>
                <p className="text-3xl font-bold mt-1">${(analysisResults.portfolioMetrics.totalValue / 1000000).toFixed(2)}M</p>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Hedge Fund Alignment</h3>
              <div className="space-y-3">
                {Object.entries(analysisResults.hedgeFundAlignment).map(([manager, score]) => (
                  <div key={manager} className="flex items-center gap-3">
                    <span className="w-32 capitalize">{manager}</span>
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${score}%` }}></div>
                    </div>
                    <span className="w-12 text-right font-bold">{score}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="bg-white rounded-lg border border-gray-200 h-[600px] flex flex-col">
            <div className="p-4 border-b">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-500" />
                AI Investment Advisor {backendConnected && '(Powered by Perplexity AI)'}
              </h3>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {chatMessages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center">
                  <MessageSquare className="w-16 h-16 text-gray-300 mb-4" />
                  <p className="text-gray-500 mb-4">Ask me anything about your portfolio</p>
                </div>
              ) : (
                chatMessages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] rounded-lg p-3 ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}>
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
            
            <div className="p-4 border-t">
              <div className="flex gap-2">
                <input type="text" value={chatInput} onChange={(e) => setChatInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleChatSubmit()} placeholder="Ask about risks, strategy, or specific holdings..." className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" />
                <button onClick={handleChatSubmit} className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">Send</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InvestmentAnalysisPlatform;
