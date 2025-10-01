import React, { useState, useRef, useEffect } from 'react';
import { Upload, TrendingUp, BarChart3, Activity, CheckCircle, Database, Award, Target, Shield, LineChart, PieChart, Users, Search, Leaf, AlertTriangle, DollarSign, Globe, Server, MessageSquare, Bell } from 'lucide-react';

const WhartonCompetitionPlatform = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [portfolioData, setPortfolioData] = useState(null);
  const [quantMetrics, setQuantMetrics] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [esgScores, setEsgScores] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [insiderTrades, setInsiderTrades] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Initialize competition scores on mount
    const timer = setTimeout(() => {
      // Simulated initialization
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  const handlePortfolioUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsAnalyzing(true);
    
    setTimeout(() => {
      const mockPortfolio = {
        positions: [
          { ticker: 'NVDA', shares: 150, costBasis: 412.50, currentPrice: 487.23, value: 73084.50, weight: 8.5 },
          { ticker: 'META', shares: 200, costBasis: 325.40, currentPrice: 378.92, value: 75784.00, weight: 8.8 },
          { ticker: 'GOOGL', shares: 400, costBasis: 125.30, currentPrice: 142.18, value: 56872.00, weight: 6.6 },
          { ticker: 'MSFT', shares: 180, costBasis: 285.67, currentPrice: 378.92, value: 68205.60, weight: 7.9 },
          { ticker: 'AAPL', shares: 250, costBasis: 145.32, currentPrice: 178.45, value: 44612.50, weight: 5.2 }
        ],
        totalValue: 318353.60
      };

      setPortfolioData(mockPortfolio);
      calculateQuantMetrics(mockPortfolio);
    }, 1500);
  };

  const calculateQuantMetrics = (portfolio) => {
    const metrics = {
      riskMetrics: {
        portfolioBeta: 1.23,
        portfolioVolatility: 18.4,
        sharpeRatio: 1.87,
        sortinoRatio: 2.34,
        maxDrawdown: -12.3,
        var95: -8.2,
        cvar95: -11.4
      },
      performanceMetrics: {
        ytdReturn: 24.7,
        oneYearReturn: 32.4,
        alpha: 6.8,
        informationRatio: 1.34
      },
      diversificationMetrics: {
        concentrationScore: 72,
        effectiveNumStocks: 11.2,
        sectorDiversification: 68
      }
    };

    setQuantMetrics(metrics);
    setIsAnalyzing(false);
  };

  const loadESGData = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setEsgScores({
        portfolioESGScore: 72,
        environmental: 68,
        social: 74,
        governance: 75,
        holdings: [
          { ticker: 'NVDA', esgScore: 76, environmental: 72, social: 78, governance: 78 },
          { ticker: 'META', esgScore: 68, environmental: 64, social: 70, governance: 70 },
          { ticker: 'GOOGL', esgScore: 74, environmental: 70, social: 76, governance: 76 }
        ]
      });
      setIsAnalyzing(false);
      setActiveTab('esg');
    }, 1000);
  };

  const loadSentimentData = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setSentimentData({
        overallSentiment: 72,
        sources: {
          reddit: { score: 78, volume: 15420 },
          twitter: { score: 68, volume: 42150 },
          news: { score: 71, volume: 324 }
        },
        topStocks: [
          { ticker: 'NVDA', sentiment: 85, volume: 4200 },
          { ticker: 'META', sentiment: 72, volume: 2800 }
        ]
      });
      setIsAnalyzing(false);
      setActiveTab('sentiment');
    }, 1000);
  };

  const loadInsiderData = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setInsiderTrades([
        { ticker: 'NVDA', insider: 'Jensen Huang (CEO)', action: 'Buy', shares: 50000, value: 24350000, date: '2025-09-15' },
        { ticker: 'META', insider: 'Mark Zuckerberg (CEO)', action: 'Buy', shares: 35000, value: 13250000, date: '2025-09-20' }
      ]);
      setIsAnalyzing(false);
      setActiveTab('insider');
    }, 1000);
  };

  const runBacktest = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setBacktestResults({
        performanceMetrics: {
          totalReturn: 24.7,
          sharpeRatio: 1.87,
          alpha: 6.8,
          maxDrawdown: -12.3
        },
        monteCarloSimulation: {
          meanOutcome: 124500,
          probabilityProfit: 78.5,
          worstCase: 87400,
          bestCase: 187600
        }
      });
      setIsAnalyzing(false);
      setActiveTab('backtest');
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 shadow-lg">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold mb-2">Wharton Investment Competition Platform</h1>
            <p className="text-blue-100">Institutional-Grade Portfolio Management</p>
          </div>
          <div className="flex items-center gap-2 bg-white/20 px-4 py-2 rounded-lg">
            <Server className="w-5 h-5" />
            <span className="text-sm">Demo Mode</span>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-6">
          <div className="flex overflow-x-auto">
            <button onClick={() => setActiveTab('dashboard')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'dashboard' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              Dashboard
            </button>
            <button onClick={() => setActiveTab('quant')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'quant' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              Quant Metrics
            </button>
            <button onClick={() => setActiveTab('esg')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'esg' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              ESG
            </button>
            <button onClick={() => setActiveTab('sentiment')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'sentiment' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              Sentiment
            </button>
            <button onClick={() => setActiveTab('insider')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'insider' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              Insider
            </button>
            <button onClick={() => setActiveTab('backtest')} className={`px-6 py-4 text-sm font-medium border-b-2 ${activeTab === 'backtest' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-600'}`}>
              Backtest
            </button>
          </div>
        </div>

        {isAnalyzing && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-3">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
              <span className="text-sm font-medium text-blue-900">Analyzing...</span>
            </div>
          </div>
        )}

        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-8 text-white">
              <div className="flex items-center gap-4 mb-4">
                <Award className="w-16 h-16" />
                <div>
                  <h2 className="text-3xl font-bold">Wharton Competition 2025</h2>
                  <p className="text-blue-100 mt-1">Score: 9.2/10 - Rank #1 of 48 teams</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border-2 border-blue-200 p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Upload className="w-6 h-6 text-blue-600" />
                Upload Your Portfolio for Analysis
              </h3>
              <div className="border-2 border-dashed border-blue-300 rounded-lg p-12 text-center hover:border-blue-500 cursor-pointer bg-blue-50" onClick={() => fileInputRef.current?.click()}>
                <Upload className="w-16 h-16 mx-auto text-blue-400 mb-4" />
                <p className="text-lg font-semibold text-gray-700 mb-2">Drop portfolio file or click to upload</p>
                <p className="text-sm text-gray-600">CSV, XLSX - Get 50+ quantitative metrics instantly</p>
                <input ref={fileInputRef} type="file" accept=".csv,.xlsx" onChange={handlePortfolioUpload} className="hidden" />
              </div>
              
              {portfolioData && (
                <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <CheckCircle className="w-5 h-5 inline text-green-600 mr-2" />
                  <span className="font-semibold">Portfolio Loaded: ${portfolioData.totalValue.toLocaleString()}</span>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white cursor-pointer hover:shadow-xl" onClick={() => portfolioData && setActiveTab('quant')}>
                <BarChart3 className="w-8 h-8 mb-3" />
                <h3 className="font-bold text-lg">Quant Metrics</h3>
                <p className="text-sm text-blue-100 mt-2">Beta, Sharpe, VaR, Factor Attribution</p>
              </div>

              <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-6 text-white cursor-pointer hover:shadow-xl" onClick={loadESGData}>
                <Leaf className="w-8 h-8 mb-3" />
                <h3 className="font-bold text-lg">ESG Analysis</h3>
                <p className="text-sm text-green-100 mt-2">Sustainability scores & carbon footprint</p>
              </div>

              <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-6 text-white cursor-pointer hover:shadow-xl" onClick={loadSentimentData}>
                <Activity className="w-8 h-8 mb-3" />
                <h3 className="font-bold text-lg">Sentiment AI</h3>
                <p className="text-sm text-purple-100 mt-2">Reddit WSB, Twitter, earnings calls</p>
              </div>

              <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg p-6 text-white cursor-pointer hover:shadow-xl" onClick={loadInsiderData}>
                <Users className="w-8 h-8 mb-3" />
                <h3 className="font-bold text-lg">Insider Tracking</h3>
                <p className="text-sm text-orange-100 mt-2">CEO/CFO Form 4 monitoring</p>
              </div>

              <div className="bg-gradient-to-br from-red-500 to-red-600 rounded-lg p-6 text-white cursor-pointer hover:shadow-xl" onClick={runBacktest}>
                <LineChart className="w-8 h-8 mb-3" />
                <h3 className="font-bold text-lg">Backtesting</h3>
                <p className="text-sm text-red-100 mt-2">Monte Carlo simulations</p>
              </div>

              <div className="bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-lg p-6 text-white cursor-pointer hover:shadow-xl">
                <Globe className="w-8 h-8 mb-3" />
                <h3 className="font-bold text-lg">Macro Scenarios</h3>
                <p className="text-sm text-indigo-100 mt-2">Fed rates, inflation stress tests</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'quant' && quantMetrics && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-2xl font-bold mb-6">Quantitative Analysis</h3>
              
              <div className="mb-6">
                <h4 className="text-lg font-semibold mb-4">Risk Metrics</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(quantMetrics.riskMetrics).map(([key, value]) => (
                    <div key={key} className="p-4 bg-red-50 rounded-lg">
                      <p className="text-xs text-gray-600 capitalize">{key.replace(/([A-Z])/g, ' $1')}</p>
                      <p className="text-2xl font-bold text-red-700">{value.toFixed(2)}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mb-6">
                <h4 className="text-lg font-semibold mb-4">Performance Metrics</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(quantMetrics.performanceMetrics).map(([key, value]) => (
                    <div key={key} className="p-4 bg-green-50 rounded-lg">
                      <p className="text-xs text-gray-600 capitalize">{key.replace(/([A-Z])/g, ' $1')}</p>
                      <p className="text-2xl font-bold text-green-700">{value.toFixed(2)}%</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'esg' && esgScores && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-2xl font-bold mb-6">ESG Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="p-6 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg text-white">
                  <p className="text-green-100 text-sm">Overall ESG</p>
                  <p className="text-4xl font-bold">{esgScores.portfolioESGScore}</p>
                </div>
                <div className="p-6 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-lg text-white">
                  <p className="text-blue-100 text-sm">Environmental</p>
                  <p className="text-4xl font-bold">{esgScores.environmental}</p>
                </div>
                <div className="p-6 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg text-white">
                  <p className="text-purple-100 text-sm">Social</p>
                  <p className="text-4xl font-bold">{esgScores.social}</p>
                </div>
                <div className="p-6 bg-gradient-to-br from-orange-500 to-red-600 rounded-lg text-white">
                  <p className="text-orange-100 text-sm">Governance</p>
                  <p className="text-4xl font-bold">{esgScores.governance}</p>
                </div>
              </div>

              <h4 className="font-semibold mb-3">Holdings Breakdown</h4>
              <div className="space-y-3">
                {esgScores.holdings.map((holding, idx) => (
                  <div key={idx} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between mb-3">
                      <h5 className="font-bold text-lg">{holding.ticker}</h5>
                      <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-bold">{holding.esgScore}</span>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <div>
                        <p className="text-xs text-gray-600">Environmental</p>
                        <p className="font-bold">{holding.environmental}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600">Social</p>
                        <p className="font-bold">{holding.social}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600">Governance</p>
                        <p className="font-bold">{holding.governance}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'sentiment' && sentimentData && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-2xl font-bold mb-6">Real-Time Sentiment Analysis</h3>
              <div className="mb-6 p-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white">
                <p className="text-purple-100 text-sm">Overall Portfolio Sentiment</p>
                <p className="text-5xl font-bold">{sentimentData.overallSentiment}%</p>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                {Object.entries(sentimentData.sources).map(([source, data]) => (
                  <div key={source} className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-xs text-gray-600 capitalize">{source}</p>
                    <p className="text-2xl font-bold text-purple-600">{data.score}%</p>
                    <p className="text-xs text-gray-500">{data.volume.toLocaleString()} mentions</p>
                  </div>
                ))}
              </div>

              <h4 className="font-semibold mb-3">Top Stocks by Sentiment</h4>
              {sentimentData.topStocks.map((stock, idx) => (
                <div key={idx} className="flex justify-between p-3 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg mb-2">
                  <span className="font-bold text-blue-600">{stock.ticker}</span>
                  <span className="font-bold">{stock.sentiment}% ({stock.volume} mentions)</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'insider' && insiderTrades.length > 0 && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-2xl font-bold mb-6">Insider Trading Monitor</h3>
              <div className="space-y-3">
                {insiderTrades.map((trade, idx) => (
                  <div key={idx} className="border-2 border-green-300 bg-green-50 rounded-lg p-4">
                    <div className="flex justify-between mb-2">
                      <div>
                        <h4 className="font-bold text-xl text-blue-600">{trade.ticker}</h4>
                        <p className="text-sm font-semibold">{trade.insider}</p>
                      </div>
                      <div className="text-right">
                        <span className="px-4 py-2 bg-green-500 text-white rounded-full text-sm font-bold">{trade.action}</span>
                        <p className="text-xs text-gray-500 mt-1">{trade.date}</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-gray-600">Shares</p>
                        <p className="font-bold">{trade.shares.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600">Value</p>
                        <p className="font-bold">${(trade.value / 1000000).toFixed(1)}M</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'backtest' && backtestResults && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-2xl font-bold mb-6">Backtest Results</h3>
              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-green-50 rounded-lg">
                  <p className="text-sm text-gray-600">Total Return</p>
                  <p className="text-2xl font-bold text-green-600">+{backtestResults.performanceMetrics.totalReturn}%</p>
                </div>
                <div className="p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-gray-600">Sharpe Ratio</p>
                  <p className="text-2xl font-bold text-blue-600">{backtestResults.performanceMetrics.sharpeRatio}</p>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg">
                  <p className="text-sm text-gray-600">Alpha</p>
                  <p className="text-2xl font-bold text-purple-600">+{backtestResults.performanceMetrics.alpha}%</p>
                </div>
                <div className="p-4 bg-red-50 rounded-lg">
                  <p className="text-sm text-gray-600">Max Drawdown</p>
                  <p className="text-2xl font-bold text-red-600">{backtestResults.performanceMetrics.maxDrawdown}%</p>
                </div>
              </div>

              <h4 className="font-semibold mb-3">Monte Carlo Simulation (1000 scenarios)</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Expected Value</p>
                  <p className="text-xl font-bold">${backtestResults.monteCarloSimulation.meanOutcome.toLocaleString()}</p>
                </div>
                <div className="p-4 bg-green-50 rounded-lg">
                  <p className="text-sm text-gray-600">Probability of Profit</p>
                  <p className="text-xl font-bold text-green-600">{backtestResults.monteCarloSimulation.probabilityProfit}%</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WhartonCompetitionPlatform;
