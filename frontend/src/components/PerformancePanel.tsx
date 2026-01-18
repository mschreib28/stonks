import { useState } from 'react';
import { getTickerPerformance, runBacktest } from '../api';
import type { PerformanceAnalysisResponse, BacktestResponse, BacktestRequest } from '../types';

interface PerformancePanelProps {
  ticker?: string;
  onClose?: () => void;
}

export default function PerformancePanel({ ticker: initialTicker, onClose }: PerformancePanelProps) {
  const [expanded, setExpanded] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'performance' | 'backtest'>('performance');
  
  // Performance state
  const [ticker, setTicker] = useState<string>(initialTicker || '');
  const [performanceLoading, setPerformanceLoading] = useState<boolean>(false);
  const [performanceResults, setPerformanceResults] = useState<PerformanceAnalysisResponse | null>(null);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  
  // Backtest state
  const [backtestTicker, setBacktestTicker] = useState<string>(initialTicker || '');
  const [strategy, setStrategy] = useState<'macd' | 'rsi'>('macd');
  const [fastPeriod, setFastPeriod] = useState<number>(12);
  const [slowPeriod, setSlowPeriod] = useState<number>(26);
  const [signalPeriod, setSignalPeriod] = useState<number>(9);
  const [rsiPeriod, setRsiPeriod] = useState<number>(14);
  const [oversold, setOversold] = useState<number>(30);
  const [overbought, setOverbought] = useState<number>(70);
  const [backtestLoading, setBacktestLoading] = useState<boolean>(false);
  const [backtestResults, setBacktestResults] = useState<BacktestResponse | null>(null);
  const [backtestError, setBacktestError] = useState<string | null>(null);

  const handleAnalyzePerformance = async () => {
    if (!ticker) return;
    
    setPerformanceLoading(true);
    setPerformanceError(null);
    
    try {
      const data = await getTickerPerformance(ticker);
      setPerformanceResults(data);
    } catch (err: any) {
      setPerformanceError(err.response?.data?.detail || 'Failed to analyze performance');
    } finally {
      setPerformanceLoading(false);
    }
  };

  const handleRunBacktest = async () => {
    setBacktestLoading(true);
    setBacktestError(null);
    
    try {
      const request: BacktestRequest = {
        strategy,
        ticker: backtestTicker || undefined,
        fast_period: fastPeriod,
        slow_period: slowPeriod,
        signal_period: signalPeriod,
        rsi_period: rsiPeriod,
        oversold,
        overbought,
      };
      
      const data = await runBacktest(request);
      setBacktestResults(data);
    } catch (err: any) {
      setBacktestError(err.response?.data?.detail || 'Failed to run backtest');
    } finally {
      setBacktestLoading(false);
    }
  };

  const formatPercent = (n: number) => `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
  const formatNumber = (n: number, decimals: number = 2) => n.toFixed(decimals);

  const getColorClass = (value: number, threshold: number = 0) => {
    return value > threshold ? 'text-green-400' : value < threshold ? 'text-red-400' : 'text-gray-300';
  };

  return (
    <div className="bg-gray-800 dark:bg-gray-800 rounded-lg shadow p-4 border border-gray-700 dark:border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white dark:text-white">Performance & Backtest</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-sm text-blue-400 hover:text-blue-300"
          >
            {expanded ? 'Collapse' : 'Expand'}
          </button>
          {onClose && (
            <button onClick={onClose} className="text-gray-400 hover:text-gray-300">
              &times;
            </button>
          )}
        </div>
      </div>

      {expanded && (
        <>
          {/* Tabs */}
          <div className="flex border-b border-gray-700 mb-4">
            <button
              onClick={() => setActiveTab('performance')}
              className={`px-4 py-2 text-sm ${
                activeTab === 'performance'
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              Performance Analysis
            </button>
            <button
              onClick={() => setActiveTab('backtest')}
              className={`px-4 py-2 text-sm ${
                activeTab === 'backtest'
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              Strategy Backtest
            </button>
          </div>

          {/* Performance Tab */}
          {activeTab === 'performance' && (
            <div className="space-y-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  placeholder="Enter ticker (e.g., AAPL)"
                  className="flex-1 px-3 py-2 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                />
                <button
                  onClick={handleAnalyzePerformance}
                  disabled={performanceLoading || !ticker}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {performanceLoading ? 'Loading...' : 'Analyze'}
                </button>
              </div>

              {performanceError && (
                <div className="p-3 bg-red-900 border border-red-700 rounded text-red-200 text-sm">
                  {performanceError}
                </div>
              )}

              {performanceResults && (
                <div className="space-y-4">
                  <h3 className="text-md font-semibold text-white border-b border-gray-700 pb-2">
                    {performanceResults.ticker || 'Portfolio'} Performance
                  </h3>

                  {/* Summary */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-700 p-3 rounded">
                      <div className="text-xs text-gray-400">Total Return</div>
                      <div className={`text-lg font-semibold ${getColorClass(performanceResults.summary.total_return_pct)}`}>
                        {formatPercent(performanceResults.summary.total_return_pct)}
                      </div>
                    </div>
                    <div className="bg-gray-700 p-3 rounded">
                      <div className="text-xs text-gray-400">Annualized Return</div>
                      <div className={`text-lg font-semibold ${getColorClass(performanceResults.summary.annualized_return_pct)}`}>
                        {formatPercent(performanceResults.summary.annualized_return_pct)}
                      </div>
                    </div>
                    <div className="bg-gray-700 p-3 rounded">
                      <div className="text-xs text-gray-400">Sharpe Ratio</div>
                      <div className={`text-lg font-semibold ${getColorClass(performanceResults.risk_adjusted.sharpe_ratio, 1)}`}>
                        {formatNumber(performanceResults.risk_adjusted.sharpe_ratio)}
                      </div>
                    </div>
                    <div className="bg-gray-700 p-3 rounded">
                      <div className="text-xs text-gray-400">Max Drawdown</div>
                      <div className="text-lg font-semibold text-red-400">
                        {formatPercent(performanceResults.drawdown.max_drawdown_pct)}
                      </div>
                    </div>
                  </div>

                  {/* Detailed Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-700 p-3 rounded">
                      <h4 className="text-sm font-medium text-gray-300 mb-2">Risk-Adjusted Metrics</h4>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Sharpe Ratio</span>
                          <span className={getColorClass(performanceResults.risk_adjusted.sharpe_ratio, 1)}>
                            {formatNumber(performanceResults.risk_adjusted.sharpe_ratio)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Sortino Ratio</span>
                          <span className={getColorClass(performanceResults.risk_adjusted.sortino_ratio, 1)}>
                            {formatNumber(performanceResults.risk_adjusted.sortino_ratio)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Calmar Ratio</span>
                          <span className={getColorClass(performanceResults.risk_adjusted.calmar_ratio, 0.5)}>
                            {formatNumber(performanceResults.risk_adjusted.calmar_ratio)}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-700 p-3 rounded">
                      <h4 className="text-sm font-medium text-gray-300 mb-2">Trade Metrics</h4>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Win Rate</span>
                          <span className={getColorClass(performanceResults.trade_metrics.win_rate_pct, 50)}>
                            {formatNumber(performanceResults.trade_metrics.win_rate_pct, 1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Best Day</span>
                          <span className="text-green-400">
                            {formatPercent(performanceResults.trade_metrics.best_day_pct)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Worst Day</span>
                          <span className="text-red-400">
                            {formatPercent(performanceResults.trade_metrics.worst_day_pct)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Backtest Tab */}
          {activeTab === 'backtest' && (
            <div className="space-y-4">
              {/* Strategy Selection */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Strategy</label>
                  <select
                    value={strategy}
                    onChange={(e) => setStrategy(e.target.value as 'macd' | 'rsi')}
                    className="w-full px-3 py-2 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                  >
                    <option value="macd">MACD Crossover</option>
                    <option value="rsi">RSI Mean Reversion</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Ticker (optional)</label>
                  <input
                    type="text"
                    value={backtestTicker}
                    onChange={(e) => setBacktestTicker(e.target.value.toUpperCase())}
                    placeholder="Leave empty for all"
                    className="w-full px-3 py-2 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                  />
                </div>
              </div>

              {/* Strategy Parameters */}
              {strategy === 'macd' && (
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Fast Period</label>
                    <input
                      type="number"
                      value={fastPeriod}
                      onChange={(e) => setFastPeriod(Number(e.target.value))}
                      className="w-full px-2 py-1 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Slow Period</label>
                    <input
                      type="number"
                      value={slowPeriod}
                      onChange={(e) => setSlowPeriod(Number(e.target.value))}
                      className="w-full px-2 py-1 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Signal Period</label>
                    <input
                      type="number"
                      value={signalPeriod}
                      onChange={(e) => setSignalPeriod(Number(e.target.value))}
                      className="w-full px-2 py-1 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                    />
                  </div>
                </div>
              )}

              {strategy === 'rsi' && (
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">RSI Period</label>
                    <input
                      type="number"
                      value={rsiPeriod}
                      onChange={(e) => setRsiPeriod(Number(e.target.value))}
                      className="w-full px-2 py-1 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Oversold</label>
                    <input
                      type="number"
                      value={oversold}
                      onChange={(e) => setOversold(Number(e.target.value))}
                      className="w-full px-2 py-1 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Overbought</label>
                    <input
                      type="number"
                      value={overbought}
                      onChange={(e) => setOverbought(Number(e.target.value))}
                      className="w-full px-2 py-1 text-sm border border-gray-600 rounded bg-gray-700 text-white"
                    />
                  </div>
                </div>
              )}

              <button
                onClick={handleRunBacktest}
                disabled={backtestLoading}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                {backtestLoading ? 'Running Backtest...' : 'Run Backtest'}
              </button>

              {backtestError && (
                <div className="p-3 bg-red-900 border border-red-700 rounded text-red-200 text-sm">
                  {backtestError}
                </div>
              )}

              {backtestResults && (
                <div className="space-y-4">
                  <h3 className="text-md font-semibold text-white border-b border-gray-700 pb-2">
                    Backtest Results: {backtestResults.strategy || strategy.toUpperCase()}
                  </h3>

                  {/* Single ticker results */}
                  {backtestResults.performance && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-xs text-gray-400">Total Return</div>
                        <div className={`text-lg font-semibold ${getColorClass(backtestResults.performance.total_return)}`}>
                          {formatPercent(backtestResults.performance.total_return)}
                        </div>
                      </div>
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-xs text-gray-400">Sharpe Ratio</div>
                        <div className={`text-lg font-semibold ${getColorClass(backtestResults.performance.sharpe_ratio, 1)}`}>
                          {formatNumber(backtestResults.performance.sharpe_ratio)}
                        </div>
                      </div>
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-xs text-gray-400">Max Drawdown</div>
                        <div className="text-lg font-semibold text-red-400">
                          {formatPercent(backtestResults.performance.max_drawdown)}
                        </div>
                      </div>
                      <div className="bg-gray-700 p-3 rounded">
                        <div className="text-xs text-gray-400">Win Rate</div>
                        <div className={`text-lg font-semibold ${getColorClass(backtestResults.performance.win_rate, 50)}`}>
                          {formatNumber(backtestResults.performance.win_rate, 1)}%
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Multi-ticker results */}
                  {backtestResults.by_ticker && (
                    <div className="max-h-96 overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-gray-800">
                          <tr className="text-left text-gray-400 border-b border-gray-700">
                            <th className="py-2 px-2">Ticker</th>
                            <th className="py-2 px-2">Return</th>
                            <th className="py-2 px-2">Sharpe</th>
                            <th className="py-2 px-2">Max DD</th>
                            <th className="py-2 px-2">Win Rate</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(backtestResults.by_ticker)
                            .filter(([, result]) => result.performance)
                            .sort(([, a], [, b]) => (b.performance?.total_return || 0) - (a.performance?.total_return || 0))
                            .slice(0, 20)
                            .map(([tickerName, result]) => (
                              <tr key={tickerName} className="border-b border-gray-700">
                                <td className="py-2 px-2 text-white font-mono">{tickerName}</td>
                                <td className={`py-2 px-2 ${getColorClass(result.performance?.total_return || 0)}`}>
                                  {formatPercent(result.performance?.total_return || 0)}
                                </td>
                                <td className={`py-2 px-2 ${getColorClass(result.performance?.sharpe_ratio || 0, 1)}`}>
                                  {formatNumber(result.performance?.sharpe_ratio || 0)}
                                </td>
                                <td className="py-2 px-2 text-red-400">
                                  {formatPercent(result.performance?.max_drawdown || 0)}
                                </td>
                                <td className={`py-2 px-2 ${getColorClass(result.performance?.win_rate || 0, 50)}`}>
                                  {formatNumber(result.performance?.win_rate || 0, 1)}%
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
