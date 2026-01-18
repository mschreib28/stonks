import { useState, useEffect } from 'react';
import { getAvailableFactors, evaluateFactor } from '../api';
import type { FactorInfo, FactorEvaluationResponse } from '../types';

interface FactorEvaluationPanelProps {
  onClose?: () => void;
}

export default function FactorEvaluationPanel({ onClose }: FactorEvaluationPanelProps) {
  const [factors, setFactors] = useState<FactorInfo[]>([]);
  const [selectedFactor, setSelectedFactor] = useState<string>('');
  const [periods, setPeriods] = useState<number[]>([1, 5, 10]);
  const [quantiles, setQuantiles] = useState<number>(5);
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<FactorEvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<boolean>(false);

  useEffect(() => {
    loadFactors();
  }, []);

  const loadFactors = async () => {
    try {
      const data = await getAvailableFactors();
      setFactors(data.factors);
      if (data.factors.length > 0) {
        setSelectedFactor(data.factors[0].name);
      }
    } catch (err) {
      setError('Failed to load available factors');
    }
  };

  const handleEvaluate = async () => {
    if (!selectedFactor) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await evaluateFactor({
        factor_column: selectedFactor,
        periods,
        quantiles,
      });
      setResults(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to evaluate factor');
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (n: number, decimals: number = 4) => {
    return n.toFixed(decimals);
  };

  const getICRating = (ic: number): { label: string; color: string } => {
    const absIC = Math.abs(ic);
    if (absIC >= 0.1) return { label: 'Strong', color: 'text-green-400' };
    if (absIC >= 0.05) return { label: 'Moderate', color: 'text-yellow-400' };
    if (absIC >= 0.02) return { label: 'Weak', color: 'text-orange-400' };
    return { label: 'Very Weak', color: 'text-red-400' };
  };

  return (
    <div className="bg-gray-800 dark:bg-gray-800 rounded-lg shadow p-4 border border-gray-700 dark:border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white dark:text-white">Factor Evaluation</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-sm text-blue-400 hover:text-blue-300"
          >
            {expanded ? 'Collapse' : 'Expand'}
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-300"
            >
              &times;
            </button>
          )}
        </div>
      </div>

      {expanded && (
        <>
          <div className="space-y-4 mb-4">
            {/* Factor Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Select Factor
              </label>
              <select
                value={selectedFactor}
                onChange={(e) => setSelectedFactor(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-600 rounded bg-gray-700 text-white"
              >
                {factors.map((factor) => (
                  <option key={factor.name} value={factor.name}>
                    {factor.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Periods */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Forward Return Periods (days)
              </label>
              <div className="flex gap-2">
                {[1, 5, 10, 20].map((p) => (
                  <label key={p} className="flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={periods.includes(p)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setPeriods([...periods, p].sort((a, b) => a - b));
                        } else {
                          setPeriods(periods.filter((x) => x !== p));
                        }
                      }}
                      className="rounded"
                    />
                    <span className="text-sm text-gray-300">{p}D</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Quantiles */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Quantiles
              </label>
              <select
                value={quantiles}
                onChange={(e) => setQuantiles(Number(e.target.value))}
                className="w-full px-3 py-2 text-sm border border-gray-600 rounded bg-gray-700 text-white"
              >
                <option value={3}>3 (Terciles)</option>
                <option value={5}>5 (Quintiles)</option>
                <option value={10}>10 (Deciles)</option>
              </select>
            </div>

            {/* Evaluate Button */}
            <button
              onClick={handleEvaluate}
              disabled={loading || !selectedFactor}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Evaluating...' : 'Evaluate Factor'}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="mb-4 p-3 bg-red-900 border border-red-700 rounded text-red-200 text-sm">
              {error}
            </div>
          )}

          {/* Results */}
          {results && (
            <div className="space-y-4">
              <h3 className="text-md font-semibold text-white border-b border-gray-700 pb-2">
                Results for: {results.factor}
              </h3>

              {/* IC Analysis */}
              <div>
                <h4 className="text-sm font-medium text-gray-300 mb-2">Information Coefficient (IC)</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-400 border-b border-gray-700">
                        <th className="py-2 px-2">Period</th>
                        <th className="py-2 px-2">Mean IC</th>
                        <th className="py-2 px-2">Std</th>
                        <th className="py-2 px-2">t-stat</th>
                        <th className="py-2 px-2">% Positive</th>
                        <th className="py-2 px-2">Rating</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(results.ic_analysis).map(([period, data]) => {
                        const rating = getICRating(data.mean_ic);
                        return (
                          <tr key={period} className="border-b border-gray-700">
                            <td className="py-2 px-2 text-white">{period}</td>
                            <td className={`py-2 px-2 ${data.mean_ic > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {formatNumber(data.mean_ic)}
                            </td>
                            <td className="py-2 px-2 text-gray-300">{formatNumber(data.ic_std)}</td>
                            <td className={`py-2 px-2 ${Math.abs(data.t_stat) > 2 ? 'text-green-400' : 'text-gray-300'}`}>
                              {formatNumber(data.t_stat, 2)}
                            </td>
                            <td className="py-2 px-2 text-gray-300">{(data.positive_pct * 100).toFixed(1)}%</td>
                            <td className={`py-2 px-2 ${rating.color}`}>{rating.label}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Quantile Returns */}
              <div>
                <h4 className="text-sm font-medium text-gray-300 mb-2">Quantile Returns (Spread = Q{quantiles} - Q1)</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-400 border-b border-gray-700">
                        <th className="py-2 px-2">Period</th>
                        <th className="py-2 px-2">Spread</th>
                        {Array.from({ length: quantiles }, (_, i) => (
                          <th key={i} className="py-2 px-2">Q{i + 1}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(results.quantile_returns).map(([period, data]) => (
                        <tr key={period} className="border-b border-gray-700">
                          <td className="py-2 px-2 text-white">{period}</td>
                          <td className={`py-2 px-2 ${data.spread > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {(data.spread * 100).toFixed(3)}%
                          </td>
                          {Object.values(data.returns_by_quantile).map((ret, i) => (
                            <td key={i} className={`py-2 px-2 ${ret > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {(ret * 100).toFixed(3)}%
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Interpretation Guide */}
              <div className="mt-4 p-3 bg-gray-700 rounded text-xs text-gray-300">
                <p className="font-medium mb-1">Interpretation Guide:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>IC &gt; 0.05: Moderate predictive power</li>
                  <li>IC &gt; 0.10: Strong predictive power</li>
                  <li>t-stat &gt; 2: Statistically significant</li>
                  <li>Positive spread: Higher quantile = higher returns</li>
                </ul>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
