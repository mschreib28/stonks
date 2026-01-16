import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { getTickerData } from '../api';

interface ChartPanelProps {
  dataset: string;
  ticker: string;
  onClose: () => void;
}

export default function ChartPanel({ dataset, ticker, onClose }: ChartPanelProps) {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    loadTickerData();
  }, [dataset, ticker]);

  const loadTickerData = async () => {
    setLoading(true);
    try {
      const response = await getTickerData(dataset, ticker, 100);
      setData(response.data.reverse()); // Reverse to show chronological order
    } catch (error) {
      console.error('Failed to load ticker data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 dark:bg-gray-800 rounded-lg shadow p-6 border border-gray-700 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-white dark:text-white">Loading {ticker}...</h2>
          <button
            onClick={onClose}
            className="text-gray-400 dark:text-gray-400 hover:text-gray-300 dark:hover:text-gray-300"
          >
            ✕
          </button>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="bg-gray-800 dark:bg-gray-800 rounded-lg shadow p-6 border border-gray-700 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-white dark:text-white">{ticker}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 dark:text-gray-400 hover:text-gray-300 dark:hover:text-gray-300"
          >
            ✕
          </button>
        </div>
        <p className="text-gray-400 dark:text-gray-400">No data available for this ticker.</p>
      </div>
    );
  }

  // Determine date column
  const dateCol = data[0].date ? 'date' : data[0].week_start ? 'week_start' : Object.keys(data[0])[1];
  
  // Prepare chart data
  const chartData = data.map((row) => ({
    date: row[dateCol],
    open: row.open || row.close_w || null,
    high: row.high || null,
    low: row.low || null,
    close: row.close || row.close_w || null,
    volume: row.volume || null,
    ret_d: row.ret_d || row.ret_w || null,
    abs_ret_d: row.abs_ret_d || row.abs_ret_w || null,
    // Minute features
    day_range_pct: row.day_range_pct || null,
    max_1m_ret: row.max_1m_ret || null,
    max_5m_ret: row.max_5m_ret || null,
    max_drawdown_5m: row.max_drawdown_5m || null,
    vol_spike: row.vol_spike || null,
  }));

  return (
    <div className="bg-gray-800 dark:bg-gray-800 rounded-lg shadow p-6 space-y-6 border border-gray-700 dark:border-gray-700">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-white dark:text-white">{ticker}</h2>
        <button
          onClick={onClose}
          className="text-gray-400 dark:text-gray-400 hover:text-gray-300 dark:hover:text-gray-300 text-xl"
        >
          ✕
        </button>
      </div>

      {/* Price Chart */}
      {chartData[0].close !== null && (
        <div>
          <h3 className="text-lg font-medium text-gray-300 dark:text-gray-300 mb-4">Price Chart</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12, fill: '#9CA3AF' }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} />
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', color: '#F3F4F6' }} />
              <Legend wrapperStyle={{ color: '#F3F4F6' }} />
              {chartData[0].open !== null && (
                <Line type="monotone" dataKey="open" stroke="#8884d8" strokeWidth={1} dot={false} />
              )}
              {chartData[0].high !== null && (
                <Line type="monotone" dataKey="high" stroke="#82ca9d" strokeWidth={1} dot={false} />
              )}
              {chartData[0].low !== null && (
                <Line type="monotone" dataKey="low" stroke="#ffc658" strokeWidth={1} dot={false} />
              )}
              <Line type="monotone" dataKey="close" stroke="#ff7300" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Returns Chart */}
      {chartData[0].ret_d !== null && (
        <div>
          <h3 className="text-lg font-medium text-gray-300 dark:text-gray-300 mb-4">Returns</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12, fill: '#9CA3AF' }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} />
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', color: '#F3F4F6' }} />
              <Legend wrapperStyle={{ color: '#F3F4F6' }} />
              <Bar dataKey="ret_d" fill="#8884d8" />
              {chartData[0].abs_ret_d !== null && (
                <Bar dataKey="abs_ret_d" fill="#82ca9d" />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Volume Chart */}
      {chartData[0].volume !== null && (
        <div>
          <h3 className="text-lg font-medium text-gray-300 dark:text-gray-300 mb-4">Volume</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12, fill: '#9CA3AF' }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} />
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', color: '#F3F4F6' }} />
              <Bar dataKey="volume" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Minute Features Chart (if available) */}
      {data[0].day_range_pct !== undefined && (
        <div>
          <h3 className="text-lg font-medium text-gray-300 dark:text-gray-300 mb-4">Intraday Features</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12, fill: '#9CA3AF' }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} />
              <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', color: '#F3F4F6' }} />
              <Legend wrapperStyle={{ color: '#F3F4F6' }} />
              <Line type="monotone" dataKey="day_range_pct" stroke="#8884d8" strokeWidth={2} dot={false} name="Day Range %" />
              <Line type="monotone" dataKey="max_1m_ret" stroke="#82ca9d" strokeWidth={2} dot={false} name="Max 1m Ret" />
              <Line type="monotone" dataKey="max_5m_ret" stroke="#ffc658" strokeWidth={2} dot={false} name="Max 5m Ret" />
              <Line type="monotone" dataKey="vol_spike" stroke="#ff7300" strokeWidth={2} dot={false} name="Vol Spike" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

