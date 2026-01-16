import { useState, useEffect, useRef, useCallback } from 'react';
import { createChart, IChartApi, CandlestickData, HistogramData, Time, CandlestickSeries, HistogramSeries, LineSeries } from 'lightweight-charts';
import { getTickerData } from '../api';

interface ChartPanelProps {
  dataset: string;
  ticker: string;
  onClose: () => void;
}

type TimePeriod = '1d' | '5d' | '30d' | '6mo' | '12mo';

export default function ChartPanel({ dataset, ticker, onClose }: ChartPanelProps) {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('12mo');
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const macdSeriesRef = useRef<any>(null);
  const signalSeriesRef = useRef<any>(null);
  const histSeriesRef = useRef<any>(null);

  // Calculate date range based on selected time period
  const getDateRange = (period: TimePeriod, dataArray: any[]): { from: Time; to: Time } | null => {
    if (dataArray.length === 0) return null;

    const dateCol = dataArray[0].date ? 'date' : dataArray[0].week_start ? 'week_start' : Object.keys(dataArray[0])[1];
    
    // Find the latest date in the data
    let latestDate: Date | null = null;
    for (let i = dataArray.length - 1; i >= 0; i--) {
      const dateStr = dataArray[i][dateCol];
      if (dateStr) {
        const date = new Date(dateStr);
        if (!isNaN(date.getTime())) {
          latestDate = date;
          break;
        }
      }
    }

    if (!latestDate) return null;

    const to = (latestDate.getTime() / 1000) as Time;
    let fromDate: Date;

    switch (period) {
      case '1d':
        fromDate = new Date(latestDate);
        fromDate.setDate(fromDate.getDate() - 1);
        break;
      case '5d':
        fromDate = new Date(latestDate);
        fromDate.setDate(fromDate.getDate() - 5);
        break;
      case '30d':
        fromDate = new Date(latestDate);
        fromDate.setDate(fromDate.getDate() - 30);
        break;
      case '6mo':
        fromDate = new Date(latestDate);
        fromDate.setMonth(fromDate.getMonth() - 6);
        break;
      case '12mo':
        fromDate = new Date(latestDate);
        fromDate.setFullYear(fromDate.getFullYear() - 1);
        break;
      default:
        fromDate = new Date(latestDate);
        fromDate.setFullYear(fromDate.getFullYear() - 1);
    }

    return {
      from: (fromDate.getTime() / 1000) as Time,
      to,
    };
  };

  // Update chart visible range when time period or data changes
  const updateVisibleRange = () => {
    if (!chartRef.current || data.length === 0) return;
    
    const range = getDateRange(timePeriod, data);
    if (range) {
      chartRef.current.timeScale().setVisibleRange(range);
    }
  };

  const loadTickerData = useCallback(async () => {
    setLoading(true);
    try {
      // Load more data to ensure we have enough for longer periods
      const response = await getTickerData(dataset, ticker, 500);
      setData(response.data.reverse()); // Reverse to show chronological order
    } catch (error) {
      console.error('Failed to load ticker data:', error);
    } finally {
      setLoading(false);
    }
  }, [dataset, ticker]);

  useEffect(() => {
    // Reset data when ticker/dataset changes to trigger re-initialization
    setData([]);
    setLoading(true);
    
    // Clean up existing chart completely when ticker/dataset changes
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      candlestickSeriesRef.current = null;
      volumeSeriesRef.current = null;
      macdSeriesRef.current = null;
      signalSeriesRef.current = null;
      histSeriesRef.current = null;
    }
    
    loadTickerData();
  }, [dataset, ticker, loadTickerData]);

  // Initialize chart - reinitialize when data is loaded and ready
  useEffect(() => {
    if (!chartContainerRef.current || loading || data.length === 0) return;

    // If chart already exists for this ticker/dataset, don't reinitialize
    if (chartRef.current) return;

    // Wait for container to be ready
    const rafId = requestAnimationFrame(() => {
      if (!chartContainerRef.current) return;
      
      const container = chartContainerRef.current;
      if (container.clientWidth === 0) {
        // Retry if container not ready
        const retryId = requestAnimationFrame(() => {
          if (chartContainerRef.current && chartContainerRef.current.clientWidth > 0) {
            initializeChart(chartContainerRef.current);
          }
        });
        return () => cancelAnimationFrame(retryId);
      }
      
      initializeChart(container);
    });

    const initializeChart = (container: HTMLDivElement) => {
      const chart = createChart(container, {
        width: container.clientWidth,
        height: 600,
        layout: {
          background: { color: '#111827' },
          textColor: '#9ca3af',
          panes: {
            separatorColor: '#374151',
            separatorHoverColor: '#4b5563',
          },
        },
        grid: {
          vertLines: { color: '#1f2937' },
          horzLines: { color: '#1f2937' },
        },
        crosshair: {
          mode: 1,
        },
        rightPriceScale: {
          borderColor: '#4b5563',
        },
        timeScale: {
          borderColor: '#4b5563',
          timeVisible: true,
          secondsVisible: false,
        },
      });

      // Pane 0: Candlesticks (main price chart)
      const candlestickSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      }, 0); // Pane 0

      // Pane 1: Volume
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
      }, 1); // Pane 1

      chartRef.current = chart;
      candlestickSeriesRef.current = candlestickSeries;
      volumeSeriesRef.current = volumeSeries;

      // Set pane heights: candlesticks get ~50%, volume and MACD each get ~25%
      const panes = chart.panes();
      if (panes.length > 0) {
        panes[0].setHeight(300); // Candlesticks - main pane
      }
      if (panes.length > 1) {
        panes[1].setHeight(150); // Volume
      }
      if (panes.length > 2) {
        panes[2].setHeight(150); // MACD
      }

      // Immediately set data if available
      if (data.length > 0) {
        updateChartData(chart, candlestickSeries, volumeSeries);
      }
    };

    const updateChartData = (chart: IChartApi, candlestickSeries: any, volumeSeries: any) => {
      if (data.length === 0) return;

      // Prepare data
      const dateCol = data[0].date ? 'date' : data[0].week_start ? 'week_start' : Object.keys(data[0])[1];
      
      const candlestickData: CandlestickData[] = [];
      const volumeData: HistogramData[] = [];
      const macdData: any[] = [];
      const signalData: any[] = [];
      const histData: HistogramData[] = [];

      // Check which MACD columns are available
      const hasMacd1m = data[0].macd_last__1m !== undefined;
      const hasMacd5m = data[0].macd_last__5m !== undefined;
      const hasMacd = hasMacd1m || hasMacd5m;
      const macdCol = hasMacd1m ? 'macd_last__1m' : (hasMacd5m ? 'macd_last__5m' : null);
      const signalCol = hasMacd1m ? 'signal_last__1m' : (hasMacd5m ? 'signal_last__5m' : null);
      const histCol = hasMacd1m ? 'hist_last__1m' : (hasMacd5m ? 'hist_last__5m' : null);

      data.forEach((row) => {
        const open = row.open || row.close_w;
        const high = row.high;
        const low = row.low;
        const close = row.close || row.close_w;
        const volume = row.volume;
        const date = row[dateCol];

        if (open && high && low && close && date) {
          // Convert date to timestamp
          const timestamp = new Date(date).getTime() / 1000 as Time;
          
          candlestickData.push({
            time: timestamp,
            open: open,
            high: high,
            low: low,
            close: close,
          });

          if (volume) {
            const isUp = close >= open;
            volumeData.push({
              time: timestamp,
              value: volume,
              color: isUp ? '#26a69a' : '#ef5350',
            });
          }

          // MACD data
          if (macdCol && row[macdCol] !== null && row[macdCol] !== undefined) {
            macdData.push({
              time: timestamp,
              value: row[macdCol],
            });
          }
          if (signalCol && row[signalCol] !== null && row[signalCol] !== undefined) {
            signalData.push({
              time: timestamp,
              value: row[signalCol],
            });
          }
          if (histCol && row[histCol] !== null && row[histCol] !== undefined) {
            histData.push({
              time: timestamp,
              value: row[histCol],
              color: row[histCol] >= 0 ? '#26a69a' : '#ef5350',
            });
          }
        }
      });

      // Sort by time (ascending)
      candlestickData.sort((a, b) => (a.time as number) - (b.time as number));
      volumeData.sort((a, b) => (a.time as number) - (b.time as number));
      macdData.sort((a, b) => (a.time as number) - (b.time as number));
      signalData.sort((a, b) => (a.time as number) - (b.time as number));
      histData.sort((a, b) => (a.time as number) - (b.time as number));

      // Set data
      candlestickSeries.setData(candlestickData);
      volumeSeries.setData(volumeData);
      
      // Create MACD series if data is available and series don't exist
      if (hasMacd && macdData.length > 0) {
        if (!macdSeriesRef.current) {
          macdSeriesRef.current = chart.addSeries(LineSeries, {
            color: '#3b82f6',
            lineWidth: 1,
            title: 'MACD',
          }, 2);
        }
        if (!signalSeriesRef.current) {
          signalSeriesRef.current = chart.addSeries(LineSeries, {
            color: '#f59e0b',
            lineWidth: 1,
            title: 'Signal',
          }, 2);
        }
        if (!histSeriesRef.current) {
          histSeriesRef.current = chart.addSeries(HistogramSeries, {
            color: '#8b5cf6',
            priceFormat: {
              type: 'price',
              precision: 4,
              minMove: 0.0001,
            },
          }, 2);
        }

        macdSeriesRef.current.setData(macdData);
        signalSeriesRef.current.setData(signalData);
        histSeriesRef.current.setData(histData);

        // Set MACD pane height
        const panes = chart.panes();
        if (panes.length > 2) {
          panes[2].setHeight(150);
        }
      } else {
        // Remove MACD series if they exist but no data
        if (macdSeriesRef.current) {
          chart.removeSeries(macdSeriesRef.current);
          macdSeriesRef.current = null;
        }
        if (signalSeriesRef.current) {
          chart.removeSeries(signalSeriesRef.current);
          signalSeriesRef.current = null;
        }
        if (histSeriesRef.current) {
          chart.removeSeries(histSeriesRef.current);
          histSeriesRef.current = null;
        }
      }

      // Set visible range based on selected time period
      const range = getDateRange(timePeriod, data);
      if (range) {
        chart.timeScale().setVisibleRange(range);
      }
    };

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        candlestickSeriesRef.current = null;
        volumeSeriesRef.current = null;
        macdSeriesRef.current = null;
        signalSeriesRef.current = null;
        histSeriesRef.current = null;
      }
    };
  }, [loading, data.length]); // Reinitialize when loading completes and data is available

  // Update chart data when data changes
  useEffect(() => {
    if (!chartRef.current || !candlestickSeriesRef.current || !volumeSeriesRef.current || data.length === 0 || loading) {
      return;
    }

    // Prepare data
    const dateCol = data[0].date ? 'date' : data[0].week_start ? 'week_start' : Object.keys(data[0])[1];
    
    const candlestickData: CandlestickData[] = [];
    const volumeData: HistogramData[] = [];
    const macdData: any[] = [];
    const signalData: any[] = [];
    const histData: HistogramData[] = [];

    // Check which MACD columns are available
    const hasMacd1m = data[0].macd_last__1m !== undefined;
    const hasMacd5m = data[0].macd_last__5m !== undefined;
    const hasMacd = hasMacd1m || hasMacd5m;
    const macdCol = hasMacd1m ? 'macd_last__1m' : (hasMacd5m ? 'macd_last__5m' : null);
    const signalCol = hasMacd1m ? 'signal_last__1m' : (hasMacd5m ? 'signal_last__5m' : null);
    const histCol = hasMacd1m ? 'hist_last__1m' : (hasMacd5m ? 'hist_last__5m' : null);

    data.forEach((row) => {
      const open = row.open || row.close_w;
      const high = row.high;
      const low = row.low;
      const close = row.close || row.close_w;
      const volume = row.volume;
      const date = row[dateCol];

      if (open && high && low && close && date) {
        // Convert date to timestamp
        const timestamp = new Date(date).getTime() / 1000 as Time;
        
        candlestickData.push({
          time: timestamp,
          open: open,
          high: high,
          low: low,
          close: close,
        });

        if (volume) {
          const isUp = close >= open;
          volumeData.push({
            time: timestamp,
            value: volume,
            color: isUp ? '#26a69a' : '#ef5350',
          });
        }

        // MACD data
        if (macdCol && row[macdCol] !== null && row[macdCol] !== undefined) {
          macdData.push({
            time: timestamp,
            value: row[macdCol],
          });
        }
        if (signalCol && row[signalCol] !== null && row[signalCol] !== undefined) {
          signalData.push({
            time: timestamp,
            value: row[signalCol],
          });
        }
        if (histCol && row[histCol] !== null && row[histCol] !== undefined) {
          histData.push({
            time: timestamp,
            value: row[histCol],
            color: row[histCol] >= 0 ? '#26a69a' : '#ef5350',
          });
        }
      }
    });

    // Sort by time (ascending)
    candlestickData.sort((a, b) => (a.time as number) - (b.time as number));
    volumeData.sort((a, b) => (a.time as number) - (b.time as number));
    macdData.sort((a, b) => (a.time as number) - (b.time as number));
    signalData.sort((a, b) => (a.time as number) - (b.time as number));
    histData.sort((a, b) => (a.time as number) - (b.time as number));

    // Set data
    candlestickSeriesRef.current.setData(candlestickData);
    volumeSeriesRef.current.setData(volumeData);
    
    // Create MACD series if data is available and series don't exist
    if (hasMacd && macdData.length > 0) {
      if (!macdSeriesRef.current) {
        macdSeriesRef.current = chartRef.current.addSeries(LineSeries, {
          color: '#3b82f6',
          lineWidth: 1,
          title: 'MACD',
        }, 2);
      }
      if (!signalSeriesRef.current) {
        signalSeriesRef.current = chartRef.current.addSeries(LineSeries, {
          color: '#f59e0b',
          lineWidth: 1,
          title: 'Signal',
        }, 2);
      }
      if (!histSeriesRef.current) {
        histSeriesRef.current = chartRef.current.addSeries(HistogramSeries, {
          color: '#8b5cf6',
          priceFormat: {
            type: 'price',
            precision: 4,
            minMove: 0.0001,
          },
        }, 2);
      }

      macdSeriesRef.current.setData(macdData);
      signalSeriesRef.current.setData(signalData);
      histSeriesRef.current.setData(histData);

      // Set MACD pane height
      const panes = chartRef.current.panes();
      if (panes.length > 2) {
        panes[2].setHeight(150);
      }
    } else {
      // Remove MACD series if they exist but no data
      if (macdSeriesRef.current) {
        chartRef.current.removeSeries(macdSeriesRef.current);
        macdSeriesRef.current = null;
      }
      if (signalSeriesRef.current) {
        chartRef.current.removeSeries(signalSeriesRef.current);
        signalSeriesRef.current = null;
      }
      if (histSeriesRef.current) {
        chartRef.current.removeSeries(histSeriesRef.current);
        histSeriesRef.current = null;
      }
    }

    // Update visible range when data changes
    updateVisibleRange();
  }, [data, timePeriod]);

  if (data.length === 0 && !loading) {
    return (
      <div className="bg-gray-900 dark:bg-gray-900 rounded-lg shadow-lg border border-gray-800 dark:border-gray-800">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h2 className="text-xl font-bold text-white">{ticker}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-300 text-xl w-8 h-8 flex items-center justify-center rounded hover:bg-gray-800"
          >
            ✕
          </button>
        </div>
        <div className="p-4">
          <p className="text-gray-400">No data available for this ticker.</p>
        </div>
      </div>
    );
  }

  // Calculate current price and change
  const dateCol = data.length > 0 ? (data[0].date ? 'date' : data[0].week_start ? 'week_start' : Object.keys(data[0])[1]) : 'date';
  const latest = data.length > 0 ? data[data.length - 1] : null;
  const prev = data.length > 1 ? data[data.length - 2] : latest;
  const currentPrice = latest?.close || latest?.close_w;
  const prevPrice = prev?.close || prev?.close_w;
  const change = currentPrice && prevPrice ? currentPrice - prevPrice : 0;
  const changePercent = prevPrice !== 0 && prevPrice ? (change / prevPrice) * 100 : 0;
  const isUp = change >= 0;

  return (
    <div className="bg-gray-900 dark:bg-gray-900 rounded-lg shadow-lg border border-gray-800 dark:border-gray-800">
      {/* Header - TradingView Style */}
      <div className="flex flex-col border-b border-gray-800 dark:border-gray-800">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-baseline gap-4">
            <h2 className="text-xl font-bold text-white">{ticker}</h2>
            {currentPrice && (
              <div className="flex items-center gap-3">
                <span className={`text-2xl font-bold ${isUp ? 'text-green-400' : 'text-red-400'}`}>
                  ${currentPrice.toFixed(2)}
                </span>
                <span className={`text-sm font-medium ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {change >= 0 ? '+' : ''}{change.toFixed(2)} ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
                </span>
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 dark:text-gray-400 hover:text-gray-300 dark:hover:text-gray-300 text-xl w-8 h-8 flex items-center justify-center rounded hover:bg-gray-800 transition-colors"
          >
            ✕
          </button>
        </div>
        {/* Time Period Buttons */}
        <div className="flex items-center gap-2 px-4 pb-3">
          {(['1d', '5d', '30d', '6mo', '12mo'] as TimePeriod[]).map((period) => (
            <button
              key={period}
              onClick={() => {
                setTimePeriod(period);
                // Update range immediately
                setTimeout(() => updateVisibleRange(), 0);
              }}
              className={`px-3 py-1.5 text-sm font-medium rounded transition-colors ${
                timePeriod === period
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              {period}
            </button>
          ))}
        </div>
      </div>

      {/* TradingView-style Chart */}
      <div className="p-4">
        {loading ? (
          <div className="w-full h-[600px] flex items-center justify-center">
            <div className="text-gray-400">Loading chart data...</div>
          </div>
        ) : (
          <div ref={chartContainerRef} className="w-full" style={{ height: '600px' }} />
        )}
      </div>

      {/* Additional Info Panel */}
      {latest && latest.ret_d !== null && (
        <div className="px-4 pb-4 border-t border-gray-800 dark:border-gray-800 pt-4">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-gray-400 mb-1">Daily Return</div>
              <div className={`font-semibold ${latest.ret_d >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {latest.ret_d >= 0 ? '+' : ''}{latest.ret_d.toFixed(2)}%
              </div>
            </div>
            {latest.volume && (
              <div>
                <div className="text-gray-400 mb-1">Volume</div>
                <div className="text-white font-semibold">{latest.volume.toLocaleString()}</div>
              </div>
            )}
            {latest.high && latest.low && (
              <div>
                <div className="text-gray-400 mb-1">Range</div>
                <div className="text-white font-semibold">
                  ${latest.low.toFixed(2)} - ${latest.high.toFixed(2)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
