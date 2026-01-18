import { useState, useEffect, useRef, useCallback } from 'react';
import { createChart, IChartApi, CandlestickData, HistogramData, Time, CandlestickSeries, HistogramSeries, LineSeries, LineData } from 'lightweight-charts';
import { getTickerData } from '../api';

interface ChartPanelProps {
  dataset: string;
  ticker: string;
  onClose: () => void;
}

type TimePeriod = '1d' | '5d' | '30d' | '6mo' | '12mo';

// Technical indicator calculation functions
const calculateSMA = (data: number[], period: number): (number | null)[] => {
  const result: (number | null)[] = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += data[i - j];
      }
      result.push(sum / period);
    }
  }
  return result;
};

const calculateBollingerBands = (data: number[], period: number = 20, stdDev: number = 2): {
  upper: (number | null)[];
  middle: (number | null)[];
  lower: (number | null)[];
} => {
  const middle = calculateSMA(data, period);
  const upper: (number | null)[] = [];
  const lower: (number | null)[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1 || middle[i] === null) {
      upper.push(null);
      lower.push(null);
    } else {
      let sumSquares = 0;
      for (let j = 0; j < period; j++) {
        sumSquares += Math.pow(data[i - j] - middle[i]!, 2);
      }
      const std = Math.sqrt(sumSquares / period);
      upper.push(middle[i]! + stdDev * std);
      lower.push(middle[i]! - stdDev * std);
    }
  }

  return { upper, middle, lower };
};

const calculateRSI = (data: number[], period: number = 14): (number | null)[] => {
  const result: (number | null)[] = [];
  const gains: number[] = [];
  const losses: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i === 0) {
      result.push(null);
      gains.push(0);
      losses.push(0);
      continue;
    }

    const change = data[i] - data[i - 1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? -change : 0);

    if (i < period) {
      result.push(null);
      continue;
    }

    let avgGain = 0;
    let avgLoss = 0;

    if (i === period) {
      // First RSI calculation uses simple average
      for (let j = 1; j <= period; j++) {
        avgGain += gains[j];
        avgLoss += losses[j];
      }
      avgGain /= period;
      avgLoss /= period;
    } else {
      // Subsequent calculations use exponential average
      const prevRSI = result[i - 1];
      if (prevRSI !== null) {
        // Calculate previous averages from previous RSI
        const prevRS = (100 - prevRSI) === 0 ? Infinity : prevRSI / (100 - prevRSI);
        // This is approximate, use simple smoothing
        for (let j = i - period + 1; j <= i; j++) {
          avgGain += gains[j];
          avgLoss += losses[j];
        }
        avgGain /= period;
        avgLoss /= period;
      }
    }

    if (avgLoss === 0) {
      result.push(100);
    } else {
      const rs = avgGain / avgLoss;
      result.push(100 - 100 / (1 + rs));
    }
  }

  return result;
};

export default function ChartPanel({ dataset, ticker, onClose }: ChartPanelProps) {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [timePeriod, setTimePeriod] = useState<TimePeriod>('12mo');
  
  // Technical indicator toggles
  const [showSMA20, setShowSMA20] = useState<boolean>(false);
  const [showSMA50, setShowSMA50] = useState<boolean>(false);
  const [showSMA200, setShowSMA200] = useState<boolean>(false);
  const [showBollinger, setShowBollinger] = useState<boolean>(false);
  const [showRSI, setShowRSI] = useState<boolean>(false);
  
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const macdSeriesRef = useRef<any>(null);
  const signalSeriesRef = useRef<any>(null);
  const histSeriesRef = useRef<any>(null);
  
  // Technical indicator series refs
  const sma20SeriesRef = useRef<any>(null);
  const sma50SeriesRef = useRef<any>(null);
  const sma200SeriesRef = useRef<any>(null);
  const bbUpperSeriesRef = useRef<any>(null);
  const bbMiddleSeriesRef = useRef<any>(null);
  const bbLowerSeriesRef = useRef<any>(null);
  const rsiSeriesRef = useRef<any>(null);

  // Calculate date range based on selected time period
  const getDateRange = (period: TimePeriod, dataArray: any[]): { from: Time; to: Time } | null => {
    if (dataArray.length === 0) return null;

    const dateCol = dataArray[0].date ? 'date' : dataArray[0].week_start ? 'week_start' : Object.keys(dataArray[0])[1];
    
    // Find the earliest and latest dates in the data
    let earliestDate: Date | null = null;
    let latestDate: Date | null = null;
    
    for (let i = 0; i < dataArray.length; i++) {
      const dateStr = dataArray[i][dateCol];
      if (dateStr) {
        const date = new Date(dateStr);
        if (!isNaN(date.getTime())) {
          if (!earliestDate || date < earliestDate) {
            earliestDate = date;
          }
          if (!latestDate || date > latestDate) {
            latestDate = date;
          }
        }
      }
    }

    if (!latestDate || !earliestDate) return null;

    const to = (latestDate.getTime() / 1000) as Time;
    let requestedFromDate: Date;

    // Calculate the requested range based on period
    switch (period) {
      case '1d':
        requestedFromDate = new Date(latestDate);
        requestedFromDate.setDate(requestedFromDate.getDate() - 1);
        break;
      case '5d':
        requestedFromDate = new Date(latestDate);
        requestedFromDate.setDate(requestedFromDate.getDate() - 5);
        break;
      case '30d':
        requestedFromDate = new Date(latestDate);
        requestedFromDate.setDate(requestedFromDate.getDate() - 30);
        break;
      case '6mo':
        requestedFromDate = new Date(latestDate);
        requestedFromDate.setMonth(requestedFromDate.getMonth() - 6);
        break;
      case '12mo':
        requestedFromDate = new Date(latestDate);
        requestedFromDate.setFullYear(requestedFromDate.getFullYear() - 1);
        break;
      default:
        requestedFromDate = new Date(latestDate);
        requestedFromDate.setFullYear(requestedFromDate.getFullYear() - 1);
    }

    // Use the intersection: take the later of requested start or actual earliest date
    // This ensures we show all available data if the requested range is larger than available
    const fromDate = requestedFromDate > earliestDate ? requestedFromDate : earliestDate;

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
      // Request enough data to cover 12 months (approximately 252 trading days)
      // Use a larger limit to ensure we get historical data
      const response = await getTickerData(dataset, ticker, 5000);
      
      // If filtered dataset has limited data, try daily dataset for more history
      if (dataset === 'filtered' && response.total < 200 && response.returned < 200) {
        try {
          // Request significantly more data from daily dataset to cover historical periods
          // 12 months = ~252 trading days, but request more to be safe (e.g., 2 years = ~500 days)
          const dailyResponse = await getTickerData('daily', ticker, 10000);
          if (dailyResponse.total > response.total) {
            console.log(`Using daily dataset for ${ticker} (${dailyResponse.total} records vs ${response.total} in filtered)`);
            
            // Debug: Check date range of returned data
            if (dailyResponse.data && dailyResponse.data.length > 0) {
              const dateCol = dailyResponse.data[0].date ? 'date' : dailyResponse.data[0].week_start ? 'week_start' : Object.keys(dailyResponse.data[0])[1];
              const dates = dailyResponse.data.map((row: any) => row[dateCol]).filter(Boolean);
              if (dates.length > 0) {
                const sortedDates = [...dates].sort();
                console.log(`Date range: ${sortedDates[0]} to ${sortedDates[sortedDates.length - 1]} (${dates.length} records)`);
              }
            }
            
            setData(dailyResponse.data.reverse());
            return;
          }
        } catch (e) {
          console.warn('Could not load daily dataset, using filtered data:', e);
        }
      }
      
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
      sma20SeriesRef.current = null;
      sma50SeriesRef.current = null;
      sma200SeriesRef.current = null;
      bbUpperSeriesRef.current = null;
      bbMiddleSeriesRef.current = null;
      bbLowerSeriesRef.current = null;
      rsiSeriesRef.current = null;
    }
    
    loadTickerData();
  }, [dataset, ticker, loadTickerData]);

  // Update technical indicators when toggles change
  const updateTechnicalIndicators = useCallback(() => {
    if (!chartRef.current || data.length === 0) return;
    
    const chart = chartRef.current;
    const dateCol = data[0].date ? 'date' : data[0].week_start ? 'week_start' : Object.keys(data[0])[1];
    
    // Extract close prices and timestamps
    const closeData: { time: Time; close: number }[] = [];
    data.forEach((row) => {
      const close = row.close || row.close_w;
      const date = row[dateCol];
      if (close && date) {
        closeData.push({
          time: new Date(date).getTime() / 1000 as Time,
          close,
        });
      }
    });
    closeData.sort((a, b) => (a.time as number) - (b.time as number));
    
    const closePrices = closeData.map(d => d.close);
    const timestamps = closeData.map(d => d.time);

    // SMA 20
    if (showSMA20) {
      const sma20 = calculateSMA(closePrices, 20);
      const sma20Data: LineData[] = timestamps
        .map((t, i) => ({ time: t, value: sma20[i] }))
        .filter(d => d.value !== null) as LineData[];
      
      if (!sma20SeriesRef.current) {
        sma20SeriesRef.current = chart.addSeries(LineSeries, {
          color: '#f59e0b',
          lineWidth: 1,
          title: 'SMA 20',
        }, 0);
      }
      sma20SeriesRef.current.setData(sma20Data);
    } else if (sma20SeriesRef.current) {
      chart.removeSeries(sma20SeriesRef.current);
      sma20SeriesRef.current = null;
    }

    // SMA 50
    if (showSMA50) {
      const sma50 = calculateSMA(closePrices, 50);
      const sma50Data: LineData[] = timestamps
        .map((t, i) => ({ time: t, value: sma50[i] }))
        .filter(d => d.value !== null) as LineData[];
      
      if (!sma50SeriesRef.current) {
        sma50SeriesRef.current = chart.addSeries(LineSeries, {
          color: '#3b82f6',
          lineWidth: 1,
          title: 'SMA 50',
        }, 0);
      }
      sma50SeriesRef.current.setData(sma50Data);
    } else if (sma50SeriesRef.current) {
      chart.removeSeries(sma50SeriesRef.current);
      sma50SeriesRef.current = null;
    }

    // SMA 200
    if (showSMA200) {
      const sma200 = calculateSMA(closePrices, 200);
      const sma200Data: LineData[] = timestamps
        .map((t, i) => ({ time: t, value: sma200[i] }))
        .filter(d => d.value !== null) as LineData[];
      
      if (!sma200SeriesRef.current) {
        sma200SeriesRef.current = chart.addSeries(LineSeries, {
          color: '#8b5cf6',
          lineWidth: 1,
          title: 'SMA 200',
        }, 0);
      }
      sma200SeriesRef.current.setData(sma200Data);
    } else if (sma200SeriesRef.current) {
      chart.removeSeries(sma200SeriesRef.current);
      sma200SeriesRef.current = null;
    }

    // Bollinger Bands
    if (showBollinger) {
      const bb = calculateBollingerBands(closePrices, 20, 2);
      const bbUpperData: LineData[] = timestamps
        .map((t, i) => ({ time: t, value: bb.upper[i] }))
        .filter(d => d.value !== null) as LineData[];
      const bbMiddleData: LineData[] = timestamps
        .map((t, i) => ({ time: t, value: bb.middle[i] }))
        .filter(d => d.value !== null) as LineData[];
      const bbLowerData: LineData[] = timestamps
        .map((t, i) => ({ time: t, value: bb.lower[i] }))
        .filter(d => d.value !== null) as LineData[];
      
      if (!bbUpperSeriesRef.current) {
        bbUpperSeriesRef.current = chart.addSeries(LineSeries, {
          color: '#ef4444',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          title: 'BB Upper',
        }, 0);
      }
      if (!bbMiddleSeriesRef.current) {
        bbMiddleSeriesRef.current = chart.addSeries(LineSeries, {
          color: '#6b7280',
          lineWidth: 1,
          title: 'BB Middle',
        }, 0);
      }
      if (!bbLowerSeriesRef.current) {
        bbLowerSeriesRef.current = chart.addSeries(LineSeries, {
          color: '#22c55e',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          title: 'BB Lower',
        }, 0);
      }
      bbUpperSeriesRef.current.setData(bbUpperData);
      bbMiddleSeriesRef.current.setData(bbMiddleData);
      bbLowerSeriesRef.current.setData(bbLowerData);
    } else {
      if (bbUpperSeriesRef.current) {
        chart.removeSeries(bbUpperSeriesRef.current);
        bbUpperSeriesRef.current = null;
      }
      if (bbMiddleSeriesRef.current) {
        chart.removeSeries(bbMiddleSeriesRef.current);
        bbMiddleSeriesRef.current = null;
      }
      if (bbLowerSeriesRef.current) {
        chart.removeSeries(bbLowerSeriesRef.current);
        bbLowerSeriesRef.current = null;
      }
    }

    // RSI
    if (showRSI) {
      const rsi = calculateRSI(closePrices, 14);
      const rsiData: LineData[] = timestamps
        .map((t, i) => ({ time: t, value: rsi[i] }))
        .filter(d => d.value !== null) as LineData[];
      
      // RSI goes in pane 3 (after MACD in pane 2)
      const paneIndex = macdSeriesRef.current ? 3 : 2;
      
      if (!rsiSeriesRef.current) {
        rsiSeriesRef.current = chart.addSeries(LineSeries, {
          color: '#ec4899',
          lineWidth: 1,
          title: 'RSI 14',
          priceFormat: {
            type: 'price',
            precision: 1,
            minMove: 0.1,
          },
        }, paneIndex);
        
        // Set RSI pane height
        const panes = chart.panes();
        if (panes.length > paneIndex) {
          panes[paneIndex].setHeight(100);
        }
      }
      rsiSeriesRef.current.setData(rsiData);
    } else if (rsiSeriesRef.current) {
      chart.removeSeries(rsiSeriesRef.current);
      rsiSeriesRef.current = null;
    }
  }, [data, showSMA20, showSMA50, showSMA200, showBollinger, showRSI]);

  // Update indicators when toggles change
  useEffect(() => {
    updateTechnicalIndicators();
  }, [showSMA20, showSMA50, showSMA200, showBollinger, showRSI, updateTechnicalIndicators]);

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

      // Deduplicate by timestamp - keep last occurrence of each timestamp
      const deduplicateByTime = <T extends { time: Time }>(arr: T[]): T[] => {
        const seen = new Map<number, T>();
        for (const item of arr) {
          const timeNum = item.time as number;
          seen.set(timeNum, item); // Keep last occurrence
        }
        return Array.from(seen.values()).sort((a, b) => (a.time as number) - (b.time as number));
      };

      const deduplicatedCandlestickData = deduplicateByTime(candlestickData);
      const deduplicatedVolumeData = deduplicateByTime(volumeData);
      const deduplicatedMacdData = deduplicateByTime(macdData);
      const deduplicatedSignalData = deduplicateByTime(signalData);
      const deduplicatedHistData = deduplicateByTime(histData);

      // Set data
      candlestickSeries.setData(deduplicatedCandlestickData);
      volumeSeries.setData(deduplicatedVolumeData);
      
      // Create MACD series if data is available and series don't exist
      if (hasMacd && deduplicatedMacdData.length > 0) {
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

        macdSeriesRef.current.setData(deduplicatedMacdData);
        signalSeriesRef.current.setData(deduplicatedSignalData);
        histSeriesRef.current.setData(deduplicatedHistData);

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

    // Deduplicate by timestamp - keep last occurrence of each timestamp
    const deduplicateByTime = <T extends { time: Time }>(arr: T[]): T[] => {
      const seen = new Map<number, T>();
      for (const item of arr) {
        const timeNum = item.time as number;
        seen.set(timeNum, item); // Keep last occurrence
      }
      return Array.from(seen.values()).sort((a, b) => (a.time as number) - (b.time as number));
    };

    const deduplicatedCandlestickData = deduplicateByTime(candlestickData);
    const deduplicatedVolumeData = deduplicateByTime(volumeData);
    const deduplicatedMacdData = deduplicateByTime(macdData);
    const deduplicatedSignalData = deduplicateByTime(signalData);
    const deduplicatedHistData = deduplicateByTime(histData);

    // Set data
    candlestickSeriesRef.current.setData(deduplicatedCandlestickData);
    volumeSeriesRef.current.setData(deduplicatedVolumeData);
    
    // Create MACD series if data is available and series don't exist
    if (hasMacd && deduplicatedMacdData.length > 0) {
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

      macdSeriesRef.current.setData(deduplicatedMacdData);
      signalSeriesRef.current.setData(deduplicatedSignalData);
      histSeriesRef.current.setData(deduplicatedHistData);

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
        <div className="flex items-center gap-2 px-4 pb-2">
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
        
        {/* Technical Indicator Toggles */}
        <div className="flex flex-wrap items-center gap-2 px-4 pb-3 border-t border-gray-800 pt-2">
          <span className="text-xs text-gray-500 mr-2">Indicators:</span>
          <button
            onClick={() => setShowSMA20(!showSMA20)}
            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
              showSMA20
                ? 'bg-amber-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            SMA 20
          </button>
          <button
            onClick={() => setShowSMA50(!showSMA50)}
            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
              showSMA50
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            SMA 50
          </button>
          <button
            onClick={() => setShowSMA200(!showSMA200)}
            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
              showSMA200
                ? 'bg-purple-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            SMA 200
          </button>
          <button
            onClick={() => setShowBollinger(!showBollinger)}
            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
              showBollinger
                ? 'bg-green-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Bollinger
          </button>
          <button
            onClick={() => setShowRSI(!showRSI)}
            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
              showRSI
                ? 'bg-pink-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            RSI
          </button>
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
