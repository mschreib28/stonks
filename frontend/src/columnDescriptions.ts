/**
 * Column descriptions for tooltips and help text
 */
export const columnDescriptions: Record<string, string> = {
  // Basic columns
  date: 'Trading date',
  ticker: 'Stock ticker symbol',
  open: 'Opening price for the day',
  high: 'Highest price during the day',
  low: 'Lowest price during the day',
  close: 'Closing price for the day',
  volume: 'Total number of shares traded',
  
  // Daily returns
  ret_d: 'Daily return percentage: (close / previous_close) - 1',
  abs_ret_d: 'Absolute daily return percentage: |ret_d| (magnitude of daily change)',
  abs_change_d: 'Absolute daily price change in dollars: |close - previous_close|',
  close_in_band: 'Boolean: true if closing price is within the specified price range (default: $1.00 - $9.99)',
  
  // Weekly columns
  week_start: 'Start date of the trading week (Monday)',
  close_w: 'Weekly closing price',
  ret_w: 'Weekly return percentage: (close_w / previous_close_w) - 1',
  abs_ret_w: 'Absolute weekly return percentage: |ret_w|',
  abs_change_w: 'Absolute weekly price change in dollars: |close_w - previous_close_w|',
  week: 'Week number',
  
  // Minute features
  day_range_pct: 'Intraday volatility: (high - low) / open (percentage of opening price)',
  max_1m_ret: 'Maximum 1-minute return during the day',
  max_5m_ret: 'Maximum 5-minute return during the day',
  max_drawdown_5m: 'Maximum drawdown within any rolling 5-minute window (pain metric)',
  vol_spike: 'Volume spike ratio: current day volume / average volume of prior 20 days',
  first_5m_high: 'Highest price during the first 5 minutes of trading (9:30-9:35 AM)',
  first_5m_low: 'Lowest price during the first 5 minutes of trading (9:30-9:35 AM)',
  first_15m_high: 'Highest price during the first 15 minutes of trading (9:30-9:45 AM)',
  first_15m_low: 'Lowest price during the first 15 minutes of trading (9:30-9:45 AM)',
  breakout_5m_up: 'Boolean: true if closing price broke above the first 5-minute high',
  breakout_5m_down: 'Boolean: true if closing price broke below the first 5-minute low',
  breakout_15m_up: 'Boolean: true if closing price broke above the first 15-minute high',
  breakout_15m_down: 'Boolean: true if closing price broke below the first 15-minute low',
  range_5m_pct: 'Opening range percentage: (first_5m_high - first_5m_low) / open',
  range_15m_pct: 'Opening range percentage: (first_15m_high - first_15m_low) / open',
  
  // Partition columns
  year: 'Year (for data partitioning)',
  month: 'Month (for data partitioning)',
  
  // MACD Features (1-minute timeframe)
  close_last__1m: 'Last closing price for the day (1-minute bars)',
  macd_last__1m: 'Final MACD line value for the day: EMA(12) - EMA(26) on 1-minute bars',
  signal_last__1m: 'Final signal line value for the day: EMA(9) of MACD on 1-minute bars',
  hist_last__1m: 'Final histogram value for the day: MACD - Signal on 1-minute bars',
  hist_min__1m: 'Minimum histogram value during the day (1-minute bars)',
  hist_max__1m: 'Maximum histogram value during the day (1-minute bars)',
  hist_zero_cross_up_count__1m: 'Number of times histogram crossed above zero during the day (1-minute bars)',
  hist_zero_cross_down_count__1m: 'Number of times histogram crossed below zero during the day (1-minute bars)',
  
  // MACD Features (5-minute timeframe)
  close_last__5m: 'Last closing price for the day (5-minute bars)',
  macd_last__5m: 'Final MACD line value for the day: EMA(12) - EMA(26) on 5-minute bars',
  signal_last__5m: 'Final signal line value for the day: EMA(9) of MACD on 5-minute bars',
  hist_last__5m: 'Final histogram value for the day: MACD - Signal on 5-minute bars',
  hist_min__5m: 'Minimum histogram value during the day (5-minute bars)',
  hist_max__5m: 'Maximum histogram value during the day (5-minute bars)',
  hist_zero_cross_up_count__5m: 'Number of times histogram crossed above zero during the day (5-minute bars)',
  hist_zero_cross_down_count__5m: 'Number of times histogram crossed below zero during the day (5-minute bars)',
};

export function getColumnDescription(columnName: string): string {
  return columnDescriptions[columnName] || `Column: ${columnName}`;
}

export function getColumnDisplayName(columnName: string): string {
  // Convert snake_case to Title Case with spaces
  return columnName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}


