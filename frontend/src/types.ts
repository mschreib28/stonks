export interface ColumnInfo {
  name: string;
  type: string;
  nullable: boolean;
}

export interface Dataset {
  name: string;
  filename: string;
  rows: number;
  columns: number;
  error?: string;
}

export interface FilterValue {
  min?: number;
  max?: number;
  value?: boolean | string | number;
}

export interface Filters {
  [key: string]: FilterValue | string | number | boolean | string[];
}

export interface QueryRequest {
  dataset: string;
  filters: Filters;
  sort_by?: string;
  sort_desc: boolean;
  limit: number;
  offset: number;
}

export interface QueryResponse {
  data: Record<string, any>[];
  total: number;
  limit: number;
  offset: number;
  columns: string[];
}

export interface Stats {
  [key: string]: {
    min: number | null;
    max: number | null;
    mean: number | null;
    std: number | null;
  };
}

export interface ScoringCriteria {
  name: string;
  weight: number;
  min_value?: number;
  max_value?: number;
  invert?: boolean;
}

export interface ScoringRequest {
  dataset: string;
  filters: Filters;
  criteria: ScoringCriteria[];
  min_days?: number;
  months_back?: number;
  min_avg_volume?: number;
  min_price?: number;
  max_price?: number;
}

export interface CriterionScore {
  value: number;
  normalized: number;
  score: number;
}

export interface RankedTicker {
  ticker: string;
  total_score: number;
  metrics: Record<string, number>;
  criterion_scores: Record<string, CriterionScore>;
}

export interface ScoringResponse {
  ranked_tickers: RankedTicker[];
  total: number;
}

export interface ScoringPreset {
  id: string;
  name: string;
  description: string;
  criteria: ScoringCriteria[];
  monthsBack: number;
  minDays: number;
  minAvgVolume: number;
  minPrice?: number;
  maxPrice?: number;
  dataset?: string; // Optional dataset to use with this preset
  createdAt: string;
}

// Factor Evaluation Types
export interface FactorInfo {
  name: string;
  type: string;
}

export interface ICAnalysis {
  mean_ic: number;
  ic_std: number;
  t_stat: number;
  positive_pct: number;
}

export interface QuantileReturns {
  returns_by_quantile: Record<string, number>;
  spread: number;
}

export interface FactorEvaluationResponse {
  factor: string;
  ic_analysis: Record<string, ICAnalysis>;
  quantile_returns: Record<string, QuantileReturns>;
}

export interface FactorEvaluationRequest {
  factor_column: string;
  periods: number[];
  quantiles: number;
}

// Backtest Types
export interface BacktestRequest {
  strategy: 'macd' | 'rsi';
  ticker?: string;
  fast_period?: number;
  slow_period?: number;
  signal_period?: number;
  rsi_period?: number;
  oversold?: number;
  overbought?: number;
  initial_cash?: number;
  fees?: number;
  slippage?: number;
}

export interface BacktestPerformance {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  cagr?: number;
  sortino_ratio?: number;
  profit_factor?: number;
}

export interface BacktestTrades {
  total_trades: number;
  avg_trade_return?: number;
  best_trade?: number;
  worst_trade?: number;
}

export interface BacktestResult {
  strategy: string;
  parameters: Record<string, number>;
  performance: BacktestPerformance;
  trades: BacktestTrades;
}

export interface BacktestResponse {
  by_ticker?: Record<string, BacktestResult>;
  strategy?: string;
  parameters?: Record<string, number>;
  performance?: BacktestPerformance;
  trades?: BacktestTrades;
}

// Performance Analysis Types
export interface PerformanceSummary {
  total_return: number;
  total_return_pct: number;
  annualized_return: number;
  annualized_return_pct: number;
  annualized_volatility: number;
  annualized_volatility_pct: number;
  trading_days: number;
}

export interface RiskAdjusted {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
}

export interface DrawdownMetrics {
  max_drawdown_pct: number;
  max_drawdown_duration_days: number;
}

export interface TradeMetrics {
  win_rate_pct: number;
  profit_factor: number;
  best_day_pct: number;
  worst_day_pct: number;
  avg_daily_return_pct: number;
}

export interface PerformanceAnalysisResponse {
  ticker?: string;
  summary: PerformanceSummary;
  risk_adjusted: RiskAdjusted;
  drawdown: DrawdownMetrics;
  trade_metrics: TradeMetrics;
  benchmark_comparison?: {
    beta: number;
    alpha: number;
    alpha_pct: number;
    information_ratio: number;
    correlation: number;
    benchmark_sharpe: number;
  };
}

