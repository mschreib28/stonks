import axios from 'axios';
import type { 
  Dataset, ColumnInfo, QueryRequest, QueryResponse, Stats, ScoringRequest, ScoringResponse,
  FactorInfo, FactorEvaluationRequest, FactorEvaluationResponse,
  BacktestRequest, BacktestResponse,
  PerformanceAnalysisResponse,
} from './types';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const listDatasets = async (): Promise<Dataset[]> => {
  const response = await api.get('/datasets');
  return response.data.datasets;
};

export const getColumns = async (dataset: string): Promise<ColumnInfo[]> => {
  const response = await api.get(`/dataset/${dataset}/columns`);
  return response.data.columns;
};

export const queryData = async (request: QueryRequest): Promise<QueryResponse> => {
  const response = await api.post('/query', request);
  return response.data;
};

export const getStats = async (dataset: string): Promise<Stats> => {
  const response = await api.get(`/dataset/${dataset}/stats`);
  return response.data.stats;
};

export const getTickerData = async (dataset: string, ticker: string, limit: number = 100) => {
  const response = await api.get(`/dataset/${dataset}/ticker/${ticker}`, {
    params: { limit },
  });
  return response.data;
};

export const scoreTickers = async (request: ScoringRequest): Promise<ScoringResponse> => {
  const response = await api.post('/score-tickers', request);
  return response.data;
};

// Fast scoring using DuckDB (sub-second response on 636M+ rows)
export interface FastScoringRequest {
  months_back?: number;
  min_days?: number;
  min_avg_volume?: number;
  min_price?: number | null;
  max_price?: number | null;
  limit?: number;
}

export interface FastScoringResponse extends ScoringResponse {
  query_time_seconds?: number;
  source?: string;
}

export const scoreTickersFast = async (request: FastScoringRequest = {}): Promise<FastScoringResponse> => {
  const response = await api.post('/score-tickers-fast', request);
  return response.data;
};

// DuckDB status check
export interface DuckDBStatus {
  available: boolean;
  database_path?: string;
  database_size_mb?: number;
  tables?: Record<string, { rows: number }>;
  message?: string;
  error?: string;
}

export const getDuckDBStatus = async (): Promise<DuckDBStatus> => {
  const response = await api.get('/duckdb-status');
  return response.data;
};

// Fast ticker search
export const searchTickers = async (
  query: string, 
  minVolume?: number,
  minPrice?: number,
  maxPrice?: number,
  limit?: number
): Promise<{ results: Array<{ ticker: string; latest_close: number; avg_volume: number }>; total: number }> => {
  const response = await api.get('/ticker-search', {
    params: { 
      q: query, 
      min_volume: minVolume || 0,
      min_price: minPrice,
      max_price: maxPrice,
      limit: limit || 20,
    },
  });
  return response.data;
};

export interface DataFreshnessResponse {
  is_fresh: boolean;
  latest_data_date: string | null;
  expected_date: string;
  message: string;
}

export interface UpdateStatusResponse {
  status: 'idle' | 'checking' | 'downloading' | 'processing' | 'completed' | 'error';
  message: string;
  progress: number;
}

export const checkDataFreshness = async (forceReload: boolean = false): Promise<DataFreshnessResponse> => {
  const response = await api.get('/data-freshness', {
    params: { force_reload: forceReload }
  });
  return response.data;
};

export const getUpdateStatus = async (): Promise<UpdateStatusResponse> => {
  const response = await api.get('/update-status');
  return response.data;
};

export const triggerDataUpdate = async (): Promise<{ status: string; message: string }> => {
  const response = await api.post('/update-data');
  return response.data;
};

// Factor Evaluation API
export const getAvailableFactors = async (): Promise<{ factors: FactorInfo[] }> => {
  const response = await api.get('/available-factors');
  return response.data;
};

export const evaluateFactor = async (request: FactorEvaluationRequest): Promise<FactorEvaluationResponse> => {
  const response = await api.post('/evaluate-factor', request);
  return response.data;
};

// Backtest API
export const runBacktest = async (request: BacktestRequest): Promise<BacktestResponse> => {
  const response = await api.post('/backtest', request);
  return response.data;
};

// Performance Analysis API
export const analyzePerformance = async (ticker?: string, benchmark?: string): Promise<PerformanceAnalysisResponse> => {
  const response = await api.post('/performance-analysis', { ticker, benchmark_ticker: benchmark });
  return response.data;
};

export const getTickerPerformance = async (ticker: string): Promise<PerformanceAnalysisResponse> => {
  const response = await api.get(`/ticker/${ticker}/performance`);
  return response.data;
};
