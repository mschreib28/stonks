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