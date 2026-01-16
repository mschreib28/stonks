import axios from 'axios';
import type { Dataset, ColumnInfo, QueryRequest, QueryResponse, Stats, ScoringRequest, ScoringResponse } from './types';

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
