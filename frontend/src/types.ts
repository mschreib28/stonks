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
  createdAt: string;
}

