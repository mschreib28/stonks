import { useState, useEffect } from 'react';
import DataTable from './components/DataTable';
import FilterBar from './components/FilterBar';
import ChartPanel from './components/ChartPanel';
import DatasetSelector from './components/DatasetSelector';
import ScoringPanel from './components/ScoringPanel';
import { listDatasets, getColumns, queryData, getStats, scoreTickers } from './api';
import type { Dataset, ColumnInfo, QueryRequest, QueryResponse, Filters, Stats, ScoringCriteria, RankedTicker } from './types';

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('daily');
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [filters, setFilters] = useState<Filters>({});
  const [activeFilters, setActiveFilters] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<string>('');
  const [sortDesc, setSortDesc] = useState<boolean>(false);
  const [data, setData] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [stats, setStats] = useState<Stats>({});
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [page, setPage] = useState<number>(0);
  const [pageSize, setPageSize] = useState<number>(100);
  const [scoringCriteria, setScoringCriteria] = useState<ScoringCriteria[]>([]);
  const [monthsBack, setMonthsBack] = useState<number>(12);
  const [minDays, setMinDays] = useState<number>(60);
  const [minAvgVolume, setMinAvgVolume] = useState<number>(750000);
  const [minPrice, setMinPrice] = useState<number | undefined>(undefined);
  const [maxPrice, setMaxPrice] = useState<number | undefined>(undefined);
  const [rankedTickers, setRankedTickers] = useState<RankedTicker[]>([]);
  const [scoring, setScoring] = useState<boolean>(false);
  const [useRankedResults, setUseRankedResults] = useState<boolean>(false);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadColumns();
      loadStats();
    }
  }, [selectedDataset]);

  useEffect(() => {
    if (selectedDataset && columns.length > 0 && !useRankedResults) {
      loadData();
    }
  }, [selectedDataset, filters, sortBy, sortDesc, page, pageSize, useRankedResults]);

  const loadDatasets = async () => {
    try {
      const ds = await listDatasets();
      setDatasets(ds);
      if (ds.length > 0 && !selectedDataset) {
        setSelectedDataset(ds[0].name);
      }
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadColumns = async () => {
    try {
      const cols = await getColumns(selectedDataset);
      setColumns(cols);
    } catch (error) {
      console.error('Failed to load columns:', error);
    }
  };

  const loadStats = async () => {
    try {
      const s = await getStats(selectedDataset);
      setStats(s);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const loadData = async () => {
    setLoading(true);
    try {
      // Only apply filters that exist in the current dataset columns
      const columnNames = new Set(columns.map((col) => col.name));
      const applicableFilters: Filters = {};
      Object.entries(filters).forEach(([key, value]) => {
        if (columnNames.has(key)) {
          applicableFilters[key] = value;
        }
      });

      const request: QueryRequest = {
        dataset: selectedDataset,
        filters: applicableFilters,
        sort_by: sortBy || undefined,
        sort_desc: sortDesc,
        limit: pageSize,
        offset: page * pageSize,
      };
      const response = await queryData(request);
      setData(response);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (newFilters: Filters) => {
    setFilters(newFilters);
    setPage(0); // Reset to first page when filters change
  };

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortDesc(!sortDesc);
    } else {
      setSortBy(column);
      setSortDesc(false);
    }
    setPage(0);
  };

  const handleTickerClick = (ticker: string) => {
    setSelectedTicker(ticker);
  };

  const handleScoreTickers = async () => {
    if (scoringCriteria.length === 0) {
      return;
    }

    setScoring(true);
    try {
      // Only apply filters that exist in the current dataset columns
      const columnNames = new Set(columns.map((col) => col.name));
      const applicableFilters: Filters = {};
      Object.entries(filters).forEach(([key, value]) => {
        if (columnNames.has(key)) {
          applicableFilters[key] = value;
        }
      });

      const response = await scoreTickers({
        dataset: selectedDataset,
        filters: applicableFilters,
        criteria: scoringCriteria,
        months_back: monthsBack,
        min_days: minDays,
        min_avg_volume: minAvgVolume,
        min_price: minPrice,
        max_price: maxPrice,
      });
      setRankedTickers(response.ranked_tickers);
      setUseRankedResults(true);
      setPage(0);
    } catch (error) {
      console.error('Failed to score tickers:', error);
    } finally {
      setScoring(false);
    }
  };

  const handleUseNormalView = () => {
    setUseRankedResults(false);
    setPage(0);
  };

  return (
    <div className="min-h-screen bg-gray-900 dark:bg-gray-900">
      <header className="bg-gray-800 dark:bg-gray-800 shadow-sm border-b border-gray-700 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-3xl font-bold text-white dark:text-white">Stonks Data Explorer</h1>
          <p className="text-sm text-gray-400 dark:text-gray-400 mt-1">Explore and analyze stock market data</p>
        </div>
      </header>

      <FilterBar
        columns={columns}
        filters={filters}
        stats={stats}
        activeFilters={activeFilters}
        onFilterChange={handleFilterChange}
        onActiveFiltersChange={setActiveFilters}
      />

      <div className="w-full px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 max-w-xs flex flex-col gap-4">
            <div className="bg-gray-800 dark:bg-gray-800 rounded-lg shadow p-4 space-y-4 flex flex-col border border-gray-700 dark:border-gray-700">
              <DatasetSelector
                datasets={datasets}
                selectedDataset={selectedDataset}
                onSelect={setSelectedDataset}
              />
            </div>

            <ScoringPanel
              criteria={scoringCriteria}
              monthsBack={monthsBack}
              minDays={minDays}
              minAvgVolume={minAvgVolume}
              minPrice={minPrice}
              maxPrice={maxPrice}
              onCriteriaChange={setScoringCriteria}
              onMonthsBackChange={setMonthsBack}
              onMinDaysChange={setMinDays}
              onMinAvgVolumeChange={setMinAvgVolume}
              onMinPriceChange={setMinPrice}
              onMaxPriceChange={setMaxPrice}
              onScore={handleScoreTickers}
              scoring={scoring}
            />
          </div>

          {/* Main Content */}
          <div className="lg:col-span-4 space-y-6 min-w-0">
            {selectedTicker && (
              <ChartPanel
                dataset={selectedDataset}
                ticker={selectedTicker}
                onClose={() => setSelectedTicker(null)}
              />
            )}

            {useRankedResults && (
              <div className="bg-blue-900 border border-blue-700 rounded-lg p-4 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-blue-100">Showing Ranked Results</h3>
                  <p className="text-xs text-blue-300 mt-1">
                    {rankedTickers.length} tickers ranked by scoring criteria
                  </p>
                </div>
                <button
                  onClick={handleUseNormalView}
                  className="px-4 py-2 text-sm bg-blue-800 border border-blue-600 text-blue-100 rounded hover:bg-blue-700"
                >
                  Back to Normal View
                </button>
              </div>
            )}

            <div className="bg-gray-800 dark:bg-gray-800 rounded-lg shadow border border-gray-700 dark:border-gray-700">
              <DataTable
                data={useRankedResults ? {
                  data: rankedTickers.slice(page * pageSize, (page + 1) * pageSize).map(t => ({
                    ...t.metrics,
                    ticker: t.ticker,
                    total_score: t.total_score,
                  })),
                  total: rankedTickers.length,
                  limit: pageSize,
                  offset: page * pageSize,
                  columns: useRankedResults && rankedTickers.length > 0 ? ['ticker', 'total_score', ...Object.keys(rankedTickers[0]?.metrics || {})] : (data?.columns || []),
                } : data}
                columns={columns}
                loading={loading && !useRankedResults}
                sortBy={sortBy}
                sortDesc={sortDesc}
                onSort={handleSort}
                onTickerClick={handleTickerClick}
                page={page}
                pageSize={pageSize}
                onPageChange={setPage}
                onPageSizeChange={setPageSize}
                rankedResults={useRankedResults ? rankedTickers : undefined}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

