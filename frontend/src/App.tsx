import { useState, useEffect, useCallback } from 'react';
import DataTable from './components/DataTable';
import FilterBar from './components/FilterBar';
import ChartPanel from './components/ChartPanel';
import DatasetSelector from './components/DatasetSelector';
import ScoringPanel from './components/ScoringPanel';
import DataUpdateButton from './components/DataUpdateButton';
import FactorEvaluationPanel from './components/FactorEvaluationPanel';
import PerformancePanel from './components/PerformancePanel';
import { listDatasets, getColumns, queryData, getStats, scoreTickers, scoreTickersFast, getDuckDBStatus } from './api';
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
  const [monthsBack, setMonthsBack] = useState<number>(3);  // Default to 3 months for faster scoring
  const [minDays, setMinDays] = useState<number>(40);  // ~40 trading days in 3 months
  const [minAvgVolume, setMinAvgVolume] = useState<number>(10000);
  const [minPrice, setMinPrice] = useState<number | undefined>(undefined);
  const [maxPrice, setMaxPrice] = useState<number | undefined>(undefined);
  const [rankedTickers, setRankedTickers] = useState<RankedTicker[]>([]);
  const [scoring, setScoring] = useState<boolean>(false);
  const [useRankedResults, setUseRankedResults] = useState<boolean>(false);

  // Load state from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    
    // Restore dataset
    const urlDataset = params.get('dataset');
    if (urlDataset) {
      setSelectedDataset(urlDataset);
    }
    
    // Restore ticker
    const urlTicker = params.get('ticker');
    if (urlTicker) {
      setSelectedTicker(urlTicker);
    }
    
    // Restore filters
    const urlFilters = params.get('filters');
    if (urlFilters) {
      try {
        const parsedFilters = JSON.parse(decodeURIComponent(urlFilters));
        setFilters(parsedFilters);
      } catch (e) {
        console.error('Failed to parse filters from URL:', e);
      }
    }
    
    // Restore active filters
    const urlActiveFilters = params.get('activeFilters');
    if (urlActiveFilters) {
      try {
        const parsed = JSON.parse(decodeURIComponent(urlActiveFilters));
        setActiveFilters(parsed);
      } catch (e) {
        console.error('Failed to parse activeFilters from URL:', e);
      }
    }
    
    // Restore page
    const urlPage = params.get('page');
    if (urlPage) {
      setPage(parseInt(urlPage, 10));
    }
    
    // Restore page size
    const urlPageSize = params.get('pageSize');
    if (urlPageSize) {
      setPageSize(parseInt(urlPageSize, 10));
    }
    
    // Restore sort
    const urlSortBy = params.get('sortBy');
    if (urlSortBy) {
      setSortBy(urlSortBy);
    }
    const urlSortDesc = params.get('sortDesc');
    if (urlSortDesc) {
      setSortDesc(urlSortDesc === 'true');
    }
    
    loadDatasets();
  }, []);

  // Update URL when state changes
  useEffect(() => {
    const params = new URLSearchParams();
    
    if (selectedDataset) {
      params.set('dataset', selectedDataset);
    }
    if (selectedTicker) {
      params.set('ticker', selectedTicker);
    }
    if (Object.keys(filters).length > 0) {
      params.set('filters', encodeURIComponent(JSON.stringify(filters)));
    }
    if (activeFilters.length > 0) {
      params.set('activeFilters', encodeURIComponent(JSON.stringify(activeFilters)));
    }
    if (page > 0) {
      params.set('page', page.toString());
    }
    if (pageSize !== 100) {
      params.set('pageSize', pageSize.toString());
    }
    if (sortBy) {
      params.set('sortBy', sortBy);
    }
    if (sortDesc) {
      params.set('sortDesc', sortDesc.toString());
    }
    
    const newUrl = params.toString() ? `?${params.toString()}` : window.location.pathname;
    window.history.replaceState({}, '', newUrl);
  }, [selectedDataset, selectedTicker, filters, activeFilters, page, pageSize, sortBy, sortDesc]);

  const loadDatasets = async () => {
    try {
      const ds = await listDatasets();
      setDatasets(ds);
      if (ds.length > 0) {
        // If no dataset is selected, or the selected dataset doesn't exist, choose one
        if (!selectedDataset || !ds.find(d => d.name === selectedDataset)) {
          // Prefer 'daily' dataset if it exists (includes all 2025 + 2026 data), otherwise use the first one
          const dailyDataset = ds.find(d => d.name === 'daily');
          setSelectedDataset(dailyDataset ? dailyDataset.name : ds[0].name);
        }
      }
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadColumns = async () => {
    try {
      setColumns([]); // Clear columns first
      const cols = await getColumns(selectedDataset);
      setColumns(cols);
      
      // Clean up filters that don't exist in the new dataset
      if (cols.length > 0) {
        const columnNames = new Set(cols.map((col) => col.name));
        setActiveFilters((prev) => prev.filter((f) => columnNames.has(f)));
        setFilters((prev) => {
          const validFilters: Filters = {};
          Object.entries(prev).forEach(([key, value]) => {
            if (columnNames.has(key)) {
              validFilters[key] = value;
            }
          });
          return validFilters;
        });
      }
    } catch (error) {
      console.error('Failed to load columns:', error);
      setColumns([]); // Clear on error
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

  const loadData = useCallback(async () => {
    if (!selectedDataset || columns.length === 0) {
      return; // Don't load if dataset or columns aren't ready
    }
    
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
      setData(null); // Clear data on error
    } finally {
      setLoading(false);
    }
  }, [selectedDataset, columns, filters, sortBy, sortDesc, page, pageSize]);

  useEffect(() => {
    if (selectedDataset) {
      // Clear columns and data when dataset changes to prevent stale data
      setColumns([]);
      setData(null);
      loadColumns();
      loadStats();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDataset]); // loadColumns and loadStats are stable functions that use selectedDataset


  useEffect(() => {
    if (selectedDataset && columns.length > 0 && !useRankedResults) {
      loadData();
    }
  }, [selectedDataset, columns, filters, sortBy, sortDesc, page, pageSize, useRankedResults, loadData]);

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
      // Try fast DuckDB-based scoring first (sub-second response)
      // Falls back to slow Python-based scoring if DuckDB unavailable
      let response;
      let usedFastScoring = false;
      
      try {
        // Check if DuckDB is available
        const duckdbStatus = await getDuckDBStatus();
        
        if (duckdbStatus.available) {
          console.log('Using fast DuckDB scoring...');
          const fastResponse = await scoreTickersFast({
            months_back: monthsBack,
            min_days: minDays,
            min_avg_volume: minAvgVolume,
            min_price: minPrice || undefined,
            max_price: maxPrice || undefined,
            limit: 100,  // Return top 100 tickers
          });
          response = fastResponse;
          usedFastScoring = true;
          console.log(`Fast scoring completed in ${fastResponse.query_time_seconds?.toFixed(3)}s, returned ${fastResponse.total} tickers`);
        }
      } catch (fastError) {
        console.warn('Fast scoring failed, falling back to standard scoring:', fastError);
      }
      
      // Fall back to standard scoring if fast scoring failed or unavailable
      if (!usedFastScoring) {
        console.log('Using standard Python-based scoring (this may take several minutes)...');
        
        // Only apply filters that exist in the current dataset columns
        const columnNames = new Set(columns.map((col) => col.name));
        const applicableFilters: Filters = {};
        Object.entries(filters).forEach(([key, value]) => {
          if (columnNames.has(key)) {
            applicableFilters[key] = value;
          }
        });

        response = await scoreTickers({
          dataset: selectedDataset,
          filters: applicableFilters,
          criteria: scoringCriteria,
          months_back: monthsBack,
          min_days: minDays,
          min_avg_volume: minAvgVolume,
          min_price: minPrice,
          max_price: maxPrice,
        });
      }
      
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
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white dark:text-white">Stonks Data Explorer</h1>
              <p className="text-sm text-gray-400 dark:text-gray-400 mt-1">Explore and analyze stock market data</p>
            </div>
            <DataUpdateButton />
          </div>
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
              onDatasetChange={setSelectedDataset}
            />

            <FactorEvaluationPanel />

            <PerformancePanel ticker={selectedTicker || undefined} />
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

