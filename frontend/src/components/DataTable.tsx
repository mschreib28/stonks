import { useMemo } from 'react';
import type { ColumnInfo, QueryResponse, RankedTicker } from '../types';
import { getColumnDescription, getColumnDisplayName } from '../columnDescriptions';

interface DataTableProps {
  data: QueryResponse | null;
  columns: ColumnInfo[];
  loading: boolean;
  sortBy: string;
  sortDesc: boolean;
  onSort: (column: string) => void;
  onTickerClick: (ticker: string) => void;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  rankedResults?: RankedTicker[];
}

export default function DataTable({
  data,
  columns,
  loading,
  sortBy,
  sortDesc,
  onSort,
  onTickerClick,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
  rankedResults,
}: DataTableProps) {
  const displayColumns = useMemo(() => {
    if (!data || data.columns.length === 0) {
      return columns.map((col) => col.name);
    }
    return data.columns;
  }, [data, columns]);

  const formatValue = (value: any, colName: string): string => {
    if (value === null || value === undefined) {
      return '—';
    }
    
    // Special formatting for total_score
    if (colName === 'total_score') {
      return typeof value === 'number' ? value.toFixed(2) : String(value);
    }
    
    // Integer-like metrics that should NOT show decimals
    const integerMetrics = [
      'days_analyzed', 'year', 'month', 'week', 'volume', 'avg_volume', 
      'max_volume', 'min_volume', 'sweet_spot_range_days', 'large_swing_days'
    ];
    
    // Metrics that should show as whole numbers with commas
    const largeNumberMetrics = ['avg_volume', 'max_volume', 'min_volume', 'volume'];
    
    const col = columns.find((c) => c.name === colName);
    if (!col) {
      // For metrics columns that might not be in columns list
      if (typeof value === 'number') {
        // Check for integer metrics
        if (integerMetrics.includes(colName)) {
          return largeNumberMetrics.includes(colName) 
            ? Math.round(value).toLocaleString() 
            : Math.round(value).toString();
        }
        // Check if it's a percentage-like value
        if (colName.includes('pct') || colName.includes('return') || colName.includes('swing')) {
          return value.toFixed(2) + '%';
        }
        // Dollar amounts
        if (colName.includes('dollars') || colName.includes('price')) {
          return '$' + value.toFixed(2);
        }
        return value.toFixed(2);
      }
      return String(value);
    }
    
    const colType = col.type.toLowerCase();
    
    if (colType.includes('float') || colType.includes('float64')) {
      return typeof value === 'number' ? value.toFixed(2) : String(value);
    }
    
    if (colType.includes('int') || colType.includes('int64')) {
      return typeof value === 'number' ? value.toLocaleString() : String(value);
    }
    
    if (colType.includes('bool')) {
      return value ? '✓' : '✗';
    }
    
    if (colType.includes('date')) {
      return new Date(value).toLocaleDateString();
    }
    
    return String(value);
  };

  const SortIcon = ({ column }: { column: string }) => {
    if (sortBy !== column) {
      return <span className="text-gray-400 dark:text-gray-500">↕</span>;
    }
    return sortDesc ? <span className="text-blue-400 dark:text-blue-400">↓</span> : <span className="text-blue-400 dark:text-blue-400">↑</span>;
  };

  if (loading) {
    return (
      <div className="p-8 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400 dark:border-blue-400"></div>
        <p className="mt-2 text-gray-400 dark:text-gray-400">Loading data...</p>
      </div>
    );
  }

  if (!data || data.data.length === 0) {
    return (
      <div className="p-8 text-center text-gray-400 dark:text-gray-400">
        No data available. Try adjusting your filters.
      </div>
    );
  }

  const totalPages = Math.ceil(data.total / pageSize);

  return (
    <div className="overflow-x-auto w-full">
      <div className="p-4 border-b border-gray-700 dark:border-gray-700 flex items-center justify-between">
        <div className="text-sm text-gray-300 dark:text-gray-300">
          Showing {page * pageSize + 1} - {Math.min((page + 1) * pageSize, data.total)} of {data.total.toLocaleString()} rows
          {rankedResults && (
            <span className="ml-2 text-blue-400 dark:text-blue-400">
              (Ranked by score)
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-300 dark:text-gray-300">Page size:</label>
          <select
            value={pageSize}
            onChange={(e) => {
              onPageSizeChange(Number(e.target.value));
              onPageChange(0);
            }}
            className="px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
          >
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={250}>250</option>
            <option value={500}>500</option>
          </select>
        </div>
      </div>

      <table className="w-full divide-y divide-gray-700 dark:divide-gray-700" style={{ minWidth: 'max-content' }}>
        <thead className="bg-gray-800 dark:bg-gray-800">
          <tr>
            {displayColumns.map((col) => {
              const displayName = col === 'total_score' ? 'Total Score' : getColumnDisplayName(col);
              const description = col === 'total_score' ? 'Weighted score based on all criteria' : getColumnDescription(col);
              const isScoreColumn = col === 'total_score';
              return (
                <th
                  key={col}
                  onClick={() => !isScoreColumn && onSort(col)}
                  className={`px-4 py-3 text-left text-xs font-medium text-gray-300 dark:text-gray-300 uppercase tracking-wider select-none ${
                    isScoreColumn ? '' : 'cursor-pointer hover:bg-gray-700 dark:hover:bg-gray-700'
                  } ${isScoreColumn ? 'bg-blue-900 dark:bg-blue-900' : ''}`}
                >
                  <div className="flex items-center gap-2 group/header">
                    <span>{displayName}</span>
                    <SortIcon column={col} />
                    <div className="relative group/tooltip">
                      <svg
                        className="w-3 h-3 text-gray-400 dark:text-gray-500 hover:text-blue-400 dark:hover:text-blue-400 cursor-help opacity-0 group-hover/header:opacity-100 transition-opacity"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      <div className="absolute left-0 top-6 z-50 w-64 p-2 bg-gray-900 dark:bg-gray-900 text-white dark:text-white text-xs rounded shadow-lg opacity-0 invisible group-hover/tooltip:opacity-100 group-hover/tooltip:visible transition-all duration-200 pointer-events-none border border-gray-700 dark:border-gray-700">
                        <div className="font-semibold mb-1">{displayName}</div>
                        <div className="text-gray-300 dark:text-gray-300">{description}</div>
                      </div>
                    </div>
                  </div>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody className="bg-gray-800 dark:bg-gray-800 divide-y divide-gray-700 dark:divide-gray-700">
          {data.data.map((row, idx) => {
            const isTopRanked = rankedResults && idx < 10; // Highlight top 10
            return (
              <tr 
                key={idx} 
                className={`hover:bg-gray-700 dark:hover:bg-gray-700 ${
                  isTopRanked ? 'bg-blue-900/20 dark:bg-blue-900/20' : ''
                }`}
              >
                {displayColumns.map((col) => {
                  const value = row[col];
                  const isTicker = col === 'ticker';
                  const isScore = col === 'total_score';
                  return (
                    <td
                      key={col}
                      className={`px-4 py-3 whitespace-nowrap text-sm ${
                        isTicker 
                          ? 'font-medium text-blue-400 dark:text-blue-400 cursor-pointer hover:underline' 
                          : isScore
                          ? 'font-semibold text-green-400 dark:text-green-400'
                          : 'text-gray-200 dark:text-gray-200'
                      }`}
                      onClick={isTicker ? () => onTickerClick(String(value)) : undefined}
                    >
                      {formatValue(value, col)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>

      {totalPages > 1 && (
        <div className="p-4 border-t border-gray-700 dark:border-gray-700 flex items-center justify-between">
          <button
            onClick={() => onPageChange(page - 1)}
            disabled={page === 0}
            className="px-4 py-2 text-sm font-medium text-gray-200 dark:text-gray-200 bg-gray-700 dark:bg-gray-700 border border-gray-600 dark:border-gray-600 rounded-md hover:bg-gray-600 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="text-sm text-gray-200 dark:text-gray-200">
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={page >= totalPages - 1}
            className="px-4 py-2 text-sm font-medium text-gray-200 dark:text-gray-200 bg-gray-700 dark:bg-gray-700 border border-gray-600 dark:border-gray-600 rounded-md hover:bg-gray-600 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

