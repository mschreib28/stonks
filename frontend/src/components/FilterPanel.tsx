import { useState } from 'react';
import type { ColumnInfo, Filters, Stats } from '../types';
import { getColumnDescription, getColumnDisplayName } from '../columnDescriptions';
import Tooltip from './Tooltip';

interface FilterPanelProps {
  columns: ColumnInfo[];
  filters: Filters;
  stats: Stats;
  onFilterChange: (filters: Filters) => void;
}

export default function FilterPanel({ columns, filters, stats, onFilterChange }: FilterPanelProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggleExpanded = (col: string) => {
    setExpanded((prev) => ({ ...prev, [col]: !prev[col] }));
  };

  const updateFilter = (col: string, value: any) => {
    const newFilters = { ...filters };
    if (value === null || value === '' || (typeof value === 'object' && Object.values(value).every(v => v === null || v === ''))) {
      delete newFilters[col];
    } else {
      newFilters[col] = value;
    }
    onFilterChange(newFilters);
  };

  const getFilterValue = (col: string): any => {
    return filters[col] || null;
  };

  const renderFilterInput = (col: ColumnInfo) => {
    const colType = col.type.toLowerCase();
    const filterValue = getFilterValue(col.name);
    const colStats = stats[col.name];

    if (colType.includes('bool')) {
      const value = filterValue?.value ?? filterValue ?? null;
      return (
        <select
          value={value === null ? '' : String(value)}
          onChange={(e) => {
            const val = e.target.value === '' ? null : e.target.value === 'true';
            updateFilter(col.name, val);
          }}
          className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
        >
          <option value="">All</option>
          <option value="true">True</option>
          <option value="false">False</option>
        </select>
      );
    }

    if (colType.includes('int') || colType.includes('float')) {
      const rangeValue = typeof filterValue === 'object' && filterValue !== null && !Array.isArray(filterValue)
        ? filterValue
        : { min: null, max: null };
      
      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <input
              type="number"
              placeholder="Min"
              value={rangeValue.min ?? ''}
              onChange={(e) => {
                const val = e.target.value === '' ? null : parseFloat(e.target.value);
                updateFilter(col.name, { ...rangeValue, min: val });
              }}
              className="flex-1 px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
              step={colType.includes('int') ? 1 : 0.01}
            />
            <span className="text-gray-400 dark:text-gray-400">-</span>
            <input
              type="number"
              placeholder="Max"
              value={rangeValue.max ?? ''}
              onChange={(e) => {
                const val = e.target.value === '' ? null : parseFloat(e.target.value);
                updateFilter(col.name, { ...rangeValue, max: val });
              }}
              className="flex-1 px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
              step={colType.includes('int') ? 1 : 0.01}
            />
          </div>
          {colStats && (
            <div className="text-xs text-gray-400 dark:text-gray-400">
              Range: {colStats.min !== null ? colStats.min.toFixed(2) : 'N/A'} - {colStats.max !== null ? colStats.max.toFixed(2) : 'N/A'}
            </div>
          )}
        </div>
      );
    }

    // String filter
    const stringValue = typeof filterValue === 'string' ? filterValue : '';
    return (
      <input
        type="text"
        placeholder="Filter..."
        value={stringValue}
        onChange={(e) => updateFilter(col.name, e.target.value || null)}
        className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white placeholder-gray-400 dark:placeholder-gray-400"
      />
    );
  };

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="flex items-center justify-between flex-shrink-0">
        <h2 className="text-lg font-semibold text-white dark:text-white">Filters</h2>
        <button
          onClick={() => onFilterChange({})}
          className="text-sm text-blue-400 dark:text-blue-400 hover:text-blue-300 dark:hover:text-blue-300"
        >
          Clear All
        </button>
      </div>

      <div className="space-y-3 overflow-y-auto min-h-0">
        {columns.map((col) => {
          const isExpanded = expanded[col.name] ?? false;
          const hasFilter = col.name in filters;
          const description = getColumnDescription(col.name);
          const displayName = getColumnDisplayName(col.name);

          return (
            <div key={col.name} className="border-b border-gray-700 dark:border-gray-700 pb-2">
              <div className="flex items-start gap-2">
                <button
                  onClick={() => toggleExpanded(col.name)}
                  className="flex-1 flex items-center justify-between text-left group"
                >
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium ${hasFilter ? 'text-blue-400 dark:text-blue-400' : 'text-gray-300 dark:text-gray-300'}`}>
                      {displayName}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-500 group-hover:text-gray-400 dark:group-hover:text-gray-400">({col.name})</span>
                  </div>
                  <span className="text-xs text-gray-400 dark:text-gray-400">{col.type}</span>
                </button>
                <div className="flex-shrink-0">
                  <Tooltip
                    position="left"
                    content={
                      <>
                        <div className="font-semibold mb-1">{displayName}</div>
                        <div className="text-gray-300 dark:text-gray-300">{description}</div>
                        <div className="mt-1 pt-1 border-t border-gray-700 dark:border-gray-700 text-gray-400 dark:text-gray-400">Type: {col.type}</div>
                      </>
                    }
                  >
                    <svg
                      className="w-4 h-4 text-gray-500 dark:text-gray-500 hover:text-blue-400 dark:hover:text-blue-400 cursor-help"
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
                  </Tooltip>
                </div>
              </div>
              
              {isExpanded && (
                <div className="mt-2">
                  <div className="text-xs text-gray-400 dark:text-gray-400 mb-2 italic">{description}</div>
                  {renderFilterInput(col)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

