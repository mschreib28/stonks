import { useState, useRef, useEffect } from 'react';
import type { ColumnInfo, Filters, Stats } from '../types';
import { getColumnDescription, getColumnDisplayName } from '../columnDescriptions';
import Tooltip from './Tooltip';

interface FilterBarProps {
  columns: ColumnInfo[];
  filters: Filters;
  stats: Stats;
  activeFilters: string[];
  onFilterChange: (filters: Filters) => void;
  onActiveFiltersChange: (activeFilters: string[]) => void;
}

export default function FilterBar({
  columns,
  filters,
  stats,
  activeFilters,
  onFilterChange,
  onActiveFiltersChange,
}: FilterBarProps) {
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [openDropdowns, setOpenDropdowns] = useState<Record<string, boolean>>({});
  const addMenuRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menus when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      
      if (addMenuRef.current && !addMenuRef.current.contains(target)) {
        setShowAddMenu(false);
        setSearchQuery('');
      }
      if (menuRef.current && !menuRef.current.contains(target)) {
        setShowMenu(false);
      }
      // Close filter dropdowns when clicking outside
      const isFilterDropdown = (target as Element).closest('.filter-dropdown');
      if (!isFilterDropdown) {
        setOpenDropdowns({});
      }
    };

    // Use capture phase to handle clicks before they bubble
    document.addEventListener('click', handleClickOutside, true);
    return () => document.removeEventListener('click', handleClickOutside, true);
  }, []);

  // Get column names that exist in current dataset
  const currentColumnNames = new Set(columns.map((col) => col.name));
  
  // Only show active filters that exist in current dataset
  const visibleActiveFilters = activeFilters.filter((f) => currentColumnNames.has(f));
  
  // Available filters are those in current dataset that aren't already active
  const availableFilters = columns.filter(
    (col) => !activeFilters.includes(col.name)
  );

  const filteredAvailableFilters = availableFilters.filter((col) => {
    const displayName = getColumnDisplayName(col.name);
    const searchLower = searchQuery.toLowerCase();
    return (
      col.name.toLowerCase().includes(searchLower) ||
      displayName.toLowerCase().includes(searchLower)
    );
  });

  const addFilter = (colName: string) => {
    onActiveFiltersChange([...activeFilters, colName]);
    setShowAddMenu(false);
    setSearchQuery('');
  };

  const removeFilter = (colName: string) => {
    const newActiveFilters = activeFilters.filter((f) => f !== colName);
    onActiveFiltersChange(newActiveFilters);
    const newFilters = { ...filters };
    delete newFilters[colName];
    onFilterChange(newFilters);
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

  const isFilterActive = (col: string): boolean => {
    const value = getFilterValue(col);
    if (value === null || value === undefined || value === '') return false;
    if (typeof value === 'object' && !Array.isArray(value)) {
      return Object.values(value).some(v => v !== null && v !== '');
    }
    return true;
  };

  const resetAllFilters = () => {
    onFilterChange({});
    setShowMenu(false);
  };

  const removeInactiveFilters = () => {
    // Only remove inactive filters that are visible in current dataset
    const inactive = visibleActiveFilters.filter((f) => !isFilterActive(f));
    const newActiveFilters = activeFilters.filter((f) => !inactive.includes(f));
    onActiveFiltersChange(newActiveFilters);
    const newFilters = { ...filters };
    inactive.forEach((f) => delete newFilters[f]);
    onFilterChange(newFilters);
    setShowMenu(false);
  };

  const removeAllFilters = () => {
    onActiveFiltersChange([]);
    onFilterChange({});
    setShowMenu(false);
  };

  const toggleDropdown = (colName: string) => {
    setOpenDropdowns((prev) => ({ ...prev, [colName]: !prev[colName] }));
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

  const inactiveCount = visibleActiveFilters.filter((f) => !isFilterActive(f)).length;

  return (
    <div className="bg-gray-800 dark:bg-gray-800 border-b border-gray-700 dark:border-gray-700 px-4 py-3">
      <div className="flex items-center gap-2 flex-wrap">
        {visibleActiveFilters.map((colName) => {
          const col = columns.find((c) => c.name === colName);
          if (!col) return null;
          
          const displayName = getColumnDisplayName(colName);
          const isActive = isFilterActive(colName);
          const isOpen = openDropdowns[colName];

          return (
            <div key={colName} className="relative filter-dropdown">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleDropdown(colName);
                }}
                className={`px-3 py-1.5 text-sm rounded-md flex items-center gap-2 transition-colors ${
                  isActive
                    ? 'bg-white dark:bg-white text-gray-900 dark:text-gray-900'
                    : 'bg-blue-900 dark:bg-blue-900 text-blue-100 dark:text-blue-100'
                } hover:opacity-80`}
              >
                <span>{displayName}</span>
                <svg
                  className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFilter(colName);
                  }}
                  className="ml-1 hover:opacity-70"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </button>

              {isOpen && (
                <div className="absolute top-full left-0 mt-1 z-50 bg-gray-800 dark:bg-gray-800 border border-gray-700 dark:border-gray-700 rounded-lg shadow-lg p-3 min-w-[300px]">
                  <div className="mb-2">
                    <div className="text-xs text-gray-400 dark:text-gray-400 mb-1">
                      {getColumnDescription(colName)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500">
                      Type: {col.type}
                    </div>
                  </div>
                  {renderFilterInput(col)}
                </div>
              )}
            </div>
          );
        })}

        {/* Add Filter Button */}
        <div className="relative" ref={addMenuRef}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowAddMenu(!showAddMenu);
            }}
            className="px-3 py-1.5 text-sm rounded-md bg-gray-700 dark:bg-gray-700 text-gray-300 dark:text-gray-300 hover:bg-gray-600 dark:hover:bg-gray-600 flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>Add Filter</span>
          </button>

          {showAddMenu && (
            <div className="absolute top-full left-0 mt-1 z-50 bg-gray-800 dark:bg-gray-800 border border-gray-700 dark:border-gray-700 rounded-lg shadow-lg min-w-[300px] max-h-[400px] flex flex-col">
              <div className="p-3 border-b border-gray-700 dark:border-gray-700">
                <input
                  type="text"
                  placeholder="Search filters..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white placeholder-gray-400 dark:placeholder-gray-400"
                  autoFocus
                />
              </div>
              <div className="overflow-y-auto max-h-[350px]">
                {filteredAvailableFilters.length === 0 ? (
                  <div className="p-4 text-center text-gray-400 dark:text-gray-400 text-sm">
                    {availableFilters.length === 0
                      ? 'All filters added'
                      : 'No filters match your search'}
                  </div>
                ) : (
                  filteredAvailableFilters.map((col) => {
                    const displayName = getColumnDisplayName(col.name);
                    return (
                      <button
                        key={col.name}
                        onClick={() => addFilter(col.name)}
                        className="w-full px-4 py-2 text-left text-sm text-gray-300 dark:text-gray-300 hover:bg-gray-700 dark:hover:bg-gray-700 flex items-center justify-between group"
                      >
                        <div>
                          <div className="font-medium">{displayName}</div>
                          <div className="text-xs text-gray-500 dark:text-gray-500">{col.name}</div>
                        </div>
                        <Tooltip
                          position="left"
                          content={
                            <>
                              <div className="font-semibold mb-1">{displayName}</div>
                              <div className="text-gray-300 dark:text-gray-300">{getColumnDescription(col.name)}</div>
                              <div className="mt-1 pt-1 border-t border-gray-700 dark:border-gray-700 text-gray-400 dark:text-gray-400">Type: {col.type}</div>
                            </>
                          }
                        >
                          <svg
                            className="w-4 h-4 text-gray-500 dark:text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                        </Tooltip>
                      </button>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </div>

        {/* Menu Button */}
        {visibleActiveFilters.length > 0 && (
          <div className="relative" ref={menuRef}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className="px-3 py-1.5 text-sm rounded-md bg-gray-700 dark:bg-gray-700 text-gray-300 dark:text-gray-300 hover:bg-gray-600 dark:hover:bg-gray-600"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
              </svg>
            </button>

            {showMenu && (
              <div className="absolute top-full right-0 mt-1 z-50 bg-gray-800 dark:bg-gray-800 border border-gray-700 dark:border-gray-700 rounded-lg shadow-lg min-w-[200px] py-1">
                <button
                  onClick={resetAllFilters}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 dark:text-gray-300 hover:bg-gray-700 dark:hover:bg-gray-700"
                >
                  Reset all filters
                </button>
                {inactiveCount > 0 && (
                  <button
                    onClick={removeInactiveFilters}
                    className="w-full px-4 py-2 text-left text-sm text-gray-300 dark:text-gray-300 hover:bg-gray-700 dark:hover:bg-gray-700"
                  >
                    Remove {inactiveCount} inactive filter{inactiveCount !== 1 ? 's' : ''}
                  </button>
                )}
                <button
                  onClick={removeAllFilters}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 dark:text-gray-300 hover:bg-gray-700 dark:hover:bg-gray-700"
                >
                  Remove all filters
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
