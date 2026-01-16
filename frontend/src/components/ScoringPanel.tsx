import { useState, useEffect, useRef } from 'react';
import type { ScoringCriteria, ScoringPreset } from '../types';

interface ScoringPanelProps {
  criteria: ScoringCriteria[];
  monthsBack: number;
  minDays: number;
  minAvgVolume: number;
  minPrice: number | undefined;
  maxPrice: number | undefined;
  onCriteriaChange: (criteria: ScoringCriteria[]) => void;
  onMonthsBackChange: (months: number) => void;
  onMinDaysChange: (days: number) => void;
  onMinAvgVolumeChange: (volume: number) => void;
  onMinPriceChange: (price: number | undefined) => void;
  onMaxPriceChange: (price: number | undefined) => void;
  onScore: () => void;
  scoring: boolean;
}

const AVAILABLE_METRICS = [
  // Trend metrics
  { name: 'trend_pct', label: 'Price Trend % (6-12mo)', description: 'Overall price trend percentage over the period' },
  { name: 'total_return_pct', label: 'Total Return %', description: 'Total return percentage over the period' },
  
  // Volatility metrics
  { name: 'avg_daily_swing_pct', label: 'Avg Daily Swing %', description: 'Average daily price swing percentage' },
  { name: 'large_swing_frequency', label: 'Large Swing Frequency', description: 'Frequency of days with >=20% swings' },
  { name: 'volatility', label: 'Volatility', description: 'Standard deviation of daily returns' },
  
  // Volume metrics
  { name: 'avg_volume', label: 'Average Volume', description: 'Average daily trading volume' },
  { name: 'liquidity_multiple', label: 'Liquidity Multiple', description: 'How many times 10k shares fits in daily volume (higher = more liquid)' },
  { name: 'position_pct_of_volume', label: 'Position % of Volume', description: '10k shares as % of daily volume (lower = less market impact)' },
  
  // Daily range in dollars
  { name: 'avg_daily_range_dollars', label: 'Avg Daily Range ($)', description: 'Average daily high-low range in dollars' },
  { name: 'median_daily_range_dollars', label: 'Median Daily Range ($)', description: 'Median daily high-low range in dollars' },
  { name: 'sweet_spot_range_pct', label: 'Sweet Spot Days %', description: '% of days with $0.20-$0.60 range (ideal for swing trading)' },
  { name: 'daily_range_cv', label: 'Range Consistency (CV)', description: 'Coefficient of variation of daily range (lower = more predictable)' },
  
  // Combined scores
  { name: 'tradability_score', label: 'Tradability Score (0-100)', description: 'Combined score: liquidity + good range + consistency' },
  { name: 'profit_potential_score', label: 'Profit Potential', description: 'Daily range × liquidity (theoretical max profit potential)' },
  
  // Price metrics
  { name: 'price_range_pct', label: 'Price Range %', description: 'Price range percentage over the period' },
  { name: 'current_price', label: 'Current Price', description: 'Most recent closing price' },
];

const PRESET_STORAGE_KEY = 'stonks_scoring_presets';

export default function ScoringPanel({
  criteria,
  monthsBack,
  minDays,
  minAvgVolume,
  minPrice,
  maxPrice,
  onCriteriaChange,
  onMonthsBackChange,
  onMinDaysChange,
  onMinAvgVolumeChange,
  onMinPriceChange,
  onMaxPriceChange,
  onScore,
  scoring,
}: ScoringPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [presets, setPresets] = useState<ScoringPreset[]>([]);
  const [showPresetMenu, setShowPresetMenu] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [presetName, setPresetName] = useState('');
  const [presetDescription, setPresetDescription] = useState('');
  const [loadedPresetId, setLoadedPresetId] = useState<string | null>(null);
  const [saveMode, setSaveMode] = useState<'overwrite' | 'new'>('new');
  const presetMenuRef = useRef<HTMLDivElement>(null);

  // Close preset menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (presetMenuRef.current && !presetMenuRef.current.contains(event.target as Node)) {
        setShowPresetMenu(false);
      }
    };

    document.addEventListener('click', handleClickOutside, true);
    return () => document.removeEventListener('click', handleClickOutside, true);
  }, []);

  const addCriterion = () => {
    const newCriteria: ScoringCriteria = {
      name: 'trend_pct',
      weight: 1.0,
      min_value: undefined,
      max_value: undefined,
      invert: false,
    };
    onCriteriaChange([...criteria, newCriteria]);
  };

  const addPresetCriteria = () => {
    // Add preset criteria for intra-day/swing trading
    // Optimized for: 10k share trades, $0.20-$0.60 daily range, good liquidity
    // Using wider ranges to ensure scores are generated
    const preset: ScoringCriteria[] = [
      {
        name: 'tradability_score',
        weight: 3.0,
        min_value: 0,  // Start from 0 to include all stocks
        max_value: 100,
        invert: false,
      },
      {
        name: 'avg_daily_range_dollars',
        weight: 2.5,
        min_value: 0.10,  // Wider range
        max_value: 1.00,
        invert: false,
      },
      {
        name: 'liquidity_multiple',
        weight: 2.0,
        min_value: 0,  // Start from 0
        max_value: 100,
        invert: false,
      },
      {
        name: 'sweet_spot_range_pct',
        weight: 1.5,
        min_value: 0,
        max_value: 100,
        invert: false,
      },
      {
        name: 'daily_range_cv',
        weight: 1.0,
        min_value: 0,
        max_value: 1.0,  // Wider range
        invert: true, // Lower CV = more consistent = better
      },
      {
        name: 'current_price',
        weight: 10.0,
        min_value: 0,
        max_value: 8,
        invert: false,
      },
    ];
    onCriteriaChange([...criteria, ...preset]);
  };

  const updateCriterion = (index: number, updates: Partial<ScoringCriteria>) => {
    const newCriteria = [...criteria];
    newCriteria[index] = { ...newCriteria[index], ...updates };
    onCriteriaChange(newCriteria);
  };

  const removeCriterion = (index: number) => {
    onCriteriaChange(criteria.filter((_, i) => i !== index));
  };

  const getMetricDescription = (name: string) => {
    return AVAILABLE_METRICS.find(m => m.name === name)?.description || '';
  };

  const getMetricLabel = (name: string) => {
    return AVAILABLE_METRICS.find(m => m.name === name)?.label || name;
  };

  // Load presets from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(PRESET_STORAGE_KEY);
      if (stored) {
        setPresets(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load presets:', error);
    }
  }, []);

  // Generate description from criteria
  const generatePresetDescription = (): string => {
    if (criteria.length === 0) {
      return 'No criteria defined';
    }

    const parts: string[] = [];
    
    // Summary of criteria
    const topCriteria = criteria
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 3)
      .map(c => {
        const label = getMetricLabel(c.name);
        const range = c.min_value !== undefined || c.max_value !== undefined
          ? ` (${c.min_value ?? 'any'}-${c.max_value ?? 'any'})`
          : '';
        const invert = c.invert ? ' (inverted)' : '';
        return `${label}×${c.weight}${range}${invert}`;
      });
    
    if (topCriteria.length > 0) {
      parts.push(`Focus: ${topCriteria.join(', ')}`);
    }

    // Settings summary
    const settings: string[] = [];
    if (monthsBack !== 12) settings.push(`${monthsBack}mo period`);
    if (minDays !== 60) settings.push(`${minDays} min days`);
    if (minAvgVolume !== 750000) settings.push(`${(minAvgVolume / 1000).toFixed(0)}k vol`);
    if (minPrice !== undefined || maxPrice !== undefined) {
      settings.push(`$${minPrice ?? '0'}-$${maxPrice ?? '∞'}`);
    }
    
    if (settings.length > 0) {
      parts.push(`Settings: ${settings.join(', ')}`);
    }

    return parts.join(' | ') || 'Custom scoring preset';
  };

  const savePreset = () => {
    if (!presetName.trim()) {
      alert('Please enter a name for the preset');
      return;
    }

    const description = presetDescription.trim() || generatePresetDescription();
    
    let updatedPresets: ScoringPreset[];
    
    if (saveMode === 'overwrite' && loadedPresetId) {
      // Overwrite existing preset
      updatedPresets = presets.map(p => 
        p.id === loadedPresetId
          ? {
              ...p,
              name: presetName.trim(),
              description,
              criteria: [...criteria],
              monthsBack,
              minDays,
              minAvgVolume,
              minPrice,
              maxPrice,
            }
          : p
      );
    } else {
      // Create new preset
      const newPreset: ScoringPreset = {
        id: Date.now().toString(),
        name: presetName.trim(),
        description,
        criteria: [...criteria],
        monthsBack,
        minDays,
        minAvgVolume,
        minPrice,
        maxPrice,
        createdAt: new Date().toISOString(),
      };
      updatedPresets = [...presets, newPreset];
    }

    setPresets(updatedPresets);
    localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(updatedPresets));
    
    // If saving as new, clear the loaded preset ID since we're now working with a new preset
    if (saveMode === 'new') {
      setLoadedPresetId(null);
    }
    // If overwriting, keep the loaded preset ID the same
    
    setPresetName('');
    setPresetDescription('');
    setShowSaveDialog(false);
    setSaveMode('new');
  };

  const loadPreset = (preset: ScoringPreset) => {
    onCriteriaChange([...preset.criteria]);
    onMonthsBackChange(preset.monthsBack);
    onMinDaysChange(preset.minDays);
    onMinAvgVolumeChange(preset.minAvgVolume);
    onMinPriceChange(preset.minPrice);
    onMaxPriceChange(preset.maxPrice);
    setLoadedPresetId(preset.id);
    setShowPresetMenu(false);
  };

  const deletePreset = (id: string) => {
    if (confirm('Are you sure you want to delete this preset?')) {
      const updatedPresets = presets.filter(p => p.id !== id);
      setPresets(updatedPresets);
      localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(updatedPresets));
      if (loadedPresetId === id) {
        setLoadedPresetId(null);
      }
    }
  };

  return (
    <div className="bg-gray-800 dark:bg-gray-800 rounded-lg shadow p-4 space-y-4 border border-gray-700 dark:border-gray-700">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white dark:text-white">Ticker Scoring</h2>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-sm text-blue-400 hover:text-blue-300"
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>

      {expanded && (
        <>
          {/* Preset Management */}
          <div className="border-b border-gray-700 dark:border-gray-700 pb-3">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-300 dark:text-gray-300">Presets</h3>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setPresetDescription(generatePresetDescription());
                    if (loadedPresetId) {
                      const loadedPreset = presets.find(p => p.id === loadedPresetId);
                      if (loadedPreset) {
                        setPresetName(loadedPreset.name);
                        setPresetDescription(loadedPreset.description);
                        setSaveMode('overwrite');
                      } else {
                        setPresetName('');
                        setSaveMode('new');
                      }
                    } else {
                      setPresetName('');
                      setSaveMode('new');
                    }
                    setShowSaveDialog(true);
                  }}
                  disabled={criteria.length === 0}
                  className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Save Preset
                </button>
                <div className="relative" ref={presetMenuRef}>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowPresetMenu(!showPresetMenu);
                    }}
                    className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                  >
                    Load Preset {presets.length > 0 && `(${presets.length})`}
                  </button>
                  {showPresetMenu && presets.length > 0 && (
                    <div className="absolute top-full right-0 mt-1 z-50 bg-gray-800 dark:bg-gray-800 border border-gray-700 dark:border-gray-700 rounded-lg shadow-lg min-w-[300px] max-h-[400px] overflow-y-auto">
                      {presets.map((preset) => (
                        <div key={preset.id} className="p-3 border-b border-gray-700 dark:border-gray-700 last:border-b-0">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <div className="font-medium text-white dark:text-white text-sm mb-1">
                                {preset.name}
                              </div>
                              <div className="text-xs text-gray-400 dark:text-gray-400 mb-2">
                                {preset.description}
                              </div>
                              <div className="text-xs text-gray-500 dark:text-gray-500">
                                {preset.criteria.length} criteria • {new Date(preset.createdAt).toLocaleDateString()}
                              </div>
                            </div>
                            <div className="flex flex-col gap-1 flex-shrink-0">
                              <button
                                onClick={() => loadPreset(preset)}
                                className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                              >
                                Load
                              </button>
                              <button
                                onClick={() => deletePreset(preset.id)}
                                className="text-xs px-2 py-1 bg-red-600 text-white rounded hover:bg-red-700"
                              >
                                Delete
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Save Preset Dialog */}
          {showSaveDialog && (
            <div 
              className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
              onClick={(e) => {
                if (e.target === e.currentTarget) {
                  setShowSaveDialog(false);
                  setPresetName('');
                  setPresetDescription('');
                  setSaveMode('new');
                }
              }}
            >
              <div 
                className="bg-gray-800 dark:bg-gray-800 border border-gray-700 dark:border-gray-700 rounded-lg p-6 max-w-md w-full mx-4"
                onClick={(e) => e.stopPropagation()}
              >
                <h3 className="text-lg font-semibold text-white dark:text-white mb-4">Save Scoring Preset</h3>
                <div className="space-y-4">
                  {loadedPresetId && (
                    <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded p-3 mb-2">
                      <p className="text-sm text-blue-300 mb-3">
                        You have a preset loaded. Choose how to save:
                      </p>
                      <div className="space-y-2">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="radio"
                            name="saveMode"
                            value="overwrite"
                            checked={saveMode === 'overwrite'}
                            onChange={() => setSaveMode('overwrite')}
                            className="text-blue-600"
                          />
                          <span className="text-sm text-white">Overwrite current preset</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="radio"
                            name="saveMode"
                            value="new"
                            checked={saveMode === 'new'}
                            onChange={() => setSaveMode('new')}
                            className="text-blue-600"
                          />
                          <span className="text-sm text-white">Save as new preset</span>
                        </label>
                      </div>
                    </div>
                  )}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 dark:text-gray-300 mb-1">
                      Preset Name *
                    </label>
                    <input
                      type="text"
                      value={presetName}
                      onChange={(e) => setPresetName(e.target.value)}
                      placeholder="e.g., Swing Trading Strategy"
                      className="w-full px-3 py-2 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
                      autoFocus
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 dark:text-gray-300 mb-1">
                      Description
                    </label>
                    <textarea
                      value={presetDescription}
                      onChange={(e) => setPresetDescription(e.target.value)}
                      placeholder="Auto-generated description based on criteria..."
                      rows={3}
                      className="w-full px-3 py-2 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Leave empty to auto-generate from criteria
                    </p>
                  </div>
                  <div className="flex gap-2 justify-end">
                    <button
                      onClick={() => {
                        setShowSaveDialog(false);
                        setPresetName('');
                        setPresetDescription('');
                        setSaveMode('new');
                      }}
                      className="px-4 py-2 text-sm bg-gray-700 text-white rounded hover:bg-gray-600"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={savePreset}
                      className="px-4 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      {saveMode === 'overwrite' ? 'Overwrite' : 'Save'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-300 dark:text-gray-300 mb-1">
                Analysis Period (months)
              </label>
              <input
                type="number"
                min="1"
                max="24"
                value={monthsBack}
                onChange={(e) => onMonthsBackChange(Number(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 dark:text-gray-300 mb-1">
                Minimum Days Required
              </label>
              <input
                type="number"
                min="1"
                max="365"
                value={minDays}
                onChange={(e) => onMinDaysChange(Number(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 dark:text-gray-300 mb-1">
                Min Avg Volume (regular hours)
              </label>
              <input
                type="number"
                min="0"
                step="100000"
                value={minAvgVolume}
                onChange={(e) => onMinAvgVolumeChange(Number(e.target.value))}
                className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
              />
              <p className="text-xs text-gray-500 mt-1">Default: 750,000 shares/day</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 dark:text-gray-300 mb-1">
                Price Range (current price)
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="0"
                  step="0.5"
                  placeholder="Min $"
                  value={minPrice ?? ''}
                  onChange={(e) => onMinPriceChange(e.target.value === '' ? undefined : Number(e.target.value))}
                  className="flex-1 px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
                />
                <span className="text-gray-400">-</span>
                <input
                  type="number"
                  min="0"
                  step="0.5"
                  placeholder="Max $"
                  value={maxPrice ?? ''}
                  onChange={(e) => onMaxPriceChange(e.target.value === '' ? undefined : Number(e.target.value))}
                  className="flex-1 px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">Filter by most recent closing price</p>
            </div>
          </div>

          <div className="border-t border-gray-700 dark:border-gray-700 pt-3">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-300 dark:text-gray-300">Scoring Criteria</h3>
              <div className="flex gap-2">
                {criteria.length === 0 && (
                  <button
                    onClick={addPresetCriteria}
                    className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                    title="Add preset criteria for intra-day/swing trading"
                  >
                    Preset
                  </button>
                )}
                <button
                  onClick={addCriterion}
                  className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  + Add
                </button>
              </div>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {criteria.map((criterion, index) => (
                <div key={index} className="border border-gray-700 dark:border-gray-700 rounded p-3 space-y-2 bg-gray-700 dark:bg-gray-700">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-300 dark:text-gray-300">Criterion {index + 1}</span>
                    <button
                      onClick={() => removeCriterion(index)}
                      className="text-xs text-red-400 hover:text-red-300"
                    >
                      Remove
                    </button>
                  </div>

                  <div>
                    <label className="block text-xs text-gray-300 dark:text-gray-300 mb-1">Metric</label>
                    <select
                      value={criterion.name}
                      onChange={(e) => updateCriterion(index, { name: e.target.value })}
                      className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-800 dark:bg-gray-800 text-white dark:text-white"
                    >
                      {AVAILABLE_METRICS.map((metric) => (
                        <option key={metric.name} value={metric.name}>
                          {metric.label}
                        </option>
                      ))}
                    </select>
                    {getMetricDescription(criterion.name) && (
                      <p className="text-xs text-gray-400 dark:text-gray-400 mt-1">{getMetricDescription(criterion.name)}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-xs text-gray-300 dark:text-gray-300 mb-1">Weight</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      value={criterion.weight}
                      onChange={(e) => updateCriterion(index, { weight: Number(e.target.value) })}
                      className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-800 dark:bg-gray-800 text-white dark:text-white"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-xs text-gray-300 dark:text-gray-300 mb-1">Min Value</label>
                      <input
                        type="number"
                        step="0.01"
                        value={criterion.min_value ?? ''}
                        onChange={(e) => updateCriterion(index, { 
                          min_value: e.target.value === '' ? undefined : Number(e.target.value) 
                        })}
                        placeholder="Optional"
                        className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-800 dark:bg-gray-800 text-white dark:text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-300 dark:text-gray-300 mb-1">Max Value</label>
                      <input
                        type="number"
                        step="0.01"
                        value={criterion.max_value ?? ''}
                        onChange={(e) => updateCriterion(index, { 
                          max_value: e.target.value === '' ? undefined : Number(e.target.value) 
                        })}
                        placeholder="Optional"
                        className="w-full px-2 py-1 text-sm border border-gray-600 dark:border-gray-600 rounded bg-gray-800 dark:bg-gray-800 text-white dark:text-white"
                      />
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id={`invert-${index}`}
                      checked={criterion.invert ?? false}
                      onChange={(e) => updateCriterion(index, { invert: e.target.checked })}
                      className="rounded"
                    />
                    <label htmlFor={`invert-${index}`} className="text-xs text-gray-300 dark:text-gray-300">
                      Invert (lower values score higher)
                    </label>
                  </div>
                </div>
              ))}

              {criteria.length === 0 && (
                <p className="text-sm text-gray-400 dark:text-gray-400 text-center py-4">
                  No criteria added. Click "+ Add" to create scoring criteria.
                </p>
              )}
            </div>
          </div>

          <div className="border-t border-gray-700 dark:border-gray-700 pt-3">
            <button
              onClick={onScore}
              disabled={scoring || criteria.length === 0}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {scoring ? 'Scoring...' : 'Score & Rank Tickers'}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
