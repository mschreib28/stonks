import type { Dataset } from '../types';

interface DatasetSelectorProps {
  datasets: Dataset[];
  selectedDataset: string;
  onSelect: (dataset: string) => void;
}

export default function DatasetSelector({ datasets, selectedDataset, onSelect }: DatasetSelectorProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-white dark:text-white mb-2">
        Dataset
      </label>
      <select
        value={selectedDataset}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full px-3 py-2 border border-gray-600 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-gray-700 dark:bg-gray-700 text-white dark:text-white"
      >
        {datasets.map((ds) => (
          <option key={ds.name} value={ds.name}>
            {ds.name} ({ds.rows?.toLocaleString() || 'N/A'} rows)
          </option>
        ))}
      </select>
      {datasets.find((ds) => ds.name === selectedDataset)?.error && (
        <p className="mt-2 text-sm text-red-400 dark:text-red-400">
          {datasets.find((ds) => ds.name === selectedDataset)?.error}
        </p>
      )}
    </div>
  );
}

