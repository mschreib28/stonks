# Stonks Data Explorer - Frontend

React application for exploring and analyzing stock market data from Parquet files.

## Features

- **Dynamic Filtering**: Filter data by any column with range filters for numbers, boolean filters, and text search
- **Sortable Columns**: Click any column header to sort ascending/descending
- **Multiple Datasets**: Switch between daily, weekly, filtered, and minute features datasets
- **Interactive Charts**: Click on any ticker to view detailed price, volume, and returns charts
- **Pagination**: Navigate through large datasets with configurable page sizes
- **Real-time Stats**: View min/max/mean/std statistics for numeric columns

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The app will be available at http://localhost:3000

## Build

To build for production:
```bash
npm run build
```

## Backend

Make sure the FastAPI backend server is running on port 8000. Start it with:

```bash
python backend/api_server.py
```

Or using uvicorn directly:
```bash
uvicorn backend.api_server:app --reload
```

