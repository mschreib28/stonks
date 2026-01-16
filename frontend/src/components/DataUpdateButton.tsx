import { useState, useEffect } from 'react';
import { checkDataFreshness, triggerDataUpdate, getUpdateStatus, type DataFreshnessResponse, type UpdateStatusResponse } from '../api';

export default function DataUpdateButton() {
  const [freshness, setFreshness] = useState<DataFreshnessResponse | null>(null);
  const [updateStatus, setUpdateStatus] = useState<UpdateStatusResponse | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Check freshness on mount
  useEffect(() => {
    checkFreshness();
  }, []);

  // Poll for update status when update is in progress
  useEffect(() => {
    if (updateStatus && ['checking', 'downloading', 'processing'].includes(updateStatus.status)) {
      // Clear any existing interval
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
      
      const interval = setInterval(async () => {
        try {
          const status = await getUpdateStatus();
          setUpdateStatus(status);
          
          // Stop polling if completed or error
          if (status.status === 'completed' || status.status === 'error') {
            clearInterval(interval);
            setPollingInterval(null);
            // Refresh freshness check after completion (force reload to get latest data)
            // Wait a bit longer to ensure processing is fully complete
            setTimeout(() => {
              checkFreshness(true); // Force reload to get fresh data
            }, 3000); // Give processing more time to complete and files to be written
            
            // Reset status after 5 seconds if completed
            if (status.status === 'completed') {
              setTimeout(() => {
                setUpdateStatus(null);
              }, 5000);
            }
          }
        } catch (error) {
          console.error('Error polling update status:', error);
          clearInterval(interval);
          setPollingInterval(null);
        }
      }, 2000); // Poll every 2 seconds
      
      setPollingInterval(interval);
      
      return () => {
        clearInterval(interval);
        setPollingInterval(null);
      };
    } else if (pollingInterval && updateStatus && !['checking', 'downloading', 'processing'].includes(updateStatus.status)) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  }, [updateStatus?.status]);

  const checkFreshness = async (forceReload: boolean = false) => {
    try {
      setIsChecking(true);
      const result = await checkDataFreshness(forceReload);
      setFreshness(result);
    } catch (error) {
      console.error('Error checking data freshness:', error);
    } finally {
      setIsChecking(false);
    }
  };

  const handleUpdate = async () => {
    try {
      await triggerDataUpdate();
      // Start polling for status
      const status = await getUpdateStatus();
      setUpdateStatus(status);
    } catch (error: any) {
      console.error('Error triggering update:', error);
      if (error.response?.status === 400) {
        // Update already in progress, get current status
        const status = await getUpdateStatus();
        setUpdateStatus(status);
      }
    }
  };

  const isLoading = updateStatus?.status === 'checking' || updateStatus?.status === 'downloading' || updateStatus?.status === 'processing';
  const isCompleted = updateStatus?.status === 'completed';
  const isError = updateStatus?.status === 'error';

  return (
    <div className="flex items-center gap-2">
      {freshness && !freshness.is_fresh && (
        <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-900/30 border border-yellow-700 rounded text-yellow-200 text-sm">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <span>Data missing for {freshness.expected_date}</span>
        </div>
      )}
      
      <button
        onClick={handleUpdate}
        disabled={isLoading || isChecking}
        className={`
          px-4 py-2 rounded text-sm font-medium transition-colors
          flex items-center gap-2
          ${isLoading || isChecking
            ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
            : isCompleted
            ? 'bg-green-700 hover:bg-green-600 text-white'
            : isError
            ? 'bg-red-700 hover:bg-red-600 text-white'
            : freshness && !freshness.is_fresh
            ? 'bg-yellow-700 hover:bg-yellow-600 text-white'
            : 'bg-blue-700 hover:bg-blue-600 text-white'
          }
        `}
      >
        {isLoading || isChecking ? (
          <>
            <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>
              {updateStatus?.status === 'checking' && 'Checking...'}
              {updateStatus?.status === 'downloading' && 'Downloading...'}
              {updateStatus?.status === 'processing' && 'Processing...'}
              {isChecking && 'Checking...'}
            </span>
          </>
        ) : isCompleted ? (
          <>
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <span>Update Complete</span>
          </>
        ) : isError ? (
          <>
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <span>Update Failed</span>
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Update Data</span>
          </>
        )}
      </button>

      {updateStatus && updateStatus.message && (
        <div className="text-xs text-gray-400 max-w-xs truncate" title={updateStatus.message}>
          {updateStatus.message}
        </div>
      )}

      {updateStatus && updateStatus.progress > 0 && updateStatus.progress < 100 && (
        <div className="w-32 h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-600 transition-all duration-300"
            style={{ width: `${updateStatus.progress}%` }}
          />
        </div>
      )}
    </div>
  );
}
