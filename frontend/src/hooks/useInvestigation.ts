/**
 * Hook for investigation state management.
 *
 * Manages investigation data, loading state, and error state.
 */

import { useState, useCallback, useRef } from "react";
import { getInvestigation } from "../api/endpoints";
import type { InvestigationResponse } from "../types/api";
import { ApiErrorImpl } from "../types/api";

export function useInvestigation() {
  const [investigation, setInvestigation] = useState<InvestigationResponse | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ApiErrorImpl | null>(null);
  const requestIdRef = useRef(0);

  const investigate = useCallback(async (refundId: string) => {
    const requestId = ++requestIdRef.current;

    setError(null);
    setInvestigation(null);
    setIsLoading(true);

    try {
      const data = await getInvestigation(refundId);

      if (requestId !== requestIdRef.current) {
        return;
      }

      setInvestigation(data);
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return;
      }

      if (err instanceof ApiErrorImpl) {
        setError(err);
      } else {
        setError(
          new ApiErrorImpl(
            "UNKNOWN_ERROR",
            err instanceof Error ? err.message : "Unknown error occurred"
          )
        );
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  const reset = useCallback(() => {
    setInvestigation(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    investigation,
    isLoading,
    error,
    investigate,
    reset,
  };
}
