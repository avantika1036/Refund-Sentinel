/**
 * Investigation search component.
 *
 * Provides a form for entering a refund ID to investigate.
 */

import { useState, type FormEvent } from "react";

interface InvestigationSearchProps {
  onInvestigate: (refundId: string) => void;
  isLoading?: boolean;
}

export function InvestigationSearch({ onInvestigate, isLoading = false }: InvestigationSearchProps) {
  const [refundId, setRefundId] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmedId = refundId.trim();
    if (trimmedId && !isLoading) {
      onInvestigate(trimmedId);
    }
  };

  const isDisabled = !refundId.trim() || isLoading;

  return (
    <form className="investigation-search" onSubmit={handleSubmit}>
      <div className="search-header">
        <h2 className="search-title">Investigate Refund</h2>
        <p className="search-description">
          Enter a refund ID to view risk assessment, rule evidence, and financial
          exposure.
        </p>
      </div>
      <div className="search-form">
        <input
          type="text"
          className="search-input"
          placeholder="Enter refund ID (e.g., 1d79aa79-a21d-4077-96e0-ccd09b51cdb7)"
          value={refundId}
          onChange={(e) => setRefundId(e.target.value)}
          disabled={isLoading}
          aria-label="Refund ID"
        />
        <button
          type="submit"
          className="search-button"
          disabled={isDisabled}
        >
          {isLoading ? "Investigating..." : "Investigate"}
        </button>
      </div>
    </form>
  );
}
