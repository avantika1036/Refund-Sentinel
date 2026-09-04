/**
 * Summary view for a completed investigation.
 *
 * Displays core assessment fields. Financial exposure, ML prediction,
 * and related refunds visualization are handled in later phases.
 */

import type { InvestigationResponse, DecisionAction } from "../../types/api";
import { RiskBadge } from "../common/RiskBadge";

interface InvestigationSummaryProps {
  investigation: InvestigationResponse;
}

const ACTION_LABELS: Record<DecisionAction, string> = {
  allow: "Allow",
  review: "Review",
  investigate: "Investigate",
};

export function InvestigationSummary({ investigation }: InvestigationSummaryProps) {
  const { assessment } = investigation;
  const actionLabel = ACTION_LABELS[assessment.action];

  return (
    <section className="investigation-summary" aria-labelledby="investigation-summary-title">
      <header className="summary-header">
        <h2 id="investigation-summary-title" className="summary-title">
          Investigation Summary
        </h2>
      </header>

      <dl className="summary-grid">
        <div className="summary-item">
          <dt className="summary-label">Refund ID</dt>
          <dd className="summary-value summary-value--mono">{assessment.refund_id}</dd>
        </div>

        <div className="summary-item">
          <dt className="summary-label">Customer ID</dt>
          <dd className="summary-value summary-value--mono">{assessment.customer_id}</dd>
        </div>

        <div className="summary-item">
          <dt className="summary-label">Component ID</dt>
          <dd className="summary-value summary-value--mono">{assessment.component_id}</dd>
        </div>

        <div className="summary-item">
          <dt className="summary-label">Risk Level</dt>
          <dd className="summary-value">
            <RiskBadge level={assessment.risk_level} />
          </dd>
        </div>

        <div className="summary-item">
          <dt className="summary-label">Recommended Action</dt>
          <dd className="summary-value">
            <span
              className={`action-badge action-badge--${assessment.action}`}
              aria-label={`Recommended action: ${actionLabel}`}
            >
              {actionLabel}
            </span>
          </dd>
        </div>
      </dl>

      <div className="summary-explanation">
        <h3 className="summary-explanation-title">Explanation</h3>
        <p className="summary-explanation-text">{assessment.explanation}</p>
      </div>
    </section>
  );
}
