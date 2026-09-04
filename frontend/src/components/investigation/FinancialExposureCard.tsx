/**
 * Displays risk-weighted financial exposure for a refund investigation.
 *
 * The backend computes three separate exposure buckets, each multiplied
 * by the final risk score. All values are stored as integer paise and
 * must be divided by 100 to obtain rupees.
 *
 * The three values intentionally represent distinct concepts and are
 * NOT summed into a single total — see backend/app/finance/exposure.py.
 */

import type { FinancialExposureResponse } from "../../types/api";
import { formatINR } from "../../utils/formatters";

interface FinancialExposureCardProps {
  exposure: FinancialExposureResponse;
}

interface ExposureRowProps {
  label: string;
  description: string;
  amountPaise: number;
}

function ExposureRow({ label, description, amountPaise }: ExposureRowProps) {
  return (
    <div className="exposure-row">
      <div className="exposure-row-info">
        <span className="exposure-row-label">{label}</span>
        <span className="exposure-row-description">{description}</span>
      </div>
      <span className="exposure-row-amount">{formatINR(amountPaise)}</span>
    </div>
  );
}

export function FinancialExposureCard({ exposure }: FinancialExposureCardProps) {
  return (
    <section
      className="financial-exposure-card"
      aria-labelledby="financial-exposure-title"
    >
      <header className="financial-exposure-header">
        <h2 id="financial-exposure-title" className="financial-exposure-title">
          Financial Exposure
        </h2>
        <p className="financial-exposure-subtitle">
          Risk-weighted amounts based on the final risk score
        </p>
      </header>

      <div className="exposure-rows">
        <ExposureRow
          label="Realized Suspicious Amount"
          description="Already refunded through processed transactions"
          amountPaise={exposure.realized_suspicious_amount_paise}
        />

        <ExposureRow
          label="Pending Refund Exposure"
          description="Pending requested refunds in this connected component"
          amountPaise={exposure.pending_refund_exposure_paise}
        />

        <ExposureRow
          label="Remaining Refundable Exposure"
          description="Could still be refunded from eligible payments"
          amountPaise={exposure.remaining_refundable_exposure_paise}
        />
      </div>
    </section>
  );
}
