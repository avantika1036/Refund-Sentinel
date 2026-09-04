/**
 * Displays risk score components for an investigation.
 */

import type { RiskScoreResponse } from "../../types/api";
import { formatScoreAsPercent } from "../../utils/formatters";

interface RiskScoreCardProps {
  riskScore: RiskScoreResponse;
}

interface ScoreRowProps {
  label: string;
  value: number;
}

function ScoreRow({ label, value }: ScoreRowProps) {
  return (
    <div className="risk-score-row">
      <span className="risk-score-row-label">{label}</span>
      <span className="risk-score-row-value">{formatScoreAsPercent(value)}</span>
    </div>
  );
}

export function RiskScoreCard({ riskScore }: RiskScoreCardProps) {
  return (
    <section className="risk-score-card" aria-labelledby="risk-score-title">
      <header className="risk-score-header">
        <h2 id="risk-score-title" className="risk-score-title">
          Risk Score
        </h2>
      </header>

      <div className="risk-score-final">
        <span className="risk-score-final-label">Final Score</span>
        <span className="risk-score-final-value" aria-label={`Final risk score: ${formatScoreAsPercent(riskScore.final_score)}`}>
          {formatScoreAsPercent(riskScore.final_score)}
        </span>
      </div>

      <div className="risk-score-breakdown">
        <ScoreRow label="Rule Signal" value={riskScore.rule_signal_component} />
        <ScoreRow
          label="Behavioral Confirmation"
          value={riskScore.behavioral_confirmation_score}
        />
        <ScoreRow label="Cluster Signal" value={riskScore.cluster_signal_component} />
      </div>
    </section>
  );
}
