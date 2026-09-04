/**
 * Accessible badge for displaying risk level.
 */

import type { RiskLevel } from "../../types/api";

interface RiskBadgeProps {
  level: RiskLevel;
}

const RISK_LABELS: Record<RiskLevel, string> = {
  low: "Low Risk",
  medium: "Medium Risk",
  high: "High Risk",
};

export function RiskBadge({ level }: RiskBadgeProps) {
  const label = RISK_LABELS[level];

  return (
    <span className={`risk-badge risk-badge--${level}`} aria-label={`Risk level: ${label}`}>
      {label}
    </span>
  );
}
