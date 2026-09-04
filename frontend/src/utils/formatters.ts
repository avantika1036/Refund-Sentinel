/**
 * Human-readable formatting for risk scores and rule evidence.
 */

export function formatScoreAsPercent(score: number): string {
  return `${(score * 100).toFixed(1)}%`;
}

const EVIDENCE_TYPE_LABELS: Record<string, string> = {
  latency_hours: "Latency",
  boolean: "Boolean",
  rate: "Rate",
  count: "Count",
  cluster_flag_count: "Cluster flags",
};

export function formatEvidenceType(type: string): string {
  return EVIDENCE_TYPE_LABELS[type] ?? type.replace(/_/g, " ");
}

export function formatEvidenceValue(
  type: string,
  value: number | boolean | null
): string {
  if (value === null) {
    return "Not available";
  }

  switch (type) {
    case "latency_hours":
      return `${Number(value).toFixed(1)} hours`;
    case "boolean":
      return value === true || value === 1 ? "Yes" : "No";
    case "rate":
      return `${(Number(value) * 100).toFixed(1)}%`;
    case "count":
    case "cluster_flag_count":
      return String(Math.round(Number(value)));
    default:
      if (typeof value === "boolean") {
        return value ? "Yes" : "No";
      }
      return String(value);
  }
}

export function formatEvidenceThreshold(
  type: string,
  threshold: number | null
): string {
  if (threshold === null) {
    return "—";
  }

  switch (type) {
    case "latency_hours":
      return `${Number(threshold).toFixed(1)} hours`;
    case "boolean":
      return threshold >= 0.99 ? "Yes" : String(threshold);
    case "rate":
      return `${(Number(threshold) * 100).toFixed(1)}%`;
    case "count":
    case "cluster_flag_count":
      return String(threshold);
    default:
      return String(threshold);
  }
}

export function formatRuleId(ruleId: string): string {
  const match = ruleId.match(/^(R\d+)_(.+)$/);
  if (!match) {
    return ruleId;
  }

  const code = match[1];
  const name = match[2]
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

  return `${code} — ${name}`;
}

export function formatSignalWeight(weight: number): string {
  return `${(weight * 100).toFixed(0)}%`;
}

/**
 * Convert an integer paise amount to a formatted Indian Rupee string.
 *
 * Backend stores all monetary values as integer paise (1 INR = 100 paise).
 * This function divides by 100 before formatting.
 *
 * Zero is a valid value and will be rendered as "₹0.00".
 */
export function formatINR(amountPaise: number): string {
  const rupees = amountPaise / 100;
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(rupees);
}

/**
 * Format a probability in the range [0.0, 1.0] as a percentage string
 * with one decimal place, e.g. 0.847 → "84.7%".
 *
 * Consistent with the existing formatScoreAsPercent helper but intended
 * specifically for ML probability values.
 */
export function formatProbability(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}
