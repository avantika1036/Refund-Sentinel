/**
 * Displays deterministic rule evidence for an investigation.
 */

import type { RuleEvidenceResponse } from "../../types/api";
import {
  formatEvidenceThreshold,
  formatEvidenceType,
  formatEvidenceValue,
  formatRuleId,
  formatSignalWeight,
} from "../../utils/formatters";

interface RuleEvidenceListProps {
  ruleOutputs: RuleEvidenceResponse[];
}

function sortRules(rules: RuleEvidenceResponse[]): RuleEvidenceResponse[] {
  return [...rules].sort((a, b) => {
    if (a.triggered !== b.triggered) {
      return a.triggered ? -1 : 1;
    }
    return a.rule_id.localeCompare(b.rule_id);
  });
}

interface RuleEvidenceItemProps {
  rule: RuleEvidenceResponse;
}

function RuleEvidenceItem({ rule }: RuleEvidenceItemProps) {
  const statusLabel = rule.triggered ? "Triggered" : "Not triggered";

  return (
    <article
      className={`rule-evidence-item ${rule.triggered ? "rule-evidence-item--triggered" : "rule-evidence-item--not-triggered"}`}
      aria-label={`Rule ${formatRuleId(rule.rule_id)}: ${statusLabel}`}
    >
      <header className="rule-evidence-header">
        <h3 className="rule-evidence-name">{formatRuleId(rule.rule_id)}</h3>
        <span
          className={`rule-status-badge ${rule.triggered ? "rule-status-badge--triggered" : "rule-status-badge--not-triggered"}`}
        >
          {statusLabel}
        </span>
      </header>

      <dl className="rule-evidence-details">
        <div className="rule-evidence-detail">
          <dt>Evidence type</dt>
          <dd>{formatEvidenceType(rule.evidence_type)}</dd>
        </div>
        <div className="rule-evidence-detail">
          <dt>Evidence value</dt>
          <dd>{formatEvidenceValue(rule.evidence_type, rule.evidence_value)}</dd>
        </div>
        <div className="rule-evidence-detail">
          <dt>Threshold</dt>
          <dd>{formatEvidenceThreshold(rule.evidence_type, rule.evidence_threshold)}</dd>
        </div>
        <div className="rule-evidence-detail">
          <dt>Signal weight</dt>
          <dd>{formatSignalWeight(rule.base_signal_weight)}</dd>
        </div>
      </dl>

      {rule.notes && (
        <p className="rule-evidence-notes">{rule.notes}</p>
      )}
    </article>
  );
}

export function RuleEvidenceList({ ruleOutputs }: RuleEvidenceListProps) {
  const sortedRules = sortRules(ruleOutputs);

  return (
    <section className="rule-evidence-list" aria-labelledby="rule-evidence-title">
      <header className="rule-evidence-list-header">
        <h2 id="rule-evidence-title" className="rule-evidence-list-title">
          Rule Evidence
        </h2>
        <p className="rule-evidence-list-description">
          {sortedRules.filter((rule) => rule.triggered).length} of {sortedRules.length} rules triggered
        </p>
      </header>

      {sortedRules.length === 0 ? (
        <p className="rule-evidence-empty">No rule evidence available for this investigation.</p>
      ) : (
        <div className="rule-evidence-items">
          {sortedRules.map((rule) => (
            <RuleEvidenceItem key={rule.rule_id} rule={rule} />
          ))}
        </div>
      )}
    </section>
  );
}
