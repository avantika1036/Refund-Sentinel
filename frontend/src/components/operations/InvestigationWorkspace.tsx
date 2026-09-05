import { useEffect, useMemo, useState } from "react";
import { useInvestigation } from "../../hooks/useInvestigation";
import { formatEvidenceThreshold, formatEvidenceValue, formatINR, formatRuleId, formatScoreAsPercent } from "../../utils/formatters";
import type { RuleEvidenceResponse } from "../../types/api";

interface InvestigationWorkspaceProps {
  refundId: string;
  onBack: () => void;
  onSelectRelated: (refundId: string) => void;
}

function ScoreBar({ label, value, tone }: { label: string; value: number; tone: string }) {
  return <div className="score-row"><div className="score-row-heading"><span>{label}</span><strong>{formatScoreAsPercent(value)}</strong></div><div className="score-track"><span className={`score-fill score-fill--${tone}`} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div></div>;
}

function EvidenceRow({ evidence, triggered }: { evidence: RuleEvidenceResponse; triggered: boolean }) {
  const content = <><div className="evidence-row-top"><span className={`evidence-indicator ${triggered ? "evidence-indicator--triggered" : ""}`}>{triggered ? "!" : "✓"}</span><strong>{formatRuleId(evidence.rule_id)}</strong><span className={`evidence-state ${triggered ? "evidence-state--triggered" : ""}`}>{triggered ? "Triggered" : "Checked"}</span></div><p>{evidence.notes}</p><div className="evidence-values"><span>Observed <b>{formatEvidenceValue(evidence.evidence_type, evidence.evidence_value)}</b></span><span>Threshold <b>{formatEvidenceThreshold(evidence.evidence_type, evidence.evidence_threshold)}</b></span><span>Weight <b>{(evidence.base_signal_weight * 100).toFixed(0)}%</b></span></div></>;
  return triggered ? <div className="evidence-row evidence-row--triggered">{content}</div> : <details className="evidence-row evidence-row--checked"><summary>{content}</summary></details>;
}

function shortId(value: string) {
  return `${value.slice(0, 8)}…${value.slice(-6)}`;
}

export function InvestigationWorkspace({ refundId, onBack, onSelectRelated }: InvestigationWorkspaceProps) {
  const { investigation, isLoading, error, investigate, reset } = useInvestigation();
  const [showAllEvidence, setShowAllEvidence] = useState(false);

  useEffect(() => {
    void investigate(refundId);
    return reset;
  }, [investigate, refundId, reset]);

  const triggeredEvidence = useMemo(() => investigation?.assessment.rule_outputs.filter((rule) => rule.triggered) ?? [], [investigation]);
  const checkedEvidence = useMemo(() => investigation?.assessment.rule_outputs.filter((rule) => !rule.triggered) ?? [], [investigation]);

  if (isLoading) return <div className="loading-panel"><span className="loading-orbit" /><p>Opening investigation workspace…</p><span className="loading-note">Assessing evidence for {shortId(refundId)}</span></div>;
  if (error) return <div className="empty-panel empty-panel--error"><span className="empty-icon">!</span><h2>Investigation unavailable</h2><p>{error.message}</p><button className="button button--secondary" onClick={onBack}>Back to queue</button><button className="button button--primary" onClick={() => void investigate(refundId)}>Retry</button></div>;
  if (!investigation) return null;

  const { assessment, exposure, component_refund_ids: relatedRefunds, ml_prediction: mlPrediction, evidence_bundle: evidenceBundle } = investigation;
  const riskLevel = assessment.risk_level;
  return (
    <section className="investigation-page">
      <button className="back-link" onClick={onBack}>← Back to investigation queue</button>
      <div className="investigation-header">
        <div><p className="eyebrow">Investigation workspace</p><h1>{shortId(assessment.refund_id)}</h1><p className="investigation-id">{assessment.refund_id}</p></div>
        <div className="investigation-actions"><span className={`risk-pill risk-pill--${riskLevel}`}>{riskLevel} risk</span><span className={`recommendation recommendation--${assessment.action}`}>{assessment.action}</span></div>
      </div>
      <div className="summary-grid">
        <article className="summary-card summary-card--risk"><span className="summary-label">Final risk score</span><strong>{formatScoreAsPercent(assessment.risk_score.final_score)}</strong><span>Deterministic priority</span></article>
        <article className="summary-card"><span className="summary-label">Pending exposure</span><strong>{formatINR(exposure.pending_refund_exposure_paise)}</strong><span>Risk-weighted cluster amount</span></article>
        <article className="summary-card"><span className="summary-label">Related refunds</span><strong>{relatedRefunds.length}</strong><span>Connected component members</span></article>
        <article className={`summary-card ${mlPrediction ? "summary-card--ml" : "summary-card--muted"}`}><span className="summary-label">ML probability</span><strong>{mlPrediction ? formatScoreAsPercent(mlPrediction.probability) : "Unavailable"}</strong><span>{mlPrediction ? "Additional analytical signal" : "Model not configured"}</span></article>
      </div>

      <div className="investigation-grid">
        <div className="investigation-main">
          <article className="workspace-card explanation-card">
            <div className="card-heading">
              <div>
                <p className="eyebrow">Analyst brief</p>
                <h2>{investigation.explanation_summary?.headline || "What is suspicious here?"}</h2>
              </div>
              <span className="brief-mark">↗</span>
            </div>
            <p className="explanation-text">
              {investigation.explanation_summary?.narrative_summary || assessment.explanation}
            </p>
            {investigation.explanation_summary?.key_risk_drivers && investigation.explanation_summary.key_risk_drivers.length > 0 && (
              <div style={{ marginTop: "12px", marginBottom: "12px" }}>
                <strong style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.05em", opacity: 0.8 }}>Key Risk Drivers:</strong>
                <ul style={{ margin: "6px 0 0 18px", padding: 0, fontSize: "13px", lineHeight: "1.6" }}>
                  {investigation.explanation_summary.key_risk_drivers.map((driver, idx) => (
                    <li key={idx}>{driver}</li>
                  ))}
                </ul>
              </div>
            )}
            {investigation.explanation_summary?.suggested_action_rationale && (
              <p style={{ fontSize: "13px", fontStyle: "italic", opacity: 0.9, marginTop: "8px" }}>
                💡 Recommendation: {investigation.explanation_summary.suggested_action_rationale}
              </p>
            )}
            <div className="entity-strip">
              <div><span>Customer</span><strong>{shortId(assessment.customer_id)}</strong></div>
              <div><span>Component</span><strong>{assessment.component_id}</strong></div>
              <div><span>Action</span><strong>{assessment.action}</strong></div>
            </div>
          </article>
          <article className="workspace-card evidence-register"><div className="card-heading"><div><p className="eyebrow">Evidence register</p><h2>Signals behind the score</h2></div><span className="evidence-count">{triggeredEvidence.length} triggered</span></div>{triggeredEvidence.length > 0 ? <div className="evidence-group"><div className="group-label group-label--alert">Triggered evidence</div>{triggeredEvidence.map((evidence) => <EvidenceRow key={evidence.rule_id} evidence={evidence} triggered />)}</div> : <div className="subtle-state">No deterministic rules triggered for this refund.</div>}<div className="evidence-group evidence-group--secondary"><button className="group-toggle" onClick={() => setShowAllEvidence((current) => !current)}><span>Evidence checked but not triggered</span><span>{showAllEvidence ? "Collapse ↑" : `${checkedEvidence.length} checks ↓`}</span></button>{showAllEvidence && checkedEvidence.map((evidence) => <EvidenceRow key={evidence.rule_id} evidence={evidence} triggered={false} />)}</div></article>
          <article className="workspace-card"><div className="card-heading"><div><p className="eyebrow">Risk composition</p><h2>Score breakdown</h2></div></div><div className="score-breakdown"><ScoreBar label="Deterministic risk signal" value={assessment.risk_score.rule_signal_component} tone="red" /><ScoreBar label="Behavioral confirmation" value={assessment.risk_score.behavioral_confirmation_score} tone="amber" />{mlPrediction && <ScoreBar label="ML probability" value={mlPrediction.probability} tone="blue" />}</div></article>
          {evidenceBundle && <article className="workspace-card"><div className="card-heading"><div><p className="eyebrow">Evidence bundle</p><h2>Customer, graph and model evidence</h2></div></div><div className="entity-strip"><div><span>Orders</span><strong>{evidenceBundle.customer_profile.total_order_count}</strong></div><div><span>Refunds</span><strong>{evidenceBundle.customer_profile.total_refund_count}</strong></div><div><span>Cluster size</span><strong>{evidenceBundle.graph_topology.cluster_size}</strong></div><div><span>Shared devices</span><strong>{evidenceBundle.graph_topology.shared_device_fingerprints.length}</strong></div></div><div className="evidence-group"><div className="group-label">Top feature contributions</div>{evidenceBundle.feature_contributions.map((feature) => <div className="evidence-row" key={feature.feature_name}><div className="evidence-row-top"><strong>{feature.feature_name}</strong><span className="evidence-state">{feature.direction}</span></div><p>{feature.description}</p><div className="evidence-values"><span>Contribution <b>{feature.value.toFixed(4)}</b></span></div></div>)}</div></article>}
        </div>
        <aside className="investigation-side">
          <article className="workspace-card exposure-card"><div className="card-heading"><div><p className="eyebrow">Financial exposure</p><h2>Money at risk</h2></div><span className="currency-mark">₹</span></div><div className="exposure-list"><div><span>Realized suspicious amount</span><strong>{formatINR(exposure.realized_suspicious_amount_paise)}</strong></div><div><span>Pending refund exposure</span><strong>{formatINR(exposure.pending_refund_exposure_paise)}</strong></div><div><span>Remaining refundable</span><strong>{formatINR(exposure.remaining_refundable_exposure_paise)}</strong></div></div><p className="card-footnote">Calculated from the current reconstructed financial state of the connected component. Each exposure bucket is a distinct risk-weighted amount.</p></article>
          <article className="workspace-card related-card"><div className="card-heading"><div><p className="eyebrow">Network view</p><h2>Related refunds</h2></div><span className="connection-icon">⌘</span></div><p className="card-copy">Refunds connected through the same customer component.</p><div className="related-list">{relatedRefunds.map((relatedId) => <button key={relatedId} className={`related-item ${relatedId === assessment.refund_id ? "related-item--current" : ""}`} onClick={() => relatedId !== assessment.refund_id && onSelectRelated(relatedId)}><span className="related-status" /><span>{shortId(relatedId)}</span>{relatedId === assessment.refund_id ? <small>Current</small> : <span className="related-arrow">→</span>}</button>)}</div></article>
        </aside>
      </div>
    </section>
  );
}