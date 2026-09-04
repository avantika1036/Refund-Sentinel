import { useMemo, useState } from "react";
import type { InvestigationQueueResponse, QueueCaseResponse } from "../../types/api";
import { formatINR, formatScoreAsPercent } from "../../utils/formatters";

type QueueFilter = "all" | "high" | "medium" | "low" | "investigate" | "review" | "triggered" | "ml" | "clustered";

interface QueuePageProps {
  queue: InvestigationQueueResponse | null;
  isLoading: boolean;
  error: string | null;
  onSelectCase: (refundId: string) => void;
  onRetry: () => void;
}

const filterOptions: Array<{ value: QueueFilter; label: string }> = [
  { value: "all", label: "All cases" },
  { value: "high", label: "High risk" },
  { value: "medium", label: "Medium risk" },
  { value: "low", label: "Low risk" },
  { value: "investigate", label: "Investigation recommended" },
  { value: "review", label: "Review recommended" },
  { value: "triggered", label: "Has triggered rules" },
  { value: "ml", label: "ML high-risk signal" },
  { value: "clustered", label: "Connected refunds" },
];

function shortId(value: string) {
  return `${value.slice(0, 8)}…${value.slice(-6)}`;
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat("en-IN", { day: "2-digit", month: "short", year: "numeric" }).format(date);
}

function MetricCard({ label, value, detail, tone }: { label: string; value: string; detail: string; tone?: string }) {
  return (
    <article className={`metric-card ${tone ? `metric-card--${tone}` : ""}`}>
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        <span className="metric-dash" />
      </div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-detail">{detail}</span>
    </article>
  );
}

function CaseCard({ caseItem, onSelect }: { caseItem: QueueCaseResponse; onSelect: () => void }) {
  const riskLabel = `${caseItem.risk_level.charAt(0).toUpperCase()}${caseItem.risk_level.slice(1)} risk`;
  return (
    <button className={`case-card case-card--${caseItem.risk_level}`} onClick={onSelect}>
      <div className="case-card-main">
        <div className="case-heading">
          <span className={`risk-pill risk-pill--${caseItem.risk_level}`}>{riskLabel}</span>
          <span className="case-date">{formatDate(caseItem.requested_at)}</span>
        </div>
        <h3>{shortId(caseItem.refund_id)}</h3>
        <p className="case-meta">Customer {shortId(caseItem.customer_id)} <span>·</span> Component {caseItem.component_id}</p>
      </div>
      <div className="case-score">
        <span className="case-score-label">Risk score</span>
        <strong>{formatScoreAsPercent(caseItem.risk_score)}</strong>
        <span className={`action-label action-label--${caseItem.action}`}>{caseItem.action}</span>
      </div>
      <div className="case-detail">
        <span className="case-detail-label">Refund amount</span>
        <strong>{formatINR(caseItem.requested_amount_paise)}</strong>
        <span className="case-detail-sub">{caseItem.status}</span>
      </div>
      <div className="case-detail">
        <span className="case-detail-label">Signals</span>
        <strong>{caseItem.triggered_rule_ids.length || "None"}</strong>
        <span className="case-detail-sub">{caseItem.triggered_rule_ids.length === 1 ? "rule triggered" : "rules triggered"}</span>
      </div>
      <div className="case-connection">
        <span className="connection-icon" aria-hidden="true">⌘</span>
        <strong>{caseItem.component_refund_count}</strong>
        <span>connected<br />refunds</span>
        <span className="case-arrow" aria-hidden="true">→</span>
      </div>
    </button>
  );
}

export function QueuePage({ queue, isLoading, error, onSelectCase, onRetry }: QueuePageProps) {
  const [filter, setFilter] = useState<QueueFilter>("all");
  const [search, setSearch] = useState("");

  const filteredCases = useMemo(() => {
    if (!queue) return [];
    const query = search.trim().toLowerCase();
    return queue.cases.filter((caseItem) => {
      const matchesSearch = !query || [caseItem.refund_id, caseItem.customer_id, caseItem.component_id].some((value) => value.toLowerCase().includes(query));
      const matchesFilter =
        filter === "all" ||
        caseItem.risk_level === filter ||
        caseItem.action === filter ||
        (filter === "triggered" && caseItem.triggered_rule_ids.length > 0) ||
        (filter === "ml" && caseItem.ml_prediction?.is_high_risk === true) ||
        (filter === "clustered" && caseItem.component_refund_count > 1);
      return matchesSearch && matchesFilter;
    });
  }, [filter, queue, search]);

  if (isLoading) {
    return <div className="loading-panel"><span className="loading-orbit" /><p>Reconstructing investigation queue…</p><span className="loading-note">Running real risk assessments from the event ledger</span></div>;
  }

  if (error) {
    return <div className="empty-panel empty-panel--error"><span className="empty-icon">!</span><h2>Queue unavailable</h2><p>{error}</p><button className="button button--primary" onClick={onRetry}>Retry connection</button></div>;
  }

  const metrics = queue?.metrics;
  return (
    <section>
      <div className="page-header">
        <div>
          <p className="eyebrow">Risk operations / workspace</p>
          <h1>Investigation queue</h1>
          <p className="page-description">Prioritized refund cases surfaced from the live event ledger and deterministic risk engine.</p>
        </div>
        <div className="data-trust"><span className="trust-dot" /> Live backend data</div>
      </div>

      <div className="metric-grid">
        <MetricCard label="Open queue" value={String(metrics?.open_case_count ?? 0)} detail={`${metrics?.high_risk_count ?? 0} high-risk cases`} tone="blue" />
        <MetricCard label="Pending exposure" value={formatINR(metrics?.pending_refund_exposure_paise ?? 0)} detail="Risk-weighted, per refund" tone="amber" />
        <MetricCard label="Realized suspicious" value={formatINR(metrics?.realized_suspicious_amount_paise ?? 0)} detail={`${metrics?.triggered_case_count ?? 0} cases with evidence`} tone="red" />
        <MetricCard label="Connected cases" value={String(metrics?.clustered_case_count ?? 0)} detail="Cases with related refunds" tone="violet" />
      </div>

      <div className="queue-toolbar">
        <div>
          <h2>Cases requiring attention</h2>
          <p>{filteredCases.length} of {queue?.cases.length ?? 0} cases · sorted by risk score and evidence</p>
        </div>
        <div className="queue-controls">
          <label className="search-box">
            <span aria-hidden="true">⌕</span>
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search refund, customer or component" aria-label="Search cases" />
          </label>
          <select value={filter} onChange={(event) => setFilter(event.target.value as QueueFilter)} aria-label="Filter investigation cases">
            {filterOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </div>
      </div>

      {filteredCases.length > 0 ? (
        <div className="case-list">
          {filteredCases.map((caseItem) => <CaseCard key={caseItem.refund_id} caseItem={caseItem} onSelect={() => onSelectCase(caseItem.refund_id)} />)}
        </div>
      ) : (
        <div className="empty-panel"><span className="empty-icon">⌕</span><h2>No matching cases</h2><p>Try clearing the search or choosing a broader queue filter. No cases were fabricated for this view.</p></div>
      )}
    </section>
  );
}