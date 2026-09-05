import type { ComparativeBaselineMetrics, ModelEvaluationResponse } from "../../types/api";

interface ModelEvaluationPageProps {
  data: ModelEvaluationResponse | null;
  isLoading: boolean;
  error: string | null;
  onRetry: () => void;
}

function formatMetricName(name: string) {
  return name.replace(/_/g, " ").replace(/\\b\\w/g, (letter) => letter.toUpperCase());
}

function formatModelName(name: string) {
  return name
    .replace(/^baseline_[abc]_/, "")
    .replace(/_/g, " ")
    .replace(/\\b\\w/g, (letter) => letter.toUpperCase());
}

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function inr(value: number) {
  return `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

function BenchmarkRow({
  name,
  metrics,
}: {
  name: string;
  metrics: ComparativeBaselineMetrics;
}) {
  const isFull = name === "baseline_c_full_multi_signal";
  return (
    <tr className={isFull ? "benchmark-row benchmark-row--primary" : "benchmark-row"}>
      <td>
        <strong>{formatModelName(name)}</strong>
        {isFull && <span className="benchmark-badge">Deployed</span>}
      </td>
      <td>{pct(metrics.precision)}</td>
      <td>{pct(metrics.recall)}</td>
      <td>{pct(metrics.f1_score)}</td>
      <td>{metrics.false_positive}</td>
      <td>{metrics.false_negative}</td>
      <td>{inr(metrics.abuse_exposure_captured_inr)}</td>
    </tr>
  );
}

export function ModelEvaluationPage({
  data,
  isLoading,
  error,
  onRetry,
}: ModelEvaluationPageProps) {
  if (isLoading) {
    return (
      <div className="loading-panel">
        <span className="loading-orbit" />
        <p>Reading model runtime status…</p>
        <span className="loading-note">Loading persisted held-out benchmark results</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="empty-panel empty-panel--error">
        <span className="empty-icon">!</span>
        <h2>Model status unavailable</h2>
        <p>{error}</p>
        <button className="button button--primary" onClick={onRetry}>
          Retry connection
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <section>
      <div className="page-header">
        <div>
          <p className="eyebrow">Risk operations / intelligence</p>
          <h1>Model evaluation</h1>
          <p className="page-description">
            Understand what the analytical model can contribute to an investigation — and where it cannot.
          </p>
        </div>
        <div
          className={`model-status model-status--${
            data.model_available ? "available" : "unavailable"
          }`}
        >
          <span className="status-dot" />
          {data.model_available ? "Model loaded" : "Model unavailable"}
        </div>
      </div>

      <div className="evaluation-grid">
        <article className="evaluation-card evaluation-card--wide">
          <div className="card-heading">
            <div>
              <p className="eyebrow">Runtime signal</p>
              <h2>{data.model_available ? "Analytical risk model active" : "Deterministic scoring only"}</h2>
            </div>
            <span className="signal-mark">{data.model_available ? "ML" : "—"}</span>
          </div>
          <p className="card-copy">{data.data_note}</p>
          <div className="model-facts">
            <div><span>Runtime status</span><strong>{data.status}</strong></div>
            <div><span>Artifact version</span><strong>{data.artifact_version ?? "Unavailable"}</strong></div>
            <div><span>Feature inputs</span><strong>{data.feature_count ?? "Unavailable"}</strong></div>
          </div>
        </article>

        <article className="evaluation-card evaluation-card--notice">
          <span className="notice-icon">i</span>
          <div>
            <h2>Signal, not ground truth</h2>
            <p>
              Model probability is one signal alongside deterministic evidence. Held-out benchmark
              results below are comparative measurements, not proof that any individual customer is fraudulent.
            </p>
          </div>
        </article>
      </div>

      {data.benchmark_available ? (
        <article className="evaluation-card metrics-card">
          <div className="card-heading">
            <div>
              <p className="eyebrow">Held-out evaluation</p>
              <h2>Comparative baseline performance</h2>
            </div>
            <span className="available-label">Persisted</span>
          </div>
          <p className="card-copy">
            Thresholds were selected on validation data and the held-out partition was evaluated once.
            False positives and missed abuse remain visible rather than being hidden behind a single score.
          </p>

          <div className="benchmark-table-wrap">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>FP</th>
                  <th>FN</th>
                  <th>Captured exposure</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.benchmark_summary).map(([name, metrics]) => (
                  <BenchmarkRow key={name} name={name} metrics={metrics} />
                ))}
              </tbody>
            </table>
          </div>

          <div className="evaluation-metrics evaluation-metrics--benchmark-summary">
            <div className="evaluation-metric">
              <span>Full-system recall</span>
              <strong>
                {data.benchmark_summary.baseline_c_full_multi_signal
                  ? pct(data.benchmark_summary.baseline_c_full_multi_signal.recall)
                  : "—"}
              </strong>
            </div>
            <div className="evaluation-metric">
              <span>Full-system FP count</span>
              <strong>
                {data.benchmark_summary.baseline_c_full_multi_signal?.false_positive ?? "—"}
              </strong>
            </div>
            <div className="evaluation-metric">
              <span>Abuse exposure captured</span>
              <strong>
                {data.benchmark_summary.baseline_c_full_multi_signal
                  ? inr(data.benchmark_summary.baseline_c_full_multi_signal.abuse_exposure_captured_inr)
                  : "—"}
              </strong>
            </div>
            <div className="evaluation-metric">
              <span>Exposure capture rate</span>
              <strong>
                {data.benchmark_summary.baseline_c_full_multi_signal
                  ? pct(data.benchmark_summary.baseline_c_full_multi_signal.abuse_exposure_capture_rate)
                  : "—"}
              </strong>
            </div>
          </div>
          {data.benchmark_summary.baseline_c_full_multi_signal && (
            <div className="evaluation-metrics evaluation-metrics--benchmark-summary evaluation-metrics--secondary">
              <div className="evaluation-metric">
                <span>Total held-out abuse exposure</span>
                <strong>{inr(data.benchmark_summary.baseline_c_full_multi_signal.total_test_abuse_exposure_inr)}</strong>
              </div>
              <div className="evaluation-metric">
                <span>Graph-only FP count</span>
                <strong>{data.benchmark_summary.baseline_b_graph_structural_only?.false_positive ?? "—"}</strong>
              </div>
              <div className="evaluation-metric">
                <span>Graph-only exposure captured</span>
                <strong>{inr(data.benchmark_summary.baseline_b_graph_structural_only?.abuse_exposure_captured_inr ?? 0)}</strong>
              </div>
              <div className="evaluation-metric">
                <span>Operating threshold</span>
                <strong>{data.benchmark_summary.baseline_c_full_multi_signal.operating_threshold ?? "—"}</strong>
              </div>
            </div>
          )}

          {Object.keys(data.benchmark_protocol).length > 0 && (
            <p className="card-footnote">
              {String(data.benchmark_protocol.heldout_design ?? "Held-out scenario families")}{" "}
              Threshold policy: {String(data.benchmark_protocol.threshold_selection ?? "validation only")}.
            </p>
          )}
        </article>
      ) : data.evaluation_metrics_available ? (
        <article className="evaluation-card metrics-card">
          <div className="card-heading">
            <div><p className="eyebrow">Persisted evaluation</p><h2>Validation metrics</h2></div>
            <span className="available-label">Available</span>
          </div>
          <div className="evaluation-metrics">
            {Object.entries(data.metrics).map(([name, value]) => (
              <div key={name} className="evaluation-metric">
                <span>{formatMetricName(name)}</span>
                <strong>{pct(value)}</strong>
              </div>
            ))}
          </div>
        </article>
      ) : (
        <article className="evaluation-card evaluation-card--unavailable">
          <div className="unavailable-graphic" aria-hidden="true">∿</div>
          <div>
            <p className="eyebrow">Evaluation dataset</p>
            <h2>Held-out benchmark unavailable</h2>
            <p>
              The runtime model is loaded, but no persisted comparative benchmark was found.
              Run <code>python scripts/run_evaluation.py</code> from the repository root and refresh this page.
            </p>
          </div>
          <span className="unavailable-label">No benchmark results</span>
        </article>
      )}
    </section>
  );
}
