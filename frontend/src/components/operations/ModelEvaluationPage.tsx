import type { ModelEvaluationResponse } from "../../types/api";

interface ModelEvaluationPageProps {
  data: ModelEvaluationResponse | null;
  isLoading: boolean;
  error: string | null;
  onRetry: () => void;
}

function formatMetricName(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function ModelEvaluationPage({ data, isLoading, error, onRetry }: ModelEvaluationPageProps) {
  if (isLoading) {
    return <div className="loading-panel"><span className="loading-orbit" /><p>Reading model runtime status…</p><span className="loading-note">Only persisted metadata will be shown</span></div>;
  }
  if (error) {
    return <div className="empty-panel empty-panel--error"><span className="empty-icon">!</span><h2>Model status unavailable</h2><p>{error}</p><button className="button button--primary" onClick={onRetry}>Retry connection</button></div>;
  }
  if (!data) return null;

  return (
    <section>
      <div className="page-header">
        <div>
          <p className="eyebrow">Risk operations / intelligence</p>
          <h1>Model evaluation</h1>
          <p className="page-description">Understand what the analytical model can contribute to an investigation — and where it cannot.</p>
        </div>
        <div className={`model-status model-status--${data.model_available ? "available" : "unavailable"}`}><span className="status-dot" />{data.model_available ? "Model loaded" : "Model unavailable"}</div>
      </div>

      <div className="evaluation-grid">
        <article className="evaluation-card evaluation-card--wide">
          <div className="card-heading"><div><p className="eyebrow">Runtime signal</p><h2>{data.model_available ? "Analytical risk model active" : "Deterministic scoring only"}</h2></div><span className="signal-mark">{data.model_available ? "ML" : "—"}</span></div>
          <p className="card-copy">{data.data_note}</p>
          <div className="model-facts">
            <div><span>Runtime status</span><strong>{data.status}</strong></div>
            <div><span>Artifact version</span><strong>{data.artifact_version ?? "Unavailable"}</strong></div>
            <div><span>Feature inputs</span><strong>{data.feature_count ?? "Unavailable"}</strong></div>
          </div>
        </article>
        <article className="evaluation-card evaluation-card--notice">
          <span className="notice-icon">i</span>
          <div><h2>Signal, not ground truth</h2><p>Use model probability as one input alongside deterministic rule evidence. This interface avoids “fraud confirmed” language.</p></div>
        </article>
      </div>

      {data.evaluation_metrics_available ? (
        <article className="evaluation-card metrics-card">
          <div className="card-heading"><div><p className="eyebrow">Persisted evaluation</p><h2>Validation metrics</h2></div><span className="available-label">Available</span></div>
          <div className="evaluation-metrics">
            {Object.entries(data.metrics).map(([name, value]) => <div key={name} className="evaluation-metric"><span>{formatMetricName(name)}</span><strong>{(value * 100).toFixed(1)}%</strong></div>)}
          </div>
        </article>
      ) : (
        <article className="evaluation-card evaluation-card--unavailable">
          <div className="unavailable-graphic" aria-hidden="true">∿</div>
          <div><p className="eyebrow">Evaluation dataset</p><h2>Evaluation metrics unavailable</h2><p>The deployed artifact contains model weights for inference, but no persisted validation metadata. Train a model with evaluation metadata to populate this section.</p></div>
          <span className="unavailable-label">No persisted metrics</span>
        </article>
      )}
    </section>
  );
}