/**
 * Displays the ML model prediction for a refund investigation.
 *
 * The ML prediction is an optional supplement to the deterministic
 * rule-based assessment. It must not be shown when null.
 *
 * Schema (MLPredictionResponse):
 *   probability  — float [0.0, 1.0]: predicted fraud probability
 *   is_high_risk — boolean: probability >= classification_threshold (default 0.5)
 *
 * The component does not invent additional fields (model name, confidence
 * interval, feature importances) that are not returned by the API.
 */

import type { MLPredictionResponse } from "../../types/api";
import { formatProbability } from "../../utils/formatters";

interface MLPredictionCardProps {
  mlPrediction: MLPredictionResponse;
}

export function MLPredictionCard({ mlPrediction }: MLPredictionCardProps) {
  const { probability, is_high_risk } = mlPrediction;

  return (
    <section
      className="ml-prediction-card"
      aria-labelledby="ml-prediction-title"
    >
      <header className="ml-prediction-header">
        <h2 id="ml-prediction-title" className="ml-prediction-title">
          ML Prediction
        </h2>
        <p className="ml-prediction-subtitle">
          Learned probability signal — supplementary to rule-based assessment
        </p>
      </header>

      <div className="ml-prediction-body">
        <div className="ml-prediction-probability">
          <span className="ml-prediction-probability-label">
            Fraud Probability
          </span>
          <span
            className="ml-prediction-probability-value"
            aria-label={`ML fraud probability: ${formatProbability(probability)}`}
          >
            {formatProbability(probability)}
          </span>
        </div>

        <div className="ml-prediction-classification">
          <span className="ml-prediction-classification-label">
            Classification
          </span>
          <span
            className={`ml-classification-badge ml-classification-badge--${
              is_high_risk ? "high-risk" : "low-risk"
            }`}
            aria-label={`ML classification: ${is_high_risk ? "High Risk" : "Not High Risk"}`}
          >
            {is_high_risk ? "High Risk" : "Not High Risk"}
          </span>
        </div>
      </div>
    </section>
  );
}
