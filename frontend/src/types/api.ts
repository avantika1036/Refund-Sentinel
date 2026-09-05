/**
 * TypeScript types matching the Refund Sentinel backend API schemas.
 *
 * These types are derived directly from backend/app/api/schemas.py
 * and should be kept in sync with backend changes.
 */

// Risk level enumeration
export type RiskLevel = "low" | "medium" | "high";

// Decision action enumeration
export type DecisionAction = "allow" | "review" | "investigate";

// Health check response
export interface HealthResponse {
  status: string;
}

// Rule evidence response
export interface RuleEvidenceResponse {
  rule_id: string;
  triggered: boolean;
  evidence_type: string;
  evidence_value: number | boolean | null;
  evidence_threshold: number | null;
  base_signal_weight: number;
  notes: string;
}

// Risk score response
export interface RiskScoreResponse {
  rule_signal_component: number;
  behavioral_confirmation_score: number;
  cluster_signal_component: number;
  final_score: number;
}

// Assessment response
export interface AssessmentResponse {
  refund_id: string;
  customer_id: string;
  component_id: string;
  risk_level: RiskLevel;
  action: DecisionAction;
  triggered_rule_ids: string[];
  behavioral_confirmation_score: number;
  risk_score: RiskScoreResponse;
  rule_outputs: RuleEvidenceResponse[];
  explanation: string;
}

// Financial exposure response
export interface FinancialExposureResponse {
  realized_suspicious_amount_paise: number;
  pending_refund_exposure_paise: number;
  remaining_refundable_exposure_paise: number;
}

// ML prediction response
export interface MLPredictionResponse {
  probability: number;
  is_high_risk: boolean;
}

export interface CustomerProfileResponse {
  customer_id: string;
  email?: string | null;
  phone?: string | null;
  created_at?: string | null;
  total_order_count: number;
  total_refund_count: number;
  total_paid_paise: number;
  total_refunded_paise: number;
  refund_rate_by_count: number;
  refund_rate_by_amount: number;
}

export interface GraphTopologyResponse {
  cluster_id: string;
  cluster_size: number;
  connected_customer_ids: string[];
  connected_refund_ids: string[];
  shared_ip_addresses: string[];
  shared_device_fingerprints: string[];
  shared_bank_accounts: string[];
  is_multi_entity_cluster: boolean;
}

export interface FeatureContributionResponse {
  feature_name: string;
  value: number;
  direction: string;
  description: string;
}

export interface EvidenceBundleResponse {
  refund_id: string;
  assessed_at: string;
  risk_level: string;
  action: string;
  final_risk_score: number;
  behavioral_confirmation_score: number;
  customer_profile: CustomerProfileResponse;
  financial_exposure: FinancialExposureResponse;
  graph_topology: GraphTopologyResponse;
  rule_violations: RuleEvidenceResponse[];
  feature_contributions: FeatureContributionResponse[];
}

export interface InvestigationExplanationResponse {
  headline: string;
  narrative_summary: string;
  key_risk_drivers: string[];
  suggested_action_rationale: string;
  is_llm_generated: boolean;
}

// Investigation response
export interface InvestigationResponse {
  assessment: AssessmentResponse;
  exposure: FinancialExposureResponse;
  component_refund_ids: string[];
  ml_prediction: MLPredictionResponse | null;
  evidence_bundle?: EvidenceBundleResponse | null;
  explanation_summary?: InvestigationExplanationResponse | null;
}

export interface QueueCaseResponse {
  refund_id: string;
  customer_id: string;
  component_id: string;
  status: string;
  requested_at: string;
  requested_amount_paise: number;
  risk_level: RiskLevel;
  action: DecisionAction;
  risk_score: number;
  triggered_rule_ids: string[];
  component_refund_count: number;
  exposure: FinancialExposureResponse;
  ml_prediction: MLPredictionResponse | null;
}

export interface QueueMetricsResponse {
  open_case_count: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  triggered_case_count: number;
  clustered_case_count: number;
  pending_refund_exposure_paise: number;
  realized_suspicious_amount_paise: number;
}

export interface InvestigationQueueResponse {
  cases: QueueCaseResponse[];
  metrics: QueueMetricsResponse;
}

export interface ModelEvaluationResponse {
  model_available: boolean;
  status: string;
  artifact_version: number | null;
  feature_count: number | null;
  evaluation_metrics_available: boolean;
  metrics: Record<string, number>;
  data_note: string;
}

// API error types
export type ApiErrorCode =
  | "INVALID_REFUND_ID"
  | "AUTHENTICATION_FAILED"
  | "REFUND_NOT_FOUND"
  | "NETWORK_ERROR"
  | "UNKNOWN_ERROR";

export interface ApiError {
  code: ApiErrorCode;
  message: string;
  status?: number;
  details?: unknown;
}

// Helper to create API errors
export class ApiErrorImpl extends Error implements ApiError {
  code: ApiErrorCode;
  status?: number;
  details?: unknown;

  constructor(code: ApiErrorCode, message: string, status?: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}
