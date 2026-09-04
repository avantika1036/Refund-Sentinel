/**
 * API endpoint functions for Refund Sentinel.
 *
 * Provides typed functions for each backend API endpoint.
 */

import { apiRequest } from "./client";
import type {
  AssessmentResponse,
  HealthResponse,
  InvestigationQueueResponse,
  InvestigationResponse,
  ModelEvaluationResponse,
} from "../types/api";

/**
 * Get health check status from the backend.
 *
 * @returns Health check response
 * @throws ApiErrorImpl for network or HTTP errors
 */
export async function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health");
}

/**
 * Get risk assessment for a specific refund.
 *
 * @param refundId - UUID string of the refund to assess
 * @returns Assessment response with risk score and rule outputs
 * @throws ApiErrorImpl for invalid refund ID, authentication, or network errors
 */
export async function getAssessment(
  refundId: string
): Promise<AssessmentResponse> {
  return apiRequest<AssessmentResponse>(
    `/api/v1/assessments/${refundId}`
  );
}

/**
 * Get complete investigation for a specific refund.
 *
 * @param refundId - UUID string of the refund to investigate
 * @returns Investigation response including assessment, exposure, and optional ML prediction
 * @throws ApiErrorImpl for invalid refund ID, authentication, or network errors
 */
export async function getInvestigation(
  refundId: string
): Promise<InvestigationResponse> {
  return apiRequest<InvestigationResponse>(
    `/api/v1/investigations/${refundId}`
  );
}

export async function getInvestigationQueue(): Promise<InvestigationQueueResponse> {
  return apiRequest<InvestigationQueueResponse>("/api/v1/investigations");
}

export async function getModelEvaluation(): Promise<ModelEvaluationResponse> {
  return apiRequest<ModelEvaluationResponse>("/api/v1/model-evaluation");
}
