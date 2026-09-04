/**
 * Centralized API client for Refund Sentinel.
 *
 * Handles authentication, error handling, and consistent request/response processing.
 */

import { ApiErrorImpl, type ApiErrorCode } from "../types/api";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "";

const API_KEY = import.meta.env.VITE_API_KEY || "";

/**
 * Normalize base URL to avoid double slashes in request URLs.
 */
function normalizeBaseUrl(url: string): string {
  return url.replace(/\/$/, "");
}

/**
 * Build full URL from base URL and path.
 */
function buildUrl(path: string): string {
  const normalizedBase = normalizeBaseUrl(API_BASE_URL);
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${normalizedBase}${normalizedPath}`;
}

/**
 * Create error object from HTTP response.
 */
function createApiError(
  status: number,
  defaultMessage: string,
  details?: unknown
): ApiErrorImpl {
  let code: ApiErrorCode = "UNKNOWN_ERROR";
  let message = defaultMessage;

  switch (status) {
    case 400:
      code = "INVALID_REFUND_ID";
      message = "Invalid refund ID format";
      break;
    case 401:
      code = "AUTHENTICATION_FAILED";
      message = "Authentication failed";
      break;
    case 404:
      code = "REFUND_NOT_FOUND";
      message = "Refund not found";
      break;
    default:
      if (status >= 500) {
        message = "Server error occurred";
      }
  }

  return new ApiErrorImpl(code, message, status, details);
}

/**
 * Make an authenticated API request.
 *
 * @param path - API endpoint path (e.g., "/health" or "/api/v1/investigations/123")
 * @param options - Fetch options
 * @returns Parsed JSON response
 * @throws ApiErrorImpl for HTTP errors or network failures
 */
export async function apiRequest<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = buildUrl(path);

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  // Add API key header only if configured
  if (API_KEY) {
    headers["X-API-Key"] = API_KEY;
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorDetails: unknown;
      try {
        errorDetails = await response.json();
      } catch {
        // If response is not JSON, use status text
        errorDetails = response.statusText;
      }

      throw createApiError(
        response.status,
        `HTTP error: ${response.status}`,
        errorDetails
      );
    }

    // Parse JSON response
    const data = await response.json();
    return data as T;
  } catch (error) {
    // Re-throw ApiErrorImpl as-is
    if (error instanceof ApiErrorImpl) {
      throw error;
    }

    // Handle network errors
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new ApiErrorImpl(
        "NETWORK_ERROR",
        "Network error: unable to connect to the backend",
        undefined,
        error
      );
    }

    // Handle other unexpected errors
    throw new ApiErrorImpl(
      "UNKNOWN_ERROR",
      error instanceof Error ? error.message : "Unknown error occurred",
      undefined,
      error
    );
  }
}
