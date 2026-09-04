/**
 * Error message component.
 *
 * Displays user-friendly error messages based on API error codes.
 */

import type { ApiErrorImpl } from "../../types/api";

interface ErrorMessageProps {
  error: ApiErrorImpl;
  onDismiss?: () => void;
}

function getErrorMessage(error: ApiErrorImpl): string {
  switch (error.code) {
    case "INVALID_REFUND_ID":
      return "The refund ID format is invalid. Please enter a valid UUID.";
    case "AUTHENTICATION_FAILED":
      return "Authentication failed. Please check your API key configuration.";
    case "REFUND_NOT_FOUND":
      return "The refund was not found in the system.";
    case "NETWORK_ERROR":
      return "Unable to connect to the backend. Please check your network connection.";
    case "UNKNOWN_ERROR":
    default:
      return error.message || "An unexpected error occurred. Please try again.";
  }
}

export function ErrorMessage({ error, onDismiss }: ErrorMessageProps) {
  const message = getErrorMessage(error);

  return (
    <div className="error-message">
      <p className="error-text">{message}</p>
      {onDismiss && (
        <button
          type="button"
          className="error-dismiss"
          onClick={onDismiss}
          aria-label="Dismiss error"
        >
          Dismiss
        </button>
      )}
    </div>
  );
}
