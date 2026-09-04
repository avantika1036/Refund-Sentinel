/**
 * Displays refunds belonging to the same connected component / cluster.
 *
 * The API returns `component_refund_ids`, which includes all refunds
 * associated with customers in the graph component — including the
 * currently investigated refund ID itself.
 *
 * This component separates the current refund from related refunds,
 * provides cluster metrics, and allows seamless exploration.
 */

interface RelatedRefundsListProps {
  currentRefundId: string;
  componentRefundIds: string[];
  onSelectRefund?: (refundId: string) => void;
}

export function RelatedRefundsList({
  currentRefundId,
  componentRefundIds = [],
  onSelectRefund,
}: RelatedRefundsListProps) {
  const totalClusterCount = componentRefundIds.length;
  const relatedRefundIds = componentRefundIds.filter(
    (id) => id !== currentRefundId
  );
  const relatedCount = relatedRefundIds.length;

  return (
    <section
      className="related-refunds-card"
      aria-labelledby="related-refunds-title"
    >
      <header className="related-refunds-header">
        <div className="related-refunds-title-group">
          <h2 id="related-refunds-title" className="related-refunds-title">
            Component Refunds
          </h2>
          <span
            className="related-refunds-count-badge"
            aria-label={`${totalClusterCount} total refunds in component cluster`}
          >
            {totalClusterCount} Total ({relatedCount} Related)
          </span>
        </div>
        <p className="related-refunds-subtitle">
          All refunds associated with customers in this graph cluster
        </p>
      </header>

      {componentRefundIds.length === 0 ? (
        <p className="related-refunds-empty">
          No component refunds found for this investigation.
        </p>
      ) : (
        <div className="related-refunds-body">
          <ul className="related-refunds-list" role="list">
            {componentRefundIds.map((id) => {
              const isCurrent = id === currentRefundId;

              return (
                <li
                  key={id}
                  className={`related-refund-item ${
                    isCurrent ? "related-refund-item--current" : ""
                  }`}
                >
                  <div className="related-refund-info">
                    <span className="related-refund-id summary-value--mono">
                      {id}
                    </span>
                    <span
                      className={`related-refund-tag ${
                        isCurrent
                          ? "related-refund-tag--current"
                          : "related-refund-tag--related"
                      }`}
                    >
                      {isCurrent ? "Current" : "Related"}
                    </span>
                  </div>

                  {!isCurrent && onSelectRefund && (
                    <button
                      type="button"
                      className="related-refund-action-button"
                      onClick={() => onSelectRefund(id)}
                      aria-label={`Investigate related refund ${id}`}
                    >
                      Investigate →
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}
