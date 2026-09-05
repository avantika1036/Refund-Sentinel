import { useCallback, useEffect, useState } from "react";
import { getHealth, getInvestigationQueue, getModelEvaluation } from "./api/endpoints";
import { ModelEvaluationPage } from "./components/operations/ModelEvaluationPage";
import { InvestigationWorkspace } from "./components/operations/InvestigationWorkspace";
import { QueuePage } from "./components/operations/QueuePage";
import { Sidebar } from "./components/operations/Sidebar";
import type { InvestigationQueueResponse, ModelEvaluationResponse } from "./types/api";
import { ApiErrorImpl } from "./types/api";
import "./App.css";

type View = "queue" | "evaluation";
type HealthState = "checking" | "ok" | "unavailable";

function errorMessage(error: unknown) {
  if (error instanceof ApiErrorImpl) return error.message;
  return error instanceof Error ? error.message : "Unable to connect to the backend";
}

function App() {
  const [activeView, setActiveView] = useState<View>("queue");
  const [selectedRefundId, setSelectedRefundId] = useState<string | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [health, setHealth] = useState<HealthState>("checking");
  const [queue, setQueue] = useState<InvestigationQueueResponse | null>(null);
  const [queueLoading, setQueueLoading] = useState(true);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [modelData, setModelData] = useState<ModelEvaluationResponse | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);

  const loadQueue = useCallback(async () => {
    setQueueLoading(true);
    setQueueError(null);
    try {
      setQueue(await getInvestigationQueue());
    } catch (error) {
      setQueueError(errorMessage(error));
    } finally {
      setQueueLoading(false);
    }
  }, []);

  const loadModelData = useCallback(async () => {
    setModelLoading(true);
    setModelError(null);
    try {
      setModelData(await getModelEvaluation());
    } catch (error) {
      setModelError(errorMessage(error));
    } finally {
      setModelLoading(false);
    }
  }, []);

  useEffect(() => {
    // This effect intentionally kicks off external I/O on mount; the async
    // loader owns the resulting loading/error/data state transitions.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadQueue();
    getHealth()
      .then((response) => setHealth(response.status === "ok" ? "ok" : "unavailable"))
      .catch(() => setHealth("unavailable"));
  }, [loadQueue]);

  useEffect(() => {
    if (activeView === "evaluation" && !modelData && !modelLoading) {
      // This effect intentionally kicks off external I/O when the view opens.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      void loadModelData();
    }
  }, [activeView, loadModelData, modelData, modelLoading]);

  const navigate = (view: View) => {
    setActiveView(view);
    setSelectedRefundId(null);
  };

  const showQueue = activeView === "queue" && selectedRefundId === null;
  return (
    <div className="app-shell">
      <Sidebar
        activeView={activeView}
        onNavigate={navigate}
        health={health}
        mobileOpen={mobileNavOpen}
        onClose={() => setMobileNavOpen(false)}
      />
      <div className="app-main">
        <header className="mobile-header">
          <button className="menu-button" onClick={() => setMobileNavOpen(true)} aria-label="Open navigation">☰</button>
          <span>Refund Sentinel</span>
          <span className="mobile-header-mark">RS</span>
        </header>
        <main className="main-content">
          {showQueue && (
            <QueuePage
              queue={queue}
              isLoading={queueLoading}
              error={queueError}
              onSelectCase={(refundId) => setSelectedRefundId(refundId)}
              onRetry={() => void loadQueue()}
            />
          )}
          {activeView === "evaluation" && (
            <ModelEvaluationPage
              data={modelData}
              isLoading={modelLoading}
              error={modelError}
              onRetry={() => void loadModelData()}
            />
          )}
          {activeView === "queue" && selectedRefundId !== null && (
            <InvestigationWorkspace
              refundId={selectedRefundId}
              onBack={() => setSelectedRefundId(null)}
              onSelectRelated={setSelectedRefundId}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;