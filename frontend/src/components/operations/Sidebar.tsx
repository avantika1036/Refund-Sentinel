interface SidebarProps {
  activeView: "queue" | "evaluation";
  onNavigate: (view: "queue" | "evaluation") => void;
  health: "checking" | "ok" | "unavailable";
  mobileOpen: boolean;
  onClose: () => void;
}

function NavIcon({ children }: { children: string }) {
  return <span className="nav-icon" aria-hidden="true">{children}</span>;
}

export function Sidebar({
  activeView,
  onNavigate,
  health,
  mobileOpen,
  onClose,
}: SidebarProps) {
  const healthCopy = {
    checking: "Checking system",
    ok: "System operational",
    unavailable: "System unavailable",
  }[health];

  return (
    <>
      {mobileOpen && <button className="sidebar-scrim" onClick={onClose} aria-label="Close navigation" />}
      <aside className={`sidebar ${mobileOpen ? "sidebar--open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">RS</div>
          <div>
            <p className="brand-name">Refund Sentinel</p>
            <p className="brand-subtitle">Risk operations</p>
          </div>
          <button className="mobile-close" onClick={onClose} aria-label="Close navigation">×</button>
        </div>

        <div className="sidebar-section-label">Workspace</div>
        <nav className="sidebar-nav" aria-label="Primary navigation">
          <button
            className={`nav-item ${activeView === "queue" ? "nav-item--active" : ""}`}
            onClick={() => { onNavigate("queue"); onClose(); }}
          >
            <NavIcon>▦</NavIcon>
            <span>Investigation Queue</span>
          </button>
          <button
            className={`nav-item ${activeView === "evaluation" ? "nav-item--active" : ""}`}
            onClick={() => { onNavigate("evaluation"); onClose(); }}
          >
            <NavIcon>◌</NavIcon>
            <span>Model Evaluation</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="system-status">
            <span className={`status-dot status-dot--${health}`} />
            <div>
              <strong>{healthCopy}</strong>
              <span>{health === "ok" ? "Scoring pipeline active" : "Backend health check"}</span>
            </div>
          </div>
          <p className="sidebar-version">Refund Sentinel · Risk intelligence</p>
        </div>
      </aside>
    </>
  );
}