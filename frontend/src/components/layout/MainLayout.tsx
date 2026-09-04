/**
 * Main layout component for Refund Sentinel.
 *
 * Provides the application shell with header and responsive content area.
 */

import { Header } from "./Header";

interface MainLayoutProps {
  children: React.ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="main-layout">
      <Header />
      <main className="main-content">
        <div className="container">{children}</div>
      </main>
    </div>
  );
}
