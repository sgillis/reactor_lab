import { useEffect, useState } from "react";
import { Atom, Boxes, LayoutDashboard, Menu, Plus, X } from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

const navigation = [
  { to: "/", label: "Run dashboard", icon: LayoutDashboard, end: true },
  { to: "/runs/new", label: "Launch a run", icon: Plus },
  { to: "/models", label: "Model catalog", icon: Boxes },
];

export function AppShell() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => setMenuOpen(false), [location.pathname]);

  return (
    <div className="app-shell">
      <button
        type="button"
        className={`nav-scrim ${menuOpen ? "is-visible" : ""}`}
        aria-label="Close navigation"
        onClick={() => setMenuOpen(false)}
      />
      <aside className={`sidebar ${menuOpen ? "is-open" : ""}`}>
        <div className="brand-row">
          <NavLink to="/" className="brand" aria-label="Reactor Lab dashboard">
            <span className="brand-mark"><Atom size={24} strokeWidth={1.7} /></span>
            <span>
              <strong>Reactor Lab</strong>
              <small>Scientific workspace</small>
            </span>
          </NavLink>
          <button
            type="button"
            className="icon-button sidebar-close"
            aria-label="Close navigation"
            onClick={() => setMenuOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        <div className="workspace-label">Workspace</div>
        <nav className="primary-nav" aria-label="Primary navigation">
          {navigation.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              <Icon size={18} strokeWidth={1.8} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-note">
          <span className="note-kicker">COMPUTATIONAL REACTOR</span>
          <p>Explore synthetic particle physics, train neural surrogates, and validate better operating regimes.</p>
        </div>
        <div className="sidebar-footer">
          <span className="environment-dot" />
          Local research environment
        </div>
      </aside>

      <div className="main-column">
        <header className="mobile-header">
          <button
            type="button"
            className="icon-button"
            aria-label="Open navigation"
            onClick={() => setMenuOpen(true)}
          >
            <Menu size={21} />
          </button>
          <NavLink to="/" className="mobile-brand">
            <Atom size={21} /> Reactor Lab
          </NavLink>
          <span className="mobile-header-spacer" />
        </header>
        <main className="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
