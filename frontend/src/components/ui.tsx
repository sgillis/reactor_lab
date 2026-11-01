import type { ReactNode } from "react";
import { useState } from "react";
import {
  Activity,
  AlertCircle,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock3,
  Copy,
  Inbox,
  LoaderCircle,
  RefreshCw,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { Link } from "react-router-dom";
import type { RunStatus, RunType } from "../domain/config";
import { humanize, shortId } from "../lib/format";

export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

const statusIcons: Record<RunStatus, LucideIcon> = {
  queued: Clock3,
  running: Activity,
  succeeded: CheckCircle2,
  failed: XCircle,
};

export function StatusBadge({ status }: { status: RunStatus }) {
  const Icon = statusIcons[status] ?? CircleDot;
  return (
    <span className={`status-badge status-${status}`}>
      <Icon size={13} className={status === "running" ? "status-pulse" : ""} />
      {humanize(status)}
    </span>
  );
}

export function TypeBadge({ type }: { type: RunType }) {
  return <span className={`type-badge type-${type}`}>{humanize(type)}</span>;
}

export function Panel({ title, description, actions, children, className = "" }: {
  title?: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || description || actions) && (
        <div className="panel-header">
          <div>
            {title && <h2>{title}</h2>}
            {description && <p>{description}</p>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function LoadingState({ label = "Loading scientific data" }: { label?: string }) {
  return (
    <div className="state-container" role="status">
      <LoaderCircle className="spin" size={25} />
      <span>{label}</span>
    </div>
  );
}

export function ErrorState({ title = "Unable to load data", message, onRetry }: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="state-container error-state" role="alert">
      <span className="state-icon"><AlertCircle size={23} /></span>
      <h3>{title}</h3>
      <p>{message}</p>
      {onRetry && (
        <button type="button" className="button button-secondary button-small" onClick={onRetry}>
          <RefreshCw size={15} /> Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, description, action, icon: Icon = Inbox }: {
  title: string;
  description: string;
  action?: ReactNode;
  icon?: LucideIcon;
}) {
  return (
    <div className="state-container empty-state">
      <span className="state-icon"><Icon size={24} /></span>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}

export function PendingState({ status }: { status: RunStatus }) {
  return (
    <div className="pending-state" role="status">
      <div className="pending-visual"><LoaderCircle size={28} className={status === "running" ? "spin" : ""} /></div>
      <div>
        <h3>{status === "running" ? "Computation in progress" : "Waiting for a worker"}</h3>
        <p>{status === "running"
          ? "Numerical results and charts will appear here when the run completes."
          : "This run is queued. You can stay on this page; its status updates automatically."}</p>
      </div>
    </div>
  );
}

export function CopyId({ id, label = "run ID" }: { id: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(id);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };
  return (
    <button type="button" className="copy-id" onClick={copy} title={`Copy ${label}`}>
      <code>{shortId(id)}</code>
      {copied ? <Check size={13} /> : <Copy size={13} />}
      <span className="sr-only">{copied ? "Copied" : `Copy ${label}`}</span>
    </button>
  );
}

export function TextLink({ to, children }: { to: string; children: ReactNode }) {
  return <Link className="text-link" to={to}>{children}<ChevronRight size={14} /></Link>;
}
