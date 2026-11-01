import { ArrowUpRight, FlaskConical } from "lucide-react";
import { Link } from "react-router-dom";
import type { RunSummary } from "../domain/runs";
import { formatDate, relativeTime, shortId } from "../lib/format";
import { EmptyState, StatusBadge, TypeBadge } from "./ui";

export function RunTable({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return (
      <EmptyState
        title="No runs match this view"
        description="Launch a computational experiment or adjust the current filters."
        icon={FlaskConical}
        action={<Link to="/runs/new" className="button button-primary button-small">Launch a run <ArrowUpRight size={15} /></Link>}
      />
    );
  }
  return (
    <div className="table-scroll">
      <table className="data-table run-table">
        <thead><tr><th>Run</th><th>Type</th><th>Status</th><th>Created</th><th>Last update</th><th><span className="sr-only">Open</span></th></tr></thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td data-label="Run"><Link to={`/runs/${run.id}`} className="run-id-link"><code>{shortId(run.id)}</code><ArrowUpRight size={13} /></Link></td>
              <td data-label="Type"><TypeBadge type={run.type} /></td>
              <td data-label="Status"><StatusBadge status={run.status} /></td>
              <td data-label="Created"><span title={formatDate(run.created_at)}>{relativeTime(run.created_at)}</span></td>
              <td data-label="Last update"><span title={formatDate(run.updated_at)}>{formatDate(run.updated_at)}</span></td>
              <td><Link to={`/runs/${run.id}`} className="row-action" aria-label={`Open run ${run.id}`}><ArrowUpRight size={17} /></Link></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
