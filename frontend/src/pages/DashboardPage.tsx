import { Activity, ArrowUpRight, Boxes, CheckCircle2, FlaskConical, RefreshCw } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { useModels, useRuns } from "../api/hooks";
import { RunTable } from "../components/RunTable";
import { ErrorState, LoadingState, PageHeader, Panel } from "../components/ui";
import type { RunStatus, RunType } from "../domain/config";

const typeOptions: Array<{ value: RunType | "all"; label: string }> = [
  { value: "all", label: "All runs" },
  { value: "simulation", label: "Simulation" },
  { value: "training", label: "Training" },
  { value: "optimization", label: "Optimization" },
];

export function DashboardPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedType = searchParams.get("type");
  const requestedStatus = searchParams.get("status");
  const type = typeOptions.some((option) => option.value === requestedType) ? requestedType as RunType | "all" : "all";
  const status = ["queued", "running", "succeeded", "failed"].includes(requestedStatus ?? "") ? requestedStatus as RunStatus : "all";
  const runsQuery = useRuns(type, status);
  const modelsQuery = useModels();
  const runs = runsQuery.data ?? [];
  const activeCount = runs.filter((run) => run.status === "queued" || run.status === "running").length;
  const succeededCount = runs.filter((run) => run.status === "succeeded").length;

  const updateFilter = (key: "type" | "status", value: string) => {
    const next = new URLSearchParams(searchParams);
    if (value === "all") next.delete(key);
    else next.set(key, value);
    setSearchParams(next, { replace: true });
  };

  return (
    <div className="page dashboard-page">
      <PageHeader
        eyebrow="Computational workspace"
        title="Experiment dashboard"
        description="Monitor numerical experiments, inspect scientific results, and trace trained model lineage."
        actions={<Link to="/runs/new" className="button button-primary"><FlaskConical size={17} /> Launch a run</Link>}
      />

      <section className="stats-grid" aria-label="Workspace summary">
        <article className="stat-card"><span className="stat-icon teal"><Activity size={19} /></span><div><span>Visible runs</span><strong>{runsQuery.isLoading ? "—" : runs.length}</strong><small>in the current view</small></div></article>
        <article className="stat-card"><span className="stat-icon amber"><Activity size={19} /></span><div><span>Active work</span><strong>{runsQuery.isLoading ? "—" : activeCount}</strong><small>queued or running</small></div></article>
        <article className="stat-card"><span className="stat-icon green"><CheckCircle2 size={19} /></span><div><span>Completed</span><strong>{runsQuery.isLoading ? "—" : succeededCount}</strong><small>successful runs</small></div></article>
        <article className="stat-card"><span className="stat-icon violet"><Boxes size={19} /></span><div><span>Trained models</span><strong>{modelsQuery.isLoading ? "—" : modelsQuery.data?.length ?? "—"}</strong><small>ready for optimization</small></div></article>
      </section>

      <div className="filter-bar">
        <div className="filter-tabs" aria-label="Filter runs by type">
          {typeOptions.map((option) => <button key={option.value} type="button" className={type === option.value ? "active" : ""} onClick={() => updateFilter("type", option.value)}>{option.label}</button>)}
        </div>
        <label className="status-filter">Status
          <select value={status} onChange={(event) => updateFilter("status", event.target.value)}>
            <option value="all">Any status</option><option value="queued">Queued</option><option value="running">Running</option><option value="succeeded">Succeeded</option><option value="failed">Failed</option>
          </select>
        </label>
      </div>

      <Panel
        title="Experiment runs"
        description={activeCount ? `${activeCount} active ${activeCount === 1 ? "run" : "runs"} · updates automatically` : "Newest computational experiments first"}
        actions={<button type="button" className="button button-quiet button-small" onClick={() => void runsQuery.refetch()} disabled={runsQuery.isFetching}><RefreshCw size={15} className={runsQuery.isRefetching ? "spin" : ""} /> Refresh</button>}
        className="runs-panel"
      >
        {runsQuery.isLoading ? <LoadingState label="Loading experiment runs" /> : runsQuery.isError ? (
          <ErrorState message={runsQuery.error.message} onRetry={() => void runsQuery.refetch()} />
        ) : <RunTable runs={runs} />}
      </Panel>

      <div className="dashboard-footnote"><span>Ready to explore a new operating regime?</span><Link to="/runs/new">Configure an experiment <ArrowUpRight size={14} /></Link></div>
    </div>
  );
}
