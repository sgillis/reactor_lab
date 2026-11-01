import { AlertCircle, ArrowLeft, ArrowUpRight, Boxes, Clock3, LoaderCircle } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { useRun, useRunResults } from "../api/hooks";
import { ArtifactsList } from "../components/ArtifactsList";
import { ConfigView, MetricsGrid } from "../components/ConfigView";
import { OptimizationCharts } from "../components/OptimizationCharts";
import { OptimizationReport } from "../components/OptimizationReport";
import { SimulationCharts } from "../components/SimulationCharts";
import { TrainingCharts } from "../components/TrainingCharts";
import { CopyId, ErrorState, LoadingState, PageHeader, Panel, PendingState, StatusBadge, TypeBadge } from "../components/ui";
import { formatDate, humanize } from "../lib/format";

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const runQuery = useRun(runId);
  const run = runQuery.data;
  const resultsQuery = useRunResults(run);

  if (runQuery.isLoading) return <div className="page"><LoadingState label="Loading run and scientific results" /></div>;
  if (runQuery.isError) return <div className="page"><Link to="/" className="back-link"><ArrowLeft size={15} /> Run dashboard</Link><div className="panel"><ErrorState title="Run unavailable" message={runQuery.error.message} onRetry={() => void runQuery.refetch()} /></div></div>;
  if (!run) return null;

  return (
    <div className="page run-detail-page">
      <Link to="/" className="back-link"><ArrowLeft size={15} /> Run dashboard</Link>
      <PageHeader
        eyebrow={`${humanize(run.type)} experiment`}
        title={`Run ${run.id.slice(0, 8)}`}
        description={`Created ${formatDate(run.created_at)} · full identifier ${run.id}`}
        actions={<><CopyId id={run.id} /><StatusBadge status={run.status} /></>}
      />

      <section className="run-meta-strip" aria-label="Run metadata">
        <div><span>Workflow</span><TypeBadge type={run.type} /></div>
        <div><span>Current status</span><StatusBadge status={run.status} /></div>
        <div><span>Created</span><strong>{formatDate(run.created_at)}</strong></div>
        <div><span>Last updated</span><strong>{formatDate(run.updated_at)}</strong></div>
        {run.model_id && <div><span>Model lineage</span><Link to={`/models/${run.model_id}`}><Boxes size={14} /><code>{run.model_id.slice(0, 8)}</code><ArrowUpRight size={13} /></Link></div>}
      </section>

      {run.status === "failed" && (
        <div className="failure-banner" role="alert"><span><AlertCircle size={22} /></span><div><h2>Run failed</h2><p>{run.error || "The worker reported an unspecified error. Inspect the run configuration and try again."}</p></div></div>
      )}
      {(run.status === "queued" || run.status === "running") && <PendingState status={run.status} />}

      {run.status === "succeeded" && (
        <div className="results-section">
          {resultsQuery.isLoading ? <Panel><LoadingState label="Loading numerical results" /></Panel> : resultsQuery.isError ? (
            <Panel><ErrorState title="Results unavailable" message={resultsQuery.error.message} onRetry={() => void resultsQuery.refetch()} /></Panel>
          ) : !resultsQuery.data ? (
            <div className="pending-state"><div className="pending-visual"><LoaderCircle size={28} /></div><div><h3>Finalizing result storage</h3><p>The run succeeded, but its numerical result object is not available yet. Retry in a moment.</p><button type="button" className="button button-secondary button-small" onClick={() => void resultsQuery.refetch()}>Check results</button></div></div>
          ) : resultsQuery.data.type === "simulation" ? (
            <>
              <Panel title="Simulation summary" description="Final physical outcomes from the direct numerical trajectory."><MetricsGrid metrics={resultsQuery.data.data.summary} /></Panel>
              <SimulationCharts result={resultsQuery.data.data} />
            </>
          ) : resultsQuery.data.type === "training" ? (
            <>
              {run.metrics && Object.keys(run.metrics).length > 0 && <Panel title="Evaluation metrics" description="Held-out test and separated-regime performance in physical units."><MetricsGrid metrics={run.metrics} /></Panel>}
              <TrainingCharts result={resultsQuery.data.data} />
            </>
          ) : (
            <>
              <OptimizationReport result={resultsQuery.data.data} />
              <OptimizationCharts result={resultsQuery.data.data} />
            </>
          )}
        </div>
      )}

      <div className="detail-two-column run-detail-bottom">
        <Panel title="Run configuration" description="Complete typed inputs submitted to the scientific worker."><ConfigView config={run.config} /></Panel>
        <Panel title="Artifacts" description={run.artifacts.length ? `${run.artifacts.length} downloadable ${run.artifacts.length === 1 ? "object" : "objects"} stored for this run` : "Objects produced by this run"}>
          <ArtifactsList artifacts={run.artifacts} />
        </Panel>
      </div>
      <div className="run-detail-footer"><Clock3 size={14} /> Queued and running runs refresh automatically; completed views remain stable.</div>
    </div>
  );
}
