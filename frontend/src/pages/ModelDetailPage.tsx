import { ArrowLeft, ArrowUpRight, BrainCircuit, Link2, Workflow } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { useModel } from "../api/hooks";
import { ConfigView, MetricsGrid } from "../components/ConfigView";
import { CopyId, ErrorState, LoadingState, PageHeader, Panel } from "../components/ui";
import { formatDate } from "../lib/format";

export function ModelDetailPage() {
  const { modelId = "" } = useParams();
  const modelQuery = useModel(modelId);

  if (modelQuery.isLoading) return <div className="page"><LoadingState label="Loading model details" /></div>;
  if (modelQuery.isError) return <div className="page"><Link to="/models" className="back-link"><ArrowLeft size={15} /> Model catalog</Link><div className="panel"><ErrorState title="Model unavailable" message={modelQuery.error.message} onRetry={() => void modelQuery.refetch()} /></div></div>;
  const model = modelQuery.data;
  if (!model) return null;

  return (
    <div className="page model-detail-page">
      <Link to="/models" className="back-link"><ArrowLeft size={15} /> Model catalog</Link>
      <PageHeader
        eyebrow="Trained neural surrogate"
        title={`Model ${model.id.slice(0, 8)}`}
        description={`Created ${formatDate(model.created_at)} · full identifier ${model.id}`}
        actions={<>
          <CopyId id={model.id} label="model ID" />
          <Link to={`/runs/new?type=optimization&model_id=${model.id}`} className="button button-primary"><Workflow size={17} /> Use for optimization</Link>
        </>}
      />
      <section className="model-detail-banner">
        <span className="model-detail-icon"><BrainCircuit size={28} /></span>
        <div><strong>Two-output neural surrogate</strong><p>Predicts final oxide fraction and maximum particle temperature in physical units.</p></div>
        <span className="model-state"><span /> Ready</span>
      </section>
      <Panel title="Evaluation metrics" description="Held-out test and separated-regime performance for both prediction targets.">
        <MetricsGrid metrics={model.metrics} />
      </Panel>
      <div className="detail-two-column">
        <Panel title="Training configuration" description="Architecture and optimization settings stored with the checkpoint."><ConfigView config={model.config} /></Panel>
        <Panel title="Model lineage" description="This model is reproducibly linked to its originating run.">
          <div className="lineage-card">
            <span className="lineage-icon"><Link2 size={20} /></span>
            <div><span>Originating training run</span><Link to={`/runs/${model.training_run_id}`}><code>{model.training_run_id}</code><ArrowUpRight size={15} /></Link><small>{formatDate(model.created_at)}</small></div>
          </div>
          <Link to={`/runs/${model.training_run_id}`} className="button button-secondary button-small">Inspect training results <ArrowUpRight size={15} /></Link>
        </Panel>
      </div>
    </div>
  );
}
