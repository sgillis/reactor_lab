import { ArrowUpRight, Boxes, BrainCircuit, CalendarDays, Plus, Workflow } from "lucide-react";
import { Link } from "react-router-dom";
import { useModels } from "../api/hooks";
import { EmptyState, ErrorState, LoadingState, PageHeader } from "../components/ui";
import { formatDate, shortId } from "../lib/format";

export function ModelsPage() {
  const modelsQuery = useModels();
  const models = modelsQuery.data ?? [];
  return (
    <div className="page models-page">
      <PageHeader
        eyebrow="Surrogate registry"
        title="Model catalog"
        description="Inspect trained neural surrogates, evaluation metrics, and their originating training runs."
        actions={<Link to="/runs/new?type=training" className="button button-primary"><Plus size={17} /> Train a model</Link>}
      />
      {modelsQuery.isLoading ? <div className="panel"><LoadingState label="Loading trained models" /></div> : modelsQuery.isError ? (
        <div className="panel"><ErrorState message={modelsQuery.error.message} onRetry={() => void modelsQuery.refetch()} /></div>
      ) : models.length === 0 ? (
        <div className="panel"><EmptyState
          title="No trained models yet"
          description="Launch a training run to create a reusable neural surrogate for optimization."
          icon={Boxes}
          action={<Link to="/runs/new?type=training" className="button button-primary button-small">Launch training <ArrowUpRight size={15} /></Link>}
        /></div>
      ) : (
        <div className="model-grid">
          {models.map((model) => (
            <article className="model-card" key={model.id}>
              <div className="model-card-top"><span className="model-icon"><BrainCircuit size={22} /></span><span className="model-state"><span /> Ready</span></div>
              <div className="model-card-title"><span>NEURAL SURROGATE</span><h2>{shortId(model.id)}</h2><code>{model.id}</code></div>
              <div className="model-card-meta"><CalendarDays size={15} /><span>Trained {formatDate(model.created_at)}</span></div>
              <div className="model-lineage"><span>Training run</span><Link to={`/runs/${model.training_run_id}`}><code>{shortId(model.training_run_id)}</code><ArrowUpRight size={13} /></Link></div>
              <div className="model-card-actions">
                <Link className="button button-secondary button-small" to={`/models/${model.id}`}>View model</Link>
                <Link className="button button-quiet button-small" to={`/runs/new?type=optimization&model_id=${model.id}`}><Workflow size={15} /> Optimize</Link>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
