import { BrainCircuit, FlaskConical, Workflow } from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useModels, useSubmitRun } from "../api/hooks";
import { OptimizationForm, SimulationForm, TrainingForm } from "../components/RunForms";
import { PageHeader } from "../components/ui";
import type { OptimizationRequest, RunType, SimulationRequest, TrainingRequest } from "../domain/config";

const runTypes = [
  { type: "simulation" as const, label: "Simulation", description: "Numerical trajectory", icon: FlaskConical },
  { type: "training" as const, label: "Training", description: "Neural surrogate", icon: BrainCircuit },
  { type: "optimization" as const, label: "Optimization", description: "Validated search", icon: Workflow },
];

export function LaunchRunPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedType = searchParams.get("type");
  const type: RunType = runTypes.some((item) => item.type === requestedType) ? requestedType as RunType : "simulation";
  const initialModelId = searchParams.get("model_id") ?? "";
  const navigate = useNavigate();
  const modelsQuery = useModels(type === "optimization");
  const submitRun = useSubmitRun();

  const selectType = (nextType: RunType) => {
    const next = new URLSearchParams(searchParams);
    next.set("type", nextType);
    if (nextType !== "optimization") next.delete("model_id");
    setSearchParams(next);
    submitRun.reset();
  };

  const submit = (request: SimulationRequest | TrainingRequest | OptimizationRequest) => {
    const onSuccess = (run: { id: string }) => navigate(`/runs/${run.id}`);
    if (type === "simulation") {
      submitRun.mutate({ type, request: request as SimulationRequest }, { onSuccess });
    } else if (type === "training") {
      submitRun.mutate({ type, request: request as TrainingRequest }, { onSuccess });
    } else {
      submitRun.mutate({ type, request: request as OptimizationRequest }, { onSuccess });
    }
  };

  const error = submitRun.error?.message ?? null;
  return (
    <div className="page launch-page">
      <PageHeader eyebrow="New computational experiment" title="Launch a run" description="Configure a scientific workflow. Every run executes asynchronously and preserves its full configuration." />
      <div className="run-type-selector" role="tablist" aria-label="Run type">
        {runTypes.map(({ type: itemType, label, description, icon: Icon }) => (
          <button key={itemType} type="button" role="tab" aria-selected={type === itemType} className={type === itemType ? "active" : ""} onClick={() => selectType(itemType)}>
            <span className="run-type-icon"><Icon size={20} /></span><span><strong>{label}</strong><small>{description}</small></span>
          </button>
        ))}
      </div>
      <div className="form-card">
        {type === "simulation" && <SimulationForm onSubmit={submit} isPending={submitRun.isPending} error={error} />}
        {type === "training" && <TrainingForm onSubmit={submit} isPending={submitRun.isPending} error={error} />}
        {type === "optimization" && <OptimizationForm
          key={initialModelId}
          initialModelId={initialModelId}
          onSubmit={submit}
          isPending={submitRun.isPending}
          error={error}
          models={modelsQuery.data ?? []}
          modelsLoading={modelsQuery.isLoading}
          modelsError={modelsQuery.error?.message ?? null}
          onRetryModels={() => void modelsQuery.refetch()}
        />}
      </div>
    </div>
  );
}
