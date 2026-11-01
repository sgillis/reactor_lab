import type { ArtifactItem } from "../components/ArtifactsList";
import type { RunStatus, RunType } from "./config";
import type { OptimizationResultView, SimulationResultView, TrainingResultView } from "./results";

export interface RunSummary {
  id: string;
  type: RunType;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  model_id: string | null;
}

export interface RunDetail extends RunSummary {
  error: string | null;
  config: Record<string, unknown>;
  metrics: Record<string, unknown> | null;
  artifacts: ArtifactItem[];
}

export type RunResults =
  | { type: "simulation"; data: SimulationResultView }
  | { type: "training"; data: TrainingResultView }
  | { type: "optimization"; data: OptimizationResultView };

export interface ModelSummary {
  id: string;
  training_run_id: string;
  created_at: string;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
}
