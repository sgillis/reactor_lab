import createClient from "openapi-fetch";
import type { components, paths } from "./schema";
import type {
  OptimizationRequest,
  RunStatus,
  RunType,
  SimulationRequest,
  TrainingRequest,
} from "../domain/config";
import type { CandidateView, OptimizationResultView } from "../domain/results";
import type { ModelSummary, RunDetail, RunResults } from "../domain/runs";
import { isRecord } from "../lib/type-guards";

const api = createClient<paths>({ baseUrl: "/" });
type ApiResults = components["schemas"]["SimulationResults"] | components["schemas"]["TrainingResults"] | components["schemas"]["OptimizationResults"];
type ApiResponse<T> = { data?: T; error?: unknown; response: Response };

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function errorMessage(error: unknown, status: number): string {
  if (typeof error === "string") return error;
  if (Array.isArray(error)) {
    const messages = error.map((item) => errorMessage(item, status)).filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  if (isRecord(error)) {
    if ("detail" in error) return errorMessage(error.detail, status);
    if ("message" in error) return errorMessage(error.message, status);
    if ("msg" in error && typeof error.msg === "string") {
      const location = Array.isArray(error.loc) ? error.loc.filter((part) => part !== "body").join(".") : "";
      return location ? `${location}: ${error.msg}` : error.msg;
    }
  }
  return `The API request failed with HTTP ${status}.`;
}

async function unwrap<T>(request: Promise<ApiResponse<T>>): Promise<T> {
  const { data, error, response } = await request;
  if (!response.ok || error !== undefined || data === undefined) {
    throw new ApiError(errorMessage(error, response.status), response.status);
  }
  return data;
}

export async function listRuns(type: RunType | "all", status: RunStatus | "all"): Promise<RunDetail[]> {
  const runs = await unwrap(api.GET("/api/v1/runs", {
    params: { query: { type: type === "all" ? undefined : type, status: status === "all" ? undefined : status } },
  }));
  return runs;
}

export function getRun(runId: string): Promise<RunDetail> {
  return unwrap(api.GET("/api/v1/runs/{run_id}", { params: { path: { run_id: runId } } }));
}

interface OptimizationReportPayload {
  surrogate_feasible_count: number;
  validated_candidates: CandidateView[];
  recommendation: CandidateView | null;
  [key: string]: unknown;
}

function adaptResults(data: ApiResults): RunResults {
  if (data.type === "simulation") {
    return {
      type: "simulation",
      data: {
        summary: { ...data.summary },
        trajectory: {
          time: data.trajectory.time,
          temperature: data.trajectory.temperature,
          oxide_fraction: data.trajectory.oxide_fraction,
          gas_temperature: data.trajectory.gas_temperature,
        },
      },
    };
  }
  if (data.type === "training") {
    return {
      type: "training",
      data: {
        history: data.history,
        simulated: { test: data.test.simulated, ood: data.ood.simulated },
        predicted: { test: data.test.predicted, ood: data.ood.predicted },
      },
    };
  }
  const report = data.report as OptimizationReportPayload;
  const optimization: OptimizationResultView = {
    surrogate_feasible_count: report.surrogate_feasible_count,
    validated_candidates: report.validated_candidates,
    recommendation: report.recommendation,
    report: data.report,
  };
  return { type: "optimization", data: optimization };
}

export async function getRunResults(runId: string): Promise<RunResults | null> {
  try {
    const results = await unwrap(api.GET("/api/v1/runs/{run_id}/results", { params: { path: { run_id: runId } } }));
    return adaptResults(results);
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) return null;
    throw error;
  }
}

export function listModels(): Promise<ModelSummary[]> {
  return unwrap(api.GET("/api/v1/models"));
}

export function getModel(modelId: string): Promise<ModelSummary> {
  return unwrap(api.GET("/api/v1/models/{model_id}", { params: { path: { model_id: modelId } } }));
}

export type SubmitRunVariables =
  | { type: "simulation"; request: SimulationRequest }
  | { type: "training"; request: TrainingRequest }
  | { type: "optimization"; request: OptimizationRequest };

export async function submitRun(variables: SubmitRunVariables): Promise<RunDetail> {
  if (variables.type === "simulation") {
    return unwrap(api.POST("/api/v1/simulation-runs", { body: variables.request }));
  }
  if (variables.type === "training") {
    return unwrap(api.POST("/api/v1/training-runs", { body: variables.request }));
  }
  return unwrap(api.POST("/api/v1/optimization-runs", { body: variables.request }));
}
