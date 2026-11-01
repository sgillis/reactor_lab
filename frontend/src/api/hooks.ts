import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { RunStatus, RunType } from "../domain/config";
import type { RunDetail } from "../domain/runs";
import {
  getModel,
  getRun,
  getRunResults,
  listModels,
  listRuns,
  submitRun,
} from "./client";

function pollingInterval(failureCount: number, base: number): number {
  return Math.min(15_000, base * 2 ** failureCount);
}

export function useRuns(type: RunType | "all", status: RunStatus | "all") {
  return useQuery({
    queryKey: ["runs", type, status],
    queryFn: () => listRuns(type, status),
    refetchInterval: (query) => {
      const runs = query.state.data;
      return runs?.some((run) => run.status === "queued" || run.status === "running")
        ? pollingInterval(query.state.fetchFailureCount, 4_000)
        : false;
    },
  });
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => getRun(runId),
    enabled: Boolean(runId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "queued" || status === "running"
        ? pollingInterval(query.state.fetchFailureCount, 2_500)
        : false;
    },
  });
}

export function useRunResults(run: RunDetail | undefined) {
  return useQuery({
    queryKey: ["run-results", run?.id],
    queryFn: () => {
      if (!run) throw new Error("Cannot load results without a run");
      return getRunResults(run.id);
    },
    enabled: run?.status === "succeeded",
    staleTime: Number.POSITIVE_INFINITY,
    refetchInterval: false,
  });
}

export function useModels(enabled = true) {
  return useQuery({
    queryKey: ["models"],
    queryFn: listModels,
    enabled,
  });
}

export function useModel(modelId: string) {
  return useQuery({
    queryKey: ["model", modelId],
    queryFn: () => getModel(modelId),
    enabled: Boolean(modelId),
  });
}

export function useSubmitRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: submitRun,
    onSuccess: (run) => {
      queryClient.setQueryData(["run", run.id], run);
      void queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}
