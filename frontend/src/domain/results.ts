export interface SimulationResultView {
  summary: Record<string, unknown>;
  trajectory: {
    time: number[];
    temperature: number[];
    oxide_fraction: number[];
    gas_temperature: number[];
  };
}

export interface TrainingResultView {
  history: Record<string, number[]>;
  simulated: Record<string, number[][]>;
  predicted: Record<string, number[][]>;
}

export interface CandidateView {
  features: number[];
  surrogate_final_oxide_fraction: number;
  surrogate_maximum_temperature: number;
  simulator_final_oxide_fraction: number;
  simulator_maximum_temperature: number;
  simulator_reached_melting: boolean;
  simulator_feasible: boolean;
}

export interface OptimizationResultView {
  surrogate_feasible_count: number;
  validated_candidates: CandidateView[];
  recommendation: CandidateView | null;
  report: Record<string, unknown>;
}
