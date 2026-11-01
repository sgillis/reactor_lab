export type RunType = "simulation" | "training" | "optimization";
export type RunStatus = "queued" | "running" | "succeeded" | "failed";

export interface Bounds {
  lower: number;
  upper: number;
}

export interface InputBounds {
  peak_temperature: Bounds;
  residence_time: Bounds;
  hydrogen_concentration: Bounds;
  diameter: Bounds;
  initial_oxide_fraction: Bounds;
}

export interface MaterialConfig {
  density: number;
  heat_capacity: number;
  heat_transfer_coefficient: number;
  emissivity: number;
  surroundings_temperature: number;
  melting_temperature: number;
  reaction_pre_exponential: number;
  activation_energy: number;
}

export interface SolverConfig {
  num_steps: number;
}

export interface ReactorConfig {
  peak_temperature: number;
  residence_time: number;
  hydrogen_concentration: number;
}

export interface ParticleConfig {
  diameter: number;
  initial_temperature: number;
  initial_oxide_fraction: number;
}

export interface SimulationRequest {
  reactor: ReactorConfig;
  particle: ParticleConfig;
  material: MaterialConfig;
  solver: SolverConfig;
}

export interface SamplingConfig {
  sampling_bounds: InputBounds;
  ood_bounds: InputBounds;
  initial_temperature: number;
  material: MaterialConfig;
  solver: SolverConfig;
  train_samples: number;
  validation_samples: number;
  test_samples: number;
  ood_samples: number;
  seed: number;
}

export interface TrainingConfig {
  hidden_width: number;
  hidden_depth: number;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  seed: number;
}

export interface TrainingRequest {
  sampling_config: SamplingConfig;
  training_config: TrainingConfig;
}

export interface OptimizationConfig {
  sampling_bounds: InputBounds;
  initial_temperature: number;
  material: MaterialConfig;
  solver: SolverConfig;
  candidate_count: number;
  validation_count: number;
  max_final_oxide_fraction: number;
  seed: number;
}

export interface OptimizationRequest {
  model_id: string;
  optimization_config: OptimizationConfig;
}

export const defaultMaterial = (): MaterialConfig => ({
  density: 5000,
  heat_capacity: 500,
  heat_transfer_coefficient: 100,
  emissivity: 0.8,
  surroundings_temperature: 300,
  melting_temperature: 1800,
  reaction_pre_exponential: 0.05,
  activation_energy: 30000,
});

export const defaultInputBounds = (): InputBounds => ({
  peak_temperature: { lower: 1800, upper: 2800 },
  residence_time: { lower: 0.2, upper: 0.8 },
  hydrogen_concentration: { lower: 20, upper: 100 },
  diameter: { lower: 30e-6, upper: 80e-6 },
  initial_oxide_fraction: { lower: 0.6, upper: 1 },
});

export const defaultOodBounds = (): InputBounds => ({
  ...defaultInputBounds(),
  peak_temperature: { lower: 2800, upper: 3400 },
});

export const defaultSimulationRequest = (): SimulationRequest => ({
  reactor: {
    peak_temperature: 2300,
    residence_time: 0.5,
    hydrogen_concentration: 60,
  },
  particle: {
    diameter: 50e-6,
    initial_temperature: 300,
    initial_oxide_fraction: 0.8,
  },
  material: defaultMaterial(),
  solver: { num_steps: 1000 },
});

export const defaultTrainingRequest = (): TrainingRequest => ({
  sampling_config: {
    sampling_bounds: defaultInputBounds(),
    ood_bounds: defaultOodBounds(),
    initial_temperature: 300,
    material: defaultMaterial(),
    solver: { num_steps: 200 },
    train_samples: 128,
    validation_samples: 32,
    test_samples: 32,
    ood_samples: 32,
    seed: 2025,
  },
  training_config: {
    hidden_width: 32,
    hidden_depth: 2,
    learning_rate: 0.003,
    batch_size: 64,
    epochs: 50,
    seed: 2025,
  },
});

export const defaultOptimizationRequest = (): OptimizationRequest => ({
  model_id: "",
  optimization_config: {
    sampling_bounds: defaultInputBounds(),
    initial_temperature: 300,
    material: defaultMaterial(),
    solver: { num_steps: 200 },
    candidate_count: 4096,
    validation_count: 16,
    max_final_oxide_fraction: 0.75,
    seed: 2025,
  },
});
