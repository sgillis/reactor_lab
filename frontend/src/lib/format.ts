const labels: Record<string, string> = {
  id: "Run ID",
  model_id: "Model",
  training_run_id: "Training run",
  created_at: "Created",
  updated_at: "Last updated",
  peak_temperature: "Peak temperature",
  residence_time: "Residence time",
  hydrogen_concentration: "Hydrogen concentration",
  initial_temperature: "Initial temperature",
  initial_oxide_fraction: "Initial oxide fraction",
  final_oxide_fraction: "Final oxide fraction",
  maximum_temperature: "Maximum temperature",
  max_final_oxide_fraction: "Maximum final oxide fraction",
  melting_temperature: "Melting temperature",
  surroundings_temperature: "Surroundings temperature",
  heat_capacity: "Heat capacity",
  heat_transfer_coefficient: "Heat transfer coefficient",
  reaction_pre_exponential: "Reaction pre-exponential",
  activation_energy: "Activation energy",
  num_steps: "Solver steps",
  hidden_width: "Hidden width",
  hidden_depth: "Hidden depth",
  learning_rate: "Learning rate",
  batch_size: "Batch size",
  train_samples: "Training samples",
  validation_samples: "Validation samples",
  test_samples: "Test samples",
  ood_samples: "OOD samples",
  candidate_count: "Candidate count",
  validation_count: "Validation count",
  reached_melting: "Reached melting",
  surrogate_feasible_count: "Surrogate-feasible candidates",
  train_loss: "Training loss",
  validation_loss: "Validation loss",
  mae: "MAE",
  rmse: "RMSE",
  r2: "R²",
  media_type: "Media type",
  size_bytes: "Size",
  download_url: "Download URL",
  sampling_bounds: "In-distribution bounds",
  ood_bounds: "Separated-regime bounds",
  optimization_config: "Optimization configuration",
  sampling_config: "Sampling configuration",
  training_config: "Network & training",
};

const units: Record<string, string> = {
  peak_temperature: "K",
  initial_temperature: "K",
  maximum_temperature: "K",
  melting_temperature: "K",
  surroundings_temperature: "K",
  residence_time: "s",
  diameter: "m",
  density: "kg/m³",
  heat_capacity: "J/(kg·K)",
  heat_transfer_coefficient: "W/(m²·K)",
  activation_energy: "J/mol",
  size_bytes: "bytes",
};

export function humanize(value: string): string {
  return labels[value] ?? value
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function unitFor(value: string): string | undefined {
  return units[value];
}

export function formatNumber(value: number, maximumFractionDigits = 4): string {
  if (!Number.isFinite(value)) return String(value);
  if (value !== 0 && (Math.abs(value) < 0.001 || Math.abs(value) >= 1_000_000)) {
    return value.toExponential(3);
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(value);
}

export function formatValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    const formatted = formatNumber(value, key.includes("fraction") ? 6 : 4);
    const unit = unitFor(key);
    return unit ? `${formatted} ${unit}` : formatted;
  }
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function relativeTime(value: string): string {
  const date = new Date(value);
  const seconds = Math.round((date.getTime() - Date.now()) / 1000);
  if (Number.isNaN(seconds)) return value;
  const formatter = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (Math.abs(seconds) < 60) return formatter.format(seconds, "second");
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return formatter.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return formatter.format(hours, "hour");
  return formatter.format(Math.round(hours / 24), "day");
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "—";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[index]}`;
}

export function shortId(value: string): string {
  return value.length > 12 ? value.slice(0, 8) : value;
}
