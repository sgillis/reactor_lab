import { useState, type FormEvent } from "react";
import { BrainCircuit, FlaskConical, Info, LoaderCircle, Play, Workflow } from "lucide-react";
import { Link } from "react-router-dom";
import {
  defaultOptimizationRequest,
  defaultSimulationRequest,
  defaultTrainingRequest,
  type OptimizationRequest,
  type SimulationRequest,
  type TrainingRequest,
} from "../domain/config";
import { formatDate, shortId } from "../lib/format";
import {
  AdvancedSection,
  FormError,
  FormSection,
  InputBoundsFields,
  MaterialFields,
  NumberField,
  validateBounds,
} from "./forms";

interface FormProps<T> {
  onSubmit: (request: T) => void;
  isPending: boolean;
  error: string | null;
}

function SubmitButton({ pending, label }: { pending: boolean; label: string }) {
  return (
    <button type="submit" className="button button-primary submit-run" disabled={pending}>
      {pending ? <LoaderCircle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}
      {pending ? "Submitting run…" : label}
    </button>
  );
}

export function SimulationForm({ onSubmit, isPending, error }: FormProps<SimulationRequest>) {
  const [request, setRequest] = useState(defaultSimulationRequest);
  const [localError, setLocalError] = useState<string | null>(null);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (request.reactor.peak_temperature < request.material.surroundings_temperature) {
      setLocalError("Peak temperature must be at least the surroundings temperature.");
      return;
    }
    setLocalError(null);
    onSubmit(request);
  };

  return (
    <form className="run-form" onSubmit={submit}>
      <div className="form-intro">
        <span className="form-intro-icon"><FlaskConical size={22} /></span>
        <div><h2>Direct numerical simulation</h2><p>Integrate particle temperature and oxide reduction through a synthetic reactor trajectory.</p></div>
      </div>
      <FormSection title="Reactor conditions" description="Thermal environment and gas composition.">
        <NumberField name="peak_temperature" value={request.reactor.peak_temperature} min={0.000001} onChange={(peak_temperature) => setRequest({ ...request, reactor: { ...request.reactor, peak_temperature } })} />
        <NumberField name="residence_time" value={request.reactor.residence_time} min={0.000001} onChange={(residence_time) => setRequest({ ...request, reactor: { ...request.reactor, residence_time } })} />
        <NumberField name="hydrogen_concentration" value={request.reactor.hydrogen_concentration} min={0} onChange={(hydrogen_concentration) => setRequest({ ...request, reactor: { ...request.reactor, hydrogen_concentration } })} />
      </FormSection>
      <FormSection title="Particle initial state" description="Initial particle geometry, temperature, and composition.">
        <NumberField name="diameter" value={request.particle.diameter} min={0.000000001} step="any" onChange={(diameter) => setRequest({ ...request, particle: { ...request.particle, diameter } })} />
        <NumberField name="initial_temperature" value={request.particle.initial_temperature} min={0.000001} onChange={(initial_temperature) => setRequest({ ...request, particle: { ...request.particle, initial_temperature } })} />
        <NumberField name="initial_oxide_fraction" value={request.particle.initial_oxide_fraction} min={0} max={1} onChange={(initial_oxide_fraction) => setRequest({ ...request, particle: { ...request.particle, initial_oxide_fraction } })} />
      </FormSection>
      <FormSection title="Numerical solver" description="Fixed-step integration over the residence time." columns={2}>
        <NumberField name="num_steps" value={request.solver.num_steps} min={1} step={1} onChange={(num_steps) => setRequest({ ...request, solver: { num_steps: Math.round(num_steps) } })} />
      </FormSection>
      <AdvancedSection title="Advanced material model" description="Override the default density, transport, and reaction parameters.">
        <MaterialFields value={request.material} onChange={(material) => setRequest({ ...request, material })} />
      </AdvancedSection>
      <FormError message={localError ?? error} />
      <div className="form-footer">
        <span className="async-note"><Info size={15} /> Runs asynchronously in a dedicated worker.</span>
        <SubmitButton pending={isPending} label="Launch simulation" />
      </div>
    </form>
  );
}

export function TrainingForm({ onSubmit, isPending, error }: FormProps<TrainingRequest>) {
  const [request, setRequest] = useState(defaultTrainingRequest);
  const [localError, setLocalError] = useState<string | null>(null);
  const sampling = request.sampling_config;
  const training = request.training_config;
  const setSampling = (sampling_config: typeof sampling) => setRequest({ ...request, sampling_config });
  const setTraining = (training_config: typeof training) => setRequest({ ...request, training_config });
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const boundsError = validateBounds(sampling.sampling_bounds) ?? validateBounds(sampling.ood_bounds);
    if (boundsError) {
      setLocalError(boundsError);
      return;
    }
    setLocalError(null);
    onSubmit(request);
  };

  return (
    <form className="run-form" onSubmit={submit}>
      <div className="form-intro">
        <span className="form-intro-icon"><BrainCircuit size={22} /></span>
        <div><h2>Train a neural surrogate</h2><p>Generate reproducible simulation data, train a two-output MLP, and evaluate a separated regime.</p></div>
      </div>
      <FormSection title="Synthetic dataset" description="Sample counts for each reproducible data split." columns={4}>
        <NumberField name="train_samples" value={sampling.train_samples} min={1} step={1} onChange={(train_samples) => setSampling({ ...sampling, train_samples: Math.round(train_samples) })} />
        <NumberField name="validation_samples" value={sampling.validation_samples} min={1} step={1} onChange={(validation_samples) => setSampling({ ...sampling, validation_samples: Math.round(validation_samples) })} />
        <NumberField name="test_samples" value={sampling.test_samples} min={1} step={1} onChange={(test_samples) => setSampling({ ...sampling, test_samples: Math.round(test_samples) })} />
        <NumberField name="ood_samples" value={sampling.ood_samples} min={1} step={1} onChange={(ood_samples) => setSampling({ ...sampling, ood_samples: Math.round(ood_samples) })} />
      </FormSection>
      <FormSection title="Network & optimization" description="MLP architecture and deterministic Optax training settings.">
        <NumberField name="hidden_width" value={training.hidden_width} min={1} step={1} onChange={(hidden_width) => setTraining({ ...training, hidden_width: Math.round(hidden_width) })} />
        <NumberField name="hidden_depth" value={training.hidden_depth} min={1} step={1} onChange={(hidden_depth) => setTraining({ ...training, hidden_depth: Math.round(hidden_depth) })} />
        <NumberField name="learning_rate" value={training.learning_rate} min={0.000000001} onChange={(learning_rate) => setTraining({ ...training, learning_rate })} />
        <NumberField name="batch_size" value={training.batch_size} min={1} step={1} onChange={(batch_size) => setTraining({ ...training, batch_size: Math.round(batch_size) })} />
        <NumberField name="epochs" value={training.epochs} min={1} step={1} onChange={(epochs) => setTraining({ ...training, epochs: Math.round(epochs) })} />
        <NumberField name="seed" label="Training seed" value={training.seed} min={0} step={1} onChange={(seed) => setTraining({ ...training, seed: Math.round(seed) })} />
      </FormSection>
      <AdvancedSection title="Sampling, solver & material" description="Edit input distributions, OOD bounds, integration, and physical parameters.">
        <div className="advanced-stack">
          <div className="bounds-grid">
            <InputBoundsFields value={sampling.sampling_bounds} onChange={(sampling_bounds) => setSampling({ ...sampling, sampling_bounds })} />
            <InputBoundsFields title="Separated-regime (OOD) bounds" value={sampling.ood_bounds} onChange={(ood_bounds) => setSampling({ ...sampling, ood_bounds })} />
          </div>
          <FormSection title="Data generation" description="Shared physical and numerical settings.">
            <NumberField name="initial_temperature" value={sampling.initial_temperature} min={0.000001} onChange={(initial_temperature) => setSampling({ ...sampling, initial_temperature })} />
            <NumberField name="num_steps" value={sampling.solver.num_steps} min={1} step={1} onChange={(num_steps) => setSampling({ ...sampling, solver: { num_steps: Math.round(num_steps) } })} />
            <NumberField name="seed" label="Sampling seed" value={sampling.seed} min={0} step={1} onChange={(seed) => setSampling({ ...sampling, seed: Math.round(seed) })} />
          </FormSection>
          <MaterialFields value={sampling.material} onChange={(material) => setSampling({ ...sampling, material })} />
        </div>
      </AdvancedSection>
      <FormError message={localError ?? error} />
      <div className="form-footer">
        <span className="async-note"><Info size={15} /> The trained model is added to the catalog on success.</span>
        <SubmitButton pending={isPending} label="Launch training" />
      </div>
    </form>
  );
}

export interface ModelOption {
  id: string;
  created_at: string;
}

export function OptimizationForm({ onSubmit, isPending, error, models, modelsLoading, modelsError, onRetryModels, initialModelId }: FormProps<OptimizationRequest> & {
  models: ModelOption[];
  modelsLoading: boolean;
  modelsError: string | null;
  onRetryModels: () => void;
  initialModelId: string;
}) {
  const [request, setRequest] = useState(() => ({ ...defaultOptimizationRequest(), model_id: initialModelId }));
  const [localError, setLocalError] = useState<string | null>(null);
  const config = request.optimization_config;
  const setConfig = (optimization_config: typeof config) => setRequest({ ...request, optimization_config });
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!request.model_id) {
      setLocalError("Select a trained surrogate model.");
      return;
    }
    const boundsError = validateBounds(config.sampling_bounds);
    if (boundsError) {
      setLocalError(boundsError);
      return;
    }
    setLocalError(null);
    onSubmit(request);
  };

  return (
    <form className="run-form" onSubmit={submit}>
      <div className="form-intro">
        <span className="form-intro-icon"><Workflow size={22} /></span>
        <div><h2>Screen & validate operating conditions</h2><p>Rank surrogate-feasible candidates, then validate the leading batch with the real simulator.</p></div>
      </div>
      <FormSection title="Surrogate lineage" description="Choose the trained model used for candidate screening." columns={2}>
        <label className="field model-select-field">
          <span className="field-label">Trained model</span>
          <select required value={request.model_id} disabled={modelsLoading || models.length === 0} onChange={(event) => setRequest({ ...request, model_id: event.target.value })}>
            <option value="">{modelsLoading ? "Loading models…" : models.length === 0 ? "No trained models available" : "Select a model"}</option>
            {models.map((model) => <option value={model.id} key={model.id}>{shortId(model.id)} · {formatDate(model.created_at)}</option>)}
          </select>
        </label>
        <div className="field-selection-note">Only successfully trained models can be used for optimization.</div>
      </FormSection>
      {modelsError && <div className="inline-alert"><span>{modelsError}</span><button type="button" className="text-button" onClick={onRetryModels}>Retry</button></div>}
      {!modelsLoading && !modelsError && models.length === 0 && (
        <div className="inline-alert"><span>Train a surrogate before launching optimization.</span><Link to="/runs/new?type=training">Launch training</Link></div>
      )}
      <FormSection title="Search & validation" description="Surrogate sample budget and real-simulator acceptance criteria.">
        <NumberField name="candidate_count" value={config.candidate_count} min={1} step={1} onChange={(candidate_count) => setConfig({ ...config, candidate_count: Math.round(candidate_count) })} />
        <NumberField name="validation_count" value={config.validation_count} min={1} step={1} onChange={(validation_count) => setConfig({ ...config, validation_count: Math.round(validation_count) })} />
        <NumberField name="max_final_oxide_fraction" value={config.max_final_oxide_fraction} min={0} max={1} onChange={(max_final_oxide_fraction) => setConfig({ ...config, max_final_oxide_fraction })} />
      </FormSection>
      <AdvancedSection title="Search bounds, solver & material" description="Constrain the operating regime and override physical defaults.">
        <div className="advanced-stack">
          <InputBoundsFields title="Candidate sampling bounds" value={config.sampling_bounds} onChange={(sampling_bounds) => setConfig({ ...config, sampling_bounds })} />
          <FormSection title="Simulation settings" description="Shared settings used to validate selected candidates.">
            <NumberField name="initial_temperature" value={config.initial_temperature} min={0.000001} onChange={(initial_temperature) => setConfig({ ...config, initial_temperature })} />
            <NumberField name="num_steps" value={config.solver.num_steps} min={1} step={1} onChange={(num_steps) => setConfig({ ...config, solver: { num_steps: Math.round(num_steps) } })} />
            <NumberField name="seed" value={config.seed} min={0} step={1} onChange={(seed) => setConfig({ ...config, seed: Math.round(seed) })} />
          </FormSection>
          <MaterialFields value={config.material} onChange={(material) => setConfig({ ...config, material })} />
        </div>
      </AdvancedSection>
      <FormError message={localError ?? error} />
      <div className="form-footer">
        <span className="async-note"><Info size={15} /> Recommendation is validated by numerical simulation.</span>
        <SubmitButton pending={isPending} label="Launch optimization" />
      </div>
    </form>
  );
}
