import { useId, type ReactNode } from "react";
import { AlertCircle, Settings2 } from "lucide-react";
import type { InputBounds, MaterialConfig } from "../domain/config";
import { humanize, unitFor } from "../lib/format";

export function NumberField({ name, label, value, onChange, min, max, step = "any", help, disabled = false }: {
  name: string;
  label?: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number | "any";
  help?: string;
  disabled?: boolean;
}) {
  const id = useId();
  const unit = unitFor(name);
  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{label ?? humanize(name)}{unit && <span className="field-unit">{unit}</span>}</span>
      <input
        id={id}
        name={name}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        required
        disabled={disabled}
        onChange={(event) => onChange(event.target.value === "" ? 0 : event.target.valueAsNumber)}
      />
      {help && <span className="field-help">{help}</span>}
    </label>
  );
}

export function FormSection({ title, description, children, columns = 3 }: {
  title: string;
  description?: string;
  children: ReactNode;
  columns?: 2 | 3 | 4;
}) {
  return (
    <section className="form-section">
      <div className="form-section-heading">
        <h3>{title}</h3>
        {description && <p>{description}</p>}
      </div>
      <div className={`form-grid form-grid-${columns}`}>{children}</div>
    </section>
  );
}

export function AdvancedSection({ title, description, children }: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <details className="advanced-section">
      <summary>
        <span className="advanced-icon"><Settings2 size={17} /></span>
        <span><strong>{title}</strong><small>{description}</small></span>
      </summary>
      <div className="advanced-content">{children}</div>
    </details>
  );
}

const boundFields = [
  "peak_temperature",
  "residence_time",
  "hydrogen_concentration",
  "diameter",
  "initial_oxide_fraction",
] as const;

export function InputBoundsFields({ value, onChange, title = "Input bounds" }: {
  value: InputBounds;
  onChange: (value: InputBounds) => void;
  title?: string;
}) {
  return (
    <div className="bounds-editor">
      <h4>{title}</h4>
      <div className="bounds-header" aria-hidden="true"><span>Parameter</span><span>Lower</span><span>Upper</span></div>
      {boundFields.map((name) => (
        <div className="bounds-row" key={name}>
          <span className="bounds-name">{humanize(name)}{unitFor(name) && <small>{unitFor(name)}</small>}</span>
          {(["lower", "upper"] as const).map((bound) => (
            <label key={bound}>
              <span className="sr-only">{humanize(name)} {bound} bound</span>
              <input
                aria-label={`${humanize(name)} ${bound} bound`}
                type="number"
                required
                step="any"
                min={name === "initial_oxide_fraction" || name === "hydrogen_concentration" ? 0 : undefined}
                max={name === "initial_oxide_fraction" ? 1 : undefined}
                value={value[name][bound]}
                onChange={(event) => onChange({
                  ...value,
                  [name]: {
                    ...value[name],
                    [bound]: event.target.value === "" ? 0 : event.target.valueAsNumber,
                  },
                })}
              />
            </label>
          ))}
        </div>
      ))}
    </div>
  );
}

export function MaterialFields({ value, onChange }: {
  value: MaterialConfig;
  onChange: (value: MaterialConfig) => void;
}) {
  const update = (key: keyof MaterialConfig) => (next: number) => onChange({ ...value, [key]: next });
  return (
    <FormSection title="Material properties" description="Illustrative SI-unit material and reaction parameters." columns={3}>
      <NumberField name="density" value={value.density} min={0.000001} onChange={update("density")} />
      <NumberField name="heat_capacity" value={value.heat_capacity} min={0.000001} onChange={update("heat_capacity")} />
      <NumberField name="heat_transfer_coefficient" value={value.heat_transfer_coefficient} min={0} onChange={update("heat_transfer_coefficient")} />
      <NumberField name="emissivity" value={value.emissivity} min={0} max={1} onChange={update("emissivity")} />
      <NumberField name="surroundings_temperature" value={value.surroundings_temperature} min={0.000001} onChange={update("surroundings_temperature")} />
      <NumberField name="melting_temperature" value={value.melting_temperature} min={0.000001} onChange={update("melting_temperature")} />
      <NumberField name="reaction_pre_exponential" value={value.reaction_pre_exponential} min={0} onChange={update("reaction_pre_exponential")} />
      <NumberField name="activation_energy" value={value.activation_energy} min={0} onChange={update("activation_energy")} />
    </FormSection>
  );
}

export function FormError({ message }: { message: string | null }) {
  if (!message) return null;
  return <div className="form-error" role="alert"><AlertCircle size={18} /><span>{message}</span></div>;
}

export function validateBounds(bounds: InputBounds): string | null {
  for (const name of boundFields) {
    if (bounds[name].lower >= bounds[name].upper) {
      return `${humanize(name)} lower bound must be less than its upper bound.`;
    }
  }
  return null;
}
