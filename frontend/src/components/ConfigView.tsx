import { formatValue, humanize } from "../lib/format";
import { isRecord } from "../lib/type-guards";

export function ConfigView({ config }: { config: unknown }) {
  if (!isRecord(config)) {
    return <pre className="json-view">{JSON.stringify(config, null, 2)}</pre>;
  }

  const scalarEntries = Object.entries(config).filter(([, value]) => !isRecord(value));
  const nestedEntries = Object.entries(config).filter((entry): entry is [string, Record<string, unknown>] => isRecord(entry[1]));

  return (
    <div className="config-view">
      {scalarEntries.length > 0 && (
        <dl className="definition-grid">
          {scalarEntries.map(([key, value]) => (
            <div className="definition-item" key={key}>
              <dt>{humanize(key)}</dt>
              <dd className={typeof value === "string" && value.includes("-") ? "mono break-all" : ""}>
                {formatValue(key, value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {nestedEntries.length > 0 && (
        <div className="config-sections">
          {nestedEntries.map(([key, value]) => (
            <section className="config-section" key={key}>
              <h3>{humanize(key)}</h3>
              <ConfigView config={value} />
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

export function MetricsGrid({ metrics }: { metrics: unknown }) {
  if (!isRecord(metrics)) return null;
  const entries: Array<{ key: string; context: string; value: number | null }> = [];

  const visit = (value: Record<string, unknown>, context: string[] = []) => {
    for (const [key, child] of Object.entries(value)) {
      if (typeof child === "number" || child === null) {
        entries.push({ key, context: context.map(humanize).join(" · "), value: child });
      } else if (isRecord(child)) {
        visit(child, [...context, key]);
      }
    }
  };
  visit(metrics);

  if (entries.length === 0) return null;
  return (
    <div className="metric-grid">
      {entries.map(({ key, context, value }, index) => (
        <div className="metric-card" key={`${context}-${key}-${index}`}>
          {context && <span className="metric-context">{context}</span>}
          <span className="metric-label">{humanize(key)}</span>
          <strong>{formatValue(key, value)}</strong>
        </div>
      ))}
    </div>
  );
}
