import type { Data } from "plotly.js";
import type { OptimizationResultView } from "../domain/results";
import { ScientificPlot } from "./ScientificPlot";
import { EmptyState, Panel } from "./ui";

function comparisonData(result: OptimizationResultView, target: "oxide" | "temperature"): Data[] {
  const candidates = result.validated_candidates;
  const actualKey = target === "oxide" ? "simulator_final_oxide_fraction" : "simulator_maximum_temperature";
  const predictedKey = target === "oxide" ? "surrogate_final_oxide_fraction" : "surrogate_maximum_temperature";
  const recommended = result.recommendation;
  return [
    {
      x: candidates.map((candidate) => candidate[predictedKey]),
      y: candidates.map((candidate) => candidate[actualKey]),
      type: "scatter",
      mode: "markers",
      name: "Validated candidates",
      marker: {
        color: candidates.map((candidate) => candidate.simulator_feasible ? "#168a78" : "#b7c2bf"),
        size: 9,
        opacity: 0.78,
        line: { color: "#ffffff", width: 1 },
      },
      hovertemplate: "Surrogate=%{x:.6g}<br>Simulator=%{y:.6g}<extra></extra>",
    },
    ...(recommended ? [{
      x: [recommended[predictedKey]],
      y: [recommended[actualKey]],
      type: "scatter" as const,
      mode: "markers" as const,
      name: "Recommendation",
      marker: { color: "#e07a3f", size: 14, symbol: "star" as const, line: { color: "#ffffff", width: 1 } },
      hovertemplate: "Surrogate=%{x:.6g}<br>Simulator=%{y:.6g}<extra>Recommendation</extra>",
    }] : []),
  ];
}

function ComparisonPlot({ result, target }: { result: OptimizationResultView; target: "oxide" | "temperature" }) {
  const data = comparisonData(result, target);
  const candidates = result.validated_candidates;
  const predictedKey = target === "oxide" ? "surrogate_final_oxide_fraction" : "surrogate_maximum_temperature";
  const actualKey = target === "oxide" ? "simulator_final_oxide_fraction" : "simulator_maximum_temperature";
  const values = candidates.flatMap((candidate) => [candidate[predictedKey], candidate[actualKey]]);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const padding = (maximum - minimum || 1) * 0.08;
  const identity = [minimum - padding, maximum + padding];
  data.push({
    x: identity,
    y: identity,
    type: "scatter",
    mode: "lines",
    name: "Ideal parity",
    line: { color: "#71817e", dash: "dash", width: 1.5 },
    hoverinfo: "skip",
  });
  const unit = target === "oxide" ? "" : " (K)";
  return (
    <ScientificPlot
      ariaLabel={`${target} surrogate and simulator comparison`}
      data={data}
      height={360}
      layout={{
        xaxis: { title: { text: `Surrogate${unit}` }, range: identity },
        yaxis: { title: { text: `Simulator${unit}` }, range: identity, scaleanchor: "x" },
      }}
    />
  );
}

export function OptimizationCharts({ result }: { result: OptimizationResultView }) {
  if (result.validated_candidates.length === 0) {
    return <EmptyState title="No validated candidates" description="No surrogate-feasible candidates were available for simulator validation." />;
  }
  return (
    <div className="chart-grid">
      <Panel title="Final oxide comparison" description="Surrogate screening estimates against real simulator outputs.">
        <ComparisonPlot result={result} target="oxide" />
      </Panel>
      <Panel title="Maximum temperature comparison" description="Validated thermal outcomes for the leading candidates.">
        <ComparisonPlot result={result} target="temperature" />
      </Panel>
    </div>
  );
}
