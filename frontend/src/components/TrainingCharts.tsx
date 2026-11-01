import { useState } from "react";
import type { Data } from "plotly.js";
import type { TrainingResultView } from "../domain/results";
import { ScientificPlot } from "./ScientificPlot";
import { EmptyState, Panel } from "./ui";

const targets = [
  { name: "Final oxide fraction", unit: "", decimals: 6 },
  { name: "Maximum temperature", unit: " K", decimals: 2 },
] as const;
const splits = [
  { key: "test", label: "Test", color: "#168a78" },
  { key: "ood", label: "Separated regime", color: "#e07a3f" },
] as const;

function pairs(simulated: number[][], predicted: number[][], targetIndex: number) {
  const actual = simulated.map((row) => row[targetIndex]).filter((value): value is number => typeof value === "number");
  const estimate = predicted.map((row) => row[targetIndex]).filter((value): value is number => typeof value === "number");
  const count = Math.min(actual.length, estimate.length);
  return { actual: actual.slice(0, count), estimate: estimate.slice(0, count) };
}

export function TrainingCharts({ result }: { result: TrainingResultView }) {
  const [targetIndex, setTargetIndex] = useState(0);
  const target = targets[targetIndex] ?? targets[0];
  const historyEntries = Object.entries(result.history).filter(([, values]) => values.length > 0);
  const parityPairs = splits.map((split) => ({ split, ...pairs(result.simulated[split.key] ?? [], result.predicted[split.key] ?? [], targetIndex) }));
  const allParityValues = parityPairs.flatMap(({ actual, estimate }) => [...actual, ...estimate]);
  const minimum = allParityValues.length ? Math.min(...allParityValues) : 0;
  const maximum = allParityValues.length ? Math.max(...allParityValues) : 1;
  const padding = (maximum - minimum || 1) * 0.05;
  const identity = [minimum - padding, maximum + padding];

  const parityData: Data[] = parityPairs.map(({ split, actual, estimate }) => ({
    x: actual,
    y: estimate,
    type: "scatter",
    mode: "markers",
    name: split.label,
    marker: { color: split.color, size: 7, opacity: 0.68, line: { width: 0 } },
    hovertemplate: `Simulated=%{x:.${target.decimals}f}${target.unit}<br>Predicted=%{y:.${target.decimals}f}${target.unit}<extra>${split.label}</extra>`,
  }));
  parityData.push({
    x: identity,
    y: identity,
    type: "scatter",
    mode: "lines",
    name: "Ideal parity",
    line: { color: "#71817e", width: 1.5, dash: "dash" },
    hoverinfo: "skip",
  });

  const residualData: Data[] = parityPairs.map(({ split, actual, estimate }) => ({
    x: estimate,
    y: actual.map((value, index) => value - (estimate[index] ?? 0)),
    type: "scatter",
    mode: "markers",
    name: split.label,
    marker: { color: split.color, size: 7, opacity: 0.68, line: { width: 0 } },
    hovertemplate: `Predicted=%{x:.${target.decimals}f}${target.unit}<br>Residual=%{y:.${target.decimals}f}${target.unit}<extra>${split.label}</extra>`,
  }));
  residualData.push({
    x: identity,
    y: [0, 0],
    type: "scatter",
    mode: "lines",
    name: "Zero residual",
    line: { color: "#71817e", width: 1.5, dash: "dash" },
    hoverinfo: "skip",
  });

  return (
    <div className="chart-stack">
      <Panel title="Optimization history" description="Normalized mean-squared error after every epoch.">
        {historyEntries.length ? (
          <ScientificPlot
            ariaLabel="Training and validation loss history"
            data={historyEntries.map(([name, values]) => ({
              x: values.map((_, index) => index + 1),
              y: values,
              type: "scatter",
              mode: "lines",
              name: name === "train_loss" ? "Training loss" : name === "validation_loss" ? "Validation loss" : name,
              line: { width: 2 },
              hovertemplate: "Epoch %{x}<br>Loss %{y:.6g}<extra></extra>",
            }))}
            layout={{
              xaxis: { title: { text: "Epoch" }, rangemode: "tozero" },
              yaxis: { title: { text: "Normalized MSE" }, type: "log" },
              hovermode: "x unified",
            }}
          />
        ) : <EmptyState title="No loss history" description="The training result does not contain epoch history." />}
      </Panel>

      <div className="chart-toolbar">
        <div><h2>Prediction diagnostics</h2><p>Compare physical-unit surrogate predictions with held-out simulations.</p></div>
        <div className="segmented-control" aria-label="Prediction target">
          {targets.map((item, index) => (
            <button type="button" key={item.name} className={targetIndex === index ? "active" : ""} onClick={() => setTargetIndex(index)}>{item.name}</button>
          ))}
        </div>
      </div>

      <div className="chart-grid">
        <Panel title={`${target.name} parity`} description="Points on the dashed line indicate perfect agreement.">
          {allParityValues.length ? <ScientificPlot
            ariaLabel={`${target.name} prediction parity chart`}
            data={parityData}
            layout={{
              xaxis: { title: { text: `Simulated${target.unit}` }, range: identity },
              yaxis: { title: { text: `Predicted${target.unit}` }, range: identity, scaleanchor: "x" },
            }}
            height={380}
          /> : <EmptyState title="No parity samples" description="No paired test or OOD predictions are available." />}
        </Panel>
        <Panel title={`${target.name} residuals`} description="Simulated minus predicted; centered is better.">
          {allParityValues.length ? <ScientificPlot
            ariaLabel={`${target.name} prediction residual chart`}
            data={residualData}
            layout={{
              xaxis: { title: { text: `Predicted${target.unit}` } },
              yaxis: { title: { text: `Residual${target.unit}` }, zeroline: true },
            }}
            height={380}
          /> : <EmptyState title="No residual samples" description="No paired test or OOD predictions are available." />}
        </Panel>
      </div>
    </div>
  );
}
