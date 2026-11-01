import { LineChart } from "lucide-react";
import type { Data } from "plotly.js";
import type { SimulationResultView } from "../domain/results";
import { ScientificPlot } from "./ScientificPlot";
import { EmptyState, Panel } from "./ui";

export function SimulationCharts({ result }: { result: SimulationResultView }) {
  const { time, temperature, oxide_fraction, gas_temperature } = result.trajectory;
  if (time.length === 0) {
    return <EmptyState title="No trajectory samples" description="The simulation completed without numerical trajectory samples." icon={LineChart} />;
  }

  const temperatureData: Data[] = [
    {
      x: time,
      y: temperature,
      type: "scatter",
      mode: "lines",
      name: "Particle temperature",
      line: { color: "#168a78", width: 2.5 },
      hovertemplate: "t=%{x:.4f} s<br>T=%{y:.2f} K<extra>Particle</extra>",
    },
    {
      x: time,
      y: gas_temperature,
      type: "scatter",
      mode: "lines",
      name: "Gas temperature",
      line: { color: "#e07a3f", width: 2 },
      hovertemplate: "t=%{x:.4f} s<br>T=%{y:.2f} K<extra>Gas</extra>",
    },
  ];

  return (
    <div className="chart-stack">
      <Panel title="Thermal trajectory" description="Particle and reactor gas temperature over residence time.">
        <ScientificPlot
          ariaLabel="Particle and gas temperature trajectory chart"
          data={temperatureData}
          layout={{
            xaxis: { title: { text: "Time (s)" } },
            yaxis: { title: { text: "Temperature (K)" } },
            hovermode: "x unified",
          }}
        />
      </Panel>
      <Panel title="Oxide reduction" description="Remaining oxide fraction as the particle reacts.">
        <ScientificPlot
          ariaLabel="Oxide fraction trajectory chart"
          data={[{
            x: time,
            y: oxide_fraction,
            type: "scatter",
            mode: "lines",
            name: "Oxide fraction",
            fill: "tozeroy",
            fillcolor: "rgba(22, 138, 120, 0.09)",
            line: { color: "#168a78", width: 2.5 },
            hovertemplate: "t=%{x:.4f} s<br>Oxide=%{y:.6f}<extra></extra>",
          }]}
          layout={{
            xaxis: { title: { text: "Time (s)" } },
            yaxis: { title: { text: "Oxide fraction" }, rangemode: "tozero" },
            showlegend: false,
          }}
          height={320}
        />
      </Panel>
    </div>
  );
}
