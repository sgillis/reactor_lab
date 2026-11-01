import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-basic-dist-min";
import type { Config, Data, Layout } from "plotly.js";

const Plot = createPlotlyComponent(Plotly);

const plotConfig: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  scrollZoom: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

export function ScientificPlot({ data, layout, height = 340, ariaLabel }: {
  data: Data[];
  layout: Partial<Layout>;
  height?: number;
  ariaLabel: string;
}) {
  return (
    <div className="scientific-plot" role="img" aria-label={ariaLabel} style={{ height }}>
      <Plot
        data={data}
        layout={{
          ...layout,
          autosize: true,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "#ffffff",
          font: { family: "Inter, ui-sans-serif, system-ui, sans-serif", size: 12, color: "#536563" },
          colorway: ["#168a78", "#e07a3f", "#6574cd", "#b04f6c"],
          margin: { l: 64, r: 24, t: 28, b: 56, ...layout.margin },
          hovermode: "closest",
          showlegend: true,
          legend: { orientation: "h", y: 1.14, x: 0, ...layout.legend },
          xaxis: {
            gridcolor: "#e8eeeb",
            zerolinecolor: "#b8c8c3",
            linecolor: "#cbd7d3",
            tickfont: { size: 11 },
            ...layout.xaxis,
          },
          yaxis: {
            gridcolor: "#e8eeeb",
            zerolinecolor: "#b8c8c3",
            linecolor: "#cbd7d3",
            tickfont: { size: 11 },
            ...layout.yaxis,
          },
        }}
        config={plotConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
