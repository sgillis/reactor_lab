import { Award, CheckCircle2, XCircle } from "lucide-react";
import type { CandidateView, OptimizationResultView } from "../domain/results";
import { formatNumber } from "../lib/format";
import { Panel } from "./ui";

const featureLabels = ["Peak temperature (K)", "Residence time (s)", "Hydrogen concentration", "Diameter (m)", "Initial oxide fraction"];

function CandidateDetails({ candidate }: { candidate: CandidateView }) {
  return (
    <dl className="recommendation-grid">
      {candidate.features.map((feature, index) => (
        <div key={featureLabels[index] ?? index}><dt>{featureLabels[index] ?? `Feature ${index + 1}`}</dt><dd>{formatNumber(feature, 6)}</dd></div>
      ))}
      <div><dt>Final oxide fraction</dt><dd>{formatNumber(candidate.simulator_final_oxide_fraction, 6)}</dd></div>
      <div><dt>Maximum temperature</dt><dd>{formatNumber(candidate.simulator_maximum_temperature, 2)} K</dd></div>
    </dl>
  );
}

export function OptimizationReport({ result }: { result: OptimizationResultView }) {
  return (
    <>
      <div className="optimization-summary">
        <div><span>Surrogate-feasible</span><strong>{formatNumber(result.surrogate_feasible_count, 0)}</strong><small>candidates</small></div>
        <div><span>Simulator-validated</span><strong>{result.validated_candidates.length}</strong><small>leading candidates</small></div>
        <div><span>Recommendation</span><strong>{result.recommendation ? "Found" : "None"}</strong><small>{result.recommendation ? "simulator feasible" : "no feasible candidate"}</small></div>
      </div>
      {result.recommendation && (
        <Panel title="Recommended operating point" description="Lowest-temperature surrogate-feasible candidate confirmed by the numerical simulator." className="recommendation-panel">
          <div className="recommendation-heading"><span><Award size={18} /> Validated recommendation</span><span className="feasible-label"><CheckCircle2 size={15} /> Feasible</span></div>
          <CandidateDetails candidate={result.recommendation} />
        </Panel>
      )}
      {result.validated_candidates.length > 0 && (
        <Panel title="Validated candidate set" description="Real simulator outcomes for every selected candidate.">
          <div className="table-scroll">
            <table className="data-table candidate-table">
              <thead><tr><th>#</th><th>Peak temperature</th><th>Final oxide</th><th>Max temperature</th><th>Melting</th><th>Feasible</th></tr></thead>
              <tbody>{result.validated_candidates.map((candidate, index) => (
                <tr key={`${candidate.features.join("-")}-${index}`}>
                  <td>{index + 1}</td>
                  <td>{formatNumber(candidate.features[0] ?? Number.NaN, 2)} K</td>
                  <td>{formatNumber(candidate.simulator_final_oxide_fraction, 6)}</td>
                  <td>{formatNumber(candidate.simulator_maximum_temperature, 2)} K</td>
                  <td>{candidate.simulator_reached_melting ? <span className="negative-label"><XCircle size={14} /> Yes</span> : "No"}</td>
                  <td>{candidate.simulator_feasible ? <span className="feasible-label"><CheckCircle2 size={14} /> Yes</span> : "No"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </Panel>
      )}
    </>
  );
}
