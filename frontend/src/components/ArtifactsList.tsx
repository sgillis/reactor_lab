import { Download, FileArchive, FileJson, FileBox } from "lucide-react";
import { formatBytes, formatDate } from "../lib/format";
import { EmptyState } from "./ui";

export interface ArtifactItem {
  id: string;
  name: string;
  media_type: string;
  size_bytes: number;
  download_url: string;
  created_at?: string;
}

export function ArtifactsList({ artifacts }: { artifacts: ArtifactItem[] }) {
  if (artifacts.length === 0) {
    return <EmptyState title="No artifacts" description="This run did not produce downloadable artifacts." icon={FileBox} />;
  }

  return (
    <div className="artifact-list">
      {artifacts.map((artifact) => {
        const Icon = artifact.media_type.includes("json") ? FileJson : FileArchive;
        return (
          <article className="artifact-row" key={artifact.id}>
            <span className="artifact-icon"><Icon size={20} /></span>
            <div className="artifact-info">
              <strong>{artifact.name}</strong>
              <span>{artifact.media_type} · {formatBytes(artifact.size_bytes)}</span>
              {artifact.created_at && <span>Created {formatDate(artifact.created_at)}</span>}
            </div>
            <a
              className="button button-secondary button-small"
              href={artifact.download_url}
              download={artifact.name}
              aria-label={`Download ${artifact.name}`}
            >
              <Download size={15} /> <span>Download</span>
            </a>
          </article>
        );
      })}
    </div>
  );
}
