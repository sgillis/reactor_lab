import { ArrowLeft, Compass } from "lucide-react";
import { Link } from "react-router-dom";
import { EmptyState } from "../components/ui";

export function NotFoundPage() {
  return (
    <div className="page not-found-page">
      <div className="panel"><EmptyState
        title="This page is outside the lab"
        description="The requested workspace route does not exist. Return to the experiment dashboard to continue."
        icon={Compass}
        action={<Link to="/" className="button button-primary button-small"><ArrowLeft size={15} /> Back to dashboard</Link>}
      /></div>
    </div>
  );
}
