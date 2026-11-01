import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { LaunchRunPage } from "./pages/LaunchRunPage";
import { ModelDetailPage } from "./pages/ModelDetailPage";
import { ModelsPage } from "./pages/ModelsPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { RunDetailPage } from "./pages/RunDetailPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<DashboardPage />} />
        <Route path="runs/new" element={<LaunchRunPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="models/:modelId" element={<ModelDetailPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
