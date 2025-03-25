import PageWrapper from "../components/ui/PageWrapper";
import StepNavigation from "../components/ui/StepNavigation";
import StepRoadmap from "../components/ui/StepRoadmap";
import StepContent from "../components/ui/StepContent";

export default function Home() {
  return (
    <PageWrapper>
      <StepRoadmap />
      <StepContent />
      <StepNavigation />
    </PageWrapper>
  );
}
