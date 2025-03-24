import Header from "./components/ui/Header";
import StepRoadmap from "./components/ui/StepRoadmap";

export default function HomePage() {
  return (
    <>
      <Header />
      <main className="bg-dark-bg bg-dark-primary max-w-3xl mx-auto text-center py-10 px-4">
        <StepRoadmap current={1} />

        <div className="mt-10">
          <h1 className="font-monoHeading text-4xl mb-2">Hello</h1>
          <p className="font-body text-base text-light-text dark:text-dark-text">
            Upload a handwriting sample and we’ll turn it into a font you can
            use anywhere.
          </p>
        </div>
      </main>
    </>
  );
}
