import PageWrapper from "../components/ui/PageWrapper";

export default function HomePage() {
  return (
    <PageWrapper>
      <h1 className="text-primary text-3xl font-heading mb-4">PageWrapper</h1>
      <p className="text-secondary text-base font-body">
        Upload a handwriting sample and we’ll turn it into a font you can use
        anywhere.
      </p>
    </PageWrapper>
  );
}
