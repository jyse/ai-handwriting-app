"use client";
import { useStepStore } from "../../state/useStepStore";
import Upload from "../steps/upload/Upload";
import Process from "../steps/Process";
import Preview from "../steps/Preview";
import Customize from "../steps/Customize";
import Download from "../steps/Download";

export default function StepContent() {
  const { step } = useStepStore();

  switch (step) {
    case 1:
      return <Upload />;
    case 2:
      return <Process />;
    // case 3:
    //   return <Preview />;
    // case 4:
    //   return <Customize />;
    // case 5:
      return <Download />;
    default:
      return <Upload />;
  }
}
