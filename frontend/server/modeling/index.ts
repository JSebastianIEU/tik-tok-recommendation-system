export { buildCandidateProfileCore } from "./step1/buildCandidateProfileCore";
export type {
  CandidateInputCore,
  CandidateProfileCore,
  ContentType,
  Objective,
  PrimaryCta
} from "./step1/types";

export { extractCandidateSignals } from "./part2/extractCandidateSignals";
export type {
  CandidateSignalHints,
  CandidateSignalProfile
} from "./part2/extractCandidateSignals";

export { buildComparableNeighborhood } from "./step2/buildComparableNeighborhood";
export type {
  BuildComparableNeighborhoodInput,
  ComparableNeighborhood,
  NeighborhoodCandidate,
  NeighborhoodConfidence,
  RankingTrace
} from "./step2/buildComparableNeighborhood";

export { buildNeighborhoodContrast } from "./step3/buildNeighborhoodContrast";
export type {
  BuildNeighborhoodContrastInput,
  ContrastClaim,
  NeighborhoodContrast
} from "./step3/buildNeighborhoodContrast";
