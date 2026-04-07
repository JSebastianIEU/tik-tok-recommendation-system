import { useEffect, useMemo, useState } from "react";
import { ComparableThumbnailImage } from "../report/components/ComparableThumbnailImage";
import {
  createLabelingSession,
  listLabelingSessions,
  listLabelingSources,
  loadLabelingSession,
  updateLabelingCandidateReview
} from "../../services/api/labelingApi";
import type {
  LabelingReviewLabel,
  LabelingSession,
  LabelingSessionCandidate,
  LabelingSessionCase,
  LabelingSessionListItem,
  LabelingSourceSummary
} from "./types";

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => asString(item)).filter(Boolean);
}

function formatDateTime(value: string): string {
  if (!value) {
    return "Unknown";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(parsed);
}

function formatPct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function caseReviewedCount(itemCase: LabelingSessionCase): number {
  return itemCase.candidates.filter((candidate) => candidate.review.label !== null).length;
}

function labelButtonText(label: LabelingReviewLabel): string {
  if (label === "saved") {
    return "Saved";
  }
  if (label === "relevant") {
    return "Relevant";
  }
  return "Not relevant";
}

function candidateVideoUrl(candidate: LabelingSessionCandidate): string {
  return (
    asString(candidate.display.video_url) ||
    asString(candidate.candidate_payload.video_url)
  );
}

function candidateCaption(candidate: LabelingSessionCandidate): string {
  return (
    asString(candidate.display.caption) ||
    asString(candidate.candidate_payload.caption) ||
    asString(candidate.candidate_payload.text)
  );
}

function candidateAuthor(candidate: LabelingSessionCandidate): string {
  return (
    asString(candidate.display.author_display_name) ||
    asString(candidate.display.author_username) ||
    asString(candidate.candidate_payload.author_id) ||
    "Unknown author"
  );
}

function candidateHashtags(candidate: LabelingSessionCandidate): string[] {
  const fromDisplay = asStringArray(candidate.display.hashtags);
  if (fromDisplay.length > 0) {
    return fromDisplay;
  }
  return asStringArray(candidate.candidate_payload.hashtags);
}

function queryCaption(itemCase: LabelingSessionCase): string {
  return (
    asString(itemCase.query.display.caption) ||
    asString(itemCase.query.query_payload.description) ||
    asString(itemCase.query.query_payload.text)
  );
}

function queryAuthor(itemCase: LabelingSessionCase): string {
  return (
    asString(itemCase.query.display.author_display_name) ||
    asString(itemCase.query.display.author_username) ||
    "Unknown author"
  );
}

function queryHashtags(itemCase: LabelingSessionCase): string[] {
  const displayTags = asStringArray(itemCase.query.display.hashtags);
  if (displayTags.length > 0) {
    return displayTags;
  }
  return asStringArray(itemCase.query.query_payload.hashtags);
}

function queryCommentsPreview(itemCase: LabelingSessionCase): string[] {
  return asStringArray(itemCase.query.display.comments_preview);
}

function preferredSourceId(sources: LabelingSourceSummary[]): string {
  return (
    sources.find((item) => item.file_name.includes("training"))?.source_id ||
    sources.find((item) => item.file_name.includes("seed"))?.source_id ||
    sources[0]?.source_id ||
    ""
  );
}

function preferredSessionId(
  sessions: LabelingSessionListItem[],
  sourceId: string
): string {
  if (!sourceId) {
    return sessions[0]?.session_id ?? "";
  }
  return (
    sessions.find((item) => item.source.source_id === sourceId)?.session_id ||
    sessions[0]?.session_id ||
    ""
  );
}

export function LabelingPage(): JSX.Element {
  const [sources, setSources] = useState<LabelingSourceSummary[]>([]);
  const [sessions, setSessions] = useState<LabelingSessionListItem[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string>("");
  const [session, setSession] = useState<LabelingSession | null>(null);
  const [activeCaseId, setActiveCaseId] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isCreatingSession, setIsCreatingSession] = useState<boolean>(false);
  const [savingCandidateId, setSavingCandidateId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  useEffect(() => {
    let isMounted = true;

    async function initialize(): Promise<void> {
      setIsLoading(true);
      setErrorMessage("");
      try {
        const [nextSources, nextSessions] = await Promise.all([
          listLabelingSources(),
          listLabelingSessions()
        ]);
        if (!isMounted) {
          return;
        }
        setSources(nextSources);
        const preferredSource = preferredSourceId(nextSources);
        setSelectedSourceId((previous) => previous || preferredSource);
        setSessions(nextSessions);

        let nextSession: LabelingSession | null = null;
        const preferredSession = preferredSessionId(nextSessions, preferredSource);
        if (preferredSession) {
          nextSession = await loadLabelingSession(preferredSession);
        } else if (preferredSource) {
          nextSession = await createLabelingSession({ source_id: preferredSource });
          const refreshedSessions = await listLabelingSessions();
          if (!isMounted) {
            return;
          }
          setSessions(refreshedSessions);
        }

        if (!isMounted) {
          return;
        }

        setSession(nextSession);
        setActiveCaseId(nextSession?.cases[0]?.case_id ?? "");
      } catch (error) {
        console.error("labeling_page_init_failed", error);
        if (isMounted) {
          setErrorMessage(
            "The local labeling workspace could not be loaded. Make sure the API server is running and a benchmark source exists."
          );
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    void initialize();

    return () => {
      isMounted = false;
    };
  }, []);

  const activeCase = useMemo(() => {
    if (!session) {
      return null;
    }
    return (
      session.cases.find((item) => item.case_id === activeCaseId) ??
      session.cases[0] ??
      null
    );
  }, [activeCaseId, session]);

  useEffect(() => {
    if (!session) {
      return;
    }
    if (activeCaseId && session.cases.some((item) => item.case_id === activeCaseId)) {
      return;
    }
    setActiveCaseId(session.cases[0]?.case_id ?? "");
  }, [activeCaseId, session]);

  async function handleCreateSession(): Promise<void> {
    if (!selectedSourceId) {
      return;
    }
    setIsCreatingSession(true);
    setErrorMessage("");
    try {
      const nextSession = await createLabelingSession({ source_id: selectedSourceId });
      const refreshedSessions = await listLabelingSessions();
      setSessions(refreshedSessions);
      setSession(nextSession);
      setActiveCaseId(nextSession.cases[0]?.case_id ?? "");
    } catch (error) {
      console.error("labeling_session_create_ui_failed", error);
      setErrorMessage("A new labeling session could not be created.");
    } finally {
      setIsCreatingSession(false);
    }
  }

  async function handleSetCandidateLabel(
    itemCase: LabelingSessionCase,
    candidate: LabelingSessionCandidate,
    label: LabelingReviewLabel | null
  ): Promise<void> {
    if (!session) {
      return;
    }
    setSavingCandidateId(candidate.candidate_id);
    setErrorMessage("");
    try {
      const nextSession = await updateLabelingCandidateReview({
        session_id: session.session_id,
        case_id: itemCase.case_id,
        candidate_id: candidate.candidate_id,
        label,
        note: candidate.review.note
      });
      setSession(nextSession);
      setSessions((previous) =>
        previous.map((item) =>
          item.session_id === nextSession.session_id
            ? {
                session_id: nextSession.session_id,
                session_name: nextSession.session_name,
                created_at: nextSession.created_at,
                updated_at: nextSession.updated_at,
                storage_path: nextSession.storage_path,
                source: nextSession.source,
                summary: nextSession.summary
              }
            : item
        )
      );
    } catch (error) {
      console.error("labeling_candidate_update_ui_failed", error);
      setErrorMessage("The label could not be saved locally.");
    } finally {
      setSavingCandidateId(null);
    }
  }

  if (isLoading) {
    return (
      <section className="glass-card labeling-shell">
        <div className="labeling-empty-state">
          <div className="spinner" aria-hidden="true" />
          <p>Preparing the local labeling workspace…</p>
        </div>
      </section>
    );
  }

  if (!session || !activeCase) {
    return (
      <section className="glass-card labeling-shell">
        <div className="labeling-empty-state">
          <p>{errorMessage || "No labeling session is available yet."}</p>
          <div className="labeling-toolbar">
            <select
              className="labeling-select"
              value={selectedSourceId}
              onChange={(event) => setSelectedSourceId(event.target.value)}
            >
              {sources.map((source) => (
                <option key={source.source_id} value={source.source_id}>
                  {source.file_name}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="labeling-primary-button"
              onClick={() => void handleCreateSession()}
              disabled={!selectedSourceId || isCreatingSession}
            >
              {isCreatingSession ? "Creating…" : "Create session"}
            </button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="glass-card labeling-shell">
      <div className="labeling-header">
        <div>
          <p className="labeling-kicker">Local training labels</p>
          <h1>Comparable labeling workspace</h1>
          <p className="labeling-subtitle">
            Labels are stored locally in a disposable session file and stay separate from the live feedback tables.
          </p>
        </div>
        <div className="labeling-header-actions">
          <button
            type="button"
            className="report-ghost-action"
            onClick={() => {
              window.location.href = "/";
            }}
          >
            Back to app
          </button>
        </div>
      </div>

      <div className="labeling-toolbar">
        <label className="labeling-field">
          <span>Benchmark source</span>
          <select
            className="labeling-select"
            value={selectedSourceId}
            onChange={(event) => setSelectedSourceId(event.target.value)}
          >
            {sources.map((source) => (
              <option key={source.source_id} value={source.source_id}>
                {source.file_name} ({source.case_count} cases)
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="labeling-primary-button"
          onClick={() => void handleCreateSession()}
          disabled={!selectedSourceId || isCreatingSession}
        >
          {isCreatingSession ? "Creating new session…" : "Create fresh session"}
        </button>
        <div className="labeling-session-meta">
          <span>Session: {session.session_name}</span>
          <span>Updated: {formatDateTime(session.updated_at)}</span>
          <span>Stored sessions: {sessions.length}</span>
        </div>
      </div>

      <div className="labeling-summary-grid">
        <div className="labeling-summary-card">
          <span>Reviewed</span>
          <strong>
            {session.summary.reviewed_count}/{session.summary.candidate_count}
          </strong>
          <small>{formatPct(session.summary.completion_ratio)} complete</small>
        </div>
        <div className="labeling-summary-card">
          <span>Saved</span>
          <strong>{session.summary.saved_count}</strong>
        </div>
        <div className="labeling-summary-card">
          <span>Relevant</span>
          <strong>{session.summary.relevant_count}</strong>
        </div>
        <div className="labeling-summary-card">
          <span>Not relevant</span>
          <strong>{session.summary.not_relevant_count}</strong>
        </div>
      </div>

      {errorMessage ? <p className="labeling-error">{errorMessage}</p> : null}

      <div className="labeling-layout">
        <aside className="labeling-case-list">
          <div className="report-section-head">
            <h3>Cases</h3>
            <p>{session.cases.length} queued</p>
          </div>
          <div className="labeling-case-items">
            {session.cases.map((itemCase) => {
              const reviewedCount = caseReviewedCount(itemCase);
              const isActive = itemCase.case_id === activeCase.case_id;
              return (
                <button
                  type="button"
                  key={itemCase.case_id}
                  className={`labeling-case-item ${isActive ? "labeling-case-item-active" : ""}`}
                  onClick={() => setActiveCaseId(itemCase.case_id)}
                >
                  <span className="labeling-case-item-top">
                    <strong>{itemCase.objective}</strong>
                    <span>
                      {reviewedCount}/{itemCase.candidates.length}
                    </span>
                  </span>
                  <span className="labeling-case-item-caption">
                    {queryCaption(itemCase) || itemCase.case_id}
                  </span>
                </button>
              );
            })}
          </div>
        </aside>

        <div className="labeling-main">
          <section className="report-section labeling-query-card">
            <div className="report-section-head">
              <h3>Query case</h3>
              <p>
                {activeCase.objective} · retrieve_k {activeCase.retrieve_k} · pool{" "}
                {activeCase.source_candidate_pool_size}
              </p>
            </div>
            <p className="labeling-query-caption">{queryCaption(activeCase)}</p>
            <p className="labeling-query-author">{queryAuthor(activeCase)}</p>
            <div className="labeling-tag-row">
              {queryHashtags(activeCase).map((tag) => (
                <span className="comparable-tag" key={`${activeCase.case_id}-${tag}`}>
                  {tag}
                </span>
              ))}
            </div>
            {queryCommentsPreview(activeCase).length > 0 ? (
              <div className="labeling-query-comments">
                <strong>Comment preview</strong>
                {queryCommentsPreview(activeCase).map((comment) => (
                  <p key={`${activeCase.case_id}-${comment}`}>{comment}</p>
                ))}
              </div>
            ) : null}
          </section>

          <section className="report-section">
            <div className="report-section-head">
              <h3>Candidates</h3>
              <p>
                Click one label per candidate. Use saved for the strongest examples you would keep as references.
              </p>
            </div>
            <div className="labeling-candidate-list" role="list">
              {activeCase.candidates.map((candidate) => {
                const videoUrl = candidateVideoUrl(candidate);
                const currentLabel = candidate.review.label;
                const isSaving = savingCandidateId === candidate.candidate_id;
                return (
                  <article
                    className="comparable-item labeling-candidate-item"
                    key={candidate.candidate_id}
                    role="listitem"
                  >
                    {videoUrl ? (
                      <a
                        href={videoUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="comparable-thumb-link"
                      >
                        <ComparableThumbnailImage
                          className="comparable-thumb-image"
                          thumbnailUrl=""
                          videoUrl={videoUrl}
                          alt="Comparable thumbnail"
                          fallbackClassName="comparable-thumb comparable-thumb-fallback"
                        />
                        <span className="comparable-thumb-overlay">Open</span>
                      </a>
                    ) : (
                      <span className="comparable-thumb comparable-thumb-fallback" aria-hidden="true" />
                    )}

                    <div className="comparable-main-shell">
                      <div className="labeling-candidate-meta">
                        <span>Baseline rank: {candidate.baseline_rank ?? "n/a"}</span>
                        <span>
                          Score:{" "}
                          {candidate.baseline_score !== null
                            ? candidate.baseline_score.toFixed(3)
                            : "n/a"}
                        </span>
                        <span>Support: {candidate.support_level ?? "unknown"}</span>
                      </div>
                      <p className="comparable-caption">{candidateCaption(candidate)}</p>
                      <p className="comparable-author">{candidateAuthor(candidate)}</p>
                      <div className="comparable-hashtags">
                        {candidateHashtags(candidate).map((tag) => (
                          <span className="comparable-tag" key={`${candidate.candidate_id}-${tag}`}>
                            {tag}
                          </span>
                        ))}
                      </div>
                      <p className="labeling-candidate-reasons">
                        Reasons:{" "}
                        {candidate.ranking_reasons.length > 0
                          ? candidate.ranking_reasons.join(", ")
                          : "Not available"}
                      </p>
                      <div className="labeling-action-row">
                        {(["saved", "relevant", "not_relevant"] as LabelingReviewLabel[]).map(
                          (label) => (
                            <button
                              key={label}
                              type="button"
                              className={`labeling-chip-button ${currentLabel === label ? "labeling-chip-button-active" : ""}`}
                              onClick={() => void handleSetCandidateLabel(activeCase, candidate, label)}
                              disabled={isSaving}
                            >
                              {labelButtonText(label)}
                            </button>
                          )
                        )}
                        <button
                          type="button"
                          className="report-ghost-action"
                          onClick={() => void handleSetCandidateLabel(activeCase, candidate, null)}
                          disabled={isSaving}
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}
