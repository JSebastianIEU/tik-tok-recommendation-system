export type BreakerState = "closed" | "open" | "half_open";

interface BreakerEvent {
  at: number;
  ok: boolean;
}

export interface CircuitBreakerPolicy {
  minRequests: number;
  errorRateThreshold: number;
  consecutiveFailureThreshold: number;
  windowMs: number;
  openMs: number;
  halfOpenMaxProbes: number;
  halfOpenSuccessToClose: number;
}

interface BreakerRecord {
  state: BreakerState;
  openedUntil: number;
  consecutiveFailures: number;
  events: BreakerEvent[];
  halfOpenProbeCount: number;
  halfOpenSuccessCount: number;
}

function newRecord(): BreakerRecord {
  return {
    state: "closed",
    openedUntil: 0,
    consecutiveFailures: 0,
    events: [],
    halfOpenProbeCount: 0,
    halfOpenSuccessCount: 0
  };
}

export class RecommenderCircuitBreakers {
  private readonly policy: CircuitBreakerPolicy;
  private readonly records = new Map<string, BreakerRecord>();

  constructor(policy: CircuitBreakerPolicy) {
    this.policy = policy;
  }

  private getRecord(key: string): BreakerRecord {
    const normalized = key.trim() || "default";
    const existing = this.records.get(normalized);
    if (existing) {
      return existing;
    }
    const created = newRecord();
    this.records.set(normalized, created);
    return created;
  }

  private trimWindow(record: BreakerRecord, now: number): void {
    const cutoff = now - this.policy.windowMs;
    while (record.events.length > 0 && record.events[0]!.at < cutoff) {
      record.events.shift();
    }
  }

  shouldAllow(key: string, now = Date.now()): { allow: boolean; state: BreakerState } {
    const record = this.getRecord(key);
    if (record.state === "open") {
      if (now >= record.openedUntil) {
        record.state = "half_open";
        record.halfOpenProbeCount = 0;
        record.halfOpenSuccessCount = 0;
      } else {
        return { allow: false, state: "open" };
      }
    }
    if (record.state === "half_open") {
      if (record.halfOpenProbeCount >= this.policy.halfOpenMaxProbes) {
        return { allow: false, state: "half_open" };
      }
      record.halfOpenProbeCount += 1;
      return { allow: true, state: "half_open" };
    }
    return { allow: true, state: record.state };
  }

  recordSuccess(key: string, now = Date.now()): BreakerState {
    const record = this.getRecord(key);
    record.events.push({ at: now, ok: true });
    this.trimWindow(record, now);
    record.consecutiveFailures = 0;
    if (record.state === "half_open") {
      record.halfOpenSuccessCount += 1;
      if (record.halfOpenSuccessCount >= this.policy.halfOpenSuccessToClose) {
        record.state = "closed";
        record.halfOpenProbeCount = 0;
        record.halfOpenSuccessCount = 0;
      }
    }
    return record.state;
  }

  recordFailure(key: string, now = Date.now()): BreakerState {
    const record = this.getRecord(key);
    record.events.push({ at: now, ok: false });
    this.trimWindow(record, now);
    record.consecutiveFailures += 1;

    const total = record.events.length;
    const failures = record.events.filter((event) => !event.ok).length;
    const errorRate = total > 0 ? failures / total : 0;
    const shouldOpen =
      record.consecutiveFailures >= this.policy.consecutiveFailureThreshold ||
      (total >= this.policy.minRequests && errorRate >= this.policy.errorRateThreshold);
    if (record.state === "half_open" || shouldOpen) {
      record.state = "open";
      record.openedUntil = now + this.policy.openMs;
      record.halfOpenProbeCount = 0;
      record.halfOpenSuccessCount = 0;
    }
    return record.state;
  }

  snapshot(key: string, now = Date.now()): {
    key: string;
    state: BreakerState;
    opened_until: number;
    consecutive_failures: number;
  } {
    const record = this.getRecord(key);
    this.trimWindow(record, now);
    return {
      key,
      state: record.state,
      opened_until: record.openedUntil,
      consecutive_failures: record.consecutiveFailures
    };
  }
}

export interface StageLatencyTracker {
  mark(stage: string): void;
  end(): Record<string, number>;
}

export function createStageLatencyTracker(): StageLatencyTracker {
  const startedAt = Date.now();
  const marks: Record<string, number> = {};
  return {
    mark(stage: string) {
      marks[stage] = Date.now() - startedAt;
    },
    end() {
      return {
        ...marks,
        total: Date.now() - startedAt
      };
    }
  };
}
