import {
  CONTENT_TYPES,
  OBJECTIVES,
  PRIMARY_CTAS,
  type CandidateInputCore,
  type CandidateProfileCore,
  type ContentType,
  type Objective,
  type PrimaryCta
} from "./types";

const STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "how",
  "i",
  "in",
  "is",
  "it",
  "my",
  "of",
  "on",
  "or",
  "our",
  "that",
  "the",
  "this",
  "to",
  "was",
  "we",
  "with",
  "you",
  "your",
  "de",
  "del",
  "el",
  "en",
  "es",
  "la",
  "las",
  "los",
  "para",
  "por",
  "que",
  "un",
  "una",
  "y"
]);

const CTA_HINT_WORDS = [
  "follow",
  "comment",
  "save",
  "share",
  "link",
  "bio",
  "dm",
  "join",
  "subscribe",
  "shop",
  "compra",
  "guarda",
  "comenta",
  "sigue"
];

const TOPIC_LEXICON: Record<string, string[]> = {
  food: ["food", "recipe", "cook", "meal", "kitchen", "pasta", "bake", "chef", "comida", "receta"],
  fitness: ["fitness", "workout", "gym", "exercise", "cardio", "muscle", "health", "salud"],
  finance: ["finance", "invest", "money", "crypto", "budget", "stock", "trading", "finanzas"],
  beauty: ["beauty", "makeup", "skincare", "hair", "cosmetic", "glow"],
  fashion: ["fashion", "outfit", "style", "lookbook", "streetwear", "wardrobe"],
  tech: ["tech", "ai", "software", "app", "coding", "gadget", "developer", "programming"],
  travel: ["travel", "trip", "vacation", "flight", "hotel", "city", "journey", "viaje"],
  education: ["learn", "lesson", "tutorial", "guide", "tips", "teach", "study", "class"],
  entertainment: ["funny", "meme", "dance", "music", "viral", "trend", "reaction"],
  lifestyle: ["lifestyle", "daily", "routine", "mindset", "productivity", "selfcare"]
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function removeDiacritics(value: string): string {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

function normalizePlainText(value: string): string {
  return normalizeWhitespace(
    removeDiacritics(value.toLowerCase()).replace(/[^\p{L}\p{N}\s#@?!._-]/gu, " ")
  );
}

function splitCompoundToken(value: string): string[] {
  const expanded = value
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/([A-Za-z])(\d)/g, "$1 $2")
    .replace(/(\d)([A-Za-z])/g, "$1 $2")
    .replace(/[_\-.]+/g, " ");

  return normalizePlainText(expanded)
    .split(" ")
    .map((token) => token.trim())
    .filter((token) => token.length >= 2);
}

function sanitizeDescription(value: string): string {
  return normalizeWhitespace(value).slice(0, 5000);
}

function normalizeLocale(locale?: string): string {
  const fallback = "en";
  if (!locale) {
    return fallback;
  }
  const cleaned = locale.trim();
  if (!cleaned) {
    return fallback;
  }
  const normalized = cleaned.replace(/_/g, "-").toLowerCase();
  return /^[a-z]{2,3}(-[a-z0-9]{2,8})?$/.test(normalized) ? normalized : fallback;
}

function normalizeTagList(
  values: string[],
  options: {
    prefix: "#" | "@";
    maxItems: number;
    allowDot?: boolean;
  }
): string[] {
  const { prefix, maxItems, allowDot = false } = options;
  const seen = new Set<string>();
  const output: string[] = [];

  for (const raw of values) {
    if (output.length >= maxItems) {
      break;
    }
    const trimmed = typeof raw === "string" ? raw.trim() : "";
    if (!trimmed) {
      continue;
    }
    const withoutPrefix = trimmed.replace(new RegExp(`^\\${prefix}+`), "");
    const lowered = removeDiacritics(withoutPrefix.toLowerCase());
    const allowedPattern = allowDot ? /[^a-z0-9._]/g : /[^a-z0-9_]/g;
    const compact = lowered.replace(/\s+/g, "").replace(allowedPattern, "");
    if (compact.length < 2) {
      continue;
    }
    const value = compact.slice(0, 40);
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    output.push(value);
  }

  return output;
}

function tokenizeDescription(normalizedDescription: string): string[] {
  return normalizedDescription
    .replace(/[#!?.,/\\()[\]{}:;"'`~]/g, " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 2);
}

function extractKeyphrases(descriptionTokens: string[], hashtagTokens: string[]): string[] {
  const phraseScores = new Map<string, number>();
  const maxN = 3;

  for (let i = 0; i < descriptionTokens.length; i += 1) {
    for (let n = 2; n <= maxN; n += 1) {
      const slice = descriptionTokens.slice(i, i + n);
      if (slice.length !== n) {
        continue;
      }
      if (STOPWORDS.has(slice[0]) || STOPWORDS.has(slice[slice.length - 1])) {
        continue;
      }
      const phrase = slice.join(" ");
      const contentTerms = slice.filter((token) => !STOPWORDS.has(token)).length;
      if (contentTerms < 2) {
        continue;
      }
      const hashtagBoost = hashtagTokens.some((token) => phrase.includes(token)) ? 0.8 : 0;
      const score = contentTerms + n * 0.4 + hashtagBoost;
      phraseScores.set(phrase, (phraseScores.get(phrase) ?? 0) + score);
    }
  }

  return [...phraseScores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([phrase]) => phrase);
}

function inferTopicTags(tokens: string[], hashtags: string[]): string[] {
  const vocabulary = new Set([...tokens, ...hashtags]);
  const out: string[] = [];

  for (const [topic, terms] of Object.entries(TOPIC_LEXICON)) {
    const score = terms.reduce((total, term) => total + (vocabulary.has(term) ? 1 : 0), 0);
    if (score > 0) {
      out.push(topic);
    }
  }

  if (out.length === 0) {
    out.push("general");
  }
  return out;
}

function inferFormatTags(description: string, tokens: string[], declaredType: ContentType): string[] {
  const out = new Set<string>([`creator_selected:${declaredType}`]);
  const text = description.toLowerCase();
  const joined = tokens.join(" ");

  if (/\b(how to|step|tutorial|guide|tips|tricks|hack)\b/.test(text)) {
    out.add("how_to");
  }
  if (/\b(\d+)\s*(ways|tips|ideas|steps)\b/.test(joined) || /\btop\s+\d+\b/.test(joined)) {
    out.add("listicle");
  }
  if (/\b(my story|journey|when i|i was|what happened)\b/.test(text)) {
    out.add("personal_story");
  }
  if (/\b(unpopular opinion|hot take|i think)\b/.test(text)) {
    out.add("opinionated");
  }
  if (/\b(duet|stitch|react|reaction)\b/.test(text)) {
    out.add("reaction");
  }
  if (/\b(before and after|results|review|demo|showcase)\b/.test(text)) {
    out.add("showcase");
  }
  if (out.size === 1) {
    out.add("general");
  }

  return [...out];
}

function hasCtaKeyword(tokens: string[]): boolean {
  return tokens.some((token) => CTA_HINT_WORDS.includes(token));
}

function pickObjective(value?: Objective): Objective {
  return value && OBJECTIVES.includes(value) ? value : "engagement";
}

function pickContentType(value?: ContentType): ContentType {
  return value && CONTENT_TYPES.includes(value) ? value : "showcase";
}

function pickPrimaryCta(value?: PrimaryCta): PrimaryCta {
  return value && PRIMARY_CTAS.includes(value) ? value : "none";
}

function buildQuality(
  options: {
    wordCount: number;
    hashtagsCount: number;
    mentionsCount: number;
    keyphraseCount: number;
    audience: string;
  }
): { confidence: number; flags: string[] } {
  const { wordCount, hashtagsCount, mentionsCount, keyphraseCount, audience } = options;
  const flags: string[] = [];
  let confidence = 1;

  if (wordCount === 0) {
    confidence -= 0.5;
    flags.push("missing_description");
  } else if (wordCount < 8) {
    confidence -= 0.2;
    flags.push("low_text_signal");
  }

  if (hashtagsCount === 0) {
    confidence -= 0.1;
    flags.push("no_hashtags");
  }
  if (hashtagsCount > 12) {
    confidence -= 0.08;
    flags.push("hashtag_spam_risk");
  }
  if (mentionsCount > 8) {
    confidence -= 0.05;
    flags.push("mention_spam_risk");
  }
  if (keyphraseCount < 2) {
    confidence -= 0.07;
    flags.push("sparse_keyphrases");
  }
  if (!audience) {
    confidence -= 0.05;
    flags.push("missing_audience");
  }

  return {
    confidence: round(clamp(confidence, 0.1, 1), 2),
    flags
  };
}

function uniq(values: string[]): string[] {
  return [...new Set(values)];
}

function buildRetrievalText(parts: {
  description: string;
  keyphrases: string[];
  hashtags: string[];
  topicTags: string[];
  formatTags: string[];
}): string {
  const hashtagText = parts.hashtags.map((tag) => `#${tag}`).join(" ");
  return normalizeWhitespace(
    [
      parts.description,
      parts.description,
      parts.keyphrases.join(" "),
      parts.keyphrases.join(" "),
      hashtagText,
      parts.topicTags.join(" "),
      parts.formatTags.join(" ")
    ]
      .filter(Boolean)
      .join(" ")
  );
}

export function buildCandidateProfileCore(input: CandidateInputCore): CandidateProfileCore {
  const rawDescription = sanitizeDescription(input.description ?? "");
  const rawHashtags = Array.isArray(input.hashtags) ? input.hashtags : [];
  const rawMentions = Array.isArray(input.mentions) ? input.mentions : [];

  const normalizedDescription = normalizePlainText(rawDescription);
  const normalizedHashtags = normalizeTagList(rawHashtags, { prefix: "#", maxItems: 30 });
  const normalizedMentions = normalizeTagList(rawMentions, {
    prefix: "@",
    maxItems: 30,
    allowDot: true
  });

  const descriptionTokens = tokenizeDescription(normalizedDescription);
  const hashtagTokens = uniq(normalizedHashtags.flatMap((tag) => splitCompoundToken(tag)));
  const mentionTokens = uniq(normalizedMentions.flatMap((tag) => splitCompoundToken(tag)));
  const keyphrases = extractKeyphrases(descriptionTokens, hashtagTokens);

  const objective = pickObjective(input.objective);
  const contentType = pickContentType(input.content_type);
  const primaryCta = pickPrimaryCta(input.primary_cta);
  const audience = normalizeWhitespace(input.audience ?? "").toLowerCase();

  const topicTags = inferTopicTags(
    [...descriptionTokens, ...hashtagTokens],
    [...normalizedHashtags, ...hashtagTokens]
  );
  const formatTags = inferFormatTags(normalizedDescription, descriptionTokens, contentType);

  const wordCount = descriptionTokens.length;
  const uniqueWordCount = new Set(descriptionTokens).size;
  const hashtagCount = normalizedHashtags.length;
  const mentionCount = normalizedMentions.length;
  const quality = buildQuality({
    wordCount,
    hashtagsCount: hashtagCount,
    mentionsCount: mentionCount,
    keyphraseCount: keyphrases.length,
    audience
  });

  return {
    profile_version: "core.v1",
    created_at: new Date().toISOString(),
    locale: normalizeLocale(input.locale),
    raw: {
      description: rawDescription,
      hashtags: rawHashtags
        .map((value) => (typeof value === "string" ? value.trim() : ""))
        .filter(Boolean)
        .slice(0, 30),
      mentions: rawMentions
        .map((value) => (typeof value === "string" ? value.trim() : ""))
        .filter(Boolean)
        .slice(0, 30)
    },
    normalized: {
      description: normalizedDescription,
      hashtags: normalizedHashtags,
      mentions: normalizedMentions
    },
    tokens: {
      description_tokens: descriptionTokens,
      hashtag_tokens: hashtagTokens,
      mention_tokens: mentionTokens,
      keyphrases
    },
    intent: {
      objective,
      audience,
      content_type: contentType,
      primary_cta: primaryCta
    },
    features: {
      word_count: wordCount,
      unique_word_count: uniqueWordCount,
      hashtag_count: hashtagCount,
      mention_count: mentionCount,
      hashtag_density: round(hashtagCount / Math.max(1, wordCount), 4),
      question_mark_present: rawDescription.includes("?"),
      exclamation_present: rawDescription.includes("!"),
      number_present: /\d/.test(rawDescription),
      cta_keyword_present: hasCtaKeyword(descriptionTokens)
    },
    tags: {
      topic_tags: topicTags,
      format_tags: formatTags
    },
    retrieval: {
      text: buildRetrievalText({
        description: normalizedDescription,
        keyphrases,
        hashtags: normalizedHashtags,
        topicTags,
        formatTags
      })
    },
    quality
  };
}
