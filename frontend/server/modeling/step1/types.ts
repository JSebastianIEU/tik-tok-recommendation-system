export const OBJECTIVES = ["reach", "engagement", "conversion", "community"] as const;
export const CONTENT_TYPES = ["tutorial", "story", "reaction", "showcase", "opinion"] as const;
export const PRIMARY_CTAS = ["follow", "comment", "save", "share", "link_click", "none"] as const;

export type Objective = (typeof OBJECTIVES)[number];
export type ContentType = (typeof CONTENT_TYPES)[number];
export type PrimaryCta = (typeof PRIMARY_CTAS)[number];

export interface CandidateInputCore {
  description: string;
  hashtags: string[];
  mentions: string[];
  objective?: Objective;
  audience?: string;
  content_type?: ContentType;
  primary_cta?: PrimaryCta;
  locale?: string;
}

export interface CandidateProfileCore {
  profile_version: "core.v1";
  created_at: string;
  locale: string;
  raw: {
    description: string;
    hashtags: string[];
    mentions: string[];
  };
  normalized: {
    description: string;
    hashtags: string[];
    mentions: string[];
  };
  tokens: {
    description_tokens: string[];
    hashtag_tokens: string[];
    mention_tokens: string[];
    keyphrases: string[];
  };
  intent: {
    objective: Objective;
    audience: string;
    content_type: ContentType;
    primary_cta: PrimaryCta;
  };
  features: {
    word_count: number;
    unique_word_count: number;
    hashtag_count: number;
    mention_count: number;
    hashtag_density: number;
    question_mark_present: boolean;
    exclamation_present: boolean;
    number_present: boolean;
    cta_keyword_present: boolean;
  };
  tags: {
    topic_tags: string[];
    format_tags: string[];
  };
  retrieval: {
    text: string;
  };
  quality: {
    confidence: number;
    flags: string[];
  };
}
