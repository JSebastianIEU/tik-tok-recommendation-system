export interface DemoVideoMetrics {
  views: number;
  likes: number;
  comments_count: number;
  shares: number;
}

export interface DemoVideoAuthor {
  author_id?: string;
  username?: string;
  followers?: number;
  [key: string]: unknown;
}

export interface DemoVideoRecord {
  video_id: string;
  caption: string;
  hashtags: string[];
  keywords: string[];
  metrics: DemoVideoMetrics;
  author: string | DemoVideoAuthor;
  comments: string[];
  [key: string]: unknown;
}
