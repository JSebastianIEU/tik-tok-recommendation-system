import { useEffect, useMemo, useState } from "react";
import { buildThumbnailProxyUrl } from "../utils/buildThumbnailProxyUrl";

interface ComparableThumbnailImageProps {
  thumbnailUrl: string;
  videoUrl: string;
  className: string;
  alt: string;
  fallbackClassName: string;
}

function uniqueNonEmpty(values: string[]): string[] {
  return Array.from(
    new Set(values.map((value) => value.trim()).filter(Boolean))
  );
}

export function ComparableThumbnailImage(
  props: ComparableThumbnailImageProps
): JSX.Element {
  const { thumbnailUrl, videoUrl, className, alt, fallbackClassName } = props;
  const [currentSourceIndex, setCurrentSourceIndex] = useState<number>(0);

  const imageSources = useMemo(() => {
    const proxyUrl = buildThumbnailProxyUrl(thumbnailUrl, videoUrl);
    return uniqueNonEmpty([proxyUrl, thumbnailUrl]);
  }, [thumbnailUrl, videoUrl]);

  useEffect(() => {
    setCurrentSourceIndex(0);
  }, [imageSources]);

  const currentSource = imageSources[currentSourceIndex];

  if (!currentSource) {
    return <span className={fallbackClassName} aria-hidden="true" />;
  }

  return (
    <img
      className={className}
      src={currentSource}
      alt={alt}
      loading="lazy"
      referrerPolicy="no-referrer"
      onError={() => {
        const hasNextSource = currentSourceIndex < imageSources.length - 1;
        if (hasNextSource) {
          setCurrentSourceIndex((previous) => previous + 1);
        } else {
          setCurrentSourceIndex(imageSources.length);
        }
      }}
    />
  );
}
