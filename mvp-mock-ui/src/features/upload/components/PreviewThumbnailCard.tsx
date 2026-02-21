import { useEffect, useRef, useState } from "react";

interface PreviewThumbnailCardProps {
  videoUrl: string | null;
  fileName: string | null;
}

export function PreviewThumbnailCard(
  props: PreviewThumbnailCardProps
): JSX.Element {
  const { videoUrl, fileName } = props;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [previewError, setPreviewError] = useState<boolean>(false);

  useEffect(() => {
    if (!videoUrl || !videoRef.current) {
      return;
    }

    const element = videoRef.current;
    const playPromise = element.play();
    if (playPromise) {
      void playPromise.catch(() => {
        setPreviewError(true);
      });
    }
  }, [videoUrl]);

  useEffect(() => {
    setPreviewError(false);
  }, [videoUrl]);

  return (
    <section className="glass-card preview-thumbnail-card">
      <h3 className="panel-title">preview</h3>

      <div className="preview-thumbnail-frame">
        {videoUrl ? (
          <video
            ref={videoRef}
            className="preview-thumbnail-video"
            src={videoUrl}
            muted
            playsInline
            autoPlay
            loop
            preload="metadata"
            onError={() => setPreviewError(true)}
          />
        ) : (
          <div className="preview-thumbnail-fallback">Preview is not available</div>
        )}

        {previewError ? (
          <div className="preview-thumbnail-fallback">
            The video preview could not be played.
          </div>
        ) : null}
      </div>

      <p className="preview-thumbnail-file" aria-live="polite">
        File: {fileName ?? "No file"}
      </p>
    </section>
  );
}
