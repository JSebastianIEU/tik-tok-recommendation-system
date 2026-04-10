import {
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type MouseEvent
} from "react";
import { LoadingOverlay } from "./LoadingOverlay";

interface UploadCardProps {
  fileName: string | null;
  isBusy: boolean;
  loadingLabel: string;
  onFileSelected: (file: File | null) => void;
}

function isVideoFile(file: File): boolean {
  return file.type.startsWith("video/");
}

export function UploadCard(props: UploadCardProps): JSX.Element {
  const { fileName, isBusy, loadingLabel, onFileSelected } = props;

  const inputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef<number>(0);

  const [isHover, setIsHover] = useState<boolean>(false);
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [fileError, setFileError] = useState<string | null>(null);

  const openFileSelector = (): void => {
    inputRef.current?.click();
  };

  const validateAndSelectFile = (file: File | null): void => {
    if (!file) {
      onFileSelected(null);
      return;
    }

    if (!isVideoFile(file)) {
      setFileError("Only video files are allowed.");
      onFileSelected(null);
      return;
    }

    setFileError(null);
    onFileSelected(file);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>): void => {
    const nextFile = event.target.files?.[0] ?? null;
    validateAndSelectFile(nextFile);
  };

  const handleDragEnter = (event: DragEvent<HTMLElement>): void => {
    event.preventDefault();
    event.stopPropagation();

    dragCounter.current += 1;
    setIsDragActive(true);
  };

  const handleDragOver = (event: DragEvent<HTMLElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    if (!isDragActive) {
      setIsDragActive(true);
    }
  };

  const handleDragLeave = (event: DragEvent<HTMLElement>): void => {
    event.preventDefault();
    event.stopPropagation();

    dragCounter.current = Math.max(0, dragCounter.current - 1);
    if (dragCounter.current === 0) {
      setIsDragActive(false);
    }
  };

  const handleDrop = (event: DragEvent<HTMLElement>): void => {
    event.preventDefault();
    event.stopPropagation();

    dragCounter.current = 0;
    setIsDragActive(false);

    const droppedFile = event.dataTransfer.files?.[0] ?? null;
    validateAndSelectFile(droppedFile);
  };

  const handleMouseEnter = (_event: MouseEvent<HTMLElement>): void => {
    setIsHover(true);
  };

  const handleMouseLeave = (_event: MouseEvent<HTMLElement>): void => {
    setIsHover(false);
  };

  const dropzoneClassNames = [
    "upload-dropzone",
    isHover ? "upload-dropzone-hover" : "",
    isDragActive ? "upload-dropzone-active" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className="glass-card upload-card-container">
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden-file-input"
        onChange={handleInputChange}
        aria-label="Select video file"
      />

      <button
        type="button"
        className={dropzoneClassNames}
        onClick={openFileSelector}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        aria-label="Upload video"
      >
        <svg
          className="upload-icon"
          width="56"
          height="56"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <path
            d="M12 4V14M12 14L8 10M12 14L16 10M5 15V18C5 19.1046 5.89543 20 7 20H17C18.1046 20 19 19.1046 19 18V15"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="upload-text">upload</span>
        <span className="upload-text">video</span>
      </button>

      <p className="selected-file-name" aria-live="polite">
        {fileName ? `File: ${fileName}` : "No video selected"}
      </p>

      {fileError ? (
        <p className="upload-file-error" role="alert">
          {fileError}
        </p>
      ) : null}

      {isBusy ? <LoadingOverlay label={loadingLabel} /> : null}
    </section>
  );
}
