interface UserAvatarButtonProps {
  className?: string;
}

export function UserAvatarButton(props: UserAvatarButtonProps): JSX.Element {
  const className = props.className ?? "";

  return (
    <button type="button" className={`avatar-button ${className}`} aria-label="User profile">
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <circle cx="12" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.6" />
        <path
          d="M5.5 19C6.8 15.8 9 14.4 12 14.4C15 14.4 17.2 15.8 18.5 19"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    </button>
  );
}
