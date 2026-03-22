import { UserAvatarButton } from "./UserAvatarButton";

export function LeftSidebar(): JSX.Element {
  return (
    <aside className="left-sidebar" aria-label="Sidebar">
      <button
        type="button"
        className="sidebar-icon-button"
        aria-label="Main menu"
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <rect
            x="3"
            y="4"
            width="18"
            height="4"
            rx="1"
            stroke="currentColor"
            strokeWidth="1.8"
          />
          <path
            d="M6 8V18H18V8"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M9 12H15"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
        </svg>
      </button>

      <UserAvatarButton className="sidebar-avatar-button" />
    </aside>
  );
}
