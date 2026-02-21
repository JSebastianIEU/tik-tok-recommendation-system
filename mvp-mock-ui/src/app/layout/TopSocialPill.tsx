import tiktokIcon from "../../../svgicons/tiktok.svg";
import instagramIcon from "../../../svgicons/instagram.svg";
import xIcon from "../../../svgicons/x.svg";
import facebookIcon from "../../../svgicons/facebook.svg";

const SOCIAL_ITEMS = [
  { label: "TikTok", icon: tiktokIcon, isPrimary: true },
  { label: "Instagram", icon: instagramIcon, isPrimary: false },
  { label: "X", icon: xIcon, isPrimary: false },
  { label: "Facebook", icon: facebookIcon, isPrimary: false }
];

export function TopSocialPill(): JSX.Element {
  return (
    <div className="top-social-pill" aria-label="Social channels">
      {SOCIAL_ITEMS.map((item) => (
        <span
          key={item.label}
          className={`social-badge ${item.isPrimary ? "social-badge-active" : "social-badge-static"}`}
          aria-label={item.label}
        >
          <img
            src={item.icon}
            alt=""
            className="social-badge-icon"
            aria-hidden="true"
          />
        </span>
      ))}
    </div>
  );
}
