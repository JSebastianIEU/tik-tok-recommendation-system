import type { ReactNode } from "react";
import { LeftSidebar } from "./layout/LeftSidebar";
import { TopSocialPill } from "./layout/TopSocialPill";
import tiktokIcon from "../../svgicons/tiktok.svg";

interface AppShellProps {
  children: ReactNode;
}

export function AppShell(props: AppShellProps): JSX.Element {
  const { children } = props;

  return (
    <div className="app-shell">
      <div className="background-decor" aria-hidden="true">
        <div className="radial-spot spot-one" />
        <div className="radial-spot spot-two" />
        <div className="tiktok-watermark">
          <img src={tiktokIcon} alt="" className="tiktok-watermark-image" />
        </div>
      </div>

      <LeftSidebar />
      <TopSocialPill />

      <main className="app-main">{children}</main>
    </div>
  );
}
