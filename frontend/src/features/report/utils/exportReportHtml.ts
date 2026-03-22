function collectStyles(): string {
  const styleNodes = document.querySelectorAll('link[rel="stylesheet"], style');
  return Array.from(styleNodes)
    .map((node) => node.outerHTML)
    .join("\n");
}

function buildExportDocument(reportHtml: string): string {
  const styles = collectStyles();
  const bodyClass = document.body.className;

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Video Report Export</title>
    ${styles}
    <style>
      body { overflow: auto !important; min-height: 100vh; }
      .export-wrap { width: min(1100px, 96vw); margin: 0 auto; padding: 24px 0; }
      .report-panel { height: auto !important; min-height: 0 !important; }
      .report-scroll-container { height: auto !important; overflow: visible !important; }
      .report-header { position: sticky; top: 0; }
      .report-drawer, .report-drawer-overlay { display: none !important; }
    </style>
  </head>
  <body class="${bodyClass}">
    <div class="app-shell">
      <main class="app-main">
        <div class="export-wrap">
          ${reportHtml}
        </div>
      </main>
    </div>
  </body>
</html>`;
}

export function exportReportHtml(reportElement: HTMLElement | null): void {
  if (!reportElement) {
    return;
  }

  const html = buildExportDocument(reportElement.outerHTML);
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const objectUrl = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = `video-report-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.html`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}
