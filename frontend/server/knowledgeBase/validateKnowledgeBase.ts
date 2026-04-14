import { KNOWLEDGE_BASE_PATH } from "../config";
import { loadKnowledgeBaseStore } from "./knowledgeBase";

async function main(): Promise<void> {
  const store = await loadKnowledgeBaseStore(KNOWLEDGE_BASE_PATH);
  const categoryCounts = new Map<string, number>();
  for (const item of store.entries) {
    categoryCounts.set(item.entry.category, (categoryCounts.get(item.entry.category) ?? 0) + 1);
  }

  const summary = Array.from(categoryCounts.entries())
    .sort((left, right) => left[0].localeCompare(right[0]))
    .map(([category, count]) => `${category}:${count}`)
    .join(", ");

  console.log(`Knowledge base valid | version=${store.version} | entries=${store.entries.length}`);
  console.log(`Categories: ${summary}`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Knowledge base validation failed: ${message}`);
  process.exitCode = 1;
});
