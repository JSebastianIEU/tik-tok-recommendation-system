import { randomBytes } from "node:crypto";

function toHex(value: number, width: number): string {
  return value.toString(16).padStart(width, "0");
}

export function createUuidV7(nowMs = Date.now()): string {
  const ts = BigInt(nowMs);
  const timeHex = ts.toString(16).padStart(12, "0").slice(-12);
  const rand = randomBytes(10);

  const hi12 = Number.parseInt(timeHex.slice(0, 8), 16) >>> 0;
  const lo48 = timeHex.slice(8);

  const randA = ((rand[0] << 8) | rand[1]) & 0x0fff;
  const randB = ((rand[2] << 24) | (rand[3] << 16) | (rand[4] << 8) | rand[5]) >>> 0;
  const randC = rand.subarray(6, 10).toString("hex");

  const seg1 = toHex(hi12, 8);
  const seg2 = lo48.slice(0, 4);
  const seg3 = `7${toHex(randA, 3)}`;
  const variantNibble = ((randB >>> 28) & 0x3) | 0x8;
  const seg4 = `${variantNibble.toString(16)}${toHex((randB >>> 16) & 0x0fff, 3)}`;
  const seg5 = `${toHex(randB & 0xffff, 4)}${randC}`;
  return `${seg1}-${seg2}-${seg3}-${seg4}-${seg5}`;
}

