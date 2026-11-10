"use client";

import { useEffect, useRef, useState } from "react";
import { agentColor, isDashedArch, palette, type PaletteMode } from "@/lib/colors";
import { engineLabelForAgent } from "@/lib/types";
import { money } from "@/lib/format";
import { ArchBadge } from "@/components/ArchBadge";
import { TableCard } from "@/components/TableCard";
import type { MatchingResult, TableResult } from "@/lib/types";

/**
 * A "living", orbit-able 3D view of one negotiation -- shippers (circles)
 * and carriers (diamonds) as two glowing clusters, one edge per bilateral
 * table, colored the business-standard way (green = won, red = lost --
 * whether that's a clean impasse or a table that closed but Gale-Shapley
 * didn't keep it -- gold = still being negotiated). Modeled on the
 * hand-rolled Canvas 2D technique from a sibling project's graph view
 * (fr8-carrier-recommender's app_v4.py `_3D_HTML`): NOT WebGL/three.js --
 * that project's own vendored three.js is dead code, unused by the actual
 * renderer -- just a hand-written rotate-then-perspective-divide camera,
 * painter's-algorithm back-to-front sorting, distance fog, and a few
 * time-driven cosmetic animations (breathing scale, expanding pulse
 * rings, particles traveling along edges) layered with additive-blend
 * glow (`globalCompositeOperation = "lighter"` on stacked radial
 * gradients, not `shadowBlur`). Node LAYOUT is fixed and structural rather
 * than force-directed -- shippers and carriers each occupy one hemisphere
 * of a single shared sphere (see `hemispherePoint` below), meeting at the
 * sphere's equator, so the WHOLE thing reads as one glowing orb from any
 * angle instead of two flat facing planes, while WHO is a shipper vs a
 * carrier (the fact this view exists to keep legible) stays exactly as
 * unambiguous as before -- only the camera moves, never which hemisphere
 * a node is in.
 *
 * Interaction model mirrors the original SVG version exactly (same
 * `activeId` semantics: a node id highlights all its tables in the panel
 * below, an edge key drops straight into that table's full `TableCard`) --
 * only the rendering technique changed, so page.tsx needs no changes.
 * Hovering (no click needed) additionally draws a small floating card
 * right on the canvas, in the same animation loop as everything else
 * (fading in over ~180ms), showing that node's or edge's live status --
 * see `hoverInfoLines`/the tooltip block inside `frame()`.
 */

const NODE_R = 9;
const SPHERE_R = 150; // shared sphere radius -- shippers fill the x<0 hemisphere, carriers x>0
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5)); // ~2.39996 rad -- even spiral coverage of a hemisphere
const FOV = 260;
const DEFAULT_DIST = 360;
const MIN_DIST = 150;
const MAX_DIST = 680;
const AUTO_ROTATE_SPEED = 0.12; // rad/s while idle
const DRAG_CLICK_THRESHOLD = 6; // px of movement before a pointerdown->up counts as a drag, not a click
const HOVER_FADE_MS = 180;

type EdgeStatus = "empty" | "active" | "impasse" | "closed-unmatched" | "matched";

function edgeStatus(t: TableResult | undefined, isMatched: boolean): EdgeStatus {
  if (!t) return "empty";
  if (t.closed) return isMatched ? "matched" : "closed-unmatched";
  if (t.impasse) return "impasse";
  if (t.history.length > 0) return "active";
  return "empty";
}

const STATUS_LABEL: Record<EdgeStatus, string> = {
  empty: "no offers yet",
  active: "en curso",
  impasse: "impasse",
  "closed-unmatched": "closed, not matched",
  matched: "🏆 matched -- final pairing",
};

type Vec3 = { x: number; y: number; z: number };

type NodeDatum = {
  id: string;
  role: "shipper" | "carrier";
  pos: Vec3;
  color: string;
  dashed: boolean;
  label: string;
};

type EdgeDatum = {
  key: string;
  shipperId: string;
  carrierId: string;
};

type PulseRing = { nodeId: string; start: number; color: string };
type Particle = { edgeKey: string; t: number; speed: number; actor: "shipper" | "carrier" };

/**
 * i-th of n points spread evenly over one hemisphere of radius R, via a
 * golden-angle spiral from the pole (i=0, densest) out to the equator
 * (i=n-1) -- the standard even-point-on-a-sphere trick, halved. `sign`
 * picks which hemisphere: -1 puts the pole at x=-R (shippers), +1 at
 * x=+R (carriers); both share the same equator plane at x=0, so the two
 * hemispheres always join into one continuous sphere.
 */
function hemispherePoint(i: number, n: number, sign: 1 | -1): Vec3 {
  const t = (i + 0.5) / n;
  const poleCoord = SPHERE_R * Math.cos((t * Math.PI) / 2);
  const ringR = SPHERE_R * Math.sin((t * Math.PI) / 2);
  const theta = i * GOLDEN_ANGLE;
  return { x: sign * poleCoord, y: ringR * Math.cos(theta), z: ringR * Math.sin(theta) };
}

function project(p: Vec3, angle: number, tilt: number, dist: number, cx: number, cy: number, zoom: number) {
  // Rotate around Y (orbit), then around X (tilt) -- standard hand-rolled
  // camera math, no matrix library needed for just two axes.
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const x1 = p.x * cosA - p.z * sinA;
  const z1 = p.x * sinA + p.z * cosA;
  const cosT = Math.cos(tilt);
  const sinT = Math.sin(tilt);
  const y2 = p.y * cosT - z1 * sinT;
  const z2 = p.y * sinT + z1 * cosT;

  const tz = z2 + dist;
  const sc = (FOV / Math.max(tz, 1)) * zoom;
  return { sx: cx + x1 * sc, sy: cy + y2 * sc, depth: tz, scale: sc };
}

/** Content for the floating hover card -- a node id summarizes all of its
 * tables (derived from `edges`, not the `shipperIds`/`carrierIds` props,
 * so it stays correct even though this whole render loop reads the
 * roster only through refs); an edge key shows that one table's status
 * and latest offer. */
function hoverInfoLines(
  id: string,
  tablesNow: Record<string, TableResult>,
  matchedCarrierOf: Record<string, string>,
  nodes: NodeDatum[],
  edges: EdgeDatum[]
): string[] {
  if (id.includes("::")) {
    const t = tablesNow[id];
    const [s, cid] = id.split("::");
    const matched = matchedCarrierOf[s] === cid;
    const status = edgeStatus(t, matched);
    const lines = [`${s.replace(/^shipper-/, "")} ↔ ${cid.replace(/^carrier-/, "")}`, STATUS_LABEL[status]];
    if (t?.closed) lines.push(`${money(t.rate)} · turn ${t.turns_to_close}`);
    else if (t && t.history.length > 0) {
      const last = t.history[t.history.length - 1];
      lines.push(last.action === "offer" ? `t${last.turn} ${last.actor}: ${money(last.amount)}` : `t${last.turn} ${last.actor} ${last.action}`);
    }
    return lines;
  }

  const node = nodes.find((n) => n.id === id);
  if (!node) return [id];
  const myEdges = edges.filter((e) => e.shipperId === id || e.carrierId === id);
  let closed = 0;
  let won = 0;
  for (const e of myEdges) {
    const t = tablesNow[e.key];
    if (t?.closed) {
      closed++;
      if (matchedCarrierOf[e.shipperId] === e.carrierId) won++;
    }
  }
  const lines = [node.label, `${myEdges.length} table${myEdges.length === 1 ? "" : "s"} · ${closed} closed`];
  if (won > 0) lines.push(`🏆 won ${won}`);
  return lines;
}

/** The floating hover card itself -- a small rounded, semi-transparent
 * panel drawn directly in the canvas (not a DOM overlay), anchored next
 * to whatever's hovered and clamped so it never runs off the edge.
 * `fade` (0..1) is driven by the render loop from how long the hover has
 * been held, so it eases in with the rest of the animation. */
function drawHoverCard(ctx: CanvasRenderingContext2D, lines: string[], x: number, y: number, fade: number, canvasW: number, canvasH: number) {
  if (fade <= 0 || lines.length === 0) return;
  ctx.save();
  ctx.globalAlpha = fade;
  const paddingX = 8;
  const paddingY = 6;
  const lineH = 15;
  ctx.font = "11px ui-monospace, monospace";
  let maxW = 0;
  for (const l of lines) maxW = Math.max(maxW, ctx.measureText(l).width);
  const boxW = maxW + paddingX * 2;
  const boxH = lines.length * lineH + paddingY * 2;
  let bx = x + 14;
  let by = y - boxH - 14;
  if (bx + boxW > canvasW - 4) bx = x - boxW - 14;
  if (by < 4) by = y + 14;
  bx = Math.max(4, Math.min(bx, canvasW - boxW - 4));

  const r = 6;
  ctx.beginPath();
  ctx.moveTo(bx + r, by);
  ctx.arcTo(bx + boxW, by, bx + boxW, by + boxH, r);
  ctx.arcTo(bx + boxW, by + boxH, bx, by + boxH, r);
  ctx.arcTo(bx, by + boxH, bx, by, r);
  ctx.arcTo(bx, by, bx + boxW, by, r);
  ctx.closePath();
  ctx.fillStyle = "rgba(10,10,16,0.92)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = "#f4f4f0";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  lines.forEach((l, i) => {
    ctx.font = i === 0 ? "bold 11px ui-monospace, monospace" : "11px ui-monospace, monospace";
    ctx.fillText(l, bx + paddingX, by + paddingY + i * lineH);
  });
  ctx.restore();
}

/** Traces a shipper (circle) or carrier (diamond) outline into `ctx`'s
 * current path, centered at (x,y) with "radius" `size` -- so role reads
 * from silhouette alone, not just which hemisphere/color a node has
 * (color already repeats across roles: `shipper-r-1` and `carrier-r-1`
 * are the same hue). Radial gradients fill either path identically, so
 * this drops into the existing glow-layer loop with no other changes. */
function traceNodeShape(ctx: CanvasRenderingContext2D, role: "shipper" | "carrier", x: number, y: number, size: number) {
  ctx.beginPath();
  if (role === "carrier") {
    ctx.moveTo(x, y - size);
    ctx.lineTo(x + size, y);
    ctx.lineTo(x, y + size);
    ctx.lineTo(x - size, y);
    ctx.closePath();
  } else {
    ctx.arc(x, y, size, 0, Math.PI * 2);
  }
}

function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const abx = bx - ax;
  const aby = by - ay;
  const len2 = abx * abx + aby * aby;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * abx + (py - ay) * aby) / len2));
  const cx = ax + t * abx;
  const cy = ay + t * aby;
  return Math.hypot(px - cx, py - cy);
}

export function NegotiationGraph({
  shipperIds,
  carrierIds,
  tables,
  matching,
  mode,
}: {
  shipperIds: string[];
  carrierIds: string[];
  tables: Record<string, TableResult>;
  matching?: MatchingResult | null;
  mode: PaletteMode;
}) {
  const c = palette[mode];
  const [activeId, setActiveId] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const activeIdRef = useRef<string | null>(null);
  const tablesRef = useRef(tables);
  const matchedCarrierOfRef = useRef<Record<string, string>>({});
  const nodesRef = useRef<NodeDatum[]>([]);
  const edgesRef = useRef<EdgeDatum[]>([]);
  const cameraRef = useRef({ angle: 0.5, tilt: 0.32, dist: DEFAULT_DIST, zoom: 1, dragging: false, dragged: false, lastX: 0, lastY: 0 });
  const projectedRef = useRef<Record<string, { sx: number; sy: number; depth: number; scale: number }>>({});
  const particlesRef = useRef<Record<string, Particle>>({});
  const pulseRingsRef = useRef<PulseRing[]>([]);
  const lastPulseSpawnRef = useRef(0);
  const hoverRef = useRef<{ id: string | null; since: number }>({ id: null, since: 0 });

  activeIdRef.current = activeId;
  tablesRef.current = tables;

  const matchedCarrierOf: Record<string, string> = {};
  for (const entry of matching?.matches ?? []) matchedCarrierOf[entry.shipper_id] = entry.carrier_id;
  matchedCarrierOfRef.current = matchedCarrierOf;

  // Rebuild the fixed 3D layout whenever the roster changes -- shippers
  // fill the x<0 hemisphere, carriers the x>0 hemisphere, of one shared
  // sphere (see `hemispherePoint`), so the shape orbiting the camera
  // reveals is a single glowing orb, not two flat facing planes.
  useEffect(() => {
    const nodes: NodeDatum[] = [];
    const place = (ids: string[], role: "shipper" | "carrier", sign: 1 | -1) => {
      ids.forEach((id, i) => {
        nodes.push({
          id,
          role,
          pos: hemispherePoint(i, ids.length, sign),
          color: agentColor(ids, id, mode),
          dashed: isDashedArch(engineLabelForAgent(id)),
          label: id.replace(/^(shipper|carrier)-/, ""),
        });
      });
    };
    place(shipperIds, "shipper", -1);
    place(carrierIds, "carrier", 1);
    nodesRef.current = nodes;

    const edges: EdgeDatum[] = [];
    for (const s of shipperIds) for (const cid of carrierIds) edges.push({ key: `${s}::${cid}`, shipperId: s, carrierId: cid });
    edgesRef.current = edges;
    particlesRef.current = {};
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shipperIds.join(","), carrierIds.join(","), mode]);

  const activeEdgeKey = activeId !== null && activeId.includes("::") ? activeId : null;
  const activeIsShipper = activeId !== null && !activeEdgeKey && shipperIds.includes(activeId);
  const activeIsCarrier = activeId !== null && !activeEdgeKey && carrierIds.includes(activeId);

  const selectedTable = activeEdgeKey ? tables[activeEdgeKey] : null;
  const activeNodeTables = activeIsShipper
    ? carrierIds.map((cid) => ({ key: `${activeId}::${cid}`, counterpart: cid, t: tables[`${activeId}::${cid}`] })).filter((row) => row.t)
    : activeIsCarrier
      ? shipperIds.map((s) => ({ key: `${s}::${activeId}`, counterpart: s, t: tables[`${s}::${activeId}`] })).filter((row) => row.t)
      : [];

  // The render loop itself: a single requestAnimationFrame subscription
  // set up once on mount, reading everything mutable through refs so
  // selecting a node or a new turn arriving never tears the loop down.
  useEffect(() => {
    // Re-bound to fresh, definitely-non-null consts: TS's null-narrowing
    // above doesn't carry into the nested `frame`/event-handler closures
    // below if they keep referencing `canvasRef.current` directly, since
    // a ref's `.current` could theoretically change between the guard and
    // the closure running -- these fixed local bindings never do.
    // Cast (not just narrow) to non-null: a plain `if (!x) return` narrows
    // `x` for the rest of THIS function body, but that flow fact does not
    // carry into the nested `frame`/event-handler closures below, which
    // TS otherwise re-treats as "possibly null". The runtime guards right
    // after still do the actual safety check; the cast only fixes the
    // static type these closures see.
    const canvasEl = canvasRef.current as HTMLCanvasElement;
    const containerEl = containerRef.current as HTMLDivElement;
    if (!canvasEl || !containerEl) return;
    const ctx2d = canvasEl.getContext("2d") as CanvasRenderingContext2D;
    if (!ctx2d) return;

    let raf = 0;
    let last = performance.now();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    function resize() {
      const w = containerEl.clientWidth;
      const h = Math.max(320, Math.min(520, w * 0.6));
      canvasEl.width = w * dpr;
      canvasEl.height = h * dpr;
      canvasEl.style.width = `${w}px`;
      canvasEl.style.height = `${h}px`;
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(containerEl);

    function edgeTouchesActive(s: string, cid: string): boolean {
      const active = activeIdRef.current;
      if (active === null) return false;
      if (active.includes("::")) return active === `${s}::${cid}`;
      return active === s || active === cid;
    }

    // Shared by click-to-select and hover-to-preview: nearest node within
    // its hit radius wins outright (nodes sit "in front of" edges for
    // picking purposes), otherwise the nearest edge within tolerance.
    function hitTestAt(p: { x: number; y: number }): string | null {
      const proj = projectedRef.current;
      let nearestNode: string | null = null;
      let nearestNodeDist = 16;
      for (const n of nodesRef.current) {
        const pp = proj[n.id];
        if (!pp) continue;
        const d = Math.hypot(p.x - pp.sx, p.y - pp.sy);
        if (d < nearestNodeDist) {
          nearestNodeDist = d;
          nearestNode = n.id;
        }
      }
      if (nearestNode) return nearestNode;

      let nearestEdge: string | null = null;
      let nearestEdgeDist = 8;
      for (const e of edgesRef.current) {
        const a = proj[e.shipperId];
        const b = proj[e.carrierId];
        if (!a || !b) continue;
        const d = distToSegment(p.x, p.y, a.sx, a.sy, b.sx, b.sy);
        if (d < nearestEdgeDist) {
          nearestEdgeDist = d;
          nearestEdge = e.key;
        }
      }
      return nearestEdge;
    }

    function frame(now: number) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const cam = cameraRef.current;
      if (!cam.dragging) cam.angle += dt * AUTO_ROTATE_SPEED;

      const w = canvasEl.width / dpr;
      const h = canvasEl.height / dpr;
      ctx2d.save();
      ctx2d.scale(dpr, dpr);
      ctx2d.globalCompositeOperation = "source-over";
      ctx2d.fillStyle = "#07070a";
      ctx2d.fillRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const nodes = nodesRef.current;
      const edges = edgesRef.current;
      const tablesNow = tablesRef.current;
      const matchedCarrierOf = matchedCarrierOfRef.current;
      const hasSelection = activeIdRef.current !== null;
      const nowSec = now / 1000;

      const proj: Record<string, { sx: number; sy: number; depth: number; scale: number }> = {};
      for (const n of nodes) proj[n.id] = project(n.pos, cam.angle, cam.tilt, cam.dist, cx, cy, cam.zoom);
      projectedRef.current = proj;

      const fogFor = (depth: number) => Math.max(0.12, Math.min(1, 1 - (depth - cam.dist + SPHERE_R) * (0.55 / cam.dist)));

      // --- edges (drawn first, nodes render on top) ---
      let anyWinner = false;
      for (const e of edges) {
        const t = tablesNow[e.key];
        const matched = matchedCarrierOf[e.shipperId] === e.carrierId;
        const status = edgeStatus(t, matched);
        const a = proj[e.shipperId];
        const b = proj[e.carrierId];
        if (!a || !b) continue;
        const fog = fogFor((a.depth + b.depth) / 2);
        const touches = edgeTouchesActive(e.shipperId, e.carrierId);
        const dimmed = hasSelection && !touches;

        // Business convention: green = won, red = lost (whether that's a
        // clean impasse or a table that closed but Gale-Shapley didn't
        // keep) -- the in-progress "en curso" state reuses the same gold
        // the winner used to be pinned to, per an explicit ask; the
        // winner itself stays gold-halo-free but green, so it's still
        // instantly readable at a glance, not just by line thickness.
        let stroke: string = c.baseline;
        let width = 1;
        let baseAlpha = 0.35;
        let dash: number[] = [3, 5];
        if (status === "active") {
          stroke = c.winner;
          width = 1.6;
          baseAlpha = 0.85;
          dash = [];
        } else if (status === "impasse") {
          stroke = c.critical;
          width = 1.6;
          baseAlpha = 0.65;
          dash = [2, 3];
        } else if (status === "closed-unmatched") {
          stroke = c.critical;
          width = 1.4;
          baseAlpha = 0.45;
          dash = [];
        } else if (status === "matched") {
          stroke = c.good;
          width = 2.6;
          baseAlpha = 1;
          dash = [];
          anyWinner = true;
        }

        ctx2d.save();
        ctx2d.globalCompositeOperation = status === "matched" ? "lighter" : "source-over";
        ctx2d.strokeStyle = stroke;
        ctx2d.setLineDash(dash);
        ctx2d.lineWidth = (touches ? width + 1.2 : width) * (0.6 + 0.4 * (a.scale + b.scale));
        ctx2d.globalAlpha = (dimmed ? 0.06 : baseAlpha) * fog;
        ctx2d.beginPath();
        ctx2d.moveTo(a.sx, a.sy);
        ctx2d.lineTo(b.sx, b.sy);
        ctx2d.stroke();
        if (status === "matched") {
          // A second, softer wide pass underneath gives the winning edge a glow halo.
          ctx2d.lineWidth *= 3;
          ctx2d.globalAlpha *= 0.25;
          ctx2d.stroke();
        }
        ctx2d.restore();

        if (status !== "empty" && !dimmed) {
          // The particle travels FROM whoever made the last move TO the
          // other side -- literally "the offer flowing to its recipient"
          // -- so it visibly flips direction every turn instead of
          // running one fixed way regardless of who's acting.
          const lastActor: "shipper" | "carrier" = t && t.history.length > 0 ? t.history[t.history.length - 1].actor : "shipper";
          const speed = status === "matched" ? 0.55 : 0.28;
          const key = e.key;
          let p = particlesRef.current[key];
          if (!p) p = particlesRef.current[key] = { edgeKey: key, t: Math.random(), speed, actor: lastActor };
          if (p.actor !== lastActor) {
            p.actor = lastActor;
            p.t = 0; // relaunch cleanly from the new offerer's side
          }
          p.speed = speed;
          p.t = (p.t + dt * p.speed) % 1;
          const [src, dst] = lastActor === "carrier" ? [b, a] : [a, b];
          const px = src.sx + (dst.sx - src.sx) * p.t;
          const py = src.sy + (dst.sy - src.sy) * p.t;
          const pr = (status === "matched" ? 3.2 : 2) * (0.6 + 0.4 * ((a.scale + b.scale) / 2));
          ctx2d.save();
          ctx2d.globalCompositeOperation = "lighter";
          ctx2d.globalAlpha = fog;
          const grad = ctx2d.createRadialGradient(px, py, 0, px, py, pr * 3);
          grad.addColorStop(0, stroke);
          grad.addColorStop(1, "rgba(0,0,0,0)");
          ctx2d.fillStyle = grad;
          ctx2d.beginPath();
          ctx2d.arc(px, py, pr * 3, 0, Math.PI * 2);
          ctx2d.fill();
          ctx2d.restore();
        }
      }

      // --- winner endpoint sonar pulses (persistent, not just on hover) ---
      if (anyWinner && now - lastPulseSpawnRef.current > 900) {
        lastPulseSpawnRef.current = now;
        for (const m of Object.entries(matchedCarrierOf)) {
          pulseRingsRef.current.push({ nodeId: m[0], start: now, color: c.good });
          pulseRingsRef.current.push({ nodeId: m[1], start: now, color: c.good });
        }
        if (activeIdRef.current && !activeIdRef.current.includes("::")) {
          pulseRingsRef.current.push({ nodeId: activeIdRef.current, start: now, color: c.series1 });
        }
      }
      pulseRingsRef.current = pulseRingsRef.current.filter((r) => now - r.start < 1100);

      // --- nodes, back-to-front (painter's algorithm) ---
      const order = [...nodes].sort((n1, n2) => proj[n2.id].depth - proj[n1.id].depth);
      for (const n of order) {
        const p = proj[n.id];
        const fog = fogFor(p.depth);
        const isActive = activeIdRef.current === n.id || (activeEdgeKeyTouches(activeIdRef.current, n.id));
        const dimmed = hasSelection && !isActive;
        const breathing = isActive ? 1 + 0.15 * Math.sin(nowSec * 3) : 1;
        const r = NODE_R * (0.55 + 0.45 * p.scale) * breathing;

        ctx2d.save();
        ctx2d.globalCompositeOperation = "lighter";
        ctx2d.globalAlpha = (dimmed ? 0.18 : 1) * fog;
        for (const [mult, alpha] of [[2.8, 0.15], [1.7, 0.4], [1, 0.95]] as [number, number][]) {
          const grad = ctx2d.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, r * mult);
          grad.addColorStop(0, n.color);
          grad.addColorStop(1, "rgba(0,0,0,0)");
          ctx2d.globalAlpha = (dimmed ? 0.18 : 1) * fog * alpha;
          ctx2d.fillStyle = grad;
          traceNodeShape(ctx2d, n.role, p.sx, p.sy, r * mult);
          ctx2d.fill();
        }
        ctx2d.restore();

        // A radial gradient's own circular falloff reads as round no
        // matter what path it's clipped to -- it fades out well before
        // reaching a diamond's corners, so the glow alone can't carry the
        // shape (confirmed by eye: shipper and carrier glows looked
        // identical without this). A small crisp, SOLID-fill core with an
        // outline -- no gradient -- is what actually shows the diamond.
        ctx2d.save();
        ctx2d.globalCompositeOperation = "source-over";
        ctx2d.globalAlpha = dimmed ? 0.25 : fog;
        ctx2d.fillStyle = n.color;
        traceNodeShape(ctx2d, n.role, p.sx, p.sy, r * 0.6);
        ctx2d.fill();
        ctx2d.strokeStyle = "rgba(255,255,255,0.85)";
        ctx2d.lineWidth = 1.1;
        ctx2d.stroke();
        ctx2d.restore();

        if (n.dashed) {
          ctx2d.save();
          ctx2d.globalAlpha = (dimmed ? 0.18 : 0.9) * fog;
          ctx2d.strokeStyle = n.color;
          ctx2d.lineWidth = 1.2;
          ctx2d.setLineDash([2, 2]);
          traceNodeShape(ctx2d, n.role, p.sx, p.sy, r + 3);
          ctx2d.stroke();
          ctx2d.restore();
        }

        ctx2d.save();
        ctx2d.globalAlpha = dimmed ? 0.25 : 1;
        ctx2d.fillStyle = "#f4f4f0";
        ctx2d.font = "10px ui-monospace, monospace";
        ctx2d.textBaseline = "middle";
        ctx2d.textAlign = n.role === "shipper" ? "right" : "left";
        ctx2d.fillText(n.label, p.sx + (n.role === "shipper" ? -r - 6 : r + 6), p.sy);
        ctx2d.restore();
      }

      // --- pulse rings on top of everything ---
      for (const ring of pulseRingsRef.current) {
        const p = proj[ring.nodeId];
        if (!p) continue;
        const progress = (now - ring.start) / 1100;
        const rr = NODE_R * (0.55 + 0.45 * p.scale) * 1.1 + progress * 34;
        ctx2d.save();
        ctx2d.globalCompositeOperation = "lighter";
        ctx2d.globalAlpha = Math.max(0, 0.5 * (1 - progress)) * fogFor(p.depth);
        ctx2d.strokeStyle = ring.color;
        ctx2d.lineWidth = 1.5;
        ctx2d.beginPath();
        ctx2d.arc(p.sx, p.sy, rr, 0, Math.PI * 2);
        ctx2d.stroke();
        ctx2d.restore();
      }

      // --- hover card, drawn last so it floats above everything else ---
      const hover = hoverRef.current;
      if (hover.id) {
        const fade = Math.min(1, (now - hover.since) / HOVER_FADE_MS);
        const lines = hoverInfoLines(hover.id, tablesNow, matchedCarrierOf, nodes, edges);
        let anchorX = w / 2;
        let anchorY = h / 2;
        if (hover.id.includes("::")) {
          const [s, cid] = hover.id.split("::");
          const a = proj[s];
          const b = proj[cid];
          if (a && b) {
            anchorX = (a.sx + b.sx) / 2;
            anchorY = (a.sy + b.sy) / 2;
          }
        } else {
          const p = proj[hover.id];
          if (p) {
            anchorX = p.sx;
            anchorY = p.sy;
          }
        }
        drawHoverCard(ctx2d, lines, anchorX, anchorY, fade, w, h);
      }

      ctx2d.restore();
      raf = requestAnimationFrame(frame);
    }

    function activeEdgeKeyTouches(active: string | null, nodeId: string): boolean {
      if (!active || !active.includes("::")) return false;
      const [s, cid] = active.split("::");
      return s === nodeId || cid === nodeId;
    }

    raf = requestAnimationFrame(frame);

    function toLocal(ev: PointerEvent) {
      const rect = canvasEl.getBoundingClientRect();
      return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
    }

    function onPointerDown(ev: PointerEvent) {
      const p = toLocal(ev);
      const cam = cameraRef.current;
      cam.dragging = true;
      cam.dragged = false;
      cam.lastX = p.x;
      cam.lastY = p.y;
      canvasEl.setPointerCapture(ev.pointerId);
    }
    function setHover(id: string | null) {
      if (hoverRef.current.id === id) return;
      hoverRef.current = { id, since: performance.now() };
    }

    function onPointerMove(ev: PointerEvent) {
      const cam = cameraRef.current;
      const p = toLocal(ev);
      if (!cam.dragging) {
        setHover(hitTestAt(p));
        return;
      }
      setHover(null);
      const dx = p.x - cam.lastX;
      const dy = p.y - cam.lastY;
      if (Math.abs(dx) + Math.abs(dy) > DRAG_CLICK_THRESHOLD) cam.dragged = true;
      cam.angle += dx * 0.006;
      cam.tilt = Math.max(-1.3, Math.min(1.3, cam.tilt + dy * 0.006));
      cam.lastX = p.x;
      cam.lastY = p.y;
    }
    function onPointerLeave() {
      setHover(null);
    }
    function onPointerDownForHover() {
      setHover(null);
    }
    function onPointerUp(ev: PointerEvent) {
      const cam = cameraRef.current;
      const wasDragged = cam.dragged;
      cam.dragging = false;
      cam.dragged = false;
      canvasEl.releasePointerCapture(ev.pointerId);
      if (wasDragged) return;

      const hit = hitTestAt(toLocal(ev));
      if (hit) setActiveId((prev) => (prev === hit ? null : hit));
    }
    function onWheel(ev: WheelEvent) {
      ev.preventDefault();
      const cam = cameraRef.current;
      cam.dist = Math.max(MIN_DIST, Math.min(MAX_DIST, cam.dist + ev.deltaY * 0.4));
    }

    canvasEl.addEventListener("pointerdown", onPointerDown);
    canvasEl.addEventListener("pointerdown", onPointerDownForHover);
    canvasEl.addEventListener("pointermove", onPointerMove);
    canvasEl.addEventListener("pointerup", onPointerUp);
    canvasEl.addEventListener("pointerleave", onPointerLeave);
    canvasEl.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvasEl.removeEventListener("pointerdown", onPointerDown);
      canvasEl.removeEventListener("pointerdown", onPointerDownForHover);
      canvasEl.removeEventListener("pointermove", onPointerMove);
      canvasEl.removeEventListener("pointerup", onPointerUp);
      canvasEl.removeEventListener("pointerleave", onPointerLeave);
      canvasEl.removeEventListener("wheel", onWheel);
    };
  }, [c, mode]);

  return (
    <div className="space-y-md">
      <div ref={containerRef} className="border border-line-light dark:border-line-dark rounded-lg overflow-hidden">
        <canvas ref={canvasRef} className="block w-full cursor-grab active:cursor-grabbing touch-none" />
      </div>

      <div className="flex flex-wrap items-center gap-lg font-mono text-xs text-ink-muted">
        <div className="flex flex-wrap items-center gap-md">
          <Legend swatch={c.baseline} dash label="no offers" />
          <Legend swatch={c.winner} label="en curso" />
          <Legend swatch={c.critical} dash label="impasse" />
          <Legend swatch={c.critical} faint label="closed, not matched" />
          <Legend swatch={c.good} bold label="🏆 matched (final pairing)" />
        </div>
        <span>drag to orbit · scroll to zoom · hover for a quick look · click to pin the details</span>
      </div>

      {(activeIsShipper || activeIsCarrier) && activeNodeTables.length > 0 && (
        <div className="border border-line-light dark:border-line-dark rounded-lg p-md">
          <h4 className="font-mono text-xs text-ink-muted uppercase tracking-wide mb-sm">
            {activeId} — {activeNodeTables.length} table{activeNodeTables.length === 1 ? "" : "s"}
          </h4>
          <div className="space-y-xs">
            {activeNodeTables.map(({ key, counterpart, t }) => {
              const isWinner =
                (activeIsShipper && matchedCarrierOf[activeId as string] === counterpart) ||
                (activeIsCarrier && matchedCarrierOf[counterpart] === activeId);
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setActiveId(key)}
                  className="w-full flex items-center justify-between gap-sm font-mono text-xs px-sm py-xs rounded hover:bg-line-light/30 dark:hover:bg-line-dark/30 text-left"
                >
                  <span>
                    {isWinner && "🏆 "}
                    ↔ {counterpart} <ArchBadge className="ml-xs" archLabel={engineLabelForAgent(counterpart)} />
                  </span>
                  {t.closed ? (
                    <span className={isWinner ? "text-status-good font-semibold" : "text-status-good"}>{money(t.rate)}</span>
                  ) : t.impasse ? (
                    <span className="text-status-critical">impasse</span>
                  ) : (
                    <span className="text-ink-muted">
                      {t.history.length} turn{t.history.length === 1 ? "" : "s"}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {selectedTable && <TableCard table={selectedTable} />}
    </div>
  );
}

function Legend({ swatch, label, dash, faint, bold }: { swatch: string; label: string; dash?: boolean; faint?: boolean; bold?: boolean }) {
  return (
    <span className={"flex items-center gap-xs" + (bold ? " font-semibold" : "")}>
      <svg width="16" height="8" aria-hidden="true">
        <line x1="0" y1="4" x2="16" y2="4" stroke={swatch} strokeWidth={bold ? 3 : 2} strokeDasharray={dash ? "2 3" : undefined} opacity={faint ? 0.4 : 1} />
      </svg>
      {label}
    </span>
  );
}
