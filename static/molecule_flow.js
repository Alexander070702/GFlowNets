/****************************************************************************
 *  DAG: Root → 3 Starters (S1–S3) → 9 Terminals (P1–P9),
 *  deterministic flows summing to 100, but particles split probabilistically
 *  at S-nodes according to the relative flows on outgoing edges.
 ****************************************************************************/
"use strict";

/*************************************************************************
 * CONFIG + GLOBALS
 *************************************************************************/
const BIG_FLOW_CONFIG = {
  rootFlow:          100,
  svgWidth:          800,
  svgHeight:         700,
  backgroundColor:   "#39568C",

  edgeColor:         "#ccc",
  edgeStrokeWidth:   2,
  arrowMarkerSize:   6,

  nodeRadius:        16,
  highlightRadius:   20,
  partialColor:      "#66ccff",
  finalColorHigh:    "#ff7766",
  finalColorLow:     "#ffaa66",
  nodeStrokeColor:   "#333",

  flowLabelFontSize: 14,
  flowLabelColor:    "#fff",

  imageMultiplier:   4.5,    // scale factor for node PNGs
  verbose:           false
};

let g_nodes         = [];
let g_edges         = [];
let g_svg           = null;
let g_mainGroup     = null;
let g_particleLayer = null;

// reuse your original “molecule” particle globals
let moleculeParticles  = [];
let moleculeParticleID = 0;
let lastTime2          = performance.now();

let edgeObjs = [];
let nodeMap  = {};

/*************************************************************************
 * HELPERS
 *************************************************************************/
function logv(msg) {
  if (BIG_FLOW_CONFIG.verbose) console.log("[DAG]", msg);
}
function translate(x, y) {
  return `translate(${x},${y})`;
}

/*************************************************************************
 * BUILD: root → S1–S3 → P1–P9
 *************************************************************************/
function buildBigDAG(){
  g_nodes = [];
  g_edges = [];

  // Root
  g_nodes.push({
    id:        "root",
    layer:     0,
    x:         50,
    y:         400,
    terminal:  false,
    reward:    0,
    flowIn:    0,
    flowOut:   0,
    shapeParts:[]
  });

  // S1–S3 (layer1), moved to x=300
  ["S1","S2","S3"].forEach((id,i)=>{
    g_nodes.push({
      id,
      layer:     1,
      x:         300,
      y:         250 + i*150,
      terminal:  false,
      reward:    0,
      flowIn:    0,
      flowOut:   0,
      shapeParts:[]
    });
  });

  // P1–P9 (layer2) as terminals
  for (let i = 1; i <= 9; i++){
    g_nodes.push({
      id:        `P${i}`,
      layer:     2,
      x:         550,
      y:         100 + i*75,
      terminal:  true,
      reward:    0,
      flowIn:    0,
      flowOut:   0,
      shapeParts:[]
    });
  }

  // helper for edges
  function addEdge(src, tgt){
    g_edges.push({
      source:  src,
      target:  tgt,
      flow:    0,
      buildPart:null,
      length:  0,
      x1:      0,
      y1:      0,
      x2:      0,
      y2:      0,
      angle:   0
    });
  }

  // root→S1,S2,S3
  ["S1","S2","S3"].forEach(s=> addEdge("root", s));
  // S1→P1–P3, S2→P4–P6, S3→P7–P9
  [["S1",1,3], ["S2",4,6], ["S3",7,9]].forEach(([s,a,b])=>{
    for (let i = a; i <= b; i++) addEdge(s, `P${i}`);
  });

  // original edge-shapeMap
  const edgeShapeMap = {
    "root->S1":"circle","root->S2":"diamond","root->S3":"hex",
    "S1->P1":"circle","S1->P2":"diamond","S1->P3":"hex",
    "S2->P4":"circle","S2->P5":"diamond","S2->P6":"hex",
    "S3->P7":"circle","S3->P8":"diamond","S3->P9":"hex"
  };
  g_edges.forEach(e=>{
    const k = `${e.source}->${e.target}`;
    e.buildPart = { shapeType: edgeShapeMap[k] || "circle" };
  });

  logv("DAG built → root→3→9");
}

/*************************************************************************
 * FLOWS: deterministic
 *************************************************************************/
function assignFlowsDeterministic(){
  const F = {
    "root->S1":55, "root->S2":37, "root->S3":8,
    "S1->P1":28,  "S1->P2":14,  "S1->P3":13,
    "S2->P4":25,  "S2->P5":6,   "S2->P6":6,
    "S3->P7":5,   "S3->P8":1,   "S3->P9":2
  };

  g_edges.forEach(e=>{
    const k = `${e.source}->${e.target}`;
    e.flow = F[k] || 0;
  });

  g_nodes.forEach(n=>{
    n.flowIn  = g_edges.filter(e => e.target === n.id).reduce((s,e) => s + e.flow, 0);
    n.flowOut = g_edges.filter(e => e.source === n.id).reduce((s,e) => s + e.flow, 0);
  });

  const root = g_nodes.find(n => n.id === "root");
  if (root) {
    root.flowIn  = BIG_FLOW_CONFIG.rootFlow;
    root.flowOut = BIG_FLOW_CONFIG.rootFlow;
  }

  logv("Flows assigned → sum terminals=100");
}

/*************************************************************************
 * STYLE & GEOMETRY
 *************************************************************************/
function styleNodesAndEdges(){
  g_nodes.forEach(n=>{
    if (n.terminal){
      n.radius      = BIG_FLOW_CONFIG.highlightRadius;
      n.color       = BIG_FLOW_CONFIG.finalColorHigh;
      n.strokeWidth = 3;
    } else {
      n.radius      = BIG_FLOW_CONFIG.nodeRadius;
      n.color       = BIG_FLOW_CONFIG.partialColor;
      n.strokeWidth = 2;
    }
    // half the displayed image size
    n.imageRadius = (n.radius * BIG_FLOW_CONFIG.imageMultiplier) / 2;
  });

  g_edges.forEach(e=>{
    const s    = g_nodes.find(n => n.id === e.source);
    const t    = g_nodes.find(n => n.id === e.target);
    const dx   = t.x - s.x, dy = t.y - s.y;
    const dist = Math.hypot(dx,dy) || 1;
    const ux   = dx / dist, uy = dy / dist;

    e.x1     = s.x + ux * s.imageRadius;
    e.y1     = s.y + uy * s.imageRadius;
    e.x2     = t.x - ux * t.imageRadius;
    e.y2     = t.y - uy * t.imageRadius;

    e.length = Math.hypot(e.x2 - e.x1, e.y2 - e.y1);
    e.angle  = Math.atan2(e.y2 - e.y1, e.x2 - e.x1);
  });
}

/*************************************************************************
 * RENDER DAG
 *************************************************************************/
function renderDAG(){
  g_mainGroup.selectAll("*").remove();

  // arrow head
  const defs = g_mainGroup.append("defs");
  defs.append("marker")
    .attr("id","arrowBig")
    .attr("viewBox","0 0 10 10")
    .attr("refX",10).attr("refY",5)
    .attr("markerWidth",BIG_FLOW_CONFIG.arrowMarkerSize)
    .attr("markerHeight",BIG_FLOW_CONFIG.arrowMarkerSize)
    .attr("orient","auto")
    .append("path")
      .attr("d","M0,0 L0,10 L10,5 z")
      .attr("fill",BIG_FLOW_CONFIG.edgeColor);

  // edges
  g_mainGroup.selectAll(".edge-line")
    .data(g_edges).enter()
    .append("line")
      .attr("class","edge-line")
      .attr("x1", d => d.x1)
      .attr("y1", d => d.y1)
      .attr("x2", d => d.x2)
      .attr("y2", d => d.y2)
      .attr("stroke", BIG_FLOW_CONFIG.edgeColor)
      .attr("stroke-width", BIG_FLOW_CONFIG.edgeStrokeWidth)
      .attr("marker-end", "url(#arrowBig)");

  // flow labels
  g_mainGroup.selectAll(".flow-label")
    .data(g_edges).enter()
    .append("text")
      .attr("class","flow-label")
      .attr("font-size", BIG_FLOW_CONFIG.flowLabelFontSize)
      .attr("fill", BIG_FLOW_CONFIG.flowLabelColor)
      .attr("text-anchor","middle")
      .attr("alignment-baseline","middle")
      .attr("x", d => (d.x1 + d.x2)/2 - 20 * Math.sin(d.angle))
      .attr("y", d => (d.y1 + d.y2)/2 + 20 * Math.cos(d.angle))
      .text(d => `flow:${d.flow}`);

  // nodes
  const nodeSel = g_mainGroup.selectAll(".dagNode")
    .data(g_nodes).enter()
    .append("g")
      .attr("class","dagNode")
      .attr("transform", d => translate(d.x, d.y));

  // images (bigger, centered)
  nodeSel.append("image")
    .attr("xlink:href", d => `molecules/${d.id}.png`)
    .attr("width",  d => d.imageRadius * 2)
    .attr("height", d => d.imageRadius * 2)
    .attr("x", d => -d.imageRadius)
    .attr("y", d => -d.imageRadius);

  // terminal % labels
  const totalTerm = g_nodes.filter(n => n.terminal)
                          .reduce((s,n) => s + n.flowIn, 0);
  nodeSel.filter(d => d.terminal)
    .append("text")
      .attr("class","terminal-flow-percent")
      .attr("font-size", BIG_FLOW_CONFIG.flowLabelFontSize + 4)
      .attr("fill", "#fff")
      .attr("font-weight", "bold")
      .attr("text-anchor", "start")
      .attr("alignment-baseline", "middle")
      .attr("x", d => d.imageRadius + 10)
      .attr("y", 0)
      .text(d => `${Math.round(d.flowIn / totalTerm * 100)}%`);
}

/*************************************************************************
 * CENTER WITH PADDING
 *************************************************************************/
function centerGraph(){
  const margin = (BIG_FLOW_CONFIG.highlightRadius 
                * BIG_FLOW_CONFIG.imageMultiplier) / 2;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  g_nodes.forEach(n => {
    if (n.x < minX) minX = n.x;
    if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.y > maxY) maxY = n.y;
  });

  minX -= margin; maxX += margin;
  minY -= margin; maxY += margin;

  const w2   = maxX - minX, h2 = maxY - minY;
  const offX = (BIG_FLOW_CONFIG.svgWidth  - w2) / 2 - minX;
  const offY = (BIG_FLOW_CONFIG.svgHeight - h2) / 2 - minY;
  g_mainGroup.attr("transform", translate(offX, offY));
}

/*************************************************************************
 * PARTICLE SYSTEM (adapted so that S→P edges only emit one particle
 * chosen probabilistically according to flow values)
 *************************************************************************/
function prepareParticleData(){
  nodeMap = {};
  g_nodes.forEach(n => nodeMap[n.id] = n);

  edgeObjs = g_edges.map(e => ({
    sourceId: e.source,   // e.g. "root" or "S2"
    targetId: e.target,   // e.g. "S2" or "P5"
    flow:     e.flow,
    x1:       e.x1,
    y1:       e.y1,
    x2:       e.x2,
    y2:       e.y2,
    length:   e.length
  }));
}

function createMoleculeParticle(edge){
  const speed = 30;
  const color = nodeMap[edge.targetId].color || "#fff";
  const id    = moleculeParticleID++;
  moleculeParticles.push({ id, edge, t: 0, speed, color, radius: 3 });
}

function spawnMoleculeParticles(dtMs){
  // Only spawn initial particles on edges whose sourceId === "root"
  const dtSec = dtMs / 1000;
  edgeObjs
    .filter(e => e.sourceId === "root")
    .forEach(e => {
      const rate     = e.flow * 0.05;    // particles per second
      const expected = rate * dtSec;
      const count    = Math.floor(expected + Math.random());
      for (let i = 0; i < count; i++) {
        createMoleculeParticle(e);
      }
    });
}

function updateMoleculeParticles(dtSec){
  for (let i = moleculeParticles.length - 1; i >= 0; i--){
    const p = moleculeParticles[i];
    const frac = (p.speed * dtSec) / p.edge.length;
    p.t += frac;

    if (p.t >= 1){
      // Particle reached the END of its edge:
      // If this edge’s targetId is S1, S2, or S3, spawn exactly one child
      // along one outgoing edge chosen randomly by their flow weights.
      const arrivedId = p.edge.targetId; // e.g. "S1"
      const children = edgeObjs.filter(e2 => e2.sourceId === arrivedId);

      if (children.length > 0) {
        // Compute total flow out of arrived node
        let totalFlow = 0;
        children.forEach(ch => { totalFlow += ch.flow; });

        // Choose one edge by sampling a uniform [0, totalFlow)
        let r = Math.random() * totalFlow;
        let cumulative = 0;
        for (let ch of children) {
          cumulative += ch.flow;
          if (r < cumulative) {
            createMoleculeParticle(ch);
            break;
          }
        }
      }
      // Remove the old particle in all cases
      moleculeParticles.splice(i, 1);
    }
  }
}

function renderMoleculeParticles(){
  const sel = g_particleLayer.selectAll(".flow-particle2")
    .data(moleculeParticles, d => d.id);

  const enter = sel.enter()
    .append("circle")
      .attr("class","flow-particle2")
      .attr("r", d => d.radius)
      .attr("fill", d => d.color)
      .attr("opacity", 0.85);

  sel.merge(enter)
    .attr("cx", d => d.edge.x1 + (d.edge.x2 - d.edge.x1) * d.t)
    .attr("cy", d => d.edge.y1 + (d.edge.y2 - d.edge.y1) * d.t);

  sel.exit().remove();
}

function animateMoleculeFlow(timestamp){
  const dtMs = timestamp - lastTime2;
  lastTime2 = timestamp;

  spawnMoleculeParticles(dtMs);
  updateMoleculeParticles(dtMs / 1000);
  renderMoleculeParticles();
  requestAnimationFrame(animateMoleculeFlow);
}

/*************************************************************************
 * MAIN INIT
 *************************************************************************/
function initMoleculeFlowDemo(){
  buildBigDAG();
  assignFlowsDeterministic();
  styleNodesAndEdges();

  g_svg = d3.select("#moleculeFlowSVG")
    .attr("viewBox", `0 0 ${BIG_FLOW_CONFIG.svgWidth} ${BIG_FLOW_CONFIG.svgHeight}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("width", "100%")
    .style("background-color", BIG_FLOW_CONFIG.backgroundColor);

  g_svg.selectAll("*").remove();
  g_mainGroup     = g_svg.append("g").attr("class","bigMainGroup");
  renderDAG();
  g_particleLayer = g_mainGroup.append("g").attr("class","bigParticleLayer");

  centerGraph();
  prepareParticleData();
  requestAnimationFrame(animateMoleculeFlow);

  logv(`init → nodes=${g_nodes.length}, edges=${g_edges.length}`);
}

// start it
initMoleculeFlowDemo();
