/****************************************************************************
 *  DAG: Root -> 3 -> 9 -> 6, deterministic flows summing to 100,
 *  3 terminal states are high-reward (very good): T1=28, T2=27, T3=25
 *  3 terminal states are low-reward (very bad):  T4=12, T5=5,  T6=3
 ****************************************************************************/
"use strict";

/*************************************************************************
 * CONFIG + GLOBALS
 *************************************************************************/
const BIG_FLOW_CONFIG = {
  rootFlow: 100,            // total flow from root
  svgWidth: 800,
  svgHeight: 900,
  backgroundColor: "#1E1E1E",

  edgeColor: "#ccc",
  edgeStrokeWidth: 2,
  arrowMarkerSize: 6,

  nodeRadius: 16,
  highlightRadius: 20,
  partialColor: "#66ccff",
  finalColorHigh: "#ff7766", // top 3
  finalColorLow:  "#ffaa66", // bottom 3
  nodeStrokeColor: "#333",

  flowLabelFontSize: 14,
  flowLabelColor: "#fff",

  partBuildDuration: 600,
  verbose: false
};

// DAG data
let g_nodes = [];
let g_edges = [];

// D3 references
let g_svg = null;
let g_mainGroup = null;
let g_particleLayer = null;

// For the new particle system
let moleculeParticles = []; 
let moleculeParticleID = 0;
let lastTime2 = performance.now();

// Edge objects for the new system
let edgeObjs = []; 
let nodeMap = {}; // nodeMap[id] -> node

/*************************************************************************
 * HELPER
 *************************************************************************/
function logv(msg) {
  if(BIG_FLOW_CONFIG.verbose) {
    console.log("[DAG]", msg);
  }
}
function translate(x,y) {
  return `translate(${x},${y})`;
}

/*************************************************************************
 * Build the DAG: Root->3->9->6
 *************************************************************************/
function buildBigDAG(){
  g_nodes = [];
  g_edges = [];

  // Root (layer 0)
  g_nodes.push({
    id: "root", layer: 0, x: 50, y: 400,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: []
  });

  // Layer1 (3 partial)
  g_nodes.push({ id: "S1", layer: 1, x: 200, y: 250,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });
  g_nodes.push({ id: "S2", layer: 1, x: 200, y: 400,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });
  g_nodes.push({ id: "S3", layer: 1, x: 200, y: 550,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });

  // Layer2 (9 partial)
  g_nodes.push({ id: "P1", layer: 2, x: 350, y: 150,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });
  g_nodes.push({ id: "P2", layer: 2, x: 350, y: 230,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });
  g_nodes.push({ id: "P3", layer: 2, x: 350, y: 310,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });

  g_nodes.push({ id: "P4", layer: 2, x: 350, y: 390,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });
  g_nodes.push({ id: "P5", layer: 2, x: 350, y: 470,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });
  g_nodes.push({ id: "P6", layer: 2, x: 350, y: 550,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });

  g_nodes.push({ id: "P7", layer: 2, x: 350, y: 630,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });
  g_nodes.push({ id: "P8", layer: 2, x: 350, y: 710,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });
  g_nodes.push({ id: "P9", layer: 2, x: 350, y: 790,
    terminal: false, reward: 0, flowIn: 0, flowOut: 0, shapeParts: [] });

  // Layer3 (Terminals)
  g_nodes.push({
    id: "T1", layer: 3, x: 550, y: 200,
    terminal: true, reward: 28, flowIn: 0, flowOut: 0, shapeParts: []
  });
  g_nodes.push({
    id: "T2", layer: 3, x: 550, y: 300,
    terminal: true, reward: 27, flowIn: 0, flowOut: 0, shapeParts: []
  });
  g_nodes.push({
    id: "T3", layer: 3, x: 550, y: 400,
    terminal: true, reward: 25, flowIn: 0, flowOut: 0, shapeParts: []
  });
  g_nodes.push({
    id: "T4", layer: 3, x: 550, y: 500,
    terminal: true, reward: 12, flowIn: 0, flowOut: 0, shapeParts: []
  });
  g_nodes.push({
    id: "T5", layer: 3, x: 550, y: 600,
    terminal: true, reward: 5, flowIn: 0, flowOut: 0, shapeParts: []
  });
  g_nodes.push({
    id: "T6", layer: 3, x: 550, y: 700,
    terminal: true, reward: 3, flowIn: 0, flowOut: 0, shapeParts: []
  });

  // Edges
  function addEdge(src, tgt) {
    g_edges.push({
      source: src,
      target: tgt,
      flow: 0,
      buildPart: null,
      length: 0,
      x1: 0, y1: 0,
      x2: 0, y2: 0
    });
  }

  addEdge("root", "S1");
  addEdge("root", "S2");
  addEdge("root", "S3");

  addEdge("S1", "P1");
  addEdge("S1", "P2");
  addEdge("S1", "P3");

  addEdge("S2", "P4");
  addEdge("S2", "P5");
  addEdge("S2", "P6");

  addEdge("S3", "P7");
  addEdge("S3", "P8");
  addEdge("S3", "P9");

  addEdge("P1", "T1");
  addEdge("P2", "T2");
  addEdge("P3", "T2");
  addEdge("P4", "T3");
  addEdge("P5", "T4");
  addEdge("P6", "T4");
  addEdge("P7", "T5");
  addEdge("P8", "T6");
  addEdge("P9", "T6");

  // Assign shape type to each edge (used if you animate shapes on partial states)
  const edgeShapeMap = {
    "root->S1":"circle","root->S2":"diamond","root->S3":"hex",
    "S1->P1":"circle","S1->P2":"diamond","S1->P3":"hex",
    "S2->P4":"circle","S2->P5":"diamond","S2->P6":"hex",
    "S3->P7":"circle","S3->P8":"diamond","S3->P9":"hex",
    "P1->T1":"hex","P2->T2":"circle","P3->T2":"diamond",
    "P4->T3":"hex","P5->T4":"circle","P6->T4":"diamond",
    "P7->T5":"circle","P8->T6":"diamond","P9->T6":"hex"
  };
  g_edges.forEach(e => {
    const key = e.source + "->" + e.target;
    e.buildPart = { shapeType: edgeShapeMap[key] || "circle" };
  });

  logv("DAG built => root->3->9->6, final states = T1=28, T2=27, T3=25, T4=12, T5=5, T6=3");
}

/*************************************************************************
 * DETERMINE FLOWS (no randomness)
 * We hard-code flows so the same DAG flows appear on every load.
 *************************************************************************/
function assignFlowsDeterministic(){

  // 3 "very good" terminals: T1=28, T2=27, T3=25
  // 3 "very bad": T4=12, T5=5, T6=3
  // Sum of final flows = 100

  // We'll assign flows top-down:
  //  root -> S1=55
  //  root -> S2=37
  //  root -> S3=8

  //  S1 -> P1=28, S1 -> P2=14, S1 -> P3=13  => sum=55
  //  S2 -> P4=25, S2 -> P5=6,  S2 -> P6=6   => sum=37
  //  S3 -> P7=5,  S3 -> P8=1,  S3 -> P9=2   => sum=8

  //  P1 -> T1=28
  //  P2 -> T2=14
  //  P3 -> T2=13  => T2= 27 total
  //  P4 -> T3=25
  //  P5 -> T4=6
  //  P6 -> T4=6   => T4=12 total
  //  P7 -> T5=5
  //  P8 -> T6=1
  //  P9 -> T6=2   => T6=3 total

  const flowAssignments = {
    "root->S1": 55,
    "root->S2": 37,
    "root->S3": 8,

    "S1->P1": 28,
    "S1->P2": 14,
    "S1->P3": 13,

    "S2->P4": 25,
    "S2->P5": 6,
    "S2->P6": 6,

    "S3->P7": 5,
    "S3->P8": 1,
    "S3->P9": 2,

    "P1->T1": 28,
    "P2->T2": 14,
    "P3->T2": 13,
    "P4->T3": 25,
    "P5->T4": 6,
    "P6->T4": 6,
    "P7->T5": 5,
    "P8->T6": 1,
    "P9->T6": 2
  };

  // Apply flows to each edge in g_edges:
  g_edges.forEach(e => {
    const key = e.source + "->" + e.target;
    if(flowAssignments[key] != null) {
      e.flow = flowAssignments[key];
    } else {
      e.flow = 0; // default fallback
    }
  });

  // Now set node inflows and outflows:
  g_nodes.forEach(n => {
    const inbound = g_edges.filter(ed => ed.target === n.id);
    n.flowIn = inbound.reduce((acc, ed) => acc + ed.flow, 0);

    const outbound = g_edges.filter(ed => ed.source === n.id);
    n.flowOut = outbound.reduce((acc, ed) => acc + ed.flow, 0);
  });

  // For the root, we enforce outflow= 100, inflow=0 or 100 (by convention).
  const rootNode = g_nodes.find(n => n.id === "root");
  if(rootNode) {
    rootNode.flowIn = 100; 
    rootNode.flowOut = 100;
  }

  logv("Flows assigned deterministically => sum of terminals=100");
}

/*************************************************************************
 * STYLING & GEOMETRY
 *************************************************************************/
function styleNodesAndEdges(){
  g_nodes.forEach(n=>{
    if(n.terminal){
      // "very good" if reward >= 20
      if(n.reward >= 20){
        n.radius = BIG_FLOW_CONFIG.highlightRadius;
        n.color = BIG_FLOW_CONFIG.finalColorHigh;
        n.strokeWidth = 3;
      } else {
        n.radius = BIG_FLOW_CONFIG.nodeRadius + 2;
        n.color = BIG_FLOW_CONFIG.finalColorLow;
        n.strokeWidth = 2;
      }
    } else {
      n.radius = BIG_FLOW_CONFIG.nodeRadius;
      n.color = BIG_FLOW_CONFIG.partialColor;
      n.strokeWidth = 2;
    }
  });

  // compute geometry
  g_edges.forEach(e=>{
    const src = g_nodes.find(x => x.id === e.source);
    const tgt = g_nodes.find(x => x.id === e.target);
    const dx  = tgt.x - src.x;
    const dy  = tgt.y - src.y;
    const dist= Math.hypot(dx, dy);

    if(dist < 1e-9){
      e.x1 = src.x; e.y1 = src.y;
      e.x2 = tgt.x; e.y2 = tgt.y;
      e.length = 1;
      e.angle = 0;
    } else {
      const ux = dx/dist, uy = dy/dist;
      e.x1 = src.x + ux * src.radius;
      e.y1 = src.y + uy * src.radius;
      e.x2 = tgt.x - ux * tgt.radius;
      e.y2 = tgt.y - uy * tgt.radius;
      e.length = Math.hypot(e.x2 - e.x1, e.y2 - e.y1);
      e.angle  = Math.atan2(e.y2 - e.y1, e.x2 - e.x1);
    }
  });
}

/*************************************************************************
 * RENDER DAG
 *************************************************************************/
function renderDAG(){
  g_mainGroup.selectAll("*").remove();

  // arrow definition
  const defs = g_mainGroup.append("defs");
  defs.append("marker")
    .attr("id","arrowBig")
    .attr("viewBox","0 0 10 10")
    .attr("refX", 10)
    .attr("refY", 5)
    .attr("markerWidth", BIG_FLOW_CONFIG.arrowMarkerSize)
    .attr("markerHeight",BIG_FLOW_CONFIG.arrowMarkerSize)
    .attr("orient","auto")
    .append("path")
    .attr("d","M0,0 L0,10 L10,5 z")
    .attr("fill", BIG_FLOW_CONFIG.edgeColor);

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
    .attr("marker-end","url(#arrowBig)");

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
    .text(d => "flow:" + Math.round(d.flow));

  // node groups
  const nodeSel = g_mainGroup.selectAll(".dagNode")
    .data(g_nodes).enter()
    .append("g")
    .attr("class","dagNode")
    .attr("transform", d => translate(d.x, d.y));

  // each node with an image (if you have images named molecules/<id>.png)
  nodeSel.append("image")
    .attr("xlink:href", d => `molecules/${d.id}.png`)
    .attr("width",  d => d.radius * 3.6)
    .attr("height", d => d.radius * 3.6)
    .attr("x", d => -d.radius * 1.8)
    .attr("y", d => -d.radius * 1.8);

  // add % label for terminal flows
  const totalTerminalFlow = g_nodes
    .filter(n => n.terminal)
    .reduce((sum, n) => sum + n.flowIn, 0);

  nodeSel.filter(d => d.terminal).append("text")
    .attr("class", "terminal-flow-percent")
    .attr("font-size", BIG_FLOW_CONFIG.flowLabelFontSize + 4)
    .attr("fill", "#ffffff")
    .attr("font-weight", "bold")
    .attr("text-anchor", "start")
    .attr("alignment-baseline", "middle")
    .attr("x", d => d.radius + 20)
    .attr("y", 0)
    .text(d => `${Math.round((d.flowIn / totalTerminalFlow) * 100)}%`);

  // shapeParts container (used if you animate partial shapes on each node)
  nodeSel.each(function(d){
    d3.select(this).append("g").attr("class","shapePartsGroup");
  });
}

/*************************************************************************
 * OPTIONAL: PART BUILD ANIMATION
 *************************************************************************/
function animateNodePartBuild(node, shapePart){
  // Example logic if you wanted to incrementally add shapes.
  node.shapeParts.push(shapePart);
  const nodeSel = g_mainGroup.selectAll(".dagNode").filter(d => d.id === node.id);
  let spG = nodeSel.select(".shapePartsGroup");
  if(spG.empty()){
    spG = nodeSel.append("g").attr("class","shapePartsGroup");
  }
  // ...
}

/*************************************************************************
 * PARTICLE SYSTEM
 *************************************************************************/
function prepareParticleData(){
  nodeMap = {};
  g_nodes.forEach(n => { nodeMap[n.id] = n; });

  edgeObjs = g_edges.map(e => {
    return {
      sourceId: e.source,
      targetId: e.target,
      flow: e.flow,
      x1: e.x1,
      y1: e.y1,
      x2: e.x2,
      y2: e.y2,
      length: e.length
    };
  });
}

let flowParticles = [];
let particleIdCounter = 0;
const spawnMultiplier = 0.009;

function spawnMoleculeParticles(dtMs) {
  const dtSec = dtMs / 100; 
  edgeObjs.forEach(e => {
    // higher exponent => more particles for higher flow
    const spawnRate = e.flow * spawnMultiplier;
    const expectedCount = spawnRate * dtSec;
    const countToSpawn = Math.floor(expectedCount + Math.random());

    for (let i = 0; i < countToSpawn; i++) {
      createMoleculeParticle(e);
    }
  });
}

function createMoleculeParticle(edgeObj) {
  const speed = 8 + edgeObj.flow * 2 + Math.random() * 4;
  const color = (nodeMap[edgeObj.targetId] && nodeMap[edgeObj.targetId].color) || "#fff";
  const id = particleIdCounter++;
  flowParticles.push({
    id: id,
    edge: edgeObj,
    t: 0,
    speed: speed,
    color: color,
    radius: 3
  });
}

function updateMoleculeParticles(dtSec) {
  for (let i = flowParticles.length - 1; i >= 0; i--) {
    let p = flowParticles[i];
    const fraction = (p.speed * dtSec) / p.edge.length;
    p.t += fraction;
    if (p.t >= 1) {
      // We could animate shape building on the target node, e.g.:
      // let tgtNode = g_nodes.find(n => n.id === p.edge.targetId);
      // animateNodePartBuild(tgtNode, { shapeType: "circle" });
      flowParticles.splice(i, 1);
    }
  }
}

function renderMoleculeParticles() {
  const sel = g_particleLayer.selectAll(".flow-particle2")
    .data(flowParticles, d => d.id);

  const enterSel = sel.enter().append("circle")
    .attr("class", "flow-particle2")
    .attr("r", d => d.radius)
    .attr("fill", d => d.color)
    .attr("opacity", 0.85);

  sel.merge(enterSel)
    .attr("cx", d => d.edge.x1 + (d.edge.x2 - d.edge.x1)*d.t)
    .attr("cy", d => d.edge.y1 + (d.edge.y2 - d.edge.y1)*d.t);

  sel.exit().remove();
}

function animateMoleculeFlow(timestamp) {
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
  assignFlowsDeterministic();  // deterministic flows
  styleNodesAndEdges();

  g_svg = d3.select("#moleculeFlowSVG")
    .attr("width", BIG_FLOW_CONFIG.svgWidth)
    .attr("height", BIG_FLOW_CONFIG.svgHeight)
    .style("background-color", BIG_FLOW_CONFIG.backgroundColor);

  // Clear old content
  g_svg.selectAll("*").remove();

  g_mainGroup = g_svg.append("g").attr("class","bigMainGroup");
  renderDAG(); 
  g_particleLayer = g_mainGroup.append("g").attr("class","bigParticleLayer");
  
  centerGraph();

  // prepare for the particle system
  prepareParticleData();

  // Start the new animation
  requestAnimationFrame(animateMoleculeFlow);
  logv("initMoleculeFlowDemo => #nodes=" + g_nodes.length + " #edges=" + g_edges.length);
}

/** Center the DAG in the SVG view. */
function centerGraph(){
  let minX= Infinity, maxX=-Infinity, minY= Infinity, maxY=-Infinity;
  g_nodes.forEach(n=>{
    if(n.x < minX) minX=n.x;
    if(n.x > maxX) maxX=n.x;
    if(n.y < minY) minY=n.y;
    if(n.y > maxY) maxY=n.y;
  });
  const w = maxX - minX;
  const h = maxY - minY;
  const offX = (BIG_FLOW_CONFIG.svgWidth - w)/2 - minX;
  const offY = (BIG_FLOW_CONFIG.svgHeight - h)/2 - minY;
  g_mainGroup.attr("transform", translate(offX, offY));
}

/** If you ever want to reset or rebuild the DAG. */
function reGenerateBigDAG(){
  moleculeParticles = [];
  particleIdCounter = 0;
  buildBigDAG();
  assignFlowsDeterministic();
  styleNodesAndEdges();
  renderDAG();
  centerGraph();
  prepareParticleData();
  logv("Re-generated DAG => partial≥1, sum of terminals=100. Particle system reset.");
}

// By default, we call the init function at the bottom.
initMoleculeFlowDemo();
