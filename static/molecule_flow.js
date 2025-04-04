/****************************************************************************
 *  DAG: Root -> 3 -> 9 -> 6, BFS flows ≥ 1, final inflow=100, 
 *  with the new flow-based particle animation traveling along edges.
 ****************************************************************************/
"use strict";

/*************************************************************************
 * CONFIG + GLOBALS
 *************************************************************************/
const BIG_FLOW_CONFIG = {
  rootFlow: 100,
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

  // For BFS and DAG
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

// For new particle system
let moleculeParticles = [];     // holds the active particles
let moleculeParticleID = 0;     // increments each time we spawn
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
function randBetween(min,max) {
  return min + (max-min)*Math.random();
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
  g_nodes.push({ id:"root", layer:0, x:50,  y:400,
    terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });

  // Layer1 (3 partial)
  g_nodes.push({ id:"S1", layer:1, x:200, y:250, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"S2", layer:1, x:200, y:400, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"S3", layer:1, x:200, y:550, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });

  // Layer2 (9 partial)
  g_nodes.push({ id:"P1", layer:2, x:350, y:150, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"P2", layer:2, x:350, y:230, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"P3", layer:2, x:350, y:310, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });

  g_nodes.push({ id:"P4", layer:2, x:350, y:390, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"P5", layer:2, x:350, y:470, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"P6", layer:2, x:350, y:550, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });

  g_nodes.push({ id:"P7", layer:2, x:350, y:630, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"P8", layer:2, x:350, y:710, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"P9", layer:2, x:350, y:790, terminal:false, reward:0, flowIn:0, flowOut:0, shapeParts:[] });

  // Layer3 (Terminals)
  g_nodes.push({ id:"T1", layer:3, x:550, y:200, terminal:true, reward:28, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"T2", layer:3, x:550, y:300, terminal:true, reward:27, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"T3", layer:3, x:550, y:400, terminal:true, reward:25, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"T4", layer:3, x:550, y:500, terminal:true, reward:12, flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"T5", layer:3, x:550, y:600, terminal:true, reward:5,  flowIn:0, flowOut:0, shapeParts:[] });
  g_nodes.push({ id:"T6", layer:3, x:550, y:700, terminal:true, reward:3,  flowIn:0, flowOut:0, shapeParts:[] });

  // Edges
  function addEdge(src,tgt) {
    g_edges.push({
      source: src,
      target: tgt,
      flow:0,
      buildPart:null,
      length: 0,
      x1:0, y1:0,
      x2:0, y2:0
    });
  }

  addEdge("root","S1");
  addEdge("root","S2");
  addEdge("root","S3");

  addEdge("S1","P1");
  addEdge("S1","P2");
  addEdge("S1","P3");

  addEdge("S2","P4");
  addEdge("S2","P5");
  addEdge("S2","P6");

  addEdge("S3","P7");
  addEdge("S3","P8");
  addEdge("S3","P9");

  addEdge("P1","T1");
  addEdge("P2","T2");
  addEdge("P3","T2");
  addEdge("P4","T3");
  addEdge("P5","T4");
  addEdge("P6","T4");
  addEdge("P7","T5");
  addEdge("P8","T6");
  addEdge("P9","T6");

  // Assign shape type to each edge
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
    const k= e.source+"->"+ e.target;
    e.buildPart= { shapeType: edgeShapeMap[k] || "circle" };
  });
  logv("DAG built => root->3->9->6, BFS≥1, final=28,27,25,12,5,3=100");
}

/*************************************************************************
 * BFS flows => partial≥1, final inflow=100
 *************************************************************************/
// scale finals robustly to exactly match 100
function scaleFinalsToExactly100() {
  const finals = g_nodes.filter(n => n.terminal);
  const sumRewards = finals.reduce((acc, n) => acc + n.reward, 0);

  finals.forEach(fn => {
    fn.flowIn = fn.reward;  // explicitly set final flows equal to rewards first
    const inEdges = g_edges.filter(e => e.target === fn.id);
    const sumInEdgesFlow = inEdges.reduce((acc, e) => acc + e.flow, 0);

    inEdges.forEach(e => {
      e.flow = (fn.reward / sumInEdgesFlow) * e.flow;
    });

    fn.flowIn = inEdges.reduce((acc, e) => acc + e.flow, 0);
  });

  // Final correction step to precisely get sum=100
  const correctedSum = finals.reduce((acc, n) => acc + n.flowIn, 0);
  const correctionFactor = BIG_FLOW_CONFIG.rootFlow / correctedSum;

  finals.forEach(fn => {
    const inEdges = g_edges.filter(e => e.target === fn.id);
    inEdges.forEach(e => { e.flow *= correctionFactor; });
    fn.flowIn *= correctionFactor;
  });
}

// Call this explicitly in assignFlowsBFS at the end
function assignFlowsBFS(){
  const root = g_nodes.find(n=> n.id==="root");
  if(!root) return;
  root.flowIn = BIG_FLOW_CONFIG.rootFlow;
  root.flowOut= BIG_FLOW_CONFIG.rootFlow;

  // existing code unchanged:
  const layers = [...new Set(g_nodes.map(n=> n.layer))].sort((a,b)=> a-b);
  for(let i=0; i< layers.length; i++){
    const layerNodes = g_nodes.filter(n=> n.layer=== layers[i]);
    layerNodes.forEach(nd => {
      let outEdges = g_edges.filter(e=> e.source=== nd.id);
      if(!outEdges.length) return;
      let totalIn= nd.flowIn;
      if(totalIn < outEdges.length) {
        totalIn= outEdges.length;
        nd.flowIn= totalIn;
        nd.flowOut= totalIn;
      }
      // random distribution
      let randArr = outEdges.map(()=> randBetween(0.01,1));
      let sumR = randArr.reduce((a,b)=> a+b,0);
      let flows = randArr.map(r=> (r/sumR)* totalIn);

      // ensure≥1
      let fixNeeded=true;
      while(fixNeeded){
        fixNeeded=false;
        let sumF= flows.reduce((a,b)=> a+b,0);
        for(let z=0; z< flows.length; z++){
          if(flows[z]<1){
            flows[z]=1;
            fixNeeded=true;
          }
        }
        if(fixNeeded){
          sumF= flows.reduce((a,b)=> a+b,0);
          if(sumF> totalIn){
            let scale= totalIn/sumF;
            flows= flows.map(x=> x* scale);
          }
        }
      }
      nd.flowOut= flows.reduce((a,b)=> a+b,0);
      outEdges.forEach((ed,ix)=> { ed.flow= flows[ix]; });
    });
    // next layer inflow
    if(i< layers.length-1){
      let nextL= layers[i+1];
      g_nodes.filter(n=> n.layer=== nextL).forEach(n2=>{
        let inbound= g_edges.filter(e=> e.target=== n2.id);
        n2.flowIn= inbound.reduce((acc,e)=> acc+ e.flow,0);
        n2.flowOut=0;
      });
    }
  }

  // Now explicitly call the robust scaling function:
  scaleFinalsToExactly100();

  // Redistribute flows after the finals correction
  reDistributeAfterFinal();
  logv("Flows assigned => partial≥1, final sum EXACTLY=100");
}


function reDistributeAfterFinal(){
  const layers = [...new Set(g_nodes.map(n=> n.layer))].sort((a,b)=> a-b);
  const root = g_nodes.find(n=> n.id==="root");
  if(!root) return;
  root.flowIn= BIG_FLOW_CONFIG.rootFlow;
  root.flowOut= BIG_FLOW_CONFIG.rootFlow;

  for(let i=0;i< layers.length;i++){
    const layerNodes= g_nodes.filter(n=> n.layer=== layers[i]);
    layerNodes.forEach(nd=>{
      let outEdges= g_edges.filter(e=> e.source=== nd.id);
      if(!outEdges.length) return;

      // Exactly match outgoing flow to incoming flow (no rounding losses)
      const totalOutEdgesFlow= outEdges.reduce((a,b)=> a+b.flow,0);
      const scalingFactor= nd.flowIn/totalOutEdgesFlow;
      outEdges.forEach(e=> e.flow*= scalingFactor);
      
      nd.flowOut= outEdges.reduce((a,b)=> a+b.flow,0);
    });

    // Set next layer inflows explicitly
    if(i< layers.length-1){
      let nextLayer= layers[i+1];
      g_nodes.filter(n=> n.layer=== nextLayer).forEach(nn=>{
        let inboundEdges= g_edges.filter(e=> e.target=== nn.id);
        nn.flowIn= inboundEdges.reduce((a,b)=> a+b.flow,0);
        nn.flowOut=0;
      });
    }
  }

  // Final rounding step at the very end:
  g_edges.forEach(e=> e.flow= Math.round(e.flow));
  g_nodes.forEach(n=> {
    const inEdges= g_edges.filter(e=> e.target=== n.id);
    n.flowIn= inEdges.reduce((a,b)=> a+b.flow,0);
    const outEdges= g_edges.filter(e=> e.source=== n.id);
    n.flowOut= outEdges.reduce((a,b)=> a+b.flow,0);
  });
}


/*************************************************************************
 * STYLING & GEOMETRY
 *************************************************************************/
function styleNodesAndEdges(){
  g_nodes.forEach(n=>{
    if(n.terminal){
      if(n.reward >= 20){
        n.radius= BIG_FLOW_CONFIG.highlightRadius;
        n.color= BIG_FLOW_CONFIG.finalColorHigh;
        n.strokeWidth=3;
      } else {
        n.radius= BIG_FLOW_CONFIG.nodeRadius+2;
        n.color= BIG_FLOW_CONFIG.finalColorLow;
        n.strokeWidth=2;
      }
    } else {
      n.radius= BIG_FLOW_CONFIG.nodeRadius;
      n.color= BIG_FLOW_CONFIG.partialColor;
      n.strokeWidth=2;
    }
  });

  // compute geometry
  g_edges.forEach(e=>{
    const src= g_nodes.find(x=> x.id=== e.source);
    const tgt= g_nodes.find(x=> x.id=== e.target);
    const dx= tgt.x- src.x;
    const dy= tgt.y- src.y;
    const dist= Math.hypot(dx,dy);
    if(dist< 1e-9){
      e.x1= src.x; e.y1= src.y;
      e.x2= tgt.x; e.y2= tgt.y;
      e.length=1;
      e.angle=0;
    } else {
      const ux= dx/dist, uy= dy/dist;
      e.x1= src.x+ ux* src.radius;
      e.y1= src.y+ uy* src.radius;
      e.x2= tgt.x- ux* tgt.radius;
      e.y2= tgt.y- uy* tgt.radius;
      e.length= Math.hypot(e.x2- e.x1, e.y2- e.y1);
      e.angle= Math.atan2(e.y2- e.y1, e.x2- e.x1);
    }
    // e.midX/e.midY if needed
  });
}

/*************************************************************************
 * RENDER DAG
 *************************************************************************/
function renderDAG(){
  g_mainGroup.selectAll("*").remove();

  // arrow
  const defs= g_mainGroup.append("defs");
  defs.append("marker")
    .attr("id","arrowBig")
    .attr("viewBox","0 0 10 10")
    .attr("refX",10)
    .attr("refY",5)
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
    .attr("x1", d=> d.x1)
    .attr("y1", d=> d.y1)
    .attr("x2", d=> d.x2)
    .attr("y2", d=> d.y2)
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
    .attr("x", d=> (d.x1+d.x2)/2 - 20*Math.sin(d.angle))
    .attr("y", d=> (d.y1+d.y2)/2 + 20*Math.cos(d.angle))
    .text(d=> "flow:" + Math.round(d.flow));

  // node groups
  const nodeSel= g_mainGroup.selectAll(".dagNode")
    .data(g_nodes).enter()
    .append("g")
    .attr("class","dagNode")
    .attr("transform", d=> translate(d.x, d.y));

  // Images 20% bigger => radius*3.6
  nodeSel.append("image")
    .attr("xlink:href", d=> `molecules/${d.id}.png`)
    .attr("width",  d=> d.radius*3.6)
    .attr("height", d=> d.radius*3.6)
    .attr("x", d=> -d.radius*1.8)
    .attr("y", d=> -d.radius*1.8);
// Calculate total terminal flow once explicitly:
const totalTerminalFlow = g_nodes
  .filter(n => n.terminal)
  .reduce((sum, n) => sum + n.flowIn, 0);

// Add percentages with correct rounding:
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

  // shapeParts container
  nodeSel.each(function(d){
    d3.select(this).append("g").attr("class","shapePartsGroup");
  });
}

/*************************************************************************
 * ANIMATE PART BUILD (Optional)
 *************************************************************************/
function animateNodePartBuild(node, shapePart){
  // If you still want incremental build, keep this
  node.shapeParts.push(shapePart);

  const nodeSel= g_mainGroup.selectAll(".dagNode").filter(d=> d.id=== node.id);
  let spG= nodeSel.select(".shapePartsGroup");
  if(spG.empty()){
    spG= nodeSel.append("g").attr("class","shapePartsGroup");
  }

  // We'll skip the shape details here if you want simpler code
  // Otherwise, keep your PARTIAL_SHAPES from your original snippet
  // ...
}

/*************************************************************************
 * NEW PARTICLE SYSTEM (based on your snippet), but for our g_edges
 *************************************************************************/

// We’ll define nodeMap and edgeObjs for convenience
// so we can do: nodeMap[e.targetId].color, etc.
function prepareParticleData(){
  // Build a quick nodeMap
  nodeMap = {};
  g_nodes.forEach(n => { nodeMap[n.id] = n; });
  
  // Build edgeObjs the snippet expects
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
const spawnMultiplier = 0.2;  // from your snippet

function spawnMoleculeParticles(dtMs) {
  const dtSec = dtMs / 1000;
  edgeObjs.forEach(e => {
    // bigger exponent => more or fewer particles?
    // snippet used (Math.pow(e.flow,2) * spawnMultiplier).
    // You can tweak exponent 1.5 or 2, as you like.
    const spawnRate = Math.pow(e.flow, 2) * spawnMultiplier;
    const expectedCount = spawnRate * dtSec;
    const countToSpawn = Math.floor(expectedCount + Math.random());
    for (let i = 0; i < countToSpawn; i++) {
      createMoleculeParticle(e);
    }
  });
}

function createMoleculeParticle(edgeObj) {
  const speed = 20 + edgeObj.flow * 4 + Math.random() * 8;
  // color from the target node
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
      // optional: if you want shape-building, call animateNodePartBuild
      // find the node in g_nodes
      // let tgtNode = g_nodes.find(n=> n.id=== p.edge.targetId);
      // if (tgtNode && ??? ) animateNodePartBuild(tgtNode, { shapeType: "circle" });
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

/** The main RAF loop for the molecule flow animation */
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
  assignFlowsBFS();
  styleNodesAndEdges();

  g_svg = d3.select("#moleculeFlowSVG")
    .attr("width", BIG_FLOW_CONFIG.svgWidth)
    .attr("height", BIG_FLOW_CONFIG.svgHeight)
    .style("background-color", BIG_FLOW_CONFIG.backgroundColor);

  // Clear old content
  g_svg.selectAll("*").remove();

  g_mainGroup = g_svg.append("g").attr("class","bigMainGroup");
  renderDAG(); // FIRST render edges & nodes
  g_particleLayer = g_mainGroup.append("g").attr("class","bigParticleLayer"); // THEN particles on top
  
  centerGraph();

  // prepare for new particle system
  prepareParticleData();

  // Start the new animation
  requestAnimationFrame(animateMoleculeFlow);
  logv("initMoleculeFlowDemo => #nodes="+ g_nodes.length+" #edges="+ g_edges.length);
}

/** Center the DAG in the SVG view. */
function centerGraph(){
  let minX= Infinity, maxX=-Infinity, minY= Infinity, maxY=-Infinity;
  g_nodes.forEach(n=>{
    if(n.x< minX) minX=n.x;
    if(n.x> maxX) maxX=n.x;
    if(n.y< minY) minY=n.y;
    if(n.y> maxY) maxY=n.y;
  });
  const w= maxX- minX;
  const h= maxY- minY;
  const offX= (BIG_FLOW_CONFIG.svgWidth- w)/2 - minX;
  const offY= (BIG_FLOW_CONFIG.svgHeight- h)/2 - minY;
  g_mainGroup.attr("transform", translate(offX, offY));
}

/** If you want to regenerate a new DAG at any time: */
function reGenerateBigDAG(){
  flowParticles = [];
  particleIdCounter = 0;
  buildBigDAG();
  assignFlowsBFS();
  styleNodesAndEdges();
  renderDAG();
  centerGraph();
  prepareParticleData();
  logv("Re-generated DAG => partial≥1, final sum=100. Particle system reset.");
}

// Finally, we do NOT call it at the bottom so that 
// you can call initMoleculeFlowDemo() from the HTML when ready:
initMoleculeFlowDemo();
