// flow_conservation.js

function initFlowConservationDemo() {
  // ─────────────────────────────────────────────────────────────────────────────
  // (A) Ensure D3 is loaded and that <svg id="flowConservationSVG"> exists
  // ─────────────────────────────────────────────────────────────────────────────
  if (typeof d3 === 'undefined') {
    console.error('D3.js ist nicht geladen. Bitte stelle sicher, dass du D3 v7+ eingebunden hast.');
    return;
  }
  const svg = d3.select('#flowConservationSVG');
  if (svg.empty()) {
    console.error('Kein <svg id="flowConservationSVG"> gefunden. Bitte überprüfe dein HTML.');
    return;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // (B) Configuration (Viridis‐inspired colors)
  // ─────────────────────────────────────────────────────────────────────────────
  const CONFIG = {
    rootFlow:       100,
    svgWidth:       600,
    svgHeight:      700,         // 700px Höhe, damit 3×168px-Knoten + Lücken passen
    backgroundColor:'#3b528b',   // rgb(59,82,139), dunkelblau (Viridis)
    edgeColor:      '#35b778',   // Viridis grün
    edgeStrokeWidth:2,
    arrowMarkerSize: 6,
    nodeBorderColor:'#5ec962',   // helleres Viridis grün
    nodeBackground: '#31688e',   // Viridis blau‐teal
    cellSize:       8,           // 8 px pro Zelle
    boardCols:     10,
    boardRows:     20,
    blockColor:     '#31688e',   // Stapelblöcke, Viridis‐Blau
    pieceColor:     '#fde725',   // Neu gespielte “I”-Stücke, Viridis‐Gelb
    emptyColor:     '#111',
    flowLabelColor: '#ccc',
    particleColor:  '#fff',
    particleRadius: 2,
    nodePadding:    4
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // (C) Clear any previous drawing
  // ─────────────────────────────────────────────────────────────────────────────
  svg.selectAll('*').remove();

  // ─────────────────────────────────────────────────────────────────────────────
  // (D) Define 7× (20×10) boards + “spawn/landing” info for action nodes
  // ─────────────────────────────────────────────────────────────────────────────

  // (D.1) ROOT board (layer 0)
  const rootBoard = [
    [0,0,0,0,0,2,0,0,0,0], // 0
    [0,0,0,0,0,2,0,0,0,0], // 1
    [0,0,0,0,0,2,0,0,0,0], // 2
    [0,0,0,0,0,2,0,0,0,0], // 3
    [0,0,0,0,0,0,0,0,0,0], // 4
    [0,0,0,0,0,0,0,0,0,0], // 5
    [0,0,0,0,0,0,0,0,0,0], // 6
    [0,0,0,0,0,0,0,0,0,0], // 7
    [0,0,0,0,0,0,0,0,0,0], // 8
    [0,0,0,0,0,0,0,0,0,0], // 9
    [0,0,0,0,0,0,0,0,0,0], // 10 (vertical “I” preview, col 5)
    [0,0,0,0,0,0,0,0,0,0], // 11
    [0,0,0,0,0,0, 0,0,0,0], // 12
    [0,0,0,0,0,0,0,0,0,0], // 13
    [0,0,0,0,0,0,0,0,0,0], // 14 (leer)
    [0,0,0,0,0,0,0,0,0,0], // 15 (leer)
    [1,1,1,1,1,0,1,1,1,1], // 16 (well)
    [1,1,1,1,1,0,1,1,1,1], // 17
    [1,1,1,1,1,0,1,1,1,1], // 18
    [1,1,1,1,1,0,1,1,1,1]  // 19
  ];

  // (D.2) ACTION boards (layer 1)

  // action1: vertical I → spawn at [0,5..3], land at [16..19,5]
  const action1Board = [
    [0,0,0,0,0,0,0,0,0,0], // 0
    [0,0,0,0,0,0,0,0,0,0], // 1
    [0,0,0,0,0,0,0,0,0,0], // 2
    [0,0,0,0,0,0,0,0,0,0], // 3
    [0,0,0,0,0,0,0,0,0,0], // 4
    [0,0,0,0,0,0,0,0,0,0], // 5
    [0,0,0,0,0,0,0,0,0,0], // 6
    [0,0,0,0,0,0,0,0,0,0], // 7
    [0,0,0,0,0,0,0,0,0,0], // 8
    [0,0,0,0,0,0,0,0,0,0], // 9
    [0,0,0,0,0,0,0,0,0,0], // 10
    [0,0,0,0,0,0,0,0,0,0], // 11
    [0,0,0,0,0,0,0,0,0,0], // 12
    [0,0,0,0,0,0,0,0,0,0], // 13
    [0,0,0,0,0,0,0,0,0,0], // 14
    [0,0,0,0,0,0,0,0,0,0], // 15
    [1,1,1,1,1,0,1,1,1,1], // 16 (well)
    [1,1,1,1,1,0,1,1,1,1], // 17
    [1,1,1,1,1,0,1,1,1,1], // 18
    [1,1,1,1,1,0,1,1,1,1]  // 19
  ];
  const action1Spawn   = [ [0,5], [1,5], [2,5], [3,5] ];
  const action1Landing = [ [16,5], [17,5], [18,5], [19,5] ];

  // action2: horizontal I at top-left (spawn), land at row 15 cols 0–3
  const action2Board = [
    [0,0,0,0,0,0,0,0,0,0], // 0
    [0,0,0,0,0,0,0,0,0,0], // 1
    [0,0,0,0,0,0,0,0,0,0], // 2
    [0,0,0,0,0,0,0,0,0,0], // 3
    [0,0,0,0,0,0,0,0,0,0], // 4
    [0,0,0,0,0,0,0,0,0,0], // 5
    [0,0,0,0,0,0,0,0,0,0], // 6
    [0,0,0,0,0,0,0,0,0,0], // 7
    [0,0,0,0,0,0,0,0,0,0], // 8
    [0,0,0,0,0,0,0,0,0,0], // 9
    [0,0,0,0,0,0,0,0,0,0], // 10
    [0,0,0,0,0,0,0,0,0,0], // 11
    [0,0,0,0,0,0,0,0,0,0], // 12
    [0,0,0,0,0,0,0,0,0,0], // 13
    [0,0,0,0,0,0,0,0,0,0], // 14
    [0,0,0,0,0,0,0,0,0,0], // 15 (lander row)
    [1,1,1,1,1,0,1,1,1,1], // 16 (well)
    [1,1,1,1,1,0,1,1,1,1], // 17
    [1,1,1,1,1,0,1,1,1,1], // 18
    [1,1,1,1,1,0,1,1,1,1]  // 19
  ];
  const action2Spawn   = [ [0,0], [0,1], [0,2], [0,3] ];
  const action2Landing = [ [15,0], [15,1], [15,2], [15,3] ];

  // action3: horizontal I at top-right corner (spawn), land at row 15 cols 6–9
  const action3Board = [
    [0,0,0,0,0,0,0,0,0,0], // 0
    [0,0,0,0,0,0,0,0,0,0], // 1
    [0,0,0,0,0,0,0,0,0,0], // 2
    [0,0,0,0,0,0,0,0,0,0], // 3
    [0,0,0,0,0,0,0,0,0,0], // 4
    [0,0,0,0,0,0,0,0,0,0], // 5
    [0,0,0,0,0,0,0,0,0,0], // 6
    [0,0,0,0,0,0,0,0,0,0], // 7
    [0,0,0,0,0,0,0,0,0,0], // 8
    [0,0,0,0,0,0,0,0,0,0], // 9
    [0,0,0,0,0,0,0,0,0,0], // 10
    [0,0,0,0,0,0,0,0,0,0], // 11
    [0,0,0,0,0,0,0,0,0,0], // 12
    [0,0,0,0,0,0,0,0,0,0], // 13
    [0,0,0,0,0,0,0,0,0,0], // 14
    [0,0,0,0,0,0,0,0,0,0], // 15 (lander row)
    [1,1,1,1,1,0,1,1,1,1], // 16 (well)
    [1,1,1,1,1,0,1,1,1,1], // 17
    [1,1,1,1,1,0,1,1,1,1], // 18
    [1,1,1,1,1,0,1,1,1,1]  // 19
  ];
  const action3Spawn   = [ [0,6], [0,7], [0,8], [0,9] ];
  const action3Landing = [ [15,6], [15,7], [15,8], [15,9] ];

  // (D.3) RESULT boards (layer 2)

  // result1: komplett leer (vierzeiliger Clear). Reward = 10
  const result1Board = Array.from({ length: 20 }, () => new Array(10).fill(0));

  // result2: horizontal “I” in row 15, cols 0–3. Reward = 2
  const result2Board = [
    [0,0,0,0,0,0,0,0,0,0], // 0
    [0,0,0,0,0,0,0,0,0,0], // 1
    [0,0,0,0,0,0,0,0,0,0], // 2
    [0,0,0,0,0,0,0,0,0,0], // 3
    [0,0,0,0,0,0,0,0,0,0], // 4
    [0,0,0,0,0,0,0,0,0,0], // 5
    [0,0,0,0,0,0,0,0,0,0], // 6
    [0,0,0,0,0,0,0,0,0,0], // 7
    [0,0,0,0,0,0,0,0,0,0], // 8
    [0,0,0,0,0,0,0,0,0,0], // 9
    [0,0,0,0,0,0,0,0,0,0], // 10
    [0,0,0,0,0,0,0,0,0,0], // 11
    [0,0,0,0,0,0,0,0,0,0], // 12
    [0,0,0,0,0,0,0,0,0,0], // 13
    [0,0,0,0,0,0,0,0,0,0], // 14
    [2,2,2,2,0,0,0,0,0,0], // 15 ← I (Viridis‐Gelb)
    [1,1,1,1,1,0,1,1,1,1], // 16 (well)
    [1,1,1,1,1,0,1,1,1,1], // 17
    [1,1,1,1,1,0,1,1,1,1], // 18
    [1,1,1,1,1,0,1,1,1,1]  // 19
  ];

  // result3: horizontal “I” in row 15, cols 6–9. Reward = 1
  const result3Board = [
    [0,0,0,0,0,0,0,0,0,0], // 0
    [0,0,0,0,0,0,0,0,0,0], // 1
    [0,0,0,0,0,0,0,0,0,0], // 2
    [0,0,0,0,0,0,0,0,0,0], // 3
    [0,0,0,0,0,0,0,0,0,0], // 4
    [0,0,0,0,0,0,0,0,0,0], // 5
    [0,0,0,0,0,0,0,0,0,0], // 6
    [0,0,0,0,0,0,0,0,0,0], // 7
    [0,0,0,0,0,0,0,0,0,0], // 8
    [0,0,0,0,0,0,0,0,0,0], // 9
    [0,0,0,0,0,0,0,0,0,0], // 10
    [0,0,0,0,0,0,0,0,0,0], // 11
    [0,0,0,0,0,0,0,0,0,0], // 12
    [0,0,0,0,0,0,0,0,0,0], // 13
    [0,0,0,0,0,0,0,0,0,0], // 14
    [0,0,0,0,0,0,2,2,2,2], // 15 ← I (Viridis‐Gelb)
    [1,1,1,1,1,0,1,1,1,1], // 16 (well)
    [1,1,1,1,1,0,1,1,1,1], // 17
    [1,1,1,1,1,0,1,1,1,1], // 18
    [1,1,1,1,1,0,1,1,1,1]  // 19
  ];

  // ─────────────────────────────────────────────────────────────────────────────
  // (E) Build “nodes” and “edges”
  // ─────────────────────────────────────────────────────────────────────────────
  const nodes = [
    {
      id: 'state',
      layer: 0,
      board: rootBoard,
      flowIn: CONFIG.rootFlow,
      flowOut: 0,
      terminal: false
    },
    {
      id: 'action1',
      layer: 1,
      board: action1Board,
      spawnCells: action1Spawn,
      landingCells: action1Landing,
      flowIn: 0,
      flowOut: 0,
      terminal: false
    },
    {
      id: 'action2',
      layer: 1,
      board: action2Board,
      spawnCells: action2Spawn,
      landingCells: action2Landing,
      flowIn: 0,
      flowOut: 0,
      terminal: false
    },
    {
      id: 'action3',
      layer: 1,
      board: action3Board,
      spawnCells: action3Spawn,
      landingCells: action3Landing,
      flowIn: 0,
      flowOut: 0,
      terminal: false
    },
    {
      id: 'result1',
      layer: 2,
      board: result1Board,
      reward: 10,   // ← Bestlösung: Reward = 10
      flowIn: 0,
      flowOut: 0,
      terminal: true
    },
    {
      id: 'result2',
      layer: 2,
      board: result2Board,
      reward: 2,    // ← Sekundäre Lösung: Reward = 2
      flowIn: 0,
      flowOut: 0,
      terminal: true
    },
    {
      id: 'result3',
      layer: 2,
      board: result3Board,
      reward: 1,    // ← Drittplatzierte Lösung: Reward = 1
      flowIn: 0,
      flowOut: 0,
      terminal: true
    }
  ];

  const edges = [
    { source: 'state',   target: 'action1', flow: 0 },
    { source: 'state',   target: 'action2', flow: 0 },
    { source: 'state',   target: 'action3', flow: 0 },
    { source: 'action1', target: 'result1', flow: 0 },
    { source: 'action2', target: 'result2', flow: 0 },
    { source: 'action3', target: 'result3', flow: 0 }
  ];

  // ─────────────────────────────────────────────────────────────────────────────
  // (F) Hard‐code flows: [90, 5, 5] → state→action, then same → action→result
  // ─────────────────────────────────────────────────────────────────────────────
  const flows = [90, 5, 5];  // state→action1, action2, action3
  ['action1','action2','action3'].forEach((aid, idx) => {
    // state → actionX
    edges.find(e => e.source === 'state' && e.target === aid).flow = flows[idx];
    // actionX → resultX
    edges.find(e => e.source === aid && e.target === `result${idx+1}`).flow = flows[idx];
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // (G) Compute flowIn / flowOut per node
  // ─────────────────────────────────────────────────────────────────────────────
  nodes.forEach(n => {
    n.flowIn  = edges.filter(e => e.target === n.id).reduce((sum, e) => sum + e.flow, 0);
    n.flowOut = edges.filter(e => e.source === n.id).reduce((sum, e) => sum + e.flow, 0);
    if (n.id === 'state') {
      n.flowIn  = CONFIG.rootFlow;
      n.flowOut = CONFIG.rootFlow;
    }
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // (H) Position each node by layer so none overlap
  // ─────────────────────────────────────────────────────────────────────────────
  const NODE_HEIGHT = CONFIG.boardRows * CONFIG.cellSize + 2 * CONFIG.nodePadding; // 20×8 + 2×4 = 168
  const NODE_WIDTH  = CONFIG.boardCols * CONFIG.cellSize + 2 * CONFIG.nodePadding; // 10×8 + 2×4 = 88

  const layer0 = nodes.filter(n => n.layer === 0);
  const layer1 = nodes.filter(n => n.layer === 1);
  const layer2 = nodes.filter(n => n.layer === 2);
  const xCoords = [50, 250, 450]; // x‐Positionen für Layer 0, 1, 2

  function positionLayer(arr, layerIdx) {
    arr.sort((a, b) => a.id.localeCompare(b.id));
    const count = arr.length;
    const totalNodesHeight = count * NODE_HEIGHT;
    const gap = (CONFIG.svgHeight - totalNodesHeight) / (count + 1);

    arr.forEach((n, i) => {
      n.x = xCoords[layerIdx];
      n.y = gap + i * (NODE_HEIGHT + gap);
      n.nodeWidth  = NODE_WIDTH;
      n.nodeHeight = NODE_HEIGHT;
    });
  }

  positionLayer(layer0, 0);
  positionLayer(layer1, 1);
  positionLayer(layer2, 2);

  // ─────────────────────────────────────────────────────────────────────────────
  // (I) Compute edge geometry (x1, y1, x2, y2, length, angle),
  //     so that “state→action” arrows exactly stop at the edge of each node,
  //     and the two vertical ones (action1 & action3) are shortened by 2 cm (≈ 75px).
  // ─────────────────────────────────────────────────────────────────────────────

  // Shrink amount in pixels to make arrows 2 cm shorter:
  const SHRINK_PX = 20;

  // 1) Compute “state → action” edges, with endpoints exactly on node borders
  nodes.filter(n => n.id === 'state').forEach(stateNode => {
    edges
      .filter(e => e.source === 'state')
      .forEach(e => {
        const src = stateNode;
        const tgt = nodes.find(n => n.id === e.target);

        const sx = src.x + src.nodeWidth  / 2;
        const sy = src.y + src.nodeHeight / 2;
        const tx = tgt.x + tgt.nodeWidth  / 2;
        const ty = tgt.y + tgt.nodeHeight / 2;

        // Direction vector from source‐center to target‐center
        const dx = tx - sx;
        const dy = ty - sy;
        const dist = Math.hypot(dx, dy) || 1;
        const ux = dx / dist;
        const uy = dy / dist;

        // Start exactly on source‐node border
        e.x1 = sx + ux * (src.nodeWidth  / 2);
        e.y1 = sy + uy * (src.nodeHeight / 2);

        // End exactly on target‐node border
        e.x2 = tx - ux * (tgt.nodeWidth  / 2);
        e.y2 = ty - uy * (tgt.nodeHeight / 2);

        // If this edge points to action1 (above) or action3 (below), shrink by SHRINK_PX
        if (e.target === 'action1' || e.target === 'action3') {
          const dx2 = e.x2 - e.x1;
          const dy2 = e.y2 - e.y1;
          const length = Math.hypot(dx2, dy2) || 1;
          const newLen = Math.max(0, length - SHRINK_PX);
          e.x2 = e.x1 + (dx2 / length) * newLen;
          e.y2 = e.y1 + (dy2 / length) * newLen;
        }

        e.length = Math.hypot(e.x2 - e.x1, e.y2 - e.y1);
        e.angle  = Math.atan2(e.y2 - e.y1, e.x2 - e.x1);
      });
  });

  // 2) Compute “action → result” edges, shortened slightly (4px) to avoid cutting arrowheads
  edges
    .filter(e => e.source.startsWith('action'))
    .forEach(e => {
      const s = nodes.find(n => n.id === e.source);
      const t = nodes.find(n => n.id === e.target);

      const sx = s.x + s.nodeWidth / 2;
      const sy = s.y + s.nodeHeight / 2;
      const tx = t.x + t.nodeWidth / 2;
      const ty = t.y + t.nodeHeight / 2;

      const offsetX = 4;
      const offsetY = 4;
      const dx = tx - sx;
      const dy = ty - sy;
      const dist = Math.hypot(dx, dy) || 1;
      const ux = dx / dist;
      const uy = dy / dist;

      e.x1 = sx + ux * ((s.nodeWidth / 2)  - offsetX);
      e.y1 = sy + uy * ((s.nodeHeight / 2) - offsetY);
      e.x2 = tx - ux * ((t.nodeWidth / 2)  - offsetX);
      e.y2 = ty - uy * ((t.nodeHeight / 2) - offsetY);
      e.length = Math.hypot(e.x2 - e.x1, e.y2 - e.y1);
      e.angle  = Math.atan2(e.y2 - e.y1, e.x2 - e.x1);
    });

  // ─────────────────────────────────────────────────────────────────────────────
  // (J) D3 Rendering
  // ─────────────────────────────────────────────────────────────────────────────

  // 1) Size the SVG & set background‐color
  svg
    .attr("viewBox", `0 0 ${CONFIG.svgWidth} ${CONFIG.svgHeight}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("width", "100%")
    .style("height", "auto")
    .style("background-color", CONFIG.backgroundColor);

  // 2) Define arrow marker (for edges and spawn‐arrows)
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrowFC')
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 10)
    .attr('refY', 5)
    .attr('markerWidth', CONFIG.arrowMarkerSize)
    .attr('markerHeight', CONFIG.arrowMarkerSize)
    .attr('orient', 'auto')
    .append('path')
      .attr('d', 'M0,0 L0,10 L10,5 z')
      .attr('fill', CONFIG.edgeColor);

  // 3) Main group for everything inside SVG
  const mainGroup = svg.append('g').attr('class','mainGroup');

  // 4) Draw the DAG edges (lines with arrowheads)
  mainGroup.selectAll('.edge-line')
    .data(edges)
    .enter()
    .append('line')
      .attr('class','edge-line')
      .attr('x1', d => d.x1)
      .attr('y1', d => d.y1)
      .attr('x2', d => d.x2)
      .attr('y2', d => d.y2)
      .attr('stroke', CONFIG.edgeColor)
      .attr('stroke-width', CONFIG.edgeStrokeWidth)
      .attr('marker-end', 'url(#arrowFC)');

  // 5) Label each edge with its flow (centered + slightly offset)
  mainGroup.selectAll('.flow-label')
    .data(edges)
    .enter()
    .append('text')
      .attr('class','flow-label')
      .attr('font-size', 10)
      .attr('fill', CONFIG.flowLabelColor)
      .attr('text-anchor','middle')
      .attr('alignment-baseline','middle')
      .attr('x', d => (d.x1 + d.x2)/2 - 10 * Math.sin(d.angle))
      .attr('y', d => (d.y1 + d.y2)/2 + 10 * Math.cos(d.angle))
      .text(d => d.flow.toFixed(1));

  // 6) Draw each “node” (mini‐board) as a <g> group
  const nodeGroups = mainGroup.selectAll('.node')
    .data(nodes)
    .enter()
    .append('g')
      .attr('class','node')
      .attr('transform', d => `translate(${d.x},${d.y})`);

  // 6.1) Draw the background rectangle for each node
  nodeGroups.append('rect')
    .attr('width',  d => d.nodeWidth)
    .attr('height', d => d.nodeHeight)
    .attr('fill',   CONFIG.nodeBackground)
    .attr('stroke', CONFIG.nodeBorderColor)
    .attr('stroke-width', 1);

  // 6.2) Inside each mini‐board: draw the 20×10 grid of cells
  nodeGroups.each(function(d) {
    const g = d3.select(this);

// (6.2.1) Draw the underlying “stack” (1 → blockColor, 2 → pieceColor, 0 → emptyColor)
for (let r = 0; r < CONFIG.boardRows; r++) {
  for (let c = 0; c < CONFIG.boardCols; c++) {
    const cellVal = d.board[r][c];
    let color;
    if (cellVal === 1) {
      color = CONFIG.blockColor;
    } else if (cellVal === 2) {
      color = CONFIG.pieceColor;
    } else {
      color = CONFIG.emptyColor;
    }
    g.append('rect')
      .attr('x', CONFIG.nodePadding + c * CONFIG.cellSize)
      .attr('y', CONFIG.nodePadding + r * CONFIG.cellSize)
      .attr('width',  CONFIG.cellSize)
      .attr('height', CONFIG.cellSize)
      .attr('fill',   color)
      .attr('stroke', '#333')
      .attr('stroke-width', 0.5);
  }
}


    // 6.2.2) If this is layer 1 (action‐node), draw spawnCells[] + arrow to landingCells[]
    if (d.layer === 1) {
      // Compute spawn centroid (average of all spawnCells positions)
      let spawnX = 0, spawnY = 0;
      d.spawnCells.forEach(([sr, sc]) => {
        spawnX += (CONFIG.nodePadding + sc * CONFIG.cellSize + CONFIG.cellSize / 2);
        spawnY += (CONFIG.nodePadding + sr * CONFIG.cellSize + CONFIG.cellSize / 2);
      });
      spawnX /= d.spawnCells.length;
      spawnY /= d.spawnCells.length;

      // Compute landing centroid (average of all landingCells positions)
      let landX = 0, landY = 0;
      d.landingCells.forEach(([lr, lc]) => {
        landX += (CONFIG.nodePadding + lc * CONFIG.cellSize + CONFIG.cellSize / 2);
        landY += (CONFIG.nodePadding + lr * CONFIG.cellSize + CONFIG.cellSize / 2);
      });
      landX /= d.landingCells.length;
      landY /= d.landingCells.length;

      // 6.2.2a) Draw the spawn “piece” (in bright pieceColor) at spawnCells[]
      d.spawnCells.forEach(([sr, sc]) => {
        g.append('rect')
          .attr('x', CONFIG.nodePadding + sc * CONFIG.cellSize)
          .attr('y', CONFIG.nodePadding + sr * CONFIG.cellSize)
          .attr('width',  CONFIG.cellSize)
          .attr('height', CONFIG.cellSize)
          .attr('fill',   CONFIG.pieceColor)
          .attr('stroke', '#333')
          .attr('stroke-width', 0.5);
      });

      // 6.2.2b) Draw an arrow from (spawnX,spawnY) → (landX,landY)
      g.append('line')
        .attr('x1', spawnX)
        .attr('y1', spawnY)
        .attr('x2', landX)
        .attr('y2', landY)
        .attr('stroke', CONFIG.pieceColor)
        .attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowFC)');
    }

    // 6.2.3) If this is layer 2 (result‐node), draw the landing piece (value 2)
    if (d.layer === 2) {
      // Draw each “2” cell in bright pieceColor (Viridis‐Gelb)
      d.board.forEach((row, r) => {
        row.forEach((val, c) => {
          if (val === 2) {
            g.append('rect')
              .attr('x', CONFIG.nodePadding + c * CONFIG.cellSize)
              .attr('y', CONFIG.nodePadding + r * CONFIG.cellSize)
              .attr('width',  CONFIG.cellSize)
              .attr('height', CONFIG.cellSize)
              .attr('fill',   CONFIG.pieceColor)
              .attr('stroke', '#333')
              .attr('stroke-width', 0.5);
          }
        });
      });

      // ※ Removed: “Reward: X” text in layer 2
    }
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // (K) Particle animation (adapted so that an action→result edge only emits
  //     particles if a particle has passed through its action node first)
  // ─────────────────────────────────────────────────────────────────────────────
  let particleID = 0;
  let particles  = [];

  // Build edgeObjs with source/target so we can track parent→child relationships:
  const edgeObjs = edges.map(e => ({
    source: e.source,      // e.g. "state" or "action1"
    target: e.target,      // e.g. "action1" or "result1"
    x1:     e.x1,
    y1:     e.y1,
    x2:     e.x2,
    y2:     e.y2,
    length: e.length,
    angle:  e.angle,
    flow:   e.flow
  }));

  function createParticle(edgeObj) {
    const speed = 40; // px/sec
    const id    = particleID++;
    particles.push({ id, edge: edgeObj, t: 0, speed });
  }

  function spawnParticles(dtMs) {
    // Only spawn on edges whose source is "state"
    const dtSec = dtMs / 1000;
    edgeObjs
      .filter(e => e.source === 'state')
      .forEach(e => {
        const rate     = e.flow * 0.04;    // particles per second
        const expected = rate * dtSec;
        const count    = Math.floor(expected + Math.random());
        for (let i = 0; i < count; i++) createParticle(e);
      });
  }

  function updateParticles(dtSec) {
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      // Advance t by how much fraction of the edge we travel this frame:
      const frac = (p.speed * dtSec) / p.edge.length;
      p.t += frac;

      if (p.t >= 1) {
        // Particle has reached the END of its edge:
        // If this edge’s target is an action node, immediately spawn
        // a corresponding new particle on the action→result edge(s).
        const arrivedActionId = p.edge.target; // e.g. "action1"
        // Find all child edges whose source === arrivedActionId
        const children = edgeObjs.filter(e2 => e2.source === arrivedActionId);

        children.forEach(childEdge => {
          // Start a brand‐new particle (t=0) on that child edge
          createParticle(childEdge);
        });

        // Remove the old particle from this.edge
        particles.splice(i, 1);
      }
    }
  }

  function renderParticles() {
    const sel = particleLayer.selectAll('.flow-particle')
      .data(particles, d => d.id);

    const enter = sel.enter()
      .append('circle')
      .attr('class', 'flow-particle')
      .attr('r', CONFIG.particleRadius)
      .attr('fill', CONFIG.particleColor)
      .attr('opacity', 0.8);

    sel.merge(enter)
      .attr('cx', d => d.edge.x1 + (d.edge.x2 - d.edge.x1) * d.t)
      .attr('cy', d => d.edge.y1 + (d.edge.y2 - d.edge.y1) * d.t);

    sel.exit().remove();
  }

  let lastTime = performance.now();
  const particleLayer = mainGroup.append('g').attr('class','particleLayer');

  function animate(time) {
    const dt = time - lastTime;
    lastTime = time;
    spawnParticles(dt);
    updateParticles(dt / 1000);
    renderParticles();
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}

// Initialize when DOM is ready
window.addEventListener('DOMContentLoaded', initFlowConservationDemo);