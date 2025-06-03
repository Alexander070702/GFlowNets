function initComparisonChart() {
  const container = d3.select("#comparisonChart");
  
  // ─── Increase from 200×300 to 300×400 ───
  const width = 300,
        height = 400;

  // Mock data for Traditional (single path)
  const leftData = {
    nodes: [
      { id: 'A', x: 100, y: 50 },
      { id: 'E', x:  60, y: 150 },
      { id: 'F', x: 140, y: 150 },  // included but unused visually
      { id: 'G', x:  40, y: 250 },  // included but unused visually
      { id: 'H', x: 100, y: 250 },
      { id: 'I', x: 160, y: 250 }   // included but unused visually
    ],
    edges: [
      { source: 'A', target: 'E' },
      { source: 'E', target: 'H' }
    ]
  };

  // Mock data for GFlowNet (multiple paths)
  const rightData = {
    nodes: [
      { id: 'A', x: 100, y: 50 },
      { id: 'E', x:  60, y: 150 },
      { id: 'F', x: 140, y: 150 },
      { id: 'G', x:  40, y: 250 },
      { id: 'H', x: 100, y: 250 },
      { id: 'I', x: 160, y: 250 }
    ],
    edges: [
      { source: 'A', target: 'E' },
      { source: 'A', target: 'F' },
      { source: 'E', target: 'G' },
      { source: 'E', target: 'H' },
      { source: 'F', target: 'H' },
      { source: 'F', target: 'I' }
    ]
  };

  // 1) Center each dataset so that its bounding box sits inside [0..width]×[0..height]
  centerData(leftData, width, height);
  centerData(rightData, width, height);

  // 2) Append two SVGs side by side, each 300×400
  const svgLeft = container.append("svg")
    .attr("width", width)
    .attr("height", height);

  const svgRight = container.append("svg")
    .attr("width", width)
    .attr("height", height);

  // ─── Add Viridis‐style background rect to each SVG ───
  [svgLeft, svgRight].forEach(svg => {
    svg.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "#39568C");  // Viridis mid‐tone
  });

  // 3) Define a shared arrow‐marker (lighter stroke for contrast)
  [svgLeft, svgRight].forEach(svg => {
    svg.append("defs").append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 10)
      .attr("refY", 5)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,0 L0,10 L10,5 z")
      .attr("fill", "#eee");
  });

  // 4) Draw each DAG (left = Traditional, right = GFlowNet)
  const edgesLeft  = drawDAG(svgLeft,  leftData,  "Traditional");
  const edgesRight = drawDAG(svgRight, rightData, "GFlowNet");

  // 5) Animate with a longer interval (1800ms instead of 1200ms)
  let step = 0;
  d3.interval(() => {
    step = (step + 1) % 6;

    // Traditional: highlight A→E for steps 0–2, then E→H for steps 3–5
    edgesLeft
      .transition().duration(600)
      .attr("stroke-width", d => {
        const a_e = (step < 3 && d.source === "A" && d.target === "E");
        const e_h = (step >= 3 && d.source === "E" && d.target === "H");
        return (a_e || e_h) ? 6 : 2;
      })
      .attr("stroke", d => {
        const a_e = (step < 3 && d.source === "A" && d.target === "E");
        const e_h = (step >= 3 && d.source === "E" && d.target === "H");
        return (a_e || e_h) ? "#ff6666" : "#eee";
      });

    // GFlowNet: do not highlight until step ≥3; highlight all edges in step 3–5
    edgesRight
      .transition().duration(600)
      .attr("stroke-width", step >= 3 ? 5 : 2)
      .attr("stroke", step >= 3 ? "#00bfff" : "#eee");

  }, 1800);


  /**
   * centerData(data, w, h):
   *   Compute bounding‐box of all node (x,y). Then shift every node so that
   *   the box is centered in a w×h canvas.
   */
  function centerData(data, w, h) {
    const xs = data.nodes.map(d => d.x),
          ys = data.nodes.map(d => d.y);

    const minX = d3.min(xs), maxX = d3.max(xs),
          minY = d3.min(ys), maxY = d3.max(ys);

    const dagWidth  = maxX - minX,
          dagHeight = maxY - minY;

    // Offsets to center the bounding box inside [0..w]×[0..h]
    const offsetX = (w - dagWidth) / 2 - minX,
          offsetY = (h - dagHeight) / 2 - minY;

    data.nodes.forEach(node => {
      node.x += offsetX;
      node.y += offsetY;
    });
  }

  /**
   * drawDAG(svg, data, title):
   *   - Appends a title text at the top.
   *   - Draws all edges (lines with arrow markers).
   *   - Draws all nodes (circles) and labels.
   *   - Returns the D3 selection of edges for later transitions.
   */
  function drawDAG(svg, data, title) {
    // Title in white, centered near top
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 24)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .attr("font-size", 18)
      .attr("font-weight", "bold")
      .text(title);

    // Edges (lines)
    const edges = svg.selectAll(".edge")
      .data(data.edges)
      .enter().append("line")
        .attr("class", "edge")
        .attr("x1", d => data.nodes.find(n => n.id === d.source).x)
        .attr("y1", d => data.nodes.find(n => n.id === d.source).y)
        .attr("x2", d => data.nodes.find(n => n.id === d.target).x)
        .attr("y2", d => data.nodes.find(n => n.id === d.target).y)
        .attr("stroke", "#eee")
        .attr("stroke-width", 2)
        .attr("marker-end", "url(#arrow)");

    // Nodes (circles)
    svg.selectAll(".node")
      .data(data.nodes)
      .enter().append("circle")
        .attr("class", "node")
        .attr("r", 12)
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("fill", "#eee")
        .attr("stroke", "#fff")
        .attr("stroke-width", 2);

    // Labels (node IDs)
    svg.selectAll(".label")
      .data(data.nodes)
      .enter().append("text")
        .attr("class", "label")
        .attr("x", d => d.x)
        .attr("y", d => d.y + 4)
        .attr("text-anchor", "middle")
        .attr("fill", "#39568C")
        .attr("font-weight", "bold")
        .attr("font-size", 12)
        .text(d => d.id);

    return edges;
  }
}
