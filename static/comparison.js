function initComparisonChart() {
  const container = d3.select("#chart");
  const width = 200, height = 300;

  // Mock data for Traditional (single path)
  const leftData = {
    nodes: [
      { id: 'A', x: 100, y: 50 },
      { id: 'E', x: 60,  y: 150 },
      { id: 'F', x: 140, y: 150 },  // Not used for single path, but included
      { id: 'G', x: 40,  y: 250 },  // Not used for single path, but included
      { id: 'H', x: 100, y: 250 },
      { id: 'I', x: 160, y: 250 },  // Not used for single path, but included
    ],
    edges: [
      // Only one path A -> E -> H
      { source: 'A', target: 'E' },
      { source: 'E', target: 'H' }
    ]
  };

  // Mock data for GFlowNet (multiple paths)
  const rightData = {
    nodes: [
      { id: 'A', x: 100, y: 50 },
      { id: 'E', x: 60,  y: 150 },
      { id: 'F', x: 140, y: 150 },
      { id: 'G', x: 40,  y: 250 },
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

  const svgLeft = container.append("svg")
    .attr("width", width)
    .attr("height", height);

  const svgRight = container.append("svg")
    .attr("width", width)
    .attr("height", height);

  // Arrow marker for edges
  [svgLeft, svgRight].forEach(svg => {
    svg.append("defs").append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 10)
      .attr("refY", 5)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,0 L0,10 L10,5 z")
      .attr("fill", "#bbb");
  });

  function drawDAG(svg, data, title) {
    // Title text
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .text(title);

    // Draw edges
    const edges = svg.selectAll(".edge")
      .data(data.edges)
      .enter().append("line")
      .attr("class", "edge")
      .attr("x1", d => data.nodes.find(n => n.id === d.source).x)
      .attr("y1", d => data.nodes.find(n => n.id === d.source).y)
      .attr("x2", d => data.nodes.find(n => n.id === d.target).x)
      .attr("y2", d => data.nodes.find(n => n.id === d.target).y)
      .attr("stroke", "#bbb")
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrow)");

    // Draw nodes
    svg.selectAll(".node")
      .data(data.nodes)
      .enter().append("circle")
      .attr("class", "node")
      .attr("r", 10)
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("fill", "#ccc");

    // Label nodes
    svg.selectAll(".label")
      .data(data.nodes)
      .enter().append("text")
      .attr("class", "label")
      .attr("x", d => d.x)
      .attr("y", d => d.y + 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .text(d => d.id);

    return edges;
  }

  // Render the two DAGs
  const edgesLeft = drawDAG(svgLeft, leftData, "Traditional");
  const edgesRight = drawDAG(svgRight, rightData, "GFlowNet");

  // Simple animation to highlight edges
  let step = 0;
  d3.interval(() => {
    step = (step + 1) % 6;

    // Left: single path thickens in two steps
    edgesLeft
      .transition().duration(500)
      .attr("stroke-width", d => {
        // Highlight A->E for first half, then E->H for second
        const a_e = (step < 3 && d.source === "A" && d.target === "E");
        const e_h = (step >= 3 && d.source === "E" && d.target === "H");
        return (a_e || e_h) ? 6 : 2;
      })
      .attr("stroke", d => {
        const a_e = (step < 3 && d.source === "A" && d.target === "E");
        const e_h = (step >= 3 && d.source === "E" && d.target === "H");
        return (a_e || e_h) ? "#ff6666" : "#bbb";
      });

    // Right: highlight multiple edges after step >= 3
    edgesRight
      .transition().duration(500)
      .attr("stroke-width", step >= 3 ? 5 : 2)
      .attr("stroke", step >= 3 ? "#00bfff" : "#bbb");
  }, 1200);
}
