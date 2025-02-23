// flow_conservation.js

function initFlowConservationDemo() {
  // Updated node definitions: renamed s5 -> xD
  const nodes = [
    { id: "s0", x: 100,  y: 200, terminal: false, color: "#66ccff" },
    { id: "s1", x: 250,  y: 100, terminal: false, color: "#cc66ff" },
    { id: "s2", x: 250,  y: 200, terminal: false, color: "#ffff66" },
    { id: "s3", x: 250,  y: 300, terminal: false, color: "#66ff66" },
    { id: "s4", x: 400,  y: 150, terminal: false, color: "#ff66ff" },
    { id: "xA", x: 400,  y:  50, terminal: true,  color: "#ff9999", reward: 1 },
    { id: "xB", x: 400,  y: 250, terminal: true,  color: "#ff7777", reward: 2 },
    { id: "xC", x: 400,  y: 350, terminal: true,  color: "#ff5555", reward: 3 },
    { id: "xD", x: 550,  y: 200, terminal: true,  color: "#ff3333", reward: 3 }
  ];

  // Flows remain balanced; changed the last edge to xD
  const edges = [
    { source: "s0", target: "s1", flow: 5 },
    { source: "s0", target: "s2", flow: 0.5 },
    { source: "s0", target: "s3", flow: 3 },
    { source: "s1", target: "s4", flow: 3 },
    { source: "s1", target: "xA", flow: 2 },
    { source: "s2", target: "xB", flow: 0.5 },
    { source: "s3", target: "xC", flow: 2 },
    { source: "s3", target: "s4", flow: 1 },
    { source: "s4", target: "xD", flow: 4 }
  ];

  // Make the SVG responsive
  const svg = d3.select("#flowDAG")
    .attr("width", null)
    .attr("height", null)
    .attr("viewBox", "0 0 600 400")
    .attr("preserveAspectRatio", "xMidYMid meet");

  const tooltip = d3.select("#flowTooltip");

  // Small arrow marker
  svg.append("defs").append("marker")
    .attr("id", "arrowFlow2")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 10)
    .attr("refY", 5)
    .attr("markerWidth", 4)
    .attr("markerHeight", 4)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,0 L0,10 L10,5 z")
    .attr("fill", "#999");

  // Build a lookup map for nodes
  const nodeMap = {};
  nodes.forEach(n => { nodeMap[n.id] = n; });

  const r = 16; // circle radius

  // Calculate edges so lines/particles start at circle boundary
  const edgeObjs = edges.map(e => {
    const sx = nodeMap[e.source].x,
          sy = nodeMap[e.source].y,
          tx = nodeMap[e.target].x,
          ty = nodeMap[e.target].y;
    const dx = tx - sx, dy = ty - sy;
    const length = Math.hypot(dx, dy);
    const ux = dx / length, uy = dy / length;
    const x1 = sx + ux * r, y1 = sy + uy * r;
    const x2 = tx - ux * r, y2 = ty - uy * r;
    return {
      sourceId: e.source,
      targetId: e.target,
      flow: e.flow,
      x1, y1,
      x2, y2,
      length: Math.hypot(x2 - x1, y2 - y1),
      midX: (x1 + x2) / 2,
      midY: (y1 + y2) / 2,
      angle: Math.atan2(y2 - y1, x2 - x1)
    };
  });

  // Draw edges
  svg.selectAll(".flow-edge2")
    .data(edgeObjs)
    .enter().append("line")
    .attr("class", "flow-edge2")
    .attr("x1", d => d.x1)
    .attr("y1", d => d.y1)
    .attr("x2", d => d.x2)
    .attr("y2", d => d.y2)
    .attr("stroke-width", 1.5)
    .attr("marker-end", "url(#arrowFlow2)")
    .attr("stroke", d => nodeMap[d.targetId].color);

  // Custom label offset logic to avoid overlap
  function getLabelOffsetData(flow) {
    // Base offset distance from the line
    let offset = 12;
    // Extra x/y adjustments
    let extraX = 0, extraY = 0;

    // Move Flow:5 (purple) up
    if (flow === 5) {
      extraY = -5; 
    }
    // Separate Flow:0.5 and Flow:1 from each other
    // We'll offset them in opposite directions
    else if (flow === 0.5) {
      extraX = -8;
      extraY = 8;
    }
    else if (flow === 1) {
      extraX = 10;
      extraY = -5;
    }
    // Move Flow:3 (green) a bit more to the right
    else if (flow === 3) {
      extraX = 10;
    }
    return { offset, extraX, extraY };
  }

  // Draw flow labels
  svg.selectAll(".flow-label2")
    .data(edgeObjs)
    .enter().append("text")
    .attr("class", "flow-label2")
    .attr("font-size", 11)
    .attr("fill", "#fff")
    .attr("text-anchor", "middle")
    .each(function(d) {
      const { offset, extraX, extraY } = getLabelOffsetData(d.flow);
      d.labelOffset = offset;
      d.labelExtraX = extraX;
      d.labelExtraY = extraY;
    })
    .attr("x", d => {
      // base offset from line
      const xBase = d.midX - Math.sin(d.angle) * d.labelOffset;
      return xBase + d.labelExtraX;
    })
    .attr("y", d => {
      const yBase = d.midY - Math.cos(d.angle) * d.labelOffset;
      return yBase + d.labelExtraY;
    })
    .text(d => `Flow: ${d.flow}`);

  // Node circles
  svg.selectAll(".flow-node2")
    .data(nodes)
    .enter().append("circle")
    .attr("class", "flow-node2")
    .attr("r", r)
    .attr("cx", d => d.x)
    .attr("cy", d => d.y)
    .attr("fill", d => d.color)
    .attr("stroke", "#666")
    .attr("stroke-width", 2)
    .on("mouseover", function(event, d) {
      // inflow/outflow for tooltip
      const inflow = edgeObjs
        .filter(e => e.targetId === d.id)
        .reduce((acc, e) => acc + e.flow, 0);
      const outflow = edgeObjs
        .filter(e => e.sourceId === d.id)
        .reduce((acc, e) => acc + e.flow, 0);

      let msg = `<strong>${d.id}</strong><br/>
        Inflow: ${inflow}<br/>
        Outflow: ${outflow}`;
      if (d.terminal && d.reward !== undefined) {
        msg += `<br/>Reward: ${d.reward}`;
      }
      tooltip
        .style("opacity", 1)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 28) + "px")
        .html(msg);
    })
    .on("mousemove", function(event) {
      tooltip
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 28) + "px");
    })
    .on("mouseout", function() {
      tooltip.style("opacity", 0);
    });

  // Node labels
  svg.selectAll(".flow-node-label2")
    .data(nodes)
    .enter().append("text")
    .attr("class", "flow-node-label2")
    .attr("font-size", 13)
    .attr("fill", "#fff")
    .attr("text-anchor", "middle")
    .attr("x", d => d.x)
    .attr("y", d => d.y + 4)
    .text(d => d.id);

  // Particle system
  let flowParticles = [];
  let particleIdCounter = 0;
  const particleLayer = svg.append("g").attr("class", "particle-layer2");
  let lastTime = performance.now();

  // Smaller exponent => fewer particles
  const spawnMultiplier = 0.2;
  function spawnParticles(dtMs) {
    const dtSec = dtMs / 1000;
    edgeObjs.forEach(e => {
      const spawnRate = Math.pow(e.flow, 2) * spawnMultiplier;
      const expectedCount = spawnRate * dtSec;
      const countToSpawn = Math.floor(expectedCount + Math.random());
      for (let i = 0; i < countToSpawn; i++) {
        createParticle(e);
      }
    });
  }

  function createParticle(edgeObj) {
    const speed = 20 + edgeObj.flow * 4 + Math.random() * 8;
    const color = nodeMap[edgeObj.targetId].color || "#fff";
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

  function updateParticles(dtSec) {
    for (let i = flowParticles.length - 1; i >= 0; i--) {
      let p = flowParticles[i];
      const fraction = (p.speed * dtSec) / p.edge.length;
      p.t += fraction;
      if (p.t >= 1) {
        flowParticles.splice(i, 1);
      }
    }
  }

  function renderParticles() {
    const sel = particleLayer.selectAll(".flow-particle2")
      .data(flowParticles, d => d.id);

    const enterSel = sel.enter().append("circle")
      .attr("class", "flow-particle2")
      .attr("r", d => d.radius)
      .attr("fill", d => d.color)
      .attr("opacity", 0.85);

    sel.merge(enterSel)
      .attr("cx", d => d.edge.x1 + (d.edge.x2 - d.edge.x1) * d.t)
      .attr("cy", d => d.edge.y1 + (d.edge.y2 - d.edge.y1) * d.t);

    sel.exit().remove();
  }

  function animateFlow(timestamp) {
    const dtMs = timestamp - lastTime;
    lastTime = timestamp;
    spawnParticles(dtMs);
    updateParticles(dtMs / 1000);
    renderParticles();
    requestAnimationFrame(animateFlow);
  }

  requestAnimationFrame(animateFlow);
}
