function initSankeySimulator() {
  // Define nodes and links
  let nodes = [
    { id: 0, name: "Start", type: "source" },
    { id: 1, name: "Branch A", type: "intermediate" },
    { id: 2, name: "Branch B", type: "intermediate" },
    { id: 3, name: "Branch C", type: "intermediate" },
    { id: 4, name: "Terminal A", type: "terminal", reward: 15 },
    { id: 5, name: "Terminal B", type: "terminal", reward: 15 },
    { id: 6, name: "Terminal C", type: "terminal", reward: 20 },
    { id: 7, name: "Terminal D", type: "terminal", reward: 20 },
    { id: 8, name: "Terminal E", type: "terminal", reward: 20 }
  ];

  let links = [
    { source: 0, target: 1, value: 30 }, // Start -> Branch A
    { source: 0, target: 2, value: 40 }, // Start -> Branch B
    { source: 0, target: 3, value: 20 }, // Start -> Branch C
    { source: 1, target: 4, value: 15 }, // Branch A -> Terminal A
    { source: 1, target: 5, value: 15 }, // Branch A -> Terminal B
    { source: 2, target: 6, value: 20 }, // Branch B -> Terminal C
    { source: 2, target: 7, value: 20 }, // Branch B -> Terminal D
    { source: 3, target: 8, value: 20 }  // Branch C -> Terminal E
  ];

  // Set the "logical" width/height for the diagram
  const svgWidth = 700, svgHeight = 500;

  // Create the <svg> with a viewBox, removing any fixed width/height
  const svg = d3.select("#sankeyContainer")
    .append("svg")
    .attr("viewBox", `0 0 ${svgWidth} ${svgHeight}`)
    .attr("width", null)      // remove fixed width
    .attr("height", null);    // remove fixed height

  // Make sure your CSS has something like:
  // #sankeyContainer svg { width: 100%; height: auto; display: block; }

  // Create the Sankey generator
  const sankeyGenerator = d3.sankey()
    .nodeId(d => d.id)
    .nodeWidth(20)
    .nodePadding(10)
    // Use the full "logical" area: from (0,0) to (svgWidth, svgHeight)
    .extent([[0, 0], [svgWidth, svgHeight]]);

  // Update upstream flows for intermediate nodes
  function updateUpstreamFlows() {
    // For each intermediate branch, set the link from Start = sum of outgoing
    nodes.filter(n => n.type === "intermediate").forEach(interNode => {
      const downstreamLinks = links.filter(l => l.source === interNode.id);
      const totalDownstream = d3.sum(downstreamLinks, l => l.value);
      // Link from Start to this intermediate node
      const upstreamLink = links.find(l => l.target === interNode.id);
      if (upstreamLink) {
        upstreamLink.value = totalDownstream;
      }
    });
  }

  // Initially update flows
  updateUpstreamFlows();

  // Compute initial Sankey layout
  let graph = sankeyGenerator({
    nodes: nodes.map(d => Object.assign({}, d)),
    links: links.map(d => Object.assign({}, d))
  });

  // Draw links
  const link = svg.append("g")
    .attr("fill", "none")
    .selectAll("path")
    .data(graph.links)
    .join("path")
    .attr("d", d3.sankeyLinkHorizontal())
    .attr("stroke", "#aaa")
    .attr("stroke-width", d => Math.max(1, d.width))
    .attr("stroke-opacity", 0.5);

  // Draw nodes as rectangles
  const node = svg.append("g")
    .selectAll("rect")
    .data(graph.nodes)
    .join("rect")
    .attr("x", d => d.x0)
    .attr("y", d => d.y0)
    .attr("width", d => d.x1 - d.x0)
    .attr("height", d => d.y1 - d.y0)
    .attr("fill", d => {
      if (d.type === "terminal") return "#ff6666";
      if (d.type === "intermediate") return "#66ccff";
      return "#99cc99"; // default for "Start"
    })
    .attr("stroke", "#000");

  // Node labels
  const label = svg.append("g")
    .style("font", "12px sans-serif")
    .selectAll("text")
    .data(graph.nodes)
    .join("text")
    .attr("x", d => d.x0 < (svgWidth / 2) ? d.x1 + 6 : d.x0 - 6)
    .attr("y", d => (d.y1 + d.y0) / 2)
    .attr("dy", "0.35em")
    .attr("text-anchor", d => d.x0 < (svgWidth / 2) ? "start" : "end")
    .text(d => d.name);

  // Build reward inputs
  const inputContainer = d3.select("#rewardInputs");
  inputContainer.html(""); // clear any previous

  const terminalNodes = nodes.filter(d => d.type === "terminal");
  terminalNodes.forEach(function(d) {
    inputContainer.append("div")
      .attr("class", "reward-input")
      .html(`${d.name} Reward: <input type="number" value="${d.reward}" data-node-id="${d.id}" />`);
  });

  // On input change, update flows
  d3.selectAll(".reward-input input").on("input", function() {
    const newReward = +this.value;
    const nodeId = +this.getAttribute("data-node-id");

    // Update that node's reward
    const terminalNode = nodes.find(n => n.id === nodeId);
    if (terminalNode) terminalNode.reward = newReward;

    // Update link to that terminal
    const linkToTerminal = links.find(l => l.target === nodeId);
    if (linkToTerminal) linkToTerminal.value = newReward;

    updateUpstreamFlows();

    // Recompute Sankey
    const newGraph = sankeyGenerator({
      nodes: nodes.map(d => Object.assign({}, d)),
      links: links.map(d => Object.assign({}, d))
    });

    // Update links
    link.data(newGraph.links)
      .transition().duration(500)
      .attr("d", d3.sankeyLinkHorizontal())
      .attr("stroke-width", d => Math.max(1, d.width));

    // Update nodes
    node.data(newGraph.nodes)
      .transition().duration(500)
      .attr("x", d => d.x0)
      .attr("y", d => d.y0)
      .attr("width", d => d.x1 - d.x0)
      .attr("height", d => d.y1 - d.y0);

    // Update labels
    label.data(newGraph.nodes)
      .transition().duration(500)
      .attr("x", d => d.x0 < (svgWidth / 2) ? d.x1 + 6 : d.x0 - 6)
      .attr("y", d => (d.y1 + d.y0) / 2);
  });
}

document.addEventListener("DOMContentLoaded", initSankeySimulator);
