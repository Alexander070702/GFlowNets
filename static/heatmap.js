//heatmap

function init2DGridTutorial() {
    const gridSize = 10;
    const cellSize = 30;
  
    // Create a 10×10 reward grid with two peaks
    let gridRewards = [];
    for (let y = 0; y < gridSize; y++) {
      let row = [];
      for (let x = 0; x < gridSize; x++) {
        let base = 1;
        let dist1 = Math.hypot(x - 3, y - 3); // peak near (3,3)
        let dist2 = Math.hypot(x - 7, y - 6); // peak near (7,6)
        let peak1 = 40 / (1 + dist1 * dist1);
        let peak2 = 60 / (1 + dist2);
        let r = base + peak1 + peak2;
        row.push(r);
      }
      gridRewards.push(row);
    }
  
    // Setup the <svg>
    const svgW = gridSize * cellSize;
    const svgH = gridSize * cellSize;
    const svg = d3.select("#gridTutorialSVG")
      .attr("width", svgW)
      .attr("height", svgH);
  
    // Color scale
    let maxReward = d3.max(gridRewards.flat());
    const colorScale = d3.scaleSequential(d3.interpolateOrRd)
      .domain([0, maxReward]);
  
    // Convert to a flat array for data binding
    let cellsData = [];
    for (let yy = 0; yy < gridSize; yy++) {
      for (let xx = 0; xx < gridSize; xx++) {
        cellsData.push({
          x: xx,
          y: yy,
          reward: gridRewards[yy][xx]
        });
      }
    }
  
    // Draw the grid cells as <rect> with corrected tooltip positioning
    svg.selectAll(".gridCell")
      .data(cellsData)
      .enter().append("rect")
      .attr("class", "gridCell")
      .attr("x", d => d.x * cellSize)
      .attr("y", d => d.y * cellSize)
      .attr("width", cellSize - 1)
      .attr("height", cellSize - 1)
      .attr("fill", d => colorScale(d.reward))
      .on("mouseover", function(event, d) {
        // Use relative coordinates from the SVG container
        const containerRect = this.ownerSVGElement.getBoundingClientRect();
        const x = event.clientX - containerRect.left;
        const y = event.clientY - containerRect.top;
        d3.select("#gridTooltip")
          .style("opacity", 1)
          .style("left", (x + 10) + "px")
          .style("top", (y - 20) + "px")
          .html(`Cell (${d.x}, ${d.y})<br/>Reward: ${d.reward.toFixed(1)}`);
      })
      .on("mousemove", function(event) {
        const containerRect = this.ownerSVGElement.getBoundingClientRect();
        const x = event.clientX - containerRect.left;
        const y = event.clientY - containerRect.top;
        d3.select("#gridTooltip")
          .style("left", (x + 10) + "px")
          .style("top", (y - 20) + "px");
      })
      .on("mouseout", function() {
        d3.select("#gridTooltip").style("opacity", 0);
      });
  
    // Hits layer for "particles" (ensure pointer-events are off so they don't block hover)
    const hitsLayer = svg.append("g")
      .style("pointer-events", "none");
  
    let hits = [];
    let hitIdCounter = 0;
    let lastTime = performance.now();
    requestAnimationFrame(animate);
  
    function animate(timestamp) {
      let dtMs = timestamp - lastTime;
      lastTime = timestamp;
  
      spawnHits(dtMs);
  
      // Decay hits
      for (let i = hits.length - 1; i >= 0; i--) {
        hits[i].life -= dtMs * 0.003;
        if (hits[i].life <= 0) hits.splice(i, 1);
      }
      renderHits();
      requestAnimationFrame(animate);
    }
  
    function spawnHits(dtMs) {
      let dtSec = dtMs / 1000;
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          let rVal = gridRewards[y][x];
          // Exaggerate difference => rVal^1.3
          let spawnRate = Math.pow(rVal, 1.3);
          let expectedCount = spawnRate * dtSec;
          let count = Math.floor(expectedCount + Math.random());
          for (let i = 0; i < count; i++) {
            createHit(x, y);
          }
        }
      }
    }
  
    function createHit(cx, cy) {
      let id = hitIdCounter++;
      hits.push({
        id,
        x: cx * cellSize + cellSize / 2,
        y: cy * cellSize + cellSize / 2,
        life: 1.0
      });
    }
  
    function renderHits() {
      let sel = hitsLayer.selectAll(".hit").data(hits, d => d.id);
  
      let enterSel = sel.enter().append("circle")
        .attr("class", "hit")
        .attr("r", 5)
        .attr("fill", "#fff")
        .attr("opacity", 1);
  
      sel.merge(enterSel)
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", d => 5 * d.life)
        .attr("opacity", d => d.life);
  
      sel.exit().remove();
    }
  }
  
  // Initialize the 2D Grid Tutorial once the DOM is fully loaded
  document.addEventListener("DOMContentLoaded", init2DGridTutorial);
  