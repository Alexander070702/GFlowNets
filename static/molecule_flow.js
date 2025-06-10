// static/molecule_flow.js

// configuration
const CFG = {
  width:            900,
  height:           600,
  nodeWidth:        100,
  nodePadding:      20,
  arrowSize:        4,
  imgSize:          64,
  barMaxWidth:      60,
  barHeight:        8,
  barPadding:       4,
  // PARTICLE CONFIG
  particleRate:     300,    // ms between spawns
  particleSize:     4,      // px radius
  particleDuration: 20000   // ms for full journey root→P
};

// your SMILES definitions
const SMILES = {
  root: "c1ccccc1",
  S1:   "c1cc(O)ccc1",
  S2:   "c1cc(N)ccc1",
  S3:   "c1cc(F)ccc1",
  P1:   "COc1ccccc1",
  P2:   "Oc1ccc(Cl)cc1",
  P3:   "Oc1ccc(C)cc1",
  P4:   "Nc1ccc(Cl)cc1",
  P5:   "Nc1ccc(C)cc1",
  P6:   "CNc1ccccc1",
  P7:   "Brc1ccc(F)cc1",
  P8:   "Clc1ccc(F)cc1",
  P9:   "Cc1ccc(F)cc1"
};

// human-readable names for each node
const NAMES = {
  root: "Benzene",
  S1:   "Phenol",
  S2:   "Aniline",
  S3:   "Fluorobenzene",
  P1:   "Anisole",
  P2:   "4-Chlorophenol",
  P3:   "p-Cresol",
  P4:   "4-Chloroaniline",
  P5:   "p-Toluidine",
  P6:   "N-Methylaniline",
  P7:   "4-Bromofluorobenzene",
  P8:   "4-Chlorofluorobenzene",
  P9:   "Methylfluorobenzene"
};

// terminal molecules
const P_IDS = Object.keys(SMILES).filter(id => id.startsWith("P"));

// store SVGs and descriptor values
let molSvgs     = {};
let descriptors = {};
let descStats   = {};

// normalize so scores sum to 1
function normalizeScores(raw) {
  const sum = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
  return Object.fromEntries(
    Object.entries(raw).map(([k, v]) => [k, v / sum])
  );
}

// draw Sankey + bars + particles + tooltip
function drawSankey(selector, rawScores) {
  const norm = normalizeScores(rawScores);

  // build sankey graph
  const nodes = Object.keys(SMILES).map(id => ({ id }));
  const links = [
    { source: 'root', target: 'S1' },
    { source: 'root', target: 'S2' },
    { source: 'root', target: 'S3' },
    ...['S1', 'S2', 'S3'].flatMap((s, i) => {
      const base = 1 + i * 3;
      return [0, 1, 2].map(j => ({ source: s, target: `P${base + j}` }));
    })
  ].map(l => ({ ...l, value: 1 }));

  // layout
  const sankeyGen = d3.sankey()
    .nodeId(d => d.id)
    .nodeWidth(CFG.nodeWidth)
    .nodePadding(CFG.nodePadding)
    .extent([[1, 1], [CFG.width - 1, CFG.height - 1]]);
  const graph = sankeyGen({
    nodes: nodes.map(d => ({ ...d })),
    links: links.map(d => ({ ...d }))
  });

  // clear & prepare svg
  const svg = d3.select(selector)
    .attr('viewBox', `0 0 ${CFG.width} ${CFG.height}`)
    .selectAll('*').remove() && d3.select(selector);

  // create/ensure tooltip
  let tooltip = d3.select('body').selectAll('div.mol-tooltip').data([0]);
  tooltip = tooltip.enter()
    .append('div')
      .attr('class','mol-tooltip')
      .style('position','absolute')
      .style('pointer-events','none')
      .style('background','white')
      .style('border','1px solid #ccc')
      .style('padding','8px')
      .style('font-size','12px')
      .style('border-radius','4px')
      .style('box-shadow','0 2px 5px rgba(0,0,0,0.15)')
      .style('opacity', 0)
      .style('z-index', 1000)
    .merge(tooltip);

  // particle layer
  const particleLayer = svg.append('g').attr('class','particles');

  // arrow marker
  svg.append('defs').append('marker')
    .attr('id','arrow')
    .attr('markerUnits','strokeWidth')
    .attr('markerWidth',CFG.arrowSize)
    .attr('markerHeight',CFG.arrowSize)
    .attr('refX',CFG.arrowSize)
    .attr('refY',CFG.arrowSize/2)
    .attr('orient','auto')
    .append('path')
      .attr('d',`M0,0 L${CFG.arrowSize},${CFG.arrowSize/2} L0,${CFG.arrowSize}`)
      .attr('fill','#000');

  // draw links
  svg.append('g').selectAll('path')
    .data(graph.links).enter().append('path')
      .attr('class','link')
      .attr('d', d => {
        const y0 = d.source.y0 + (d.source.y1 - d.source.y0)/2;
        const y1 = d.target.y0 + (d.target.y1 - d.target.y0)/2;
        return `M${d.source.x1},${y0} L${d.target.x0},${y1}`;
      })
      .attr('marker-end','url(#arrow)');

  // draw nodes
  const node = svg.append('g').selectAll('g')
    .data(graph.nodes).enter().append('g')
      .attr('class','node')
      .attr('transform', d => `translate(${d.x0},${d.y0})`);

  // molecule icons
  node.append('foreignObject')
    .attr('x', d => (d.x1 - d.x0 - CFG.imgSize) / 2)
    .attr('y', d => (d.y1 - d.y0 - CFG.imgSize) / 2 - CFG.barHeight)
    .attr('width', CFG.imgSize)
    .attr('height', CFG.imgSize)
    .html(d => molSvgs[d.id] || `<div>${d.id}</div>`);

  // performance bars for P nodes
  node.filter(d => d.id.startsWith('P'))
    .each(function(d) {
      const pct = norm[d.id] || 0;
      const barX = (CFG.nodeWidth - CFG.barMaxWidth) / 2;
      const barY = ((d.y1 - d.y0) - CFG.barHeight) - CFG.barPadding;
      const g = d3.select(this).append('g')
        .attr('transform', `translate(${barX},${barY})`);
      g.append('rect')
        .attr('width', CFG.barMaxWidth)
        .attr('height', CFG.barHeight)
        .attr('fill', '#eee');
      g.append('rect')
        .attr('width', CFG.barMaxWidth * pct)
        .attr('height', CFG.barHeight)
        .attr('fill', '#3b82f6');
      g.append('text')
        .attr('x', CFG.barMaxWidth/2)
        .attr('y', CFG.barHeight/2)
        .attr('dy', '0.35em')
        .attr('text-anchor','middle')
        .attr('font-size','8px')
        .text(`${(pct*100).toFixed(0)}%`);
    });

  // tooltip handlers on every node, no SMILES, show descriptors for all
  node
    .on('mouseover', (event, d) => {
      const name = NAMES[d.id] || d.id;
      let html = `<strong>${name}</strong>`;
      if (descriptors[d.id]) {
        const info = descriptors[d.id];
        html +=
          `<br/>Mol. Wt: ${info.amw.toFixed(2)}` +
          `<br/>logP: ${info.logP.toFixed(2)}` +
          `<br/>HBD: ${info.hbd}` +
          `<br/>HBA: ${info.hba}` +
          `<br/>TPSA: ${info.tpsa}` +
          `<br/>RotB: ${info.rotb}`;
      }
      tooltip.html(html)
        .style('left', (event.pageX + 12) + 'px')
        .style('top',  (event.pageY + 12) + 'px')
        .transition().duration(100).style('opacity', 1);
    })
    .on('mousemove', (event) => {
      tooltip
        .style('left', (event.pageX + 12) + 'px')
        .style('top',  (event.pageY + 12) + 'px');
    })
    .on('mouseout', () => {
      tooltip.transition().duration(100).style('opacity', 0);
    });

  // —— PARTICLE SYSTEM ——
  const pathsByP = {};
  P_IDS.forEach(pid => {
    const first  = graph.links.find(l => l.source.id === 'root'
                   && graph.links.some(l2 => l2.source.id === l.target.id && l2.target.id === pid));
    const second = graph.links.find(l => l.source.id === first.target.id && l.target.id === pid);
    const y0 = first.source.y0 + (first.source.y1 - first.source.y0) / 2;
    const y1 = first.target.y0 + (first.target.y1 - first.target.y0) / 2;
    const y2 = second.source.y0 + (second.source.y1 - second.source.y0) / 2;
    const y3 = second.target.y0 + (second.target.y1 - second.target.y0) / 2;
    pathsByP[pid] = [
      { x0: first.source.x1,  y0, x1: first.target.x0, y1 },
      { x0: second.source.x1, y0: y2, x1: second.target.x0, y1: y3 }
    ];
  });

  clearInterval(window._molParticleTimer);
  window._molParticleTimer = setInterval(() => {
    if (Object.values(norm).every(v => v === 0)) return;
    const r = Math.random(), keys = P_IDS;
    let cum = 0, chosen = keys[keys.length - 1];
    for (const k of keys) {
      cum += norm[k] || 0;
      if (r < cum) { chosen = k; break; }
    }
    const segs = pathsByP[chosen];
    const circ = particleLayer.append('circle')
      .attr('r', CFG.particleSize)
      .attr('cx', segs[0].x0)
      .attr('cy', segs[0].y0)
      .attr('fill', '#440154FF')   // Viridis yellow
      .attr('opacity', 0.8);
    circ.transition().duration(CFG.particleDuration)
      .attrTween('transform', function() {
        return t => {
          let x, y;
          if (t < 0.5) {
            const tt = t / 0.5;
            x = segs[0].x0 + (segs[0].x1 - segs[0].x0) * tt;
            y = segs[0].y0 + (segs[0].y1 - segs[0].y0) * tt;
          } else {
            const tt = (t - 0.5) / 0.5;
            x = segs[1].x0 + (segs[1].x1 - segs[1].x0) * tt;
            y = segs[1].y0 + (segs[1].y1 - segs[1].y0) * tt;
          }
          return `translate(${x - segs[0].x0},${y - segs[0].y0})`;
        };
      })
      .remove();
  }, CFG.particleRate);
}

// compute raw scores from sliders
function computeRawScores() {
  const w = {
    mw:   +document.getElementById('w-mw').value,
    logP: +document.getElementById('w-logP').value,
    hbd:  +document.getElementById('w-hbd').value,
    hba:  +document.getElementById('w-hba').value,
    tpsa: +document.getElementById('w-tpsa').value,
    rotb: +document.getElementById('w-rotb').value
  };
  const raw = {};
  P_IDS.forEach(id => {
    const d = descriptors[id];
    raw[id] =
      w.mw   * ((d.amw   - descStats.mw.min)   / descStats.mw.range) +
      w.logP * ((d.logP  - descStats.logP.min) / descStats.logP.range) +
      w.hbd  * ((d.hbd   - descStats.hbd.min)  / descStats.hbd.range) +
      w.hba  * ((d.hba   - descStats.hba.min)  / descStats.hba.range) +
      w.tpsa * ((d.tpsa  - descStats.tpsa.min) / descStats.tpsa.range) +
      w.rotb * ((d.rotb  - descStats.rotb.min) / descStats.rotb.range);
  });
  return raw;
}

// wire sliders
function initSliders() {
  document.querySelectorAll('#weights input[type=range]').forEach(slider => {
    const disp = document.getElementById(slider.id + '-val');
    slider.addEventListener('input', () => {
      disp.textContent = parseFloat(slider.value).toFixed(2);
      updateFlow();
    });
  });
}

// update
function updateFlow() {
  drawSankey('#chart', computeRawScores());
}

// entry
window.initMoleculeFlow = function(selector = '#chart') {
  window.initRDKitModule().then(RDKit => {
    Object.entries(SMILES).forEach(([id, smi]) => {
      const m = RDKit.get_mol(smi);
      if (!m) return;
      molSvgs[id] = m.get_svg(CFG.imgSize, CFG.imgSize);
      // compute and store descriptors for all nodes
      const desc = JSON.parse(m.get_descriptors());
      descriptors[id] = {
        amw:  desc.amw,
        logP: desc.CrippenClogP,
        hbd:  desc.lipinskiHBD,
        hba:  desc.lipinskiHBA,
        tpsa: desc.tpsa,
        rotb: desc.NumRotatableBonds
      };
      m.delete();
    });

    // compute min/max/range for P nodes only (for normalization)
    ['amw','logP','hbd','hba','tpsa','rotb'].forEach(key => {
      const vals = P_IDS.map(id => descriptors[id][key]);
      const min = Math.min(...vals), max = Math.max(...vals);
      descStats[key] = { min, range: (max - min) || 1 };
    });
    // mirror shorthand
    descStats.mw   = descStats.amw;
    descStats.logP = descStats.logP;
    descStats.hbd  = descStats.hbd;
    descStats.hba  = descStats.hba;
    descStats.tpsa = descStats.tpsa;
    descStats.rotb = descStats.rotb;

    initSliders();
    updateFlow();
  });
};
