let svg, g, simulation;
let nodes = [], edges = [];
let nodeData = {}, insights = {};
let selectedNode = null;

const graphDiv = d3.select("#graph");
const infoPanel = d3.select("#node-info");
const width = graphDiv.node().clientWidth;
const height = 600;

// Initialize SVG and group
svg = graphDiv.append("svg")
  .attr("width", "100%")
  .attr("height", height)
  .call(d3.zoom().on("zoom", (event) => {
    g.attr("transform", event.transform);
  }));

g = svg.append("g");

// Load JSON data
Promise.all([
  d3.json("/dashboard_data/graph.json"),
  d3.json("/dashboard_data/node_data.json"),
  d3.json("/dashboard_data/llm_insights.json")
]).then(([loadedEdges, loadedNodes, loadedInsights]) => {
  edges = loadedEdges;
  nodeData = loadedNodes;
  insights = loadedInsights;

  console.log("graph.json loaded:", edges.length, "edges");
  console.log("node_data.json loaded:", Object.keys(nodeData).length, "nodes");
  console.log("llm_insights.json loaded:", Object.keys(insights).length, "insights");

  // Auto-load top 10 anomalies
  loadTopAnomalies(10);
}).catch(err => {
  console.error("Error loading JSON:", err);
});

// Helper: load top N anomalies
function loadTopAnomalies(topN) {
  const anomalousNodes = Object.keys(nodeData)
    .filter(id => nodeData[id].anomaly_label === 1)  // <-- use anomaly_label
    .slice(0, topN);

  nodes = anomalousNodes.map(id => ({
    id,
    anomaly_score: nodeData[id].anomaly_score,
    is_anomalous: nodeData[id].anomaly_label === 1, // keep a boolean for coloring
    has_llm: nodeData[id].has_llm
  }));

  // Filter edges between selected nodes
  const nodeSet = new Set(nodes.map(n => n.id));
  const filteredEdges = edges.filter(
    e => nodeSet.has(String(e.source)) && nodeSet.has(String(e.target))
  );
  edges = filteredEdges;

  console.log("Nodes to render:", nodes.map(n => n.id));
  console.log("Edges to render:", edges.length);

  updateGraph();
}


// Update the D3 graph
function updateGraph() {
  if (simulation) simulation.stop();

  // LINKS
  const link = g.selectAll("line").data(edges, d => d.source + "-" + d.target);
  link.join(
    enter => enter.append("line").attr("stroke", "#555").attr("stroke-opacity", 0.6).attr("stroke-width", 1),
    update => update,
    exit => exit.remove()
  );

  // NODES
  const nodeSelection = g.selectAll("circle").data(nodes, d => d.id);
  nodeSelection.join(
    enter => enter.append("circle")
      .attr("r", 5)
      .attr("fill", d => d.is_anomalous ? "#e74c3c" : "#3498db")
      .attr("stroke", d => d.has_llm ? "#fff" : "#888")
      .attr("stroke-width", d => d.has_llm ? 1.5 : 0.5)
      .on("click", (event, d) => {
        selectedNode = d;
        const data = nodeData[d.id];
        const insight = insights[d.id]?.llm_reasoning || "No LLM explanation available.";
        infoPanel.html(`
          <strong>Node ID:</strong> ${d.id}<br>
          <strong>Score:</strong> ${d.anomaly_score?.toFixed(5)}<br>
          <strong>Anomalous:</strong> ${d.is_anomalous}<br><br>
          <strong>LLM Insight:</strong><br><div class="insight-text">${insight}</div>
        `);
      })
      .append("title").text(d => `Node ${d.id}`),
    update => update,
    exit => exit.remove()
  );

  // Force simulation
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(d => d.id).distance(30))
    .force("charge", d3.forceManyBody().strength(-80))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .on("tick", () => {
      g.selectAll("line")
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      g.selectAll("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
    });
}

// Button handlers
document.addEventListener("DOMContentLoaded", () => {
  d3.select("#load-anomalies").on("click", () => {
    const topN = +d3.select("#top-n").property("value") || 10;
    loadTopAnomalies(topN);
  });

  d3.select("#expand-node").on("click", () => {
    if (!selectedNode) {
      console.log("No node selected");
      return;
    }
    console.log(`Expand node ${selectedNode.id} by k hops (placeholder)`);
    alert("Expand functionality not implemented yet."); // placeholder
  });
});
