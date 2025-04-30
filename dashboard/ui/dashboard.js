const graphDiv = d3.select("#graph");
const infoPanel = d3.select("#node-info");

Promise.all([
  d3.json("../dashboard_data/graph.json"),
  d3.json("../dashboard_data/node_data.json"),
  d3.json("../dashboard_data/llm_insights.json")
]).then(([edges, nodeData, insights]) => {
  const nodes = Object.keys(nodeData).map(id => ({
    id,
    anomaly_score: nodeData[id].anomaly_score,
    is_anomalous: nodeData[id].is_anomalous,
    has_llm: nodeData[id].has_llm
  }));

  const width = graphDiv.node().clientWidth;
  const height = 600;

  const svg = graphDiv.append("svg")
    .attr("width", "100%")
    .attr("height", height)
    .call(d3.zoom().on("zoom", (event) => {
      g.attr("transform", event.transform);
    }))
    .append("g");

  const g = svg.append("g");

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(d => d.id).distance(30))
    .force("charge", d3.forceManyBody().strength(-80))
    .force("center", d3.forceCenter(width / 2, height / 2));

  const link = g.append("g")
    .attr("stroke", "#555")
    .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(edges)
    .join("line")
    .attr("stroke-width", 1);

  const node = g.append("g")
    .attr("stroke", "#000")
    .attr("stroke-width", 0.5)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", 5)
    .attr("fill", d => d.is_anomalous ? "#e74c3c" : "#3498db")
    .attr("stroke", d => d.has_llm ? "#fff" : "#888")
    .attr("stroke-width", d => d.has_llm ? 1.5 : 0.5)
    .on("click", (event, d) => {
      const data = nodeData[d.id];
      const insight = insights[d.id]?.llm_reasoning || "No LLM explanation available.";
      infoPanel.html(`
        <strong>Node ID:</strong> ${d.id}<br>
        <strong>Score:</strong> ${d.anomaly_score?.toFixed(5)}<br>
        <strong>Anomalous:</strong> ${d.is_anomalous}<br><br>
        <strong>LLM Insight:</strong><br><div class="insight-text">${insight}</div>
      `);
    });

  node.append("title").text(d => `Node ${d.id}`);

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);
  });
});
