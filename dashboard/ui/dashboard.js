const width = window.innerWidth;
const height = window.innerHeight;

const svg = d3.select("svg")
    .attr("viewBox", [0, 0, width, height])
    .call(d3.zoom().on("zoom", function (event) {
        svg.select('g').attr("transform", event.transform);
    }))
    .append("g");

let simulation;

Promise.all([
    d3.json("dashboard_data/graph.json"),
    d3.json("dashboard_data/node_data.json"),
    d3.json("dashboard_data/llm_insights.json")
]).then(([edges, nodesMeta, insights]) => {
    const nodes = Object.keys(nodesMeta).map(id => ({
        id,
        ...nodesMeta[id]
    }));

    const links = edges.map(d => ({ source: d.source, target: d.target }));

    const color = d => d.has_llm ? "#00bfff" : (d.is_anomalous ? "#ff4d4d" : "#aaaaaa");

    simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(30))
        .force("charge", d3.forceManyBody().strength(-100))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .attr("stroke", "#444")
        .attr("stroke-opacity", 0.6)
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke-width", 1);

    const node = svg.append("g")
        .selectAll("circle")
        .data(nodes)
        .join("circle")
        .attr("r", 4)
        .attr("fill", d => color(d))
        .on("click", (event, d) => showNodeDetails(d, insights))
        .append("title")
        .text(d => d.id);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        svg.selectAll("circle")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    });
});

function showNodeDetails(node, insights) {
    const sidebar = document.getElementById("node-details");
    sidebar.innerHTML = `
        <h2>Node ID: ${node.id}</h2>
        <p><strong>Anomaly Score:</strong> ${node.anomaly_score?.toFixed(4)}</p>
        <p><strong>Is Anomalous:</strong> ${node.is_anomalous ? "✅ Yes" : "❌ No"}</p>
        <h3>Feature Values</h3>
        <pre>${JSON.stringify(node.feature_values, null, 2)}</pre>
        <h3>LLM Insight</h3>
        <p>${insights[node.id]?.llm_reasoning || "No LLM interpretation available."}</p>
    `;
}
