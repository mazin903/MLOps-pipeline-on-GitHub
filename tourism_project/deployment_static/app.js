const DEFAULT_CONFIG = {
  githubRepo: "mazin903/MLOps-pipeline-on-GitHub",
  datasetRepo: "mazin903/tourism",
  modelRepo: "mazin903/tourism-package-model",
  staticModelFile: "static_model.json",
};

const appConfig = { ...DEFAULT_CONFIG, ...(window.APP_CONFIG || {}) };
const form = document.querySelector("#prediction-form");
const statusLine = document.querySelector("#model-status");
const probabilityEl = document.querySelector("#probability");
const recommendationEl = document.querySelector("#recommendation");
const scoreBar = document.querySelector("#score-bar");
const thresholdMarker = document.querySelector("#threshold-marker");
const metricAccuracy = document.querySelector("#metric-accuracy");
const metricF1 = document.querySelector("#metric-f1");
const metricThreshold = document.querySelector("#metric-threshold");
const driversList = document.querySelector("#drivers-list");

let modelArtifact = null;

function percent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function setEvidenceLinks() {
  document.querySelector("#github-link").href = `https://github.com/${appConfig.githubRepo}`;
  document.querySelector("#dataset-link").href = `https://huggingface.co/datasets/${appConfig.datasetRepo}`;
  document.querySelector("#model-link").href = `https://huggingface.co/${appConfig.modelRepo}`;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function loadModelArtifact() {
  const remoteUrl = `https://huggingface.co/${appConfig.modelRepo}/resolve/main/${appConfig.staticModelFile}`;
  const candidates = [
    { label: "Hugging Face Model Hub", url: remoteUrl },
    { label: "Space artifact", url: appConfig.staticModelFile },
  ];

  for (const candidate of candidates) {
    try {
      const artifact = await fetchJson(candidate.url);
      statusLine.textContent = `Model loaded from ${candidate.label}`;
      return artifact;
    } catch (error) {
      console.warn(`Could not load model from ${candidate.label}:`, error);
    }
  }

  throw new Error("Model artifact is unavailable.");
}

function collectInputs() {
  const data = {};
  const formData = new FormData(form);
  for (const [name, value] of formData.entries()) {
    data[name] = value;
  }
  return data;
}

function numericValue(rawValue, fallback) {
  const parsed = Number(rawValue);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function transformInputs(inputs, artifact) {
  const row = [];

  for (const spec of artifact.numeric) {
    const raw = numericValue(inputs[spec.name], spec.impute);
    row.push(Math.fround((raw - spec.mean) / spec.scale));
  }

  for (const spec of artifact.categorical) {
    const selected = inputs[spec.name] || spec.impute;
    for (const category of spec.categories) {
      row.push(Math.fround(selected === category ? 1 : 0));
    }
  }

  return row;
}

function findChild(node, nodeId) {
  return node.children.find((child) => child.nodeid === nodeId);
}

function evaluateTree(tree, row) {
  let node = tree;
  while (node.leaf === undefined) {
    const featureIndex = Number(node.split.replace("f", ""));
    const value = Math.fround(row[featureIndex]);
    const threshold = Math.fround(Number(node.split_condition));
    const nextNodeId = Number.isNaN(value)
      ? node.missing
      : value < threshold
        ? node.yes
        : node.no;
    node = findChild(node, nextNodeId);
  }
  return Number(node.leaf);
}

function predictProbability(inputs, artifact) {
  const row = transformInputs(inputs, artifact);
  const margin = artifact.trees.reduce((total, tree) => total + evaluateTree(tree, row), 0);
  return 1 / (1 + Math.exp(-margin));
}

function renderMetrics(artifact) {
  const metrics = artifact.metrics || {};
  metricAccuracy.textContent = Number.isFinite(metrics.test_accuracy)
    ? percent(metrics.test_accuracy)
    : "--";
  metricF1.textContent = Number.isFinite(metrics.test_f1) ? metrics.test_f1.toFixed(3) : "--";
  metricThreshold.textContent = Number.isFinite(artifact.threshold)
    ? artifact.threshold.toFixed(2)
    : "--";
  thresholdMarker.style.left = `${Math.min(Math.max(artifact.threshold || 0.5, 0), 1) * 100}%`;
}

function renderDrivers(artifact) {
  const drivers = (artifact.feature_importance || []).slice(0, 5);
  const maxImportance = Math.max(...drivers.map((driver) => driver.importance), 0.01);
  driversList.innerHTML = "";
  for (const driver of drivers) {
    const row = document.createElement("div");
    row.className = "driver-row";

    const label = document.createElement("span");
    label.textContent = driver.feature.replaceAll("_", " ");

    const track = document.createElement("div");
    track.className = "driver-track";
    const bar = document.createElement("i");
    bar.style.width = `${Math.max((driver.importance / maxImportance) * 100, 4)}%`;
    track.appendChild(bar);

    row.append(label, track);
    driversList.appendChild(row);
  }
}

function scoreCurrentLead() {
  if (!modelArtifact) {
    return;
  }

  const probability = predictProbability(collectInputs(), modelArtifact);
  const threshold = Number(modelArtifact.threshold || 0.5);
  const recommendPriority = probability >= threshold;

  probabilityEl.textContent = percent(probability);
  scoreBar.style.width = `${Math.round(probability * 100)}%`;
  document.documentElement.style.setProperty("--score-angle", `${probability * 360}deg`);
  recommendationEl.className = `recommendation ${recommendPriority ? "positive" : "negative"}`;
  recommendationEl.textContent = recommendPriority
    ? "Recommendation: prioritize this customer for a sales conversation."
    : "Recommendation: place this customer in a lower-touch nurture segment.";
}

async function main() {
  setEvidenceLinks();
  try {
    modelArtifact = await loadModelArtifact();
    renderMetrics(modelArtifact);
    renderDrivers(modelArtifact);
    scoreCurrentLead();
    form.addEventListener("input", scoreCurrentLead);
  } catch (error) {
    statusLine.textContent = "Model artifact could not be loaded.";
    recommendationEl.className = "recommendation error";
    recommendationEl.textContent =
      "The model artifact is not available yet. Re-run the GitHub Actions workflow after model registration.";
    console.error(error);
  }
}

main();
