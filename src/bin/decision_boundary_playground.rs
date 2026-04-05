use neural_network_study::NeuralNetwork;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

const OUTPUT_PATH: &str = "target/decision-boundary-playground.html";
const GRID_SIZE: usize = 180;

#[derive(Clone)]
struct Sample {
    x: f64,
    y: f64,
    label: f64,
}

#[derive(Clone)]
struct DatasetSpec {
    slug: &'static str,
    name: &'static str,
    description: &'static str,
    hidden_size: usize,
    learning_rate: f64,
    epochs: usize,
    samples: Vec<Sample>,
}

#[derive(Serialize)]
struct DatasetView {
    slug: &'static str,
    name: &'static str,
    description: &'static str,
    hidden_size: usize,
    learning_rate: f64,
    epochs: usize,
    accuracy: f64,
    grid_size: usize,
    boundary: Vec<u8>,
    samples: Vec<SampleView>,
}

#[derive(Serialize)]
struct SampleView {
    x: f64,
    y: f64,
    label: f64,
    prediction: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let datasets = vec![
        make_linear_dataset(),
        make_xor_dataset(),
        make_ring_dataset(),
    ];

    let views: Vec<DatasetView> = datasets
        .into_iter()
        .enumerate()
        .map(|(index, dataset)| train_and_render(dataset, index as u64 + 1))
        .collect::<Result<_, _>>()?;

    let output_path = PathBuf::from(OUTPUT_PATH);
    write_playground(&output_path, &views)?;

    println!(
        "Decision boundary playground written to {}",
        output_path.canonicalize().unwrap_or(output_path).display()
    );

    Ok(())
}

fn train_and_render(dataset: DatasetSpec, seed: u64) -> Result<DatasetView, Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut network = NeuralNetwork::new(2, dataset.hidden_size, 1, Some(&mut rng))?;
    network.set_learning_rate(dataset.learning_rate);

    let mut shuffled = dataset.samples.clone();
    for _ in 0..dataset.epochs {
        shuffled.shuffle(&mut rng);
        for sample in &shuffled {
            network.train(vec![sample.x, sample.y], vec![sample.label])?;
        }
    }

    let mut correct = 0usize;
    let mut sample_views = Vec::with_capacity(dataset.samples.len());
    for sample in &dataset.samples {
        let prediction = network.predict(vec![sample.x, sample.y])?[0];
        if classify(prediction) == classify(sample.label) {
            correct += 1;
        }
        sample_views.push(SampleView {
            x: sample.x,
            y: sample.y,
            label: sample.label,
            prediction,
        });
    }

    let mut boundary = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
    for grid_y in 0..GRID_SIZE {
        for grid_x in 0..GRID_SIZE {
            let x = grid_x as f64 / (GRID_SIZE - 1) as f64;
            let y = 1.0 - grid_y as f64 / (GRID_SIZE - 1) as f64;
            let prediction = network.predict(vec![x, y])?[0];
            boundary.push((prediction.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }

    Ok(DatasetView {
        slug: dataset.slug,
        name: dataset.name,
        description: dataset.description,
        hidden_size: dataset.hidden_size,
        learning_rate: dataset.learning_rate,
        epochs: dataset.epochs,
        accuracy: correct as f64 / dataset.samples.len() as f64,
        grid_size: GRID_SIZE,
        boundary,
        samples: sample_views,
    })
}

fn write_playground(path: &Path, datasets: &[DatasetView]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let dataset_json = serde_json::to_string(datasets)?;
    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Decision Boundary Playground</title>
  <style>
    :root {{
      --bg: #f5efe3;
      --panel: rgba(255, 252, 245, 0.92);
      --ink: #1f2430;
      --muted: #5f6572;
      --accent: #cc5a34;
      --accent-2: #17806d;
      --border: rgba(31, 36, 48, 0.12);
      --shadow: 0 20px 60px rgba(72, 47, 24, 0.14);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(204, 90, 52, 0.16), transparent 32%),
        radial-gradient(circle at bottom right, rgba(23, 128, 109, 0.2), transparent 28%),
        linear-gradient(160deg, #f8f2e8 0%, #f0e2c7 48%, #efe7d7 100%);
      padding: 32px 18px 56px;
    }}

    .shell {{
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      gap: 24px;
    }}

    .hero {{
      display: grid;
      gap: 10px;
    }}

    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.75rem;
      color: var(--accent-2);
      font-weight: 700;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(2.2rem, 5vw, 4.6rem);
      line-height: 0.95;
      max-width: 12ch;
    }}

    .lede {{
      margin: 0;
      max-width: 60ch;
      color: var(--muted);
      font-size: 1.06rem;
      line-height: 1.6;
    }}

    .layout {{
      display: grid;
      gap: 24px;
      grid-template-columns: minmax(0, 320px) minmax(0, 1fr);
      align-items: start;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 26px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}

    .controls {{
      padding: 24px;
      display: grid;
      gap: 18px;
    }}

    .label {{
      display: grid;
      gap: 8px;
      font-size: 0.9rem;
      color: var(--muted);
    }}

    select {{
      width: 100%;
      border: 1px solid rgba(31, 36, 48, 0.18);
      border-radius: 14px;
      padding: 12px 14px;
      background: #fffdfa;
      color: var(--ink);
      font: inherit;
    }}

    .dataset-title {{
      margin: 0;
      font-size: 1.75rem;
      line-height: 1.05;
    }}

    .dataset-description {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}

    .stat {{
      padding: 14px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(31, 36, 48, 0.08);
    }}

    .stat-label {{
      display: block;
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}

    .stat-value {{
      font-size: 1.2rem;
      font-weight: 700;
    }}

    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      font-size: 0.9rem;
    }}

    .legend-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}

    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
      border: 2px solid rgba(31, 36, 48, 0.16);
    }}

    .dot.class0 {{
      background: #f8df78;
    }}

    .dot.class1 {{
      background: #1389a0;
    }}

    .canvas-panel {{
      padding: 20px;
      display: grid;
      gap: 14px;
    }}

    .canvas-frame {{
      position: relative;
      aspect-ratio: 1;
      width: 100%;
      overflow: hidden;
      border-radius: 24px;
      border: 1px solid rgba(31, 36, 48, 0.14);
      background:
        linear-gradient(rgba(255,255,255,0.16) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.16) 1px, transparent 1px),
        linear-gradient(180deg, rgba(20, 25, 31, 0.04), rgba(20, 25, 31, 0.08));
      background-size: 10% 10%, 10% 10%, 100% 100%;
    }}

    canvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}

    .caption {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.95rem;
    }}

    code {{
      font-family: "Cascadia Code", "Fira Code", monospace;
      background: rgba(31, 36, 48, 0.06);
      border-radius: 8px;
      padding: 0.12rem 0.34rem;
    }}

    @media (max-width: 920px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}

      h1 {{
        max-width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <span class="eyebrow">Neural Network Study</span>
      <h1>Decision Boundary Playground</h1>
      <p class="lede">
        Tiny feedforward networks can still produce wonderfully strange maps of confidence.
        This page trains the library on a few 2D datasets, then paints the model's prediction
        across the entire plane so we can see what it learned and where it still bends awkwardly.
      </p>
    </section>

    <section class="layout">
      <aside class="panel controls">
        <label class="label" for="dataset-select">
          Dataset
          <select id="dataset-select"></select>
        </label>

        <div>
          <h2 class="dataset-title" id="dataset-title"></h2>
          <p class="dataset-description" id="dataset-description"></p>
        </div>

        <div class="stats">
          <div class="stat">
            <span class="stat-label">Training Accuracy</span>
            <span class="stat-value" id="accuracy"></span>
          </div>
          <div class="stat">
            <span class="stat-label">Hidden Layer</span>
            <span class="stat-value" id="hidden-size"></span>
          </div>
          <div class="stat">
            <span class="stat-label">Epochs</span>
            <span class="stat-value" id="epochs"></span>
          </div>
          <div class="stat">
            <span class="stat-label">Learning Rate</span>
            <span class="stat-value" id="learning-rate"></span>
          </div>
        </div>

        <div class="legend">
          <span class="legend-chip"><span class="dot class0"></span> Class 0</span>
          <span class="legend-chip"><span class="dot class1"></span> Class 1</span>
          <span class="legend-chip">Background shows model confidence</span>
        </div>
      </aside>

      <section class="panel canvas-panel">
        <div class="canvas-frame">
          <canvas id="playground-canvas" width="900" height="900"></canvas>
        </div>
        <p class="caption">
          The colored field is sampled from the network on a dense grid. Each point is a training
          sample placed in normalized coordinates <code>[0, 1] x [0, 1]</code>. Crisp boundaries
          mean the model found a clean split; muddy or wobbly transitions usually reveal the limits
          of a tiny one-hidden-layer network.
        </p>
      </section>
    </section>
  </main>

  <script>
    const datasets = {dataset_json};
    const select = document.getElementById('dataset-select');
    const title = document.getElementById('dataset-title');
    const description = document.getElementById('dataset-description');
    const accuracy = document.getElementById('accuracy');
    const hiddenSize = document.getElementById('hidden-size');
    const epochs = document.getElementById('epochs');
    const learningRate = document.getElementById('learning-rate');
    const canvas = document.getElementById('playground-canvas');
    const ctx = canvas.getContext('2d');

    const classColors = [
      [248, 223, 120],
      [19, 137, 160],
    ];

    for (const dataset of datasets) {{
      const option = document.createElement('option');
      option.value = dataset.slug;
      option.textContent = dataset.name;
      select.appendChild(option);
    }}

    function lerp(a, b, t) {{
      return a + (b - a) * t;
    }}

    function fillBoundary(dataset) {{
      const image = ctx.createImageData(canvas.width, canvas.height);
      for (let y = 0; y < canvas.height; y++) {{
        const sampleY = Math.floor(y * dataset.grid_size / canvas.height);
        for (let x = 0; x < canvas.width; x++) {{
          const sampleX = Math.floor(x * dataset.grid_size / canvas.width);
          const t = dataset.boundary[sampleY * dataset.grid_size + sampleX] / 255;
          const red = Math.round(lerp(classColors[0][0], classColors[1][0], t));
          const green = Math.round(lerp(classColors[0][1], classColors[1][1], t));
          const blue = Math.round(lerp(classColors[0][2], classColors[1][2], t));
          const offset = (y * canvas.width + x) * 4;
          image.data[offset] = red;
          image.data[offset + 1] = green;
          image.data[offset + 2] = blue;
          image.data[offset + 3] = 255;
        }}
      }}
      ctx.putImageData(image, 0, 0);
    }}

    function drawSamples(dataset) {{
      for (const sample of dataset.samples) {{
        const x = sample.x * canvas.width;
        const y = (1 - sample.y) * canvas.height;
        const color = sample.label >= 0.5 ? classColors[1] : classColors[0];
        const outlineAlpha = Math.max(0.2, Math.abs(sample.prediction - sample.label));

        ctx.beginPath();
        ctx.arc(x, y, 7.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${{color[0]}}, ${{color[1]}}, ${{color[2]}})`;
        ctx.fill();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = `rgba(31, 36, 48, ${{0.28 + outlineAlpha * 0.72}})`;
        ctx.stroke();
      }}
    }}

    function render(dataset) {{
      title.textContent = dataset.name;
      description.textContent = dataset.description;
      accuracy.textContent = `${{(dataset.accuracy * 100).toFixed(1)}}%`;
      hiddenSize.textContent = `${{dataset.hidden_size}} neurons`;
      epochs.textContent = dataset.epochs.toLocaleString();
      learningRate.textContent = dataset.learning_rate.toFixed(2);

      fillBoundary(dataset);
      drawSamples(dataset);
    }}

    select.addEventListener('change', () => {{
      const dataset = datasets.find((entry) => entry.slug === select.value);
      if (dataset) {{
        render(dataset);
      }}
    }});

    select.value = datasets[0].slug;
    render(datasets[0]);
  </script>
</body>
</html>
"#
    );

    fs::write(path, html)?;
    Ok(())
}

fn classify(value: f64) -> u8 {
    if value >= 0.5 { 1 } else { 0 }
}

fn make_linear_dataset() -> DatasetSpec {
    let mut rng = StdRng::seed_from_u64(11);
    let mut samples = Vec::with_capacity(260);
    for _ in 0..260 {
        let x = rng.random::<f64>();
        let y = rng.random::<f64>();
        let boundary = 0.72 - (0.58 * x);
        let label = if y > boundary { 1.0 } else { 0.0 };
        samples.push(Sample { x, y, label });
    }

    DatasetSpec {
        slug: "linear-slope",
        name: "Linear Slope",
        description: "A gentle warm-up. The classes are separated by a slanted line, so even a tiny network should carve out a crisp split quickly.",
        hidden_size: 6,
        learning_rate: 0.35,
        epochs: 2_000,
        samples,
    }
}

fn make_xor_dataset() -> DatasetSpec {
    let mut rng = StdRng::seed_from_u64(29);
    let mut samples = Vec::with_capacity(320);
    for _ in 0..320 {
        let x = rng.random::<f64>();
        let y = rng.random::<f64>();
        let label = if (x > 0.5) ^ (y > 0.5) { 1.0 } else { 0.0 };
        samples.push(Sample { x, y, label });
    }

    DatasetSpec {
        slug: "xor-quadrants",
        name: "XOR Quadrants",
        description: "The classic nonlinear puzzle. No single line can solve it, so the hidden layer has to bend space into two diagonal islands.",
        hidden_size: 10,
        learning_rate: 0.45,
        epochs: 6_000,
        samples,
    }
}

fn make_ring_dataset() -> DatasetSpec {
    let mut rng = StdRng::seed_from_u64(47);
    let mut samples = Vec::with_capacity(360);
    for _ in 0..360 {
        let x = rng.random::<f64>();
        let y = rng.random::<f64>();
        let dx = x - 0.5;
        let dy = y - 0.5;
        let radius = (dx * dx + dy * dy).sqrt();
        let label = if radius > 0.2 && radius < 0.34 { 1.0 } else { 0.0 };
        samples.push(Sample { x, y, label });
    }

    DatasetSpec {
        slug: "ring",
        name: "Ring Detector",
        description: "Points inside a narrow circular band count as class 1. This is a nice stress test for a small one-hidden-layer network because the target region wraps around itself.",
        hidden_size: 12,
        learning_rate: 0.32,
        epochs: 8_000,
        samples,
    }
}
