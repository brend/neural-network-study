use neural_network_study::NeuralNetwork;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    env,
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

const DEFAULT_SEED: u64 = 7;
const DEFAULT_GENERATIONS: usize = 150;
const DEFAULT_POPULATION: usize = 200;
const DEFAULT_SAVE_MODEL_PATH: &str = "target/flappy-champion.json";
const DEFAULT_HTML_OUTPUT_PATH: &str = "target/flappy-evolution-playground.html";

const INPUT_SIZE: usize = 5;
const HIDDEN_SIZE: usize = 8;
const OUTPUT_SIZE: usize = 1;
const ELITE_COUNT: usize = 20;
const MUTATION_RATE: f64 = 0.08;

const MAX_SIMULATION_FRAMES: u32 = 2_500;
const MAX_REPLAY_FRAMES: usize = 420;

const BIRD_X: f64 = 0.24;
const BIRD_RADIUS: f64 = 0.02;
const BIRD_START_Y: f64 = 0.5;
const GRAVITY: f64 = 0.0032;
const FLAP_VELOCITY: f64 = -0.03;
const MIN_VELOCITY: f64 = -0.09;
const MAX_VELOCITY: f64 = 0.09;

const PIPE_SPEED: f64 = 0.009;
const PIPE_WIDTH: f64 = 0.12;
const PIPE_GAP_HEIGHT: f64 = 0.27;
const PIPE_SPACING: f64 = 0.46;
const PIPE_MIN_GAP_CENTER_Y: f64 = 0.20;
const PIPE_MAX_GAP_CENTER_Y: f64 = 0.80;

const PIPE_PASS_BONUS: f64 = 250.0;
const ALIGNMENT_WEIGHT: f64 = 0.4;

#[derive(Clone, Debug)]
struct Config {
    seed: u64,
    generations: usize,
    population: usize,
    save_model: PathBuf,
    load_model: Option<PathBuf>,
    html_output: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            generations: DEFAULT_GENERATIONS,
            population: DEFAULT_POPULATION,
            save_model: PathBuf::from(DEFAULT_SAVE_MODEL_PATH),
            load_model: None,
            html_output: PathBuf::from(DEFAULT_HTML_OUTPUT_PATH),
        }
    }
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for CliError {}

#[derive(Clone, Debug)]
struct BirdState {
    y: f64,
    velocity: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Pipe {
    x: f64,
    gap_center_y: f64,
    passed: bool,
}

#[derive(Clone, Debug)]
struct SimulationState {
    bird: BirdState,
    pipes: Vec<Pipe>,
    frame: u32,
    pipes_passed: u32,
    alive: bool,
}

impl SimulationState {
    fn new(rng: &mut StdRng) -> Self {
        let first = Pipe {
            x: 0.92,
            gap_center_y: random_gap_center(rng),
            passed: false,
        };
        let second = Pipe {
            x: first.x + PIPE_SPACING,
            gap_center_y: random_gap_center(rng),
            passed: false,
        };

        Self {
            bird: BirdState {
                y: BIRD_START_Y,
                velocity: 0.0,
            },
            pipes: vec![first, second],
            frame: 0,
            pipes_passed: 0,
            alive: true,
        }
    }
}

#[derive(Clone, Debug)]
struct Evaluation {
    fitness: f64,
    frames_survived: u32,
    pipes_passed: u32,
    replay: Vec<ReplayFrame>,
}

#[derive(Clone)]
struct TrainingOutcome {
    view: PlaybackView,
    best_network: NeuralNetwork,
}

#[derive(Clone, Debug, Serialize)]
struct PipeSnapshot {
    x: f32,
    gap_center_y: f32,
}

#[derive(Clone, Debug, Serialize)]
struct ReplayFrame {
    bird_y: f32,
    bird_velocity: f32,
    pipes_passed: u32,
    alive: bool,
    pipes: Vec<PipeSnapshot>,
}

#[derive(Clone, Serialize)]
struct GenerationView {
    generation: usize,
    best_fitness: f64,
    average_fitness: f64,
    best_pipes_passed: u32,
    best_survival_frames: u32,
    replay: Vec<ReplayFrame>,
}

#[derive(Clone, Serialize)]
struct MetadataView {
    seed: u64,
    population: usize,
    generations: usize,
    elite_count: usize,
    mutation_rate: f64,
    input_size: usize,
    layer_sizes: Vec<usize>,
    output_size: usize,
    bird_x: f64,
    bird_radius: f64,
    pipe_width: f64,
    pipe_gap_height: f64,
}

#[derive(Clone, Serialize)]
struct PlaybackView {
    metadata: MetadataView,
    generations: Vec<GenerationView>,
    overall_best_fitness: f64,
    overall_best_generation: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args(env::args())?;
    validate_config(&config)?;

    let loaded_model = match &config.load_model {
        Some(path) => Some(load_model(path)?),
        None => None,
    };

    let outcome = run_training(&config, loaded_model)?;

    write_model_checkpoint(&config.save_model, &outcome.best_network)?;
    write_html_report(&config.html_output, &outcome.view)?;

    println!(
        "Flappy evolution report written to {}",
        config
            .html_output
            .canonicalize()
            .unwrap_or_else(|_| config.html_output.clone())
            .display()
    );
    println!(
        "Champion model written to {}",
        config
            .save_model
            .canonicalize()
            .unwrap_or_else(|_| config.save_model.clone())
            .display()
    );

    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, CliError> {
    let mut config = Config::default();
    let mut iter = args.into_iter();
    iter.next();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--seed" => {
                let value = iter
                    .next()
                    .ok_or_else(|| CliError("missing value for --seed".to_string()))?;
                config.seed = value
                    .parse::<u64>()
                    .map_err(|_| CliError(format!("invalid --seed value: {value}")))?;
            }
            "--generations" => {
                let value = iter
                    .next()
                    .ok_or_else(|| CliError("missing value for --generations".to_string()))?;
                config.generations = value
                    .parse::<usize>()
                    .map_err(|_| CliError(format!("invalid --generations value: {value}")))?;
            }
            "--population" => {
                let value = iter
                    .next()
                    .ok_or_else(|| CliError("missing value for --population".to_string()))?;
                config.population = value
                    .parse::<usize>()
                    .map_err(|_| CliError(format!("invalid --population value: {value}")))?;
            }
            "--save-model" => {
                let value = iter
                    .next()
                    .ok_or_else(|| CliError("missing value for --save-model".to_string()))?;
                config.save_model = PathBuf::from(value);
            }
            "--load-model" => {
                let value = iter
                    .next()
                    .ok_or_else(|| CliError("missing value for --load-model".to_string()))?;
                config.load_model = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                return Err(CliError(help_text()));
            }
            _ => {
                return Err(CliError(format!(
                    "unknown argument: {arg}\n\n{}",
                    help_text()
                )));
            }
        }
    }

    Ok(config)
}

fn help_text() -> String {
    [
        "Usage:",
        "  cargo run --bin flappy_evolution_playground -- [options]",
        "",
        "Options:",
        "  --seed <u64>            RNG seed (default: 7)",
        "  --generations <usize>   Number of generations (default: 150)",
        "  --population <usize>    Population size (default: 200)",
        "  --save-model <path>     Output model JSON (default: target/flappy-champion.json)",
        "  --load-model <path>     Warm-start from a saved model JSON",
        "  --help, -h              Show this help",
    ]
    .join("\n")
}

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    if config.population == 0 {
        return Err(Box::new(CliError(
            "population must be greater than zero".to_string(),
        )));
    }
    if config.generations == 0 {
        return Err(Box::new(CliError(
            "generations must be greater than zero".to_string(),
        )));
    }
    Ok(())
}

fn load_model(path: &Path) -> Result<NeuralNetwork, Box<dyn Error>> {
    let source = fs::read_to_string(path)?;
    let model: NeuralNetwork = serde_json::from_str(&source)?;

    let output = model.predict(vec![0.0; INPUT_SIZE]).map_err(|err| {
        CliError(format!(
            "model at {} is incompatible with expected input size {}: {err}",
            path.display(),
            INPUT_SIZE
        ))
    })?;

    if output.len() != OUTPUT_SIZE {
        return Err(Box::new(CliError(format!(
            "model at {} is incompatible with expected output size {}",
            path.display(),
            OUTPUT_SIZE
        ))));
    }

    Ok(model)
}

fn run_training(
    config: &Config,
    loaded_model: Option<NeuralNetwork>,
) -> Result<TrainingOutcome, Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let elite_count = ELITE_COUNT.min(config.population).max(1);

    let mut population = initialize_population(config.population, &mut rng, loaded_model)?;
    let mut generation_views = Vec::with_capacity(config.generations);

    let mut overall_best_fitness = f64::NEG_INFINITY;
    let mut overall_best_generation = 1usize;
    let mut overall_best_network = population[0].clone();

    for generation_idx in 0..config.generations {
        let scenario_seed = generation_seed(config.seed, generation_idx);
        let mut scored = Vec::with_capacity(population.len());

        for network in population {
            let evaluation = evaluate_network(&network, scenario_seed, false)?;
            scored.push((network, evaluation));
        }

        scored.sort_by(|left, right| {
            right
                .1
                .fitness
                .partial_cmp(&left.1.fitness)
                .unwrap_or(Ordering::Equal)
        });

        let best_network = scored[0].0.clone();
        let best_evaluation = &scored[0].1;
        let total_fitness: f64 = scored.iter().map(|(_, eval)| eval.fitness).sum();
        let average_fitness = total_fitness / scored.len() as f64;

        let replay_eval = evaluate_network(&best_network, scenario_seed, true)?;

        generation_views.push(GenerationView {
            generation: generation_idx + 1,
            best_fitness: best_evaluation.fitness,
            average_fitness,
            best_pipes_passed: best_evaluation.pipes_passed,
            best_survival_frames: best_evaluation.frames_survived,
            replay: replay_eval.replay,
        });

        if best_evaluation.fitness > overall_best_fitness {
            overall_best_fitness = best_evaluation.fitness;
            overall_best_generation = generation_idx + 1;
            overall_best_network = best_network.clone();
        }

        let elites: Vec<NeuralNetwork> = scored
            .iter()
            .take(elite_count)
            .map(|(network, _)| network.clone())
            .collect();

        let mut next_population = elites.clone();
        while next_population.len() < config.population {
            let parent_index = rng.random_range(0..elites.len());
            let mut child = elites[parent_index].clone();
            child.mutate(&mut rng, MUTATION_RATE);
            next_population.push(child);
        }

        population = next_population;
    }

    Ok(TrainingOutcome {
        view: PlaybackView {
            metadata: MetadataView {
                seed: config.seed,
                population: config.population,
                generations: config.generations,
                elite_count,
                mutation_rate: MUTATION_RATE,
                input_size: INPUT_SIZE,
                layer_sizes: overall_best_network.layer_sizes().to_vec(),
                output_size: OUTPUT_SIZE,
                bird_x: BIRD_X,
                bird_radius: BIRD_RADIUS,
                pipe_width: PIPE_WIDTH,
                pipe_gap_height: PIPE_GAP_HEIGHT,
            },
            generations: generation_views,
            overall_best_fitness,
            overall_best_generation,
        },
        best_network: overall_best_network,
    })
}

fn initialize_population(
    population_size: usize,
    rng: &mut StdRng,
    loaded_model: Option<NeuralNetwork>,
) -> Result<Vec<NeuralNetwork>, Box<dyn Error>> {
    let mut population = Vec::with_capacity(population_size);

    if let Some(model) = loaded_model {
        population.push(model);
    }

    while population.len() < population_size {
        population.push(NeuralNetwork::new(
            vec![INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE],
            Some(rng),
        )?);
    }

    Ok(population)
}

fn evaluate_network(
    network: &NeuralNetwork,
    scenario_seed: u64,
    capture_replay: bool,
) -> Result<Evaluation, Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(scenario_seed);
    let mut state = SimulationState::new(&mut rng);

    let mut alignment_accumulator = 0.0;
    let mut replay = if capture_replay {
        Vec::with_capacity(MAX_REPLAY_FRAMES)
    } else {
        Vec::new()
    };

    if capture_replay {
        replay.push(snapshot(&state));
    }

    while state.alive && state.frame < MAX_SIMULATION_FRAMES {
        let inputs = build_inputs(&state);
        let output = network.predict(inputs.to_vec())?;
        let flap = output[0] >= 0.5;
        alignment_accumulator += step_simulation(&mut state, flap, &mut rng);

        if capture_replay && replay.len() < MAX_REPLAY_FRAMES {
            replay.push(snapshot(&state));
        }
    }

    Ok(Evaluation {
        fitness: compute_fitness(state.frame, state.pipes_passed, alignment_accumulator),
        frames_survived: state.frame,
        pipes_passed: state.pipes_passed,
        replay,
    })
}

fn build_inputs(state: &SimulationState) -> [f64; INPUT_SIZE] {
    let next_pipe = next_pipe(state);
    let horizontal_distance = (next_pipe.x - BIRD_X).max(0.0);
    let vertical_offset = next_pipe.gap_center_y - state.bird.y;

    [
        clamp01(state.bird.y),
        normalize_velocity(state.bird.velocity),
        normalize_horizontal_distance(horizontal_distance),
        clamp01(next_pipe.gap_center_y),
        normalize_vertical_offset(vertical_offset),
    ]
}

fn step_simulation(state: &mut SimulationState, flap: bool, rng: &mut StdRng) -> f64 {
    if !state.alive {
        return 0.0;
    }

    if flap {
        state.bird.velocity = FLAP_VELOCITY;
    }

    state.bird.velocity = (state.bird.velocity + GRAVITY).clamp(MIN_VELOCITY, MAX_VELOCITY);
    state.bird.y += state.bird.velocity;
    state.frame += 1;

    advance_pipes(state, rng);

    let alignment = alignment_score(state);

    if has_collision(state) {
        state.alive = false;
    }

    alignment
}

fn advance_pipes(state: &mut SimulationState, rng: &mut StdRng) {
    for pipe in &mut state.pipes {
        pipe.x -= PIPE_SPEED;

        if !pipe.passed && pipe.x + PIPE_WIDTH / 2.0 < BIRD_X {
            pipe.passed = true;
            state.pipes_passed += 1;
        }
    }

    state.pipes.retain(|pipe| pipe.x + PIPE_WIDTH / 2.0 > -0.2);

    let rightmost_x = state.pipes.last().map(|pipe| pipe.x).unwrap_or(0.9);
    if rightmost_x < 1.0 + PIPE_SPACING {
        state.pipes.push(Pipe {
            x: rightmost_x + PIPE_SPACING,
            gap_center_y: random_gap_center(rng),
            passed: false,
        });
    }
}

fn has_collision(state: &SimulationState) -> bool {
    if state.bird.y - BIRD_RADIUS <= 0.0 || state.bird.y + BIRD_RADIUS >= 1.0 {
        return true;
    }

    for pipe in &state.pipes {
        let left = pipe.x - PIPE_WIDTH / 2.0;
        let right = pipe.x + PIPE_WIDTH / 2.0;
        let intersects_x = BIRD_X + BIRD_RADIUS > left && BIRD_X - BIRD_RADIUS < right;

        if intersects_x {
            let gap_top = pipe.gap_center_y + PIPE_GAP_HEIGHT / 2.0;
            let gap_bottom = pipe.gap_center_y - PIPE_GAP_HEIGHT / 2.0;
            let intersects_pipe =
                state.bird.y - BIRD_RADIUS < gap_bottom || state.bird.y + BIRD_RADIUS > gap_top;

            if intersects_pipe {
                return true;
            }
        }
    }

    false
}

fn next_pipe(state: &SimulationState) -> &Pipe {
    state
        .pipes
        .iter()
        .find(|pipe| pipe.x + PIPE_WIDTH / 2.0 >= BIRD_X)
        .unwrap_or_else(|| {
            state
                .pipes
                .last()
                .expect("simulation always keeps at least one pipe")
        })
}

fn alignment_score(state: &SimulationState) -> f64 {
    let pipe = next_pipe(state);
    let dy = (pipe.gap_center_y - state.bird.y).abs();
    (1.0 - (dy / 0.5).clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn compute_fitness(frames_survived: u32, pipes_passed: u32, alignment_sum: f64) -> f64 {
    frames_survived as f64
        + pipes_passed as f64 * PIPE_PASS_BONUS
        + alignment_sum * ALIGNMENT_WEIGHT
}

fn snapshot(state: &SimulationState) -> ReplayFrame {
    let pipes = state
        .pipes
        .iter()
        .filter(|pipe| pipe.x + PIPE_WIDTH / 2.0 > -0.1 && pipe.x - PIPE_WIDTH / 2.0 < 1.3)
        .take(3)
        .map(|pipe| PipeSnapshot {
            x: pipe.x as f32,
            gap_center_y: pipe.gap_center_y as f32,
        })
        .collect();

    ReplayFrame {
        bird_y: state.bird.y as f32,
        bird_velocity: state.bird.velocity as f32,
        pipes_passed: state.pipes_passed,
        alive: state.alive,
        pipes,
    }
}

fn random_gap_center(rng: &mut StdRng) -> f64 {
    rng.random_range(PIPE_MIN_GAP_CENTER_Y..PIPE_MAX_GAP_CENTER_Y)
}

fn generation_seed(base_seed: u64, generation_index: usize) -> u64 {
    base_seed.wrapping_add((generation_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn normalize_velocity(value: f64) -> f64 {
    ((value - MIN_VELOCITY) / (MAX_VELOCITY - MIN_VELOCITY)).clamp(0.0, 1.0)
}

fn normalize_horizontal_distance(value: f64) -> f64 {
    (value / (1.0 + PIPE_SPACING)).clamp(0.0, 1.0)
}

fn normalize_vertical_offset(value: f64) -> f64 {
    ((value + 1.0) * 0.5).clamp(0.0, 1.0)
}

fn write_model_checkpoint(path: &Path, network: &NeuralNetwork) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(network)?;
    fs::write(path, json)?;
    Ok(())
}

fn write_html_report(path: &Path, view: &PlaybackView) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let payload = serde_json::to_string(view)?;
    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flappy Evolution Playground</title>
  <style>
    :root {{
      --bg: #efe7d8;
      --panel: rgba(253, 249, 241, 0.92);
      --ink: #17212b;
      --muted: #5a6470;
      --accent: #197a93;
      --accent-alt: #bb5a2a;
      --line: rgba(23, 33, 43, 0.14);
      --shadow: 0 18px 60px rgba(23, 33, 43, 0.16);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at 15% 10%, rgba(25,122,147,0.18), transparent 35%),
        radial-gradient(circle at 85% 90%, rgba(187,90,42,0.20), transparent 35%),
        linear-gradient(165deg, #f5efe2 0%, #eadbc3 45%, #f3ebdc 100%);
      padding: 28px 16px 48px;
    }}

    main {{
      max-width: 1160px;
      margin: 0 auto;
      display: grid;
      gap: 20px;
    }}

    .hero h1 {{
      margin: 0;
      font-size: clamp(2rem, 5vw, 4.2rem);
      line-height: 0.95;
      letter-spacing: -0.02em;
    }}

    .hero p {{
      margin: 10px 0 0;
      max-width: 68ch;
      color: var(--muted);
      line-height: 1.5;
    }}

    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 320px) minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}

    .controls {{
      padding: 20px;
      display: grid;
      gap: 16px;
    }}

    .label {{
      font-size: 0.84rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      margin-bottom: 6px;
      display: block;
    }}

    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}

    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}

    .stat {{
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(23,33,43,0.10);
      border-radius: 14px;
      padding: 10px;
    }}

    .stat .k {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      display: block;
      margin-bottom: 4px;
    }}

    .stat .v {{
      font-size: 1.1rem;
      font-weight: 700;
      line-height: 1.2;
    }}

    .canvas-wrap {{
      padding: 14px;
      display: grid;
      gap: 10px;
    }}

    canvas {{
      width: 100%;
      height: auto;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #9fd9ef 0%, #d3f0ff 100%);
      display: block;
      aspect-ratio: 16 / 9;
    }}

    .caption {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 0.92rem;
    }}

    code {{
      background: rgba(23,33,43,0.08);
      border-radius: 6px;
      padding: 0.1rem 0.3rem;
      font-family: "Cascadia Code", "Fira Code", monospace;
    }}

    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Flappy Evolution Playground</h1>
      <p>
        A tiny feedforward neural network controls the bird. Each generation keeps the best performers,
        mutates their descendants, and tries again on the same deterministic pipe sequence.
      </p>
    </section>

    <section class="grid">
      <aside class="panel controls">
        <div>
          <span class="label">Generation</span>
          <input id="generation-slider" type="range" min="0" value="0" step="1">
          <div id="generation-label">Generation 1</div>
        </div>

        <div class="stat-grid">
          <div class="stat"><span class="k">Best Fitness</span><span class="v" id="best-fitness"></span></div>
          <div class="stat"><span class="k">Average Fitness</span><span class="v" id="avg-fitness"></span></div>
          <div class="stat"><span class="k">Pipes Passed</span><span class="v" id="pipes-passed"></span></div>
          <div class="stat"><span class="k">Survival Frames</span><span class="v" id="survival-frames"></span></div>
        </div>

        <div class="stat-grid">
          <div class="stat"><span class="k">Seed</span><span class="v" id="meta-seed"></span></div>
          <div class="stat"><span class="k">Population</span><span class="v" id="meta-pop"></span></div>
          <div class="stat"><span class="k">Generations</span><span class="v" id="meta-gens"></span></div>
          <div class="stat"><span class="k">Mutation</span><span class="v" id="meta-mutation"></span></div>
        </div>

        <p class="caption">
          Overall champion: generation <span id="overall-gen"></span>
          with fitness <span id="overall-fit"></span>.
        </p>
      </aside>

      <section class="panel canvas-wrap">
        <canvas id="replay-canvas" width="960" height="540"></canvas>
        <p class="caption">
          Replay shows the best network from the selected generation on its deterministic scenario.
          Bird inputs are <code>[y, velocity, dx_to_pipe, gap_y, dy_to_gap]</code>.
        </p>
      </section>
    </section>
  </main>

  <script>
    const payload = {payload};
    const slider = document.getElementById('generation-slider');
    const generationLabel = document.getElementById('generation-label');
    const bestFitness = document.getElementById('best-fitness');
    const avgFitness = document.getElementById('avg-fitness');
    const pipesPassed = document.getElementById('pipes-passed');
    const survivalFrames = document.getElementById('survival-frames');

    const metaSeed = document.getElementById('meta-seed');
    const metaPop = document.getElementById('meta-pop');
    const metaGens = document.getElementById('meta-gens');
    const metaMutation = document.getElementById('meta-mutation');
    const overallGen = document.getElementById('overall-gen');
    const overallFit = document.getElementById('overall-fit');

    const canvas = document.getElementById('replay-canvas');
    const ctx = canvas.getContext('2d');

    let activeGeneration = 0;
    let replayFrame = 0;
    let lastTick = 0;

    slider.max = String(Math.max(payload.generations.length - 1, 0));

    metaSeed.textContent = String(payload.metadata.seed);
    metaPop.textContent = String(payload.metadata.population);
    metaGens.textContent = String(payload.metadata.generations);
    metaMutation.textContent = payload.metadata.mutation_rate.toFixed(2);
    overallGen.textContent = String(payload.overall_best_generation);
    overallFit.textContent = payload.overall_best_fitness.toFixed(1);

    function drawScene(frame) {{
      const width = canvas.width;
      const height = canvas.height;
      const pipeWidthPx = payload.metadata.pipe_width * width;
      const gapHalfPx = (payload.metadata.pipe_gap_height * height) / 2;

      ctx.clearRect(0, 0, width, height);

      ctx.fillStyle = '#d8f1ff';
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = '#66ad4b';
      for (const pipe of frame.pipes) {{
        const centerX = pipe.x * width;
        const left = centerX - pipeWidthPx / 2;
        const gapCenterY = pipe.gap_center_y * height;
        const gapTop = gapCenterY - gapHalfPx;
        const gapBottom = gapCenterY + gapHalfPx;

        ctx.fillRect(left, 0, pipeWidthPx, Math.max(0, gapTop));
        ctx.fillRect(left, gapBottom, pipeWidthPx, Math.max(0, height - gapBottom));
      }}

      const birdX = payload.metadata.bird_x * width;
      const birdY = frame.bird_y * height;
      const birdRadius = payload.metadata.bird_radius * width;

      ctx.beginPath();
      ctx.arc(birdX, birdY, birdRadius, 0, Math.PI * 2);
      ctx.fillStyle = '#f7b733';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#5c3c15';
      ctx.stroke();

      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.font = '600 20px Georgia';
      ctx.fillText(`Score: ${{frame.pipes_passed}}`, 16, 30);
    }}

    function updateStats(generation) {{
      generationLabel.textContent = `Generation ${{generation.generation}}`;
      bestFitness.textContent = generation.best_fitness.toFixed(1);
      avgFitness.textContent = generation.average_fitness.toFixed(1);
      pipesPassed.textContent = String(generation.best_pipes_passed);
      survivalFrames.textContent = String(generation.best_survival_frames);
    }}

    function selectGeneration(index) {{
      activeGeneration = index;
      replayFrame = 0;
      const generation = payload.generations[index];
      updateStats(generation);
      drawScene(generation.replay[0]);
    }}

    slider.addEventListener('input', () => {{
      selectGeneration(Number(slider.value));
    }});

    function tick(ts) {{
      if (!lastTick) lastTick = ts;
      const elapsed = ts - lastTick;
      if (elapsed >= 45) {{
        lastTick = ts;
        const generation = payload.generations[activeGeneration];
        if (generation && generation.replay.length > 0) {{
          replayFrame = (replayFrame + 1) % generation.replay.length;
          drawScene(generation.replay[replayFrame]);
        }}
      }}
      requestAnimationFrame(tick);
    }}

    if (payload.generations.length > 0) {{
      selectGeneration(0);
      requestAnimationFrame(tick);
    }}
  </script>
</body>
</html>
"#
    );

    fs::write(path, html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_collision_geometry_correctness() {
        let in_gap = SimulationState {
            bird: BirdState {
                y: 0.5,
                velocity: 0.0,
            },
            pipes: vec![Pipe {
                x: BIRD_X,
                gap_center_y: 0.5,
                passed: false,
            }],
            frame: 0,
            pipes_passed: 0,
            alive: true,
        };
        assert!(!has_collision(&in_gap));

        let above_gap = SimulationState {
            bird: BirdState {
                y: 0.1,
                velocity: 0.0,
            },
            ..in_gap.clone()
        };
        assert!(has_collision(&above_gap));
    }

    #[test]
    fn pipe_spawning_and_advancement_invariants() {
        let mut rng = StdRng::seed_from_u64(10);
        let mut state = SimulationState::new(&mut rng);

        for _ in 0..400 {
            advance_pipes(&mut state, &mut rng);
            assert!(!state.pipes.is_empty());

            for pair in state.pipes.windows(2) {
                assert!(pair[0].x < pair[1].x);
            }
        }
    }

    #[test]
    fn input_features_stay_normalized() {
        let mut rng = StdRng::seed_from_u64(33);
        let mut state = SimulationState::new(&mut rng);

        for _ in 0..300 {
            let inputs = build_inputs(&state);
            for feature in inputs {
                assert!((0.0..=1.0).contains(&feature));
            }

            step_simulation(&mut state, rng.random::<f64>() > 0.5, &mut rng);
            if !state.alive {
                state = SimulationState::new(&mut rng);
            }
        }
    }

    #[test]
    fn fitness_sanity_checks() {
        let base = compute_fitness(100, 0, 10.0);
        let with_more_frames = compute_fitness(180, 0, 10.0);
        let with_pipe = compute_fitness(100, 1, 10.0);

        assert!(with_more_frames > base);
        assert!(with_pipe > base);
    }

    #[test]
    fn deterministic_smoke_test_outputs_and_threshold() -> Result<(), Box<dyn Error>> {
        let tmp_dir =
            std::env::temp_dir().join(format!("flappy-evolution-smoke-{}", std::process::id()));
        fs::create_dir_all(&tmp_dir)?;

        let config = Config {
            seed: 42,
            generations: 20,
            population: 50,
            save_model: tmp_dir.join("champion.json"),
            load_model: None,
            html_output: tmp_dir.join("playground.html"),
        };

        let outcome = run_training(&config, None)?;
        write_model_checkpoint(&config.save_model, &outcome.best_network)?;
        write_html_report(&config.html_output, &outcome.view)?;

        assert!(outcome.view.overall_best_fitness > 80.0);
        assert!(config.save_model.exists());
        assert!(config.html_output.exists());
        assert!(fs::metadata(&config.save_model)?.len() > 0);
        assert!(fs::metadata(&config.html_output)?.len() > 0);

        fs::remove_dir_all(&tmp_dir)?;
        Ok(())
    }
}
