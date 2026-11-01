"""Run CUDA compute and Forge image generation benchmarks without a server."""

import argparse
import logging
import os
import platform
import signal
import sys
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
RUNS = 5
MATRIX_SIZE = 4096
MATMULS_PER_RUN = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA compute and Forge generation without starting WebUI or an API server.",
        epilog="Examples: python benchmark.py; python benchmark.py --skip-generation; "
               "python benchmark.py --skip-gpu --model models/Stable-diffusion/model.safetensors",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-gpu", action="store_true", help="skip the FP64/FP32/FP16/BF16 compute benchmark")
    group.add_argument("--skip-generation", action="store_true", help="skip the image generation benchmark")
    parser.add_argument("--model", help="checkpoint name or local path; defaults to the WebUI configured model")
    return parser.parse_args(argv)


def matmul_times(torch, dtype):
    """Yield CUDA milliseconds per round, excluding allocation and warmup."""
    with torch.inference_mode():
        a = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device="cuda", dtype=dtype)
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize()
        for _ in range(RUNS):
            start.record()
            for _ in range(MATMULS_PER_RUN):
                torch.mm(a, b, out=out)
            end.record()
            end.synchronize()
            yield start.elapsed_time(end)


def benchmark_gpu(torch, report):
    report("\nGPU compute: dense 4096x4096 matmul; 3 warmups; 5 rounds of 10 matmuls; FP32 TF32 disabled")
    precisions = [("FP64", torch.float64), ("FP32", torch.float32),
                  ("FP16", torch.float16), ("BF16", torch.bfloat16)]
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    success = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for name, dtype in precisions:
            if name == "BF16" and not torch.cuda.is_bf16_supported():
                report(f"{name}: SKIPPED (not supported by this GPU/PyTorch)")
                continue
            values = []
            try:
                for i, milliseconds in enumerate(matmul_times(torch, dtype), 1):
                    tflops = 2 * MATRIX_SIZE ** 3 * MATMULS_PER_RUN / (milliseconds / 1000) / 1e12
                    values.append(tflops)
                    report(f"{name} run {i}/{RUNS}: {tflops:.6f} TFLOPS ({milliseconds:.6f} ms)")
                if len(values) != RUNS:
                    raise RuntimeError(f"expected {RUNS} measurements, got {len(values)}")
                report(f"{name} average: {mean(values):.6f} TFLOPS")
            except (NotImplementedError, RuntimeError) as exc:
                unsupported = isinstance(exc, NotImplementedError) or any(
                    text in str(exc).lower().replace("_", " ") for text in ("not implemented", "not supported")
                )
                report(f"{name}: {'SKIPPED' if unsupported else 'ERROR'} ({type(exc).__name__}: {exc})")
                success &= unsupported
            finally:
                torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return success


def select_model(sd_models, shared, model):
    requested = model if model is not None else shared.opts.sd_model_checkpoint
    if model is None and requested in (None, "", "None", "none"):
        info = sd_models.select_checkpoint()
    else:
        info = sd_models.checkpoint_aliases.get(requested)
        if info is None and Path(requested).is_file():
            info = sd_models.CheckpointInfo(str(Path(requested).resolve()))
            info.register()
    if info is None:
        raise FileNotFoundError(f"Checkpoint not found: {requested!r}")
    shared.opts.data["sd_model_checkpoint"] = info.title
    return info


def initialize_generation(model):
    # Keep benchmark arguments out of Forge's parsers and disable startup installs.
    sys.argv = [str(ROOT / "benchmark.py"), "--skip-install", "--no-prompt-history"]
    os.environ["COMMANDLINE_ARGS"] = ""
    # Infotext queries Git metadata; trust this checkout for child Git commands only.
    git_config_count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    os.environ[f"GIT_CONFIG_KEY_{git_config_count}"] = "safe.directory"
    os.environ[f"GIT_CONFIG_VALUE_{git_config_count}"] = ROOT.as_posix()
    os.environ["GIT_CONFIG_COUNT"] = str(git_config_count + 1)
    from modules_forge.initialization import initialize_forge

    initialize_forge()
    from modules import initialize

    initialize.imports()
    from modules import processing, sd_models, shared
    from modules_forge import main_entry

    configured_model = shared.opts.sd_model_checkpoint
    sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        initialize.initialize()
    finally:
        signal.signal(signal.SIGINT, sigint_handler)
    # list_models() can select model.ckpt; the benchmark defaults to the saved selection.
    shared.opts.data["sd_model_checkpoint"] = configured_model
    shared.opts.data["live_previews_enable"] = False
    info = select_model(sd_models, shared, model)
    main_entry.refresh_memory_management_settings()
    main_entry.refresh_model_loading_parameters()
    return processing, shared, info


def generate_once(torch, processing, shared, report=None):
    with closing(processing.StableDiffusionProcessingTxt2Img(
        sampler_name="Euler a", scheduler="Automatic", steps=20, width=512, height=512,
        do_not_save_samples=True, do_not_save_grid=True,
    )) as p:
        if report is not None:
            report(f"Generation settings: sampler={p.sampler_name}; scheduler={p.scheduler}; steps={p.steps}; size={p.width}x{p.height}; "
                   f"batch_size={p.batch_size}; n_iter={p.n_iter}; CFG={p.cfg_scale}; seed={p.seed}; "
                   f"prompt={p.prompt!r}; negative_prompt={p.negative_prompt!r}; hires={p.enable_hr}")
        shared.state.begin(job="benchmark")
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = processing.process_images(p)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if shared.state.interrupted or shared.state.stopping_generation:
                raise RuntimeError("image generation interrupted")
            if len(result.images) != 1 or result.images[0].size != (512, 512):
                raise RuntimeError("expected one 512x512 image")
            return seconds, result.seed
        finally:
            shared.state.end()
            shared.total_tqdm.clear()


def benchmark_generation(torch, model, report):
    report("\nImage generation: 1 warmup (excluded), followed by 5 complete generations; no image/history saving")
    processing, shared, info = initialize_generation(model)
    report(f"Model: {info.title}\nModel path: {info.filename}")
    report(f"Model settings: storage={shared.opts.forge_unet_storage_dtype}; "
           f"additional_modules={shared.opts.forge_additional_modules}; clip_skip={shared.opts.CLIP_stop_at_last_layers}")
    report(f"Memory settings: inference={shared.opts.forge_inference_memory} MB; "
           f"swap={shared.opts.forge_async_loading}; location={shared.opts.forge_pin_shared_memory}")
    generate_once(torch, processing, shared, report)
    report("Generation warmup complete (excluded from average)")
    values = []
    for i in range(1, RUNS + 1):
        seconds, seed = generate_once(torch, processing, shared)
        values.append(seconds)
        report(f"Generation run {i}/{RUNS}: {seconds:.6f} seconds (seed={seed})")
    report(f"Generation average: {mean(values):.6f} seconds")
    return True


def main(argv=None):
    args = parse_args(argv)
    model = args.model
    if model is not None and Path(model).is_file():
        model = str(Path(model).resolve())
    os.chdir(ROOT)
    started = datetime.now().astimezone()
    report_path = ROOT / f"benchmark-{started:%Y-%m-%d-%H-%M-%S-%f}.txt"
    status = 0
    with report_path.open("x", encoding="utf-8") as output:
        def report(message):
            output.write(message + "\n")
            output.flush()
            print(message, flush=True)

        report(f"Forge benchmark\nStarted: {started.isoformat()} ({started.tzname()})\nPython: {platform.python_version()}")
        try:
            import torch

            report(f"PyTorch: {torch.__version__}\nCUDA runtime: {torch.version.cuda}")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU required; CUDA is not available")
            device = torch.cuda.get_device_properties(torch.cuda.current_device())
            report(f"GPU: {device.name}\nVRAM: {device.total_memory / 2 ** 30:.2f} GiB")
            for name, skip, run in [
                ("GPU compute", args.skip_gpu, lambda: benchmark_gpu(torch, report)),
                ("Image generation", args.skip_generation, lambda: benchmark_generation(torch, model, report)),
            ]:
                if skip:
                    report(f"\n{name}: SKIPPED (command line)")
                    continue
                try:
                    if not run():
                        status = 1
                except Exception as exc:
                    logger.exception("%s benchmark failed", name)
                    report(f"{name}: ERROR ({type(exc).__name__}: {exc})")
                    status = 1
        except KeyboardInterrupt:
            report("INTERRUPTED (partial results retained)")
            status = 130
        except Exception as exc:
            logger.exception("Benchmark initialization failed")
            report(f"ERROR ({type(exc).__name__}: {exc})")
            status = 1
        finally:
            report(f"\nFinished: {datetime.now().astimezone().isoformat()}\nExit status: {status}\nResults: {report_path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
