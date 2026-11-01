import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("forge_benchmark", ROOT / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def fake_torch():
    return SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda="test"),
        float64="FP64", float32="FP32", float16="FP16", bfloat16="BF16",
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        cuda=SimpleNamespace(
            is_available=Mock(return_value=True), current_device=Mock(return_value=0),
            get_device_properties=Mock(return_value=SimpleNamespace(name="Test GPU", total_memory=12 * 2 ** 30)),
            is_bf16_supported=Mock(return_value=True), synchronize=Mock(), empty_cache=Mock(),
        ),
    )


def fake_shared():
    return SimpleNamespace(
        opts=SimpleNamespace(
            sd_model_checkpoint="configured", data={}, forge_unet_storage_dtype="Automatic",
            forge_additional_modules=[], CLIP_stop_at_last_layers=1,
            forge_inference_memory=1024, forge_async_loading="Queue", forge_pin_shared_memory="CPU",
        ),
        state=SimpleNamespace(begin=Mock(), end=Mock(), interrupted=False, stopping_generation=False),
        total_tqdm=SimpleNamespace(clear=Mock()),
    )


class ArgumentsTests(unittest.TestCase):
    def test_defaults_run_both_tests(self):
        args = benchmark.parse_args([])
        self.assertFalse(args.skip_gpu or args.skip_generation)
        self.assertIsNone(args.model)

    def test_skip_flags_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            benchmark.parse_args(["--skip-gpu", "--skip-generation"])
        self.assertEqual(caught.exception.code, 2)

    def test_help_does_not_import_torch_or_forge(self):
        with patch.dict(sys.modules, {"torch": None, "modules": None, "modules_forge": None}), \
                redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as caught:
            benchmark.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--model", output.getvalue())


class ComputeTests(unittest.TestCase):
    def test_cuda_events_measure_five_rounds_after_warmup(self):
        torch = fake_torch()
        torch.inference_mode = Mock(return_value=nullcontext())
        torch.randn, torch.randn_like, torch.empty_like, torch.mm = Mock(), Mock(), Mock(), Mock()
        start, end = Mock(), Mock()
        start.elapsed_time.return_value = 100.0
        torch.cuda.Event = Mock(side_effect=[start, end])
        self.assertEqual(list(benchmark.matmul_times(torch, torch.float32)), [100.0] * 5)
        self.assertEqual(torch.mm.call_count, 3 + 5 * 10)
        torch.randn.assert_called_once_with((4096, 4096), device="cuda", dtype="FP32")
        self.assertEqual(start.record.call_count, 5)
        self.assertEqual(end.synchronize.call_count, 5)
        torch.cuda.synchronize.assert_called_once()

    def test_tflops_calculation_and_tf32_restoration(self):
        torch, lines = fake_torch(), []

        def timings(torch, dtype):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            yield from [100.0] * 5

        with patch.object(benchmark, "matmul_times", side_effect=timings):
            self.assertTrue(benchmark.benchmark_gpu(torch, lines.append))
        # 10 square GEMMs at 2*N^3 FLOPs each, completed in 0.1 seconds.
        self.assertIn("FP32 average: 13.743895 TFLOPS", lines)
        self.assertEqual(sum("run " in line for line in lines), 20)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

    def test_partial_failure_has_no_average_and_other_precisions_continue(self):
        torch, lines = fake_torch(), []

        def timings(torch, dtype):
            yield 100.0
            if dtype == "FP32":
                raise RuntimeError("out of memory")
            yield from [100.0] * 4

        with patch.object(benchmark, "matmul_times", side_effect=timings):
            self.assertFalse(benchmark.benchmark_gpu(torch, lines.append))
        self.assertTrue(any("FP32: ERROR" in line for line in lines))
        self.assertFalse(any("FP32 average:" in line for line in lines))
        self.assertTrue(any("BF16 average:" in line for line in lines))
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

    def test_unsupported_bf16_is_skipped(self):
        torch, lines = fake_torch(), []
        torch.cuda.is_bf16_supported.return_value = False
        with patch.object(benchmark, "matmul_times", return_value=[100.0] * 5) as timings:
            self.assertTrue(benchmark.benchmark_gpu(torch, lines.append))
        self.assertEqual(timings.call_count, 3)
        self.assertTrue(any("BF16: SKIPPED" in line for line in lines))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.shared = fake_shared()
        self.info = SimpleNamespace(title="configured", filename="configured.safetensors")
        self.models = SimpleNamespace(
            checkpoint_aliases={"configured": self.info},
            select_checkpoint=Mock(return_value=self.info), CheckpointInfo=Mock(),
        )

    def test_saved_selection_is_used_by_default(self):
        self.assertIs(benchmark.select_model(self.models, self.shared, None), self.info)
        self.assertEqual(self.shared.opts.data["sd_model_checkpoint"], "configured")
        self.models.select_checkpoint.assert_not_called()

    def test_explicit_name_overrides_saved_selection(self):
        other = SimpleNamespace(title="other")
        self.models.checkpoint_aliases["other"] = other
        self.assertIs(benchmark.select_model(self.models, self.shared, "other"), other)
        self.assertEqual(self.shared.opts.data["sd_model_checkpoint"], "other")

    def test_external_path_is_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.safetensors"
            path.touch()
            info = self.models.CheckpointInfo.return_value
            self.assertIs(benchmark.select_model(self.models, self.shared, str(path)), info)
            self.models.CheckpointInfo.assert_called_once_with(str(path.resolve()))
            info.register.assert_called_once()

    def test_unconfigured_selection_uses_webui_default(self):
        self.shared.opts.sd_model_checkpoint = None
        self.assertIs(benchmark.select_model(self.models, self.shared, None), self.info)
        self.models.select_checkpoint.assert_called_once()

    def test_missing_explicit_or_configured_model_does_not_fallback(self):
        for model in ("missing-model", None):
            with self.subTest(model=model):
                self.shared.opts.sd_model_checkpoint = "missing-configured-model"
                with self.assertRaises(FileNotFoundError):
                    benchmark.select_model(self.models, self.shared, model)
        self.models.select_checkpoint.assert_not_called()


class GenerationTests(unittest.TestCase):
    def test_initialization_preserves_configured_model_and_restores_sigint(self):
        shared = fake_shared()
        shared.opts.save = Mock()
        info = SimpleNamespace(title="configured", filename="configured.safetensors")
        models = SimpleNamespace(checkpoint_aliases={"configured": info})
        forge = SimpleNamespace(initialize_forge=Mock())
        initialize = SimpleNamespace(imports=Mock(), initialize=Mock())
        main_entry = SimpleNamespace(refresh_memory_management_settings=Mock(), refresh_model_loading_parameters=Mock())
        modules = SimpleNamespace(initialize=initialize, processing=Mock(), sd_models=models, shared=shared)
        imported = {"modules_forge.initialization": forge, "modules_forge": SimpleNamespace(main_entry=main_entry),
                    "modules": modules, "webui": None}
        # Model scanning must not replace the saved selection with a root model.ckpt.
        initialize.initialize.side_effect = lambda: setattr(shared.opts, "sd_model_checkpoint", "model.ckpt")
        with patch.dict(sys.modules, imported), patch.object(benchmark.sys, "argv", ["benchmark.py", "--skip-gpu"]), \
                patch.dict(benchmark.os.environ, {"COMMANDLINE_ARGS": "--autolaunch"}), \
                patch.object(benchmark, "select_model", return_value=info), \
                patch.object(benchmark.signal, "getsignal", return_value="previous") as getsignal, \
                patch.object(benchmark.signal, "signal") as setsignal:
            self.assertIs(benchmark.initialize_generation(None)[2], info)
            self.assertIn("--no-prompt-history", sys.argv)
            self.assertIn("--skip-install", sys.argv)
            self.assertEqual(benchmark.os.environ["COMMANDLINE_ARGS"], "")
            git_config_index = int(benchmark.os.environ["GIT_CONFIG_COUNT"]) - 1
            self.assertEqual(benchmark.os.environ[f"GIT_CONFIG_VALUE_{git_config_index}"], benchmark.ROOT.as_posix())
        setsignal.assert_called_once_with(benchmark.signal.SIGINT, getsignal.return_value)
        self.assertFalse(shared.opts.data["live_previews_enable"])
        self.assertEqual(shared.opts.data["sd_model_checkpoint"], "configured")
        shared.opts.save.assert_not_called()
        main_entry.refresh_memory_management_settings.assert_called_once()
        main_entry.refresh_model_loading_parameters.assert_called_once()

    def test_complete_generation_is_synchronized_and_processing_closes(self):
        torch, shared = fake_torch(), fake_shared()
        p = SimpleNamespace(close=Mock())
        processing = SimpleNamespace(
            StableDiffusionProcessingTxt2Img=Mock(return_value=p),
            process_images=Mock(return_value=SimpleNamespace(images=[SimpleNamespace(size=(512, 512))], seed=42)),
        )
        with patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 12.5]):
            self.assertEqual(benchmark.generate_once(torch, processing, shared), (2.5, 42))
        processing.StableDiffusionProcessingTxt2Img.assert_called_once_with(
            sampler_name="Euler a", scheduler="Automatic", steps=20, width=512, height=512,
            do_not_save_samples=True, do_not_save_grid=True,
        )
        self.assertEqual(torch.cuda.synchronize.call_count, 2)
        p.close.assert_called_once()
        shared.state.end.assert_called_once()

    def test_failed_generation_still_closes_processing_and_state(self):
        p, shared = SimpleNamespace(close=Mock()), fake_shared()
        processing = SimpleNamespace(StableDiffusionProcessingTxt2Img=Mock(return_value=p),
                                     process_images=Mock(side_effect=RuntimeError("generation failed")))
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            benchmark.generate_once(fake_torch(), processing, shared)
        p.close.assert_called_once()
        shared.state.end.assert_called_once()

    def test_warmup_is_excluded_from_five_run_average(self):
        lines = []
        initialized = (Mock(), fake_shared(), SimpleNamespace(title="model", filename="model.safetensors"))
        with patch.object(benchmark, "initialize_generation", return_value=initialized), \
                patch.object(benchmark, "generate_once", side_effect=[(999, 0)] + [(i, i) for i in range(1, 6)]) as generate:
            self.assertTrue(benchmark.benchmark_generation(fake_torch(), None, lines.append))
        self.assertEqual(generate.call_count, 6)
        self.assertIn("Generation average: 3.000000 seconds", lines)
        self.assertEqual(sum("Generation run " in line for line in lines), 5)

    def test_partial_generation_has_no_success_average(self):
        lines = []
        initialized = (Mock(), fake_shared(), SimpleNamespace(title="model", filename="model.safetensors"))
        with patch.object(benchmark, "initialize_generation", return_value=initialized), \
                patch.object(benchmark, "generate_once", side_effect=[(999, 0), (2, 1), RuntimeError("out of memory")]), \
                self.assertRaisesRegex(RuntimeError, "out of memory"):
            benchmark.benchmark_generation(fake_torch(), None, lines.append)
        self.assertTrue(any("Generation run 1/5:" in line for line in lines))
        self.assertFalse(any("Generation average:" in line for line in lines))


class EntryPointTests(unittest.TestCase):
    def run_main(self, argv, gpu_error=None, cuda_available=True):
        torch = fake_torch()
        torch.cuda.is_available.return_value = cuda_available
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(benchmark, "ROOT", Path(directory)), patch.object(benchmark.os, "chdir"), \
                patch.dict(sys.modules, {"torch": torch, "modules": None, "modules_forge": None}), \
                patch.object(benchmark, "initialize_generation") as initialize, \
                patch.object(benchmark, "benchmark_gpu", return_value=True, side_effect=gpu_error) as gpu, \
                patch.object(benchmark, "benchmark_generation", return_value=True) as generation, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = benchmark.main(argv)
            files = list(Path(directory).glob("benchmark-*.txt"))
            self.assertEqual(len(files), 1)
            self.assertRegex(files[0].name, r"^benchmark-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}\.txt$")
            text = files[0].read_text(encoding="utf-8")
        return status, text, gpu, generation, initialize

    def test_compute_only_never_initializes_forge(self):
        status, text, gpu, generation, initialize = self.run_main(["--skip-generation"])
        self.assertEqual(status, 0)
        gpu.assert_called_once()
        generation.assert_not_called()
        initialize.assert_not_called()
        self.assertIn("Image generation: SKIPPED", text)
        self.assertIn("GPU: Test GPU", text)

    def test_generation_only_passes_model_without_running_compute(self):
        status, text, gpu, generation, _ = self.run_main(["--skip-gpu", "--model", "模型名称"])
        self.assertEqual(status, 0)
        gpu.assert_not_called()
        self.assertEqual(generation.call_args.args[1], "模型名称")
        self.assertIn("GPU compute: SKIPPED", text)

    def test_failure_preserves_report_and_runs_other_test(self):
        status, text, _, generation, _ = self.run_main([], gpu_error=RuntimeError("compute failed"))
        self.assertEqual(status, 1)
        self.assertIn("GPU compute: ERROR (RuntimeError: compute failed)", text)
        self.assertIn("Exit status: 1", text)
        generation.assert_called_once()

    def test_no_cuda_produces_failure_report(self):
        status, text, gpu, generation, _ = self.run_main([], cuda_available=False)
        self.assertEqual(status, 1)
        self.assertIn("CUDA GPU required", text)
        gpu.assert_not_called()
        generation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
