import ast
import importlib.util
import inspect
import io
import json
import sys
import unittest
from contextlib import closing, redirect_stderr
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path):
    spec = importlib.util.spec_from_file_location(Path(relative_path).stem, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_function(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    # Exercise the actual entry point without importing the GPU/model initialization stack.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


class ToastTestCase(unittest.TestCase):
    def setUp(self):
        self.notification = load_module("modules_forge/generation_notification.py")
        self.opts = SimpleNamespace(notification_toast=True, notification_audio=True, notification_volume=42)
        self.state = SimpleNamespace(interrupted=False, stopping_generation=False)
        self.shared = SimpleNamespace(opts=self.opts, state=self.state)
        self.errors = load_module("modules/errors.py")
        modules = ModuleType("modules")
        modules.shared = self.shared
        modules.errors = self.errors
        self.toaster = Mock()
        self.windows_toasts = ModuleType("windows_toasts")
        self.windows_toasts.Toast = Mock(side_effect=SimpleNamespace)
        self.windows_toasts.WindowsToaster = Mock(return_value=self.toaster)
        self.addCleanup(patch.stopall)
        patch.object(sys, "platform", "win32").start()
        patch.dict(sys.modules, {"modules": modules, "windows_toasts": self.windows_toasts}).start()


class NotificationTests(ToastTestCase):
    def test_sends_original_message_with_default_audio(self):
        self.notification.notify_generation_finished()
        self.windows_toasts.WindowsToaster.assert_called_once_with("Stable Diffusion WebUI Forge")
        toast = self.toaster.show_toast.call_args.args[0]
        self.assertEqual(toast.text_fields, ["WebUI Generation Finished"])
        self.assertFalse(hasattr(toast, "audio"))

    def test_audio_and_toast_switches_are_independent(self):
        for audio in (True, False):
            for toast in (True, False):
                with self.subTest(audio=audio, toast=toast):
                    self.opts.notification_audio = audio
                    self.opts.notification_toast = toast
                    self.toaster.reset_mock()
                    self.notification.notify_generation_finished()
                    self.assertEqual(self.toaster.show_toast.call_count, int(toast))
                    self.assertEqual(self.opts.notification_audio, audio)
                    self.assertEqual(self.opts.notification_volume, 42)

    def test_non_windows_does_not_import_windows_dependency(self):
        with patch.object(sys, "platform", "linux"), patch.dict(sys.modules, {"windows_toasts": None}):
            self.notification.notify_generation_finished()
        self.windows_toasts.Toast.assert_not_called()

    def test_interrupt_and_stop_do_not_notify(self):
        for flag in ("interrupted", "stopping_generation"):
            with self.subTest(flag=flag):
                setattr(self.state, flag, True)
                self.notification.notify_generation_finished()
                setattr(self.state, flag, False)
        self.windows_toasts.Toast.assert_not_called()

    def test_missing_dependency_is_reported_without_raising(self):
        with patch.dict(sys.modules, {"windows_toasts": None}), redirect_stderr(io.StringIO()) as stderr:
            self.notification.notify_generation_finished()
            self.notification.notify_generation_finished()
        self.assertEqual(stderr.getvalue().count("sending generation-finished Windows notification:"), 1)

    def test_windows_failures_are_reported_once_without_raising(self):
        self.toaster.show_toast.side_effect = OSError("notification unavailable")
        with redirect_stderr(io.StringIO()) as stderr:
            self.notification.notify_generation_finished()
            self.notification.notify_generation_finished()
        self.assertEqual(stderr.getvalue().count("sending generation-finished Windows notification:"), 1)


class GenerationEntryPointTests(ToastTestCase):
    def setUp(self):
        super().setUp()
        self.opts.__dict__.update(
            samples_log_stdout=False, do_not_show_images=False, enable_console_prompts=False,
            outdir_samples="", outdir_grids="", outdir_img2img_samples="samples", outdir_img2img_grids="grids",
            hires_button_gallery_insert=False,
        )
        self.shared.total_tqdm = Mock()
        self.shared.cmd_opts = SimpleNamespace(hide_ui_dir_config=False)
        self.p = SimpleNamespace(script_args=(), close=Mock(), extra_generation_params={}, override_settings={})
        self.processed = SimpleNamespace(
            images=[Image.new("RGB", (64, 64))], extra_images=[], info="generated", comments="",
            js=Mock(return_value='{"images": 1}'),
        )
        self.runner = SimpleNamespace(run=Mock(return_value=None))
        self.process_images = Mock(return_value=self.processed)
        self.process_batch = Mock(return_value=self.processed)
        self.namespace = {
            "gr": SimpleNamespace(Request=object), "json": json, "closing": closing,
            "modules": SimpleNamespace(scripts=SimpleNamespace(scripts_txt2img=self.runner, scripts_img2img=self.runner)),
            "processing": SimpleNamespace(process_images=self.process_images),
            "shared": self.shared, "opts": self.opts, "Image": Image,
            "images": SimpleNamespace(fix_image=lambda image: image),
            "StableDiffusionProcessingImg2Img": Mock(return_value=self.p),
            "txt2img_create_processing": Mock(return_value=self.p),
            "create_override_settings_dict": lambda text: {},
            "parse_generation_parameters": lambda text, fields: {},
            "infotext_utils": SimpleNamespace(image_from_url_text=Mock()),
            "plaintext_to_html": lambda text, **kwargs: text,
            "process_images": self.process_images, "process_batch": self.process_batch,
            "generation_notification": self.notification,
        }

    def run_entry_point(self, name, **overrides):
        path = "modules/img2img.py" if name == "img2img_function" else "modules/txt2img.py"
        function = load_function(path, name, self.namespace)
        if name == "txt2img_upscale_function":
            return function("task(test)", SimpleNamespace(username="test"), [[self.processed.images[0], ""]],
                            0, json.dumps({"infotexts": ["original"]}), *([64] * 9))
        if name == "txt2img_function":
            return function("task(test)", SimpleNamespace(username="test"))
        values = dict.fromkeys(inspect.signature(function).parameters)
        values.pop("args")
        values.update(id_task="task(test)", request=SimpleNamespace(username="test"), mode=0,
                      height=64, width=64, denoising_strength=0.5, selected_scale_tab=0,
                      init_img=self.processed.images[0], img2img_batch_source_type="upload",
                      img2img_batch_upload=["first.png", "second.png"])
        values.update(overrides)
        return function(**values)

    def test_complete_tasks_notify_once_after_processing_closes(self):
        for name in ("txt2img_function", "txt2img_upscale_function", "img2img_function"):
            with self.subTest(entry=name):
                self.toaster.reset_mock()
                self.p.close.reset_mock()
                self.toaster.show_toast.side_effect = lambda toast: self.p.close.assert_called_once()
                result = self.run_entry_point(name)
                self.assertEqual(len(result), 4)
                self.toaster.show_toast.assert_called_once()

    def test_multiple_images_still_notify_once(self):
        self.processed.images *= 4
        self.run_entry_point("txt2img_function")
        self.toaster.show_toast.assert_called_once()

    def test_img2img_batch_notifies_once_for_the_whole_batch(self):
        self.run_entry_point("img2img_function", mode=5)
        self.process_batch.assert_called_once()
        self.toaster.show_toast.assert_called_once()

    def test_hidden_results_still_notify(self):
        self.opts.do_not_show_images = True
        result = self.run_entry_point("txt2img_function")
        self.assertEqual(result[0], [])
        self.toaster.show_toast.assert_called_once()

    def test_interrupt_and_stop_suppress_all_entry_points(self):
        for name in ("txt2img_function", "txt2img_upscale_function", "img2img_function"):
            for flag in ("interrupted", "stopping_generation"):
                with self.subTest(entry=name, flag=flag):
                    setattr(self.state, flag, True)
                    self.run_entry_point(name)
                    setattr(self.state, flag, False)
        self.toaster.show_toast.assert_not_called()

    def test_generation_errors_do_not_notify(self):
        self.process_images.side_effect = RuntimeError("generation failed")
        for name in ("txt2img_function", "txt2img_upscale_function", "img2img_function"):
            with self.subTest(entry=name), self.assertRaisesRegex(RuntimeError, "generation failed"):
                self.run_entry_point(name)
        self.toaster.show_toast.assert_not_called()

    def test_result_formatting_errors_do_not_notify(self):
        self.namespace["plaintext_to_html"] = Mock(side_effect=RuntimeError("formatting failed"))
        with self.assertRaisesRegex(RuntimeError, "formatting failed"):
            self.run_entry_point("txt2img_function")
        self.toaster.show_toast.assert_not_called()

    def test_invalid_upscale_selection_does_not_notify(self):
        function = load_function("modules/txt2img.py", "txt2img_upscale_function", self.namespace)
        self.assertEqual(function("task(test)", None, [[None, ""]], 2, "{}")[2], "Bad image index: 2")
        self.toaster.show_toast.assert_not_called()

    def test_notification_failure_preserves_generation_result(self):
        self.toaster.show_toast.side_effect = OSError("notification unavailable")
        with redirect_stderr(io.StringIO()):
            result = self.run_entry_point("txt2img_function")
        self.assertEqual(result[2], "generated")
        self.assertEqual(len(result[0]), 1)


if __name__ == "__main__":
    unittest.main()
