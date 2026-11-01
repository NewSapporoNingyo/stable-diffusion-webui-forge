import ast
import importlib.metadata
import re
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from packaging.markers import default_environment


class RequirementsMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "modules/launch_utils.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = [node for node in tree.body if
                       isinstance(node, ast.FunctionDef) and node.name == "requirements_met" or
                       isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "re_requirement" for target in node.targets)]
        namespace = {"re": re}
        # Avoid launcher argument parsing and initialization in this unit test.
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
        cls.requirements_met = staticmethod(namespace["requirements_met"])

    def check_requirement(self, requirement, platform="win32", installed="1.3.1"):
        environment = default_environment()
        environment["sys_platform"] = platform
        with patch("builtins.open", mock_open(read_data=requirement)), \
                patch("packaging.markers.default_environment", return_value=environment), \
                patch.object(importlib.metadata, "version", return_value=installed) as version:
            result = self.requirements_met("requirements.txt")
        return result, version

    def test_windows_dependency_is_checked_on_windows(self):
        result, version = self.check_requirement('windows-toasts==1.3.1; sys_platform == "win32"\n')
        self.assertTrue(result)
        version.assert_called_once_with("windows-toasts")

    def test_windows_dependency_is_skipped_on_other_platforms(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                result, version = self.check_requirement('windows-toasts==1.3.1; sys_platform == "win32"\n', platform)
                self.assertTrue(result)
                version.assert_not_called()

    def test_wrong_version_is_still_rejected(self):
        self.assertFalse(self.check_requirement('windows-toasts==1.3.1; sys_platform == "win32"', installed="1.2.0")[0])

    def test_missing_package_is_still_rejected(self):
        with patch("builtins.open", mock_open(read_data="missing-package==1.0")), \
                patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            self.assertFalse(self.requirements_met("requirements.txt"))

    def test_invalid_marker_returns_false(self):
        for marker in ('sys_platform = "win32"', 'sys_platform === "win32"'):
            with self.subTest(marker=marker):
                result, version = self.check_requirement(f"windows-toasts==1.3.1; {marker}")
                self.assertFalse(result)
                version.assert_not_called()

    def test_existing_unmarked_requirements_keep_their_behavior(self):
        self.assertTrue(self.check_requirement("example==1.3.1 # existing comment\n\n")[0])
        result, version = self.check_requirement("unversioned-package\n")
        self.assertTrue(result)
        version.assert_not_called()


if __name__ == "__main__":
    unittest.main()
