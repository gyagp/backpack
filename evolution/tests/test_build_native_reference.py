import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evolution.build_native_reference import BuildPaths, build_plan, ignored_path, main, normalize_ort_install


class NativeBuildPlanTest(unittest.TestCase):
    def test_existing_ninja_builds_keep_their_generator_and_ignore_visual_studio_arch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ort_build, genai_build = root / "gitignore/ort-build/Release", root / "gitignore/genai-build"
            for directory in (ort_build, genai_build):
                directory.mkdir(parents=True)
                (directory / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Ninja\n")
            paths = BuildPaths(root, root / "gitignore/ort", root / "gitignore/genai", root / "gitignore/logs",
                               ort_build, genai_build, root / "gitignore/ort-sdk")
            plan = {item["label"]: item for item in build_plan(paths, root / "gitignore/python.exe", 8)}
            self.assertIn("Ninja", plan["ort-configure"]["argv"])
            self.assertIn("Ninja", plan["genai-configure"]["argv"])
            self.assertNotIn("-A", plan["genai-configure"]["argv"])
            self.assertIn(str(genai_build), plan["genai-build"]["argv"])
            self.assertIn(str(root / "gitignore/ort-sdk"), plan["ort-install"]["argv"])
            with self.assertRaises(ValueError):
                BuildPaths(root, paths.ort, paths.genai, paths.logs, root / "outside/Release").checked()

    def test_all_build_outputs_are_isolated_and_only_native_d3d12_is_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = BuildPaths(root, root / "gitignore/ort", root / "gitignore/genai", root / "gitignore/logs")
            plan = build_plan(paths, root / "gitignore/venv/Scripts/python.exe", 8)
            by_name = {item["label"]: item for item in plan}
            self.assertEqual(7, len(plan))
            configure = by_name["ort-configure"]["argv"]
            self.assertIn("--use_webgpu", configure)
            self.assertIn("onnxruntime_ENABLE_DAWN_BACKEND_D3D12=ON", configure)
            self.assertIn("onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=OFF", configure)
            self.assertNotIn("--build_wasm", configure)
            self.assertIn("onnxruntime_providers_shared", by_name["ort-build"]["argv"])
            self.assertEqual(str(paths.ort_home), by_name["ort-install"]["argv"][-1])
            self.assertIn("-DORT_HOME=" + str(paths.ort_home), by_name["genai-configure"]["argv"])
            self.assertIn("-DUSE_CUDA=OFF", by_name["genai-configure"]["argv"])
            self.assertTrue(all(Path(item["cwd"]).is_relative_to(root / "gitignore") for item in plan))
            self.assertFalse(any("upload" in argument.lower() for item in plan for argument in item["argv"]))

    def test_sources_and_logs_cannot_escape_ignored_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path in [root / "source", root / "gitignore/../outside", root.parent / "another-project"]:
                with self.assertRaises(ValueError):
                    ignored_path(root, path)
            with self.assertRaises(ValueError):
                build_plan(BuildPaths(root, root / "gitignore/ort", root / "gitignore/genai", root / "logs"), Path("python"), 8)

    def test_sdk_normalization_preserves_install_and_rejects_incomplete_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            (home / "bin").mkdir(); (home / "lib").mkdir(); (home / "include/onnxruntime").mkdir(parents=True)
            (home / "bin/onnxruntime.dll").write_bytes(b"dll")
            (home / "lib/onnxruntime.lib").write_bytes(b"lib")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                normalize_ort_install(home)
            (home / "include/onnxruntime/onnxruntime_c_api.h").write_bytes(b"header")
            normalize_ort_install(home)
            self.assertEqual(b"dll", (home / "lib/onnxruntime.dll").read_bytes())
            self.assertEqual(b"header", (home / "include/onnxruntime_c_api.h").read_bytes())
            self.assertTrue((home / "bin/onnxruntime.dll").is_file())
            self.assertTrue((home / "include/onnxruntime/onnxruntime_c_api.h").is_file())

    def test_remote_host_is_rejected_before_any_build_command(self):
        with patch("evolution.build_native_reference.socket.gethostname", return_value="webgfx-103"), \
             patch("evolution.build_native_reference.subprocess.run") as run, \
             patch("sys.argv", ["build", "--workspace", "x", "--ort-source", "x", "--genai-source", "x", "--log-root", "x"]):
            with self.assertRaises(SystemExit):
                main()
            run.assert_not_called()

    def test_plan_only_emits_exact_commands_without_running_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = ["build", "--workspace", str(root), "--ort-source", str(root / "gitignore/ort"),
                    "--genai-source", str(root / "gitignore/genai"), "--log-root", str(root / "gitignore/logs"), "--plan-only"]
            with patch("evolution.build_native_reference.socket.gethostname", return_value="webgfx-104"), \
                 patch("evolution.build_native_reference.subprocess.run") as run, patch("sys.argv", args):
                self.assertEqual(0, main())
                run.assert_not_called()
            self.assertEqual(7, len(json.loads((root / "gitignore/logs/build-plan.json").read_text())))


if __name__ == "__main__":
    unittest.main()
