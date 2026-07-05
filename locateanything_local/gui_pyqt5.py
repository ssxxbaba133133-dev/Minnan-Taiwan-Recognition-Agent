# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import time
import traceback
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

# On Windows, importing PyQt5 before CUDA PyTorch can make torch fail later with
# WinError 1114 while loading c10.dll. Preload torch before Qt DLLs are loaded.
PRELOAD_TORCH_ERROR = ""
try:
    import torch as _PRELOADED_TORCH
except Exception as exc:
    _PRELOADED_TORCH = None
    PRELOAD_TORCH_ERROR = f"{type(exc).__name__}: {exc}"

from PyQt5.QtCore import QObject, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from locate import (  # noqa: E402
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_LOCAL_FILES_ONLY,
    LocateAnythingRunner,
    build_prompt,
    collect_images,
    save_json,
)


class ImageView(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(QSize(360, 300))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "QLabel { border: 1px solid #cbd5e1; background: #f8fafc; color: #475569; }"
        )

    def set_image(self, path: Optional[Path], empty_text: str) -> None:
        if not path:
            self._pixmap = None
            self.setText(empty_text)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = None
            self.setText(f"无法读取图片:\n{path}")
            return
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size() - QSize(12, 12),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class LocateWorker(QObject):
    progress = pyqtSignal(str, int, int)
    image_done = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        images: List[Path],
        query: str,
        mode: str,
        full_prompt: str,
        output_dir: Path,
        model_id: str,
        device: str,
        dtype_name: str,
        max_new_tokens: int,
        generation_mode: str,
        continue_on_error: bool,
        local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    ) -> None:
        super().__init__()
        self.images = images
        self.query = query
        self.mode = mode
        self.full_prompt = full_prompt
        self.output_dir = output_dir
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype_name
        self.max_new_tokens = max_new_tokens
        self.generation_mode = generation_mode
        self.continue_on_error = continue_on_error
        self.local_files_only = local_files_only

    def run(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            prompt = self.full_prompt.strip() or build_prompt(self.query, self.mode)
            runner = LocateAnythingRunner(
                model_id=self.model_id,
                device=self.device,
                dtype_name=self.dtype_name,
                max_new_tokens=self.max_new_tokens,
                generation_mode=self.generation_mode,
                local_files_only=self.local_files_only,
            )
            results: List[Dict[str, Any]] = []
            for index, image_path in enumerate(self.images, 1):
                self.progress.emit(str(image_path), index, len(self.images))
                try:
                    item = runner.locate_image(image_path, prompt, self.output_dir)
                    item["ok"] = True
                    item["error"] = None
                except Exception as exc:
                    item = {
                        "ok": False,
                        "error": str(exc),
                        "image": str(image_path),
                        "prompt": prompt,
                        "answer": "",
                        "boxes": [],
                        "points": [],
                        "result_image": "",
                    }
                    if not self.continue_on_error:
                        results.append(item)
                        self.image_done.emit(item)
                        break
                results.append(item)
                self.image_done.emit(item)

            summary = {
                "model_id": self.model_id,
                "output_dir": str(self.output_dir),
                "query": self.query,
                "prompt": prompt,
                "count": len(results),
                "success_count": sum(1 for item in results if item.get("ok")),
                "box_count": sum(len(item.get("boxes", [])) for item in results),
                "point_count": sum(len(item.get("points", [])) for item in results),
                "results": results,
            }
            json_path = self.output_dir / f"gui_locate_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            save_json(json_path, summary)
            summary["json_path"] = str(json_path)
            self.finished.emit(summary)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LocateAnything-3B 本地可视化")
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self.images: List[Path] = []
        self.results_by_image: Dict[str, Dict[str, Any]] = {}
        self.thread: Optional[QThread] = None
        self.worker: Optional[LocateWorker] = None

        self.input_path = QLineEdit(DEFAULT_INPUT_PATH)
        self.output_dir = QLineEdit(DEFAULT_OUTPUT_DIR)
        self.query_box = QPlainTextEdit()
        self.query_box.setPlaceholderText("输入定位指令，例如：roof ridge dragon / 屋顶上的龙 / temple plaque")
        self.query_box.setFixedHeight(80)
        self.full_prompt_box = QPlainTextEdit()
        self.full_prompt_box.setPlaceholderText("可选：完整 prompt。填写后会覆盖上面的定位指令模板。")
        self.full_prompt_box.setFixedHeight(66)

        self.model_id = QLineEdit(DEFAULT_MODEL_ID)
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["", "cuda", "cuda:0", "cpu"])
        self.device.setCurrentText(DEFAULT_DEVICE)
        self.dtype = QComboBox()
        self.dtype.setEditable(True)
        self.dtype.addItems(["", "bfloat16", "float16", "float32"])
        self.dtype.setCurrentText(DEFAULT_DTYPE)
        self.mode = QComboBox()
        self.mode.addItems(["all", "single", "point"])
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(128, 32768)
        self.max_tokens.setValue(DEFAULT_MAX_NEW_TOKENS)
        self.max_tokens.setSingleStep(512)
        self.generation_mode = QLineEdit("hybrid")
        self.continue_on_error = QCheckBox("批量出错后继续")
        self.continue_on_error.setChecked(True)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)
        self.input_view = ImageView("拖拽图片到窗口，或选择输入目录")
        self.output_view = ImageView("输出图会显示在这里")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.progress = QProgressBar()

        self.refresh_btn = QPushButton("刷新输入")
        self.file_btn = QPushButton("选择图片")
        self.input_dir_btn = QPushButton("选择输入目录")
        self.output_dir_btn = QPushButton("选择输出目录")
        self.run_current_btn = QPushButton("处理当前图片")
        self.run_all_btn = QPushButton("处理全部图片")

        self.refresh_btn.clicked.connect(self.refresh_images)
        self.file_btn.clicked.connect(self.choose_file)
        self.input_dir_btn.clicked.connect(self.choose_input_dir)
        self.output_dir_btn.clicked.connect(self.choose_output_dir)
        self.run_current_btn.clicked.connect(self.run_current)
        self.run_all_btn.clicked.connect(self.run_all)

        self.setCentralWidget(self.build_ui())
        self.refresh_images()
        self.append_environment_report()

    def build_ui(self) -> QWidget:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        path_group = QGroupBox("目录配置")
        path_layout = QVBoxLayout(path_group)
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("输入"))
        input_row.addWidget(self.input_path, 1)
        input_row.addWidget(self.file_btn)
        input_row.addWidget(self.input_dir_btn)
        input_row.addWidget(self.refresh_btn)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("输出"))
        output_row.addWidget(self.output_dir, 1)
        output_row.addWidget(self.output_dir_btn)
        path_layout.addLayout(input_row)
        path_layout.addLayout(output_row)
        root_layout.addWidget(path_group)

        prompt_group = QGroupBox("输入指令")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.addWidget(self.query_box)
        prompt_layout.addWidget(self.full_prompt_box)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("模式"))
        controls.addWidget(self.mode)
        controls.addWidget(QLabel("设备"))
        controls.addWidget(self.device)
        controls.addWidget(QLabel("精度"))
        controls.addWidget(self.dtype)
        controls.addWidget(QLabel("max tokens"))
        controls.addWidget(self.max_tokens)
        controls.addWidget(QLabel("generation"))
        controls.addWidget(self.generation_mode)
        controls.addWidget(self.continue_on_error)
        controls.addStretch(1)
        controls.addWidget(self.run_current_btn)
        controls.addWidget(self.run_all_btn)
        prompt_layout.addLayout(controls)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型"))
        model_row.addWidget(self.model_id, 1)
        prompt_layout.addLayout(model_row)
        root_layout.addWidget(prompt_group)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("图片列表"))
        left_layout.addWidget(self.image_list, 1)
        splitter.addWidget(left_panel)

        compare_splitter = QSplitter(Qt.Horizontal)
        input_group = QGroupBox("输入图")
        input_layout = QVBoxLayout(input_group)
        input_layout.addWidget(self.input_view)
        output_group = QGroupBox("输出图")
        output_layout = QVBoxLayout(output_group)
        output_layout.addWidget(self.output_view)
        compare_splitter.addWidget(input_group)
        compare_splitter.addWidget(output_group)
        compare_splitter.setSizes([520, 520])
        splitter.addWidget(compare_splitter)
        splitter.setSizes([260, 1000])
        root_layout.addWidget(splitter, 1)

        bottom = QGroupBox("日志")
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.addWidget(self.progress)
        bottom_layout.addWidget(self.log)
        root_layout.addWidget(bottom)
        return root

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if not paths:
            return
        first = paths[0]
        if first.is_file() and first.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
            self.input_path.setText(str(first))
            self.refresh_images()
        elif first.is_dir():
            self.input_path.setText(str(first))
            self.refresh_images()

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            self.input_path.text() or str(PROJECT_ROOT),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if path:
            self.input_path.setText(path)
            self.refresh_images()

    def choose_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输入目录", self.input_path.text() or str(PROJECT_ROOT))
        if path:
            self.input_path.setText(path)
            self.refresh_images()

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir.text() or str(PROJECT_ROOT))
        if path:
            self.output_dir.setText(path)

    def refresh_images(self) -> None:
        self.images = collect_images(Path(self.input_path.text()).expanduser())
        self.image_list.clear()
        for image in self.images:
            item = QListWidgetItem(image.name)
            item.setToolTip(str(image))
            item.setData(Qt.UserRole, str(image))
            self.image_list.addItem(item)
        self.append_log(f"找到 {len(self.images)} 张图片。")
        if self.images:
            self.image_list.setCurrentRow(0)
        else:
            self.input_view.set_image(None, "没有找到图片。可以拖拽图片到窗口。")
            self.output_view.set_image(None, "输出图会显示在这里")

    def on_image_selected(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        del previous
        image_path = self.current_image()
        self.input_view.set_image(image_path, "没有选择输入图")
        result = self.results_by_image.get(str(image_path.resolve())) if image_path else None
        result_image = Path(result["result_image"]) if result and result.get("result_image") else None
        self.output_view.set_image(result_image, "当前图片还没有输出结果")

    def current_image(self) -> Optional[Path]:
        item = self.image_list.currentItem()
        if not item:
            return None
        return Path(item.data(Qt.UserRole))

    def run_current(self) -> None:
        image = self.current_image()
        if image is None:
            QMessageBox.warning(self, "没有图片", "请先选择或拖拽一张图片。")
            return
        self.start_worker([image])

    def run_all(self) -> None:
        if not self.images:
            QMessageBox.warning(self, "没有图片", "输入目录里没有可处理的图片。")
            return
        self.start_worker(self.images)

    def start_worker(self, images: List[Path]) -> None:
        query = self.query_box.toPlainText().strip()
        full_prompt = self.full_prompt_box.toPlainText().strip()
        if not query and not full_prompt:
            QMessageBox.warning(self, "缺少指令", "请输入定位指令，或者填写完整 prompt。")
            return
        output_dir = Path(self.output_dir.text()).expanduser()
        self.set_running(True)
        self.progress.setRange(0, len(images))
        self.progress.setValue(0)
        self.append_log(f"开始处理 {len(images)} 张图片，输出目录：{output_dir}")

        self.thread = QThread()
        self.worker = LocateWorker(
            images=images,
            query=query,
            mode=self.mode.currentText(),
            full_prompt=full_prompt,
            output_dir=output_dir,
            model_id=self.model_id.text().strip() or DEFAULT_MODEL_ID,
            device=self.device.currentText().strip(),
            dtype_name=self.dtype.currentText().strip(),
            max_new_tokens=int(self.max_tokens.value()),
            generation_mode=self.generation_mode.text().strip() or "hybrid",
            continue_on_error=self.continue_on_error.isChecked(),
            local_files_only=DEFAULT_LOCAL_FILES_ONLY,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.image_done.connect(self.on_image_done)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: self.set_running(False))
        self.thread.start()

    def on_progress(self, image: str, index: int, total: int) -> None:
        self.progress.setValue(index - 1)
        self.append_log(f"[{index}/{total}] {image}")

    def on_image_done(self, item: Dict[str, Any]) -> None:
        image = Path(str(item.get("image", "")))
        if image:
            self.results_by_image[str(image.resolve())] = item
        status = "OK" if item.get("ok") else "ERROR"
        boxes = len(item.get("boxes", []))
        points = len(item.get("points", []))
        self.append_log(f"{status}: {image.name} | boxes={boxes}, points={points}")
        if item.get("answer"):
            answer = str(item.get("answer", "")).replace("\n", " ").strip()
            if len(answer) > 500:
                answer = f"{answer[:500]}..."
            self.append_log(f"raw answer: {answer}")
        if item.get("error"):
            self.append_log(f"error: {item.get('error')}")
        current = self.current_image()
        if current and image and current.resolve() == image.resolve():
            result_image = Path(item["result_image"]) if item.get("result_image") else None
            self.output_view.set_image(result_image, "没有输出图")

    def on_finished(self, summary: Dict[str, Any]) -> None:
        self.progress.setValue(self.progress.maximum())
        self.append_log(
            f"完成：成功 {summary.get('success_count', 0)}/{summary.get('count', 0)}，"
            f"boxes={summary.get('box_count', 0)}，points={summary.get('point_count', 0)}"
        )
        self.append_log(f"结果 JSON：{summary.get('json_path', '')}")

    def on_failed(self, detail: str) -> None:
        self.append_log(detail)
        QMessageBox.critical(self, "运行失败", detail[-3000:])

    def set_running(self, running: bool) -> None:
        for widget in [
            self.refresh_btn,
            self.file_btn,
            self.input_dir_btn,
            self.output_dir_btn,
            self.run_current_btn,
            self.run_all_btn,
        ]:
            widget.setEnabled(not running)

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def append_environment_report(self) -> None:
        self.append_log(f"Python: {sys.executable}")
        for name in ["transformers", "peft", "decord", "lmdb", "PyQt5"]:
            status = "OK" if importlib.util.find_spec(name) else "MISSING"
            self.append_log(f"{name}: {status}")
        if _PRELOADED_TORCH is None:
            self.append_log(f"torch: ERROR {PRELOAD_TORCH_ERROR}")
            return
        self.append_log(f"torch: {_PRELOADED_TORCH.__version__}")
        self.append_log(f"cuda: {_PRELOADED_TORCH.cuda.is_available()} / {_PRELOADED_TORCH.version.cuda}")
        if _PRELOADED_TORCH.cuda.is_available():
            self.append_log(f"gpu: {_PRELOADED_TORCH.cuda.get_device_name(0)}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
