"""FeishuBot handler mixin -- slash commands, pipeline execution, file upload.

Separated from feishu_bot.py for size.  Mixed into FeishuBot via MRO.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateFileRequest,
    CreateFileRequestBody,
    CreateMessageRequest,
    CreateMessageRequestBody,
)

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
from nanoresearch.pipeline.workspace import Workspace

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".nanobot" / "workspace" / "research"
_CHAT_MEMORY_DIR = Path.home() / ".nanobot" / "chat_memory"


class _FeishuBotHandlersMixin:
    """Second-half mixin: command handlers, pipeline execution, file upload."""

    # ─── command handlers ───

    def _cmd_help(self, chat_id: str, message_id: str) -> None:
        help_text = (
            "NanoResearch 飞书助手\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "我是 AI 研究助理，你可以直接跟我聊天，也可以用命令：\n\n"
            "对话示例：\n"
            "  「帮我研究一下多模态情感分析」\n"
            "  「Transformer 在 NLP 中的最新进展有哪些？」\n"
            "  「记住我偏好用 PyTorch」\n\n"
            "命令列表：\n"
            "  /run <主题>  — 启动研究 pipeline\n"
            "  /status     — 查看当前任务状态\n"
            "  /list       — 列出所有历史会话\n"
            "  /stop       — 停止当前正在运行的任务\n"
            "  /export     — 重新导出最近的研究结果\n"
            "  /new        — 清除对话记忆，开始新对话\n"
            "  /help       — 显示此帮助\n\n"
            "Pipeline 阶段：\n"
            "  IDEATION → PLANNING → EXPERIMENT → FIGURE_GEN → WRITING → REVIEW\n"
            "完成后会自动推送 paper.pdf。"
        )
        self.reply_message(message_id, help_text)

    def _cmd_new(self, chat_id: str, message_id: str) -> None:
        with self._lock:
            self._pending_env_select.pop(chat_id, None)
        chat_lock = self._get_chat_lock(chat_id)
        with chat_lock:
            with self._chat_locks_lock:
                old_mem = self._memories.pop(chat_id, None)
            if old_mem is not None:
                old_mem._invalidated = True
            safe_id = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', chat_id)[:120]
            mem_file = _CHAT_MEMORY_DIR / f"{safe_id}.json"
            try:
                mem_file.unlink(missing_ok=True)
            except OSError:
                pass
        self.reply_message(message_id, "对话记忆已清除，开始新对话！")

    def _cmd_status(self, chat_id: str, message_id: str) -> None:
        with self._lock:
            task = self._running_tasks.get(chat_id)
        if not task:
            self.reply_message(message_id, "当前没有正在运行的任务。")
            return
        status_text = (
            f"当前任务状态\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"主题: {task['topic']}\n"
            f"状态: {task['status']}\n"
            f"工作目录: {task.get('workspace', 'N/A')}"
        )
        self.reply_message(message_id, status_text)

    def _cmd_list(self, chat_id: str, message_id: str) -> None:
        if not _DEFAULT_ROOT.is_dir():
            self.reply_message(message_id, "没有找到历史会话。")
            return

        try:
            all_dirs = sorted(
                [p for p in _DEFAULT_ROOT.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            self.reply_message(message_id, "没有找到历史会话。")
            return

        lines = ["历史研究会话", "━━━━━━━━━━━━━━━━━━━━"]
        count = 0
        for session_dir in all_dirs:
            manifest_path = session_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                topic = str(data.get("topic", "?"))[:60]
                stage = data.get("current_stage", "?")
                sid = data.get("session_id", "?")[:8]
                lines.append(f"  [{sid}] {stage:12s} {topic}")
                count += 1
                if count >= 10:
                    lines.append(f"  ... 还有更多（共 {len(all_dirs)} 个）")
                    break
            except (json.JSONDecodeError, OSError):
                continue

        self.reply_message(message_id, "\n".join(lines) if count > 0 else "没有找到历史会话。")

    def _cmd_stop(self, chat_id: str, message_id: str) -> None:
        with self._lock:
            self._pending_env_select.pop(chat_id, None)
            task = self._running_tasks.get(chat_id)
            if not task:
                self.reply_message(message_id, "当前没有正在运行的任务。")
                return
            task["cancel"] = True
        self.reply_message(message_id, f"正在停止任务: {task['topic'][:50]}...")

    def _cmd_export(self, chat_id: str, message_id: str) -> None:
        ws_path = None
        with self._lock:
            task = self._running_tasks.get(chat_id)
            if task and task.get("workspace"):
                ws_path = Path(task["workspace"])

        if ws_path is None or not ws_path.exists():
            if _DEFAULT_ROOT.is_dir():
                try:
                    dirs_by_mtime = sorted(
                        [p for p in _DEFAULT_ROOT.iterdir() if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                except OSError:
                    dirs_by_mtime = []
                for session_dir in dirs_by_mtime:
                    manifest_path = session_dir / "manifest.json"
                    if manifest_path.is_file():
                        try:
                            data = json.loads(manifest_path.read_text(encoding="utf-8"))
                            if data.get("current_stage") in ("done", "review", "DONE", "REVIEW"):
                                ws_path = session_dir
                                break
                        except (json.JSONDecodeError, OSError):
                            continue
                if ws_path is None:
                    for session_dir in dirs_by_mtime:
                        if (session_dir / "manifest.json").is_file():
                            ws_path = session_dir
                            break

        if ws_path is None or not ws_path.exists():
            self.reply_message(message_id, "没有找到可导出的研究会话。")
            return

        self.reply_message(message_id, f"正在导出...\n工作目录: {ws_path}")

        try:
            workspace = Workspace.load(ws_path)
            export_path = workspace.export()
            pdf_path = export_path / "paper.pdf"

            summary = (
                f"导出完成！\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"主题: {workspace.manifest.topic}\n"
                f"输出目录: {export_path}\n"
            )
            if pdf_path.exists():
                summary += f"PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)\n"

            summary += "\n生成文件:\n"
            for f in sorted(export_path.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(export_path)
                    size = f.stat().st_size
                    summary += f"  {rel} ({size / 1024:.1f} KB)\n"

            self.send_message(chat_id, summary)

            if pdf_path.exists():
                self._upload_file(chat_id, pdf_path)

        except Exception as e:
            self.send_message(chat_id, f"导出失败: {e}")

    @staticmethod
    def _discover_conda_envs() -> tuple[list[dict[str, str]], bool]:
        import subprocess
        envs: list[dict[str, str]] = []
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for env_path in data.get("envs", []):
                    p = Path(env_path)
                    name = p.name if p.name != "base" else "base"
                    if str(p) == data.get("root_prefix", ""):
                        name = "base"
                    if sys.platform == "win32":
                        py = p / "python.exe"
                    else:
                        py = p / "bin" / "python"
                    if py.exists():
                        envs.append({"name": name, "python": str(py), "path": str(p)})
        except subprocess.TimeoutExpired:
            logger.warning("conda env 发现超时 (15s)")
            return [], True
        except Exception as e:
            logger.warning("conda env 发现失败: %s", e)
        return envs, False

    def _cmd_run(self, chat_id: str, message_id: str, topic: str = "") -> None:
        with self._lock:
            if chat_id in self._running_tasks:
                existing = self._running_tasks[chat_id]
                if existing.get("status") not in ("completed", "failed", "stopped"):
                    self.reply_message(
                        message_id,
                        f"已有任务正在运行: {existing['topic'][:50]}\n"
                        f"请等待完成或发送 /stop 停止。"
                    )
                    return
            if chat_id in self._pending_env_select:
                self.reply_message(message_id, "正在等待环境选择，请先回复字母。")
                return

        envs, timed_out = self._discover_conda_envs()
        if timed_out:
            self.reply_message(
                message_id,
                "环境发现超时，将自动创建专用环境。\n如需指定环境，请在 config.json 中设置 experiment_conda_env。"
            )
            self._start_pipeline(chat_id, message_id, topic, env_name="")
            return
        if envs:
            lines = ["选择实验运行环境:"]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i, env in enumerate(envs[:10]):
                letter = letters[i]
                lines.append(f"{letter}. {env['name']}  ({env['path']})")
            lines.append(f"{letters[len(envs[:10])]}.  新建专用环境（自动创建）")
            lines.append("")
            lines.append("回复对应字母即可。")

            with self._lock:
                self._pending_env_select[chat_id] = {
                    "topic": topic,
                    "envs": envs[:10],
                    "message_id": message_id,
                }
            self.reply_message(message_id, "\n".join(lines))
            return
        else:
            logger.info("No conda envs found, using auto-fallback (venv)")
            self._start_pipeline(chat_id, message_id, topic, env_name="")

    def _handle_env_selection(
        self, chat_id: str, message_id: str, text: str, pending: dict
    ) -> None:
        choice = text.strip().upper()
        envs = pending["envs"]
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        new_env_letter = letters[len(envs)] if len(envs) < 26 else ""

        selected_env = ""
        if len(choice) == 1 and choice in letters:
            idx = letters.index(choice)
            if idx < len(envs):
                selected_env = envs[idx]["name"]
            elif choice == new_env_letter:
                selected_env = ""
            else:
                self.reply_message(message_id, f"无效选择 '{choice}'，请重新输入字母。")
                return
        else:
            for env in envs:
                if env["name"].lower() == text.strip().lower():
                    selected_env = env["name"]
                    break
            else:
                self.reply_message(message_id, "请输入对应字母（如 A/B/C）或环境名。")
                return

        with self._lock:
            if self._pending_env_select.pop(chat_id, None) is None:
                return

        env_display = selected_env if selected_env else "新建专用环境"
        self.reply_message(message_id, f"已选环境: {env_display}")
        self._start_pipeline(chat_id, message_id, pending["topic"], env_name=selected_env)

    def _start_pipeline(
        self, chat_id: str, message_id: str, topic: str, env_name: str
    ) -> None:
        with self._lock:
            self._running_tasks[chat_id] = {
                "topic": topic,
                "status": "starting",
                "cancel": False,
                "env_name": env_name,
            }

        self.reply_message(
            message_id,
            f"收到！开始研究:\n{topic}\n\n"
            f"实验环境: {env_name or '自动创建'}\n"
            f"Pipeline 已启动，我会在每个阶段结束时汇报进度。"
        )

        memory = self._get_memory(chat_id)
        memory.add_fact(f"启动了研究: {topic[:80]}")
        memory.save()

        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(chat_id, topic, env_name),
            daemon=True,
        )
        with self._lock:
            self._pipeline_threads[chat_id] = thread
        thread.start()

    # ─── pipeline execution ───

    def _run_pipeline_thread(self, chat_id: str, topic: str, env_name: str = "") -> None:
        loop = asyncio.new_event_loop()
        with self._lock:
            self._pipeline_loops[chat_id] = loop
        try:
            loop.run_until_complete(self._run_pipeline_async(chat_id, topic, env_name))
        except Exception as e:
            logger.exception("Pipeline thread crashed: %s", e)
            if not self._shutting_down:
                self.send_message(chat_id, f"Pipeline 异常退出: {e}")
            with self._lock:
                if chat_id in self._running_tasks:
                    self._running_tasks[chat_id]["status"] = "failed"
        finally:
            loop.close()
            with self._lock:
                self._pipeline_loops.pop(chat_id, None)
                self._pipeline_threads.pop(chat_id, None)

    async def _run_pipeline_async(self, chat_id: str, topic: str, env_name: str = "") -> None:
        config = ResearchConfig.load()
        if env_name:
            config = config.model_copy(update={"experiment_conda_env": env_name})
            logger.info("Pipeline 使用用户选择的 conda 环境: %s", env_name)
        workspace = Workspace.create(topic=topic, config_snapshot=config.snapshot())

        with self._lock:
            if chat_id in self._running_tasks:
                self._running_tasks[chat_id]["workspace"] = str(workspace.path)
                self._running_tasks[chat_id]["status"] = "running"

        self.send_message(
            chat_id,
            f"工作目录: {workspace.path}\n"
            f"Session: {workspace.manifest.session_id}"
        )

        stage_start_time: float = 0

        def progress_callback(stage: str, status: str, message: str) -> None:
            nonlocal stage_start_time
            with self._lock:
                task = self._running_tasks.get(chat_id, {})
                if task.get("cancel"):
                    task["status"] = "stopped"
                    raise KeyboardInterrupt("用户请求停止")
                task["status"] = f"{stage} - {status}"

            if status == "started":
                stage_start_time = time.monotonic()
                self.send_message(chat_id, f">>> 开始 {stage}...")
            elif status == "completed":
                elapsed = time.monotonic() - stage_start_time if stage_start_time else 0
                self.send_message(chat_id, f"<<< {stage} 完成 ({elapsed:.0f}s)")
            elif status == "retrying":
                self.send_message(chat_id, f"!!! {stage} 重试中: {message}")

        orchestrator = PipelineOrchestrator(workspace, config, progress_callback=progress_callback)

        try:
            result = await orchestrator.run(topic)
            await orchestrator.close()

            with self._lock:
                if chat_id in self._running_tasks:
                    self._running_tasks[chat_id]["status"] = "completed"

            try:
                export_path = workspace.export()
                pdf_path = export_path / "paper.pdf"
                tex_path = export_path / "paper.tex"

                summary = (
                    f"Pipeline 完成！\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"主题: {topic}\n"
                    f"输出目录: {export_path}\n"
                )
                if pdf_path.exists():
                    summary += f"PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)\n"
                if tex_path.exists():
                    summary += f"TeX: {tex_path}\n"

                summary += "\n生成文件:\n"
                for f in sorted(export_path.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(export_path)
                        size = f.stat().st_size
                        summary += f"  {rel} ({size / 1024:.1f} KB)\n"

                self.send_message(chat_id, summary)

                if pdf_path.exists():
                    self._upload_file(chat_id, pdf_path)

            except Exception as e:
                self.send_message(chat_id, f"Pipeline 完成，但导出失败: {e}\n原始工作目录: {workspace.path}")

        except KeyboardInterrupt:
            await orchestrator.close()
            self.send_message(chat_id, "任务已停止。")
        except Exception as e:
            await orchestrator.close()
            with self._lock:
                if chat_id in self._running_tasks:
                    self._running_tasks[chat_id]["status"] = "failed"
            self.send_message(chat_id, f"Pipeline 失败: {e}")

    # ─── file upload ───

    def _upload_file(self, chat_id: str, file_path: Path) -> None:
        try:
            with open(file_path, "rb") as f:
                request = CreateFileRequest.builder() \
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type("pdf")
                        .file_name(file_path.name)
                        .file(f)
                        .build()
                    ).build()

                response = self.client.im.v1.file.create(request)
                if response.success():
                    file_key = response.data.file_key
                    content = json.dumps({"file_key": file_key})
                    msg_request = CreateMessageRequest.builder() \
                        .receive_id_type("chat_id") \
                        .request_body(
                            CreateMessageRequestBody.builder()
                            .receive_id(chat_id)
                            .msg_type("file")
                            .content(content)
                            .build()
                        ).build()
                    self.client.im.v1.message.create(msg_request)
                else:
                    logger.error("上传文件失败: %s", response.msg)
                    self.send_message(chat_id, f"PDF 上传失败: {response.msg}\n文件路径: {file_path}")
        except Exception as e:
            logger.error("上传文件异常: %s", e)
            self.send_message(chat_id, f"PDF 上传异常: {e}\n文件路径: {file_path}")
