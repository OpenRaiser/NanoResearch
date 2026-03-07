"""飞书机器人集成 — 通过飞书消息触发 NanoResearch pipeline。

使用飞书 WebSocket 长连接模式（无需公网服务器）。

用法：
    python -m nanoresearch.feishu_bot

环境变量（或在 ~/.nanobot/config.json 中配置）：
    FEISHU_APP_ID      飞书应用 App ID
    FEISHU_APP_SECRET  飞书应用 App Secret
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Windows UTF-8 fix
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
    CreateFileRequest,
    CreateFileRequestBody,
)

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
from nanoresearch.pipeline.workspace import Workspace

logger = logging.getLogger(__name__)

# ─── Config ───
_DEFAULT_ROOT = Path.home() / ".nanobot" / "workspace" / "research"


def _load_feishu_credentials() -> tuple[str, str]:
    """从环境变量或 config.json 加载飞书凭证。"""
    app_id = os.environ.get("FEISHU_APP_ID", "")
    app_secret = os.environ.get("FEISHU_APP_SECRET", "")

    if not app_id or not app_secret:
        config_path = Path.home() / ".nanobot" / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                feishu = data.get("feishu", {})
                app_id = app_id or feishu.get("app_id", "")
                app_secret = app_secret or feishu.get("app_secret", "")
            except (json.JSONDecodeError, OSError):
                pass

    if not app_id or not app_secret:
        raise RuntimeError(
            "飞书凭证未配置。请设置环境变量 FEISHU_APP_ID / FEISHU_APP_SECRET，\n"
            "或在 ~/.nanobot/config.json 中添加：\n"
            '  "feishu": {"app_id": "cli_xxx", "app_secret": "xxx"}'
        )
    return app_id, app_secret


# ═══════════════════════════════════════════════════════════════
#  飞书消息收发
# ═══════════════════════════════════════════════════════════════

class FeishuBot:
    """飞书 NanoResearch 机器人。"""

    def __init__(self, app_id: str, app_secret: str) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        # 正在运行的任务 {chat_id: {"topic": str, "workspace": Path, "status": str}}
        self._running_tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def send_message(self, chat_id: str, text: str, msg_type: str = "open_id") -> None:
        """发送文本消息到会话。"""
        # 文本过长时截断
        if len(text) > 4000:
            text = text[:3900] + "\n\n... (消息过长，已截断)"

        content = json.dumps({"text": text})
        request = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("text")
                .content(content)
                .build()
            ).build()

        response = self.client.im.v1.message.create(request)
        if not response.success():
            logger.error("发送消息失败: %s %s", response.code, response.msg)

    def reply_message(self, message_id: str, text: str) -> None:
        """回复某条消息。"""
        if len(text) > 4000:
            text = text[:3900] + "\n\n... (消息过长，已截断)"

        content = json.dumps({"text": text})
        request = ReplyMessageRequest.builder() \
            .message_id(message_id) \
            .request_body(
                ReplyMessageRequestBody.builder()
                .msg_type("text")
                .content(content)
                .build()
            ).build()

        response = self.client.im.v1.message.reply(request)
        if not response.success():
            logger.error("回复消息失败: %s %s", response.code, response.msg)

    def send_card(self, chat_id: str, card: dict) -> None:
        """发送卡片消息（用于进度展示）。"""
        content = json.dumps(card)
        request = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("interactive")
                .content(content)
                .build()
            ).build()

        response = self.client.im.v1.message.create(request)
        if not response.success():
            logger.error("发送卡片失败: %s %s", response.code, response.msg)

    # ─── 命令解析 ───

    @staticmethod
    def _clean_text(raw: str) -> str:
        """清理飞书消息文本：去 mention、全角转半角、strip。"""
        text = raw.strip()
        # 去掉 @_user_N mention 占位符
        text = re.sub(r'@_user_\d+\s*', '', text)
        # 全角斜杠 → 半角
        text = text.replace('\uff0f', '/')
        return text.strip()

    def handle_message(self, chat_id: str, message_id: str, text: str, sender_id: str) -> None:
        """解析用户消息并执行对应命令。"""
        raw = text
        text = self._clean_text(raw)

        if not text:
            return

        logger.info("消息: chat=%s raw=%r cleaned=%r", chat_id, raw[:200], text[:100])

        # 提取命令关键词（支持 /cmd 和中文）
        lower = text.lower()

        # 命令路由表：(匹配条件, 处理函数)
        # 先检查精确命令，再检查前缀命令
        _EXACT_CMDS = {
            "/help": "help", "帮助": "help", "/帮助": "help",
            "/status": "status", "状态": "status", "/状态": "status",
            "/list": "list", "列表": "list", "/列表": "list",
            "/stop": "stop", "停止": "stop", "/停止": "stop",
            "/export": "export", "导出": "export", "/导出": "export",
        }

        cmd = _EXACT_CMDS.get(lower)
        if cmd == "help":
            self._cmd_help(chat_id, message_id)
            return
        elif cmd == "status":
            self._cmd_status(chat_id, message_id)
            return
        elif cmd == "list":
            self._cmd_list(chat_id, message_id)
            return
        elif cmd == "stop":
            self._cmd_stop(chat_id, message_id)
            return
        elif cmd == "export":
            self._cmd_export(chat_id, message_id)
            return

        # /run <topic> — 必须用命令启动研究
        if lower.startswith("/run ") or lower.startswith("研究 "):
            topic = text.split(" ", 1)[1].strip() if " " in text else ""
            if len(topic) < 5:
                self.reply_message(message_id, "主题太短了，请描述更详细一些。\n用法: /run <研究主题>")
                return
            self._cmd_run(chat_id, message_id, topic)
            return

        # 不认识的消息 — 回显原始文本帮助调试，并提示用法
        self.reply_message(
            message_id,
            f"未识别的命令: {text[:50]}\n\n"
            "发送 /help 查看帮助\n"
            "发送 /run <主题> 启动研究\n"
            "例如: /run Multimodal sarcasm detection"
        )

    def _cmd_help(self, chat_id: str, message_id: str) -> None:
        help_text = (
            "NanoResearch 飞书机器人\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "直接发送研究主题即可启动 pipeline：\n"
            "  例：Multimodal sarcasm detection with counterfactual debiasing\n\n"
            "命令列表：\n"
            "  /run <主题>  — 启动研究 pipeline\n"
            "  /status     — 查看当前任务状态\n"
            "  /list       — 列出所有历史会话\n"
            "  /stop       — 停止当前正在运行的任务\n"
            "  /export     — 重新导出最近的研究结果\n"
            "  /help       — 显示此帮助\n\n"
            "Pipeline 阶段：\n"
            "  IDEATION → PLANNING → EXPERIMENT → FIGURE_GEN → WRITING → REVIEW\n"
            "完成后会自动推送 paper.pdf。"
        )
        self.reply_message(message_id, help_text)

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

        lines = ["历史研究会话", "━━━━━━━━━━━━━━━━━━━━"]
        count = 0
        for session_dir in sorted(_DEFAULT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
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
                    lines.append(f"  ... 还有更多（共 {sum(1 for _ in _DEFAULT_ROOT.iterdir())} 个）")
                    break
            except (json.JSONDecodeError, OSError):
                continue

        self.reply_message(message_id, "\n".join(lines) if count > 0 else "没有找到历史会话。")

    def _cmd_stop(self, chat_id: str, message_id: str) -> None:
        with self._lock:
            task = self._running_tasks.get(chat_id)

        if not task:
            self.reply_message(message_id, "当前没有正在运行的任务。")
            return

        task["cancel"] = True
        self.reply_message(message_id, f"正在停止任务: {task['topic'][:50]}...")

    def _cmd_export(self, chat_id: str, message_id: str) -> None:
        """重新导出最近一次完成的研究结果。"""
        # 优先使用当前 chat 的任务 workspace
        ws_path = None
        with self._lock:
            task = self._running_tasks.get(chat_id)
            if task and task.get("workspace"):
                ws_path = Path(task["workspace"])

        # 如果没有，找最近的已完成 session（按修改时间排序）
        if ws_path is None or not ws_path.exists():
            if _DEFAULT_ROOT.is_dir():
                dirs_by_mtime = sorted(
                    _DEFAULT_ROOT.iterdir(),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
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
                # 如果还没找到 done/review 的，就取最新的
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

    def _cmd_run(self, chat_id: str, message_id: str, topic: str) -> None:
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

            self._running_tasks[chat_id] = {
                "topic": topic,
                "status": "starting",
                "cancel": False,
            }

        self.reply_message(
            message_id,
            f"收到！开始研究:\n{topic}\n\n"
            f"Pipeline 已启动，我会在每个阶段结束时汇报进度。"
        )

        # 在后台线程中运行 pipeline
        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(chat_id, topic),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_thread(self, chat_id: str, topic: str) -> None:
        """在后台线程中运行 pipeline（包含独立的事件循环）。"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_pipeline_async(chat_id, topic))
        except Exception as e:
            logger.exception("Pipeline thread crashed: %s", e)
            self.send_message(chat_id, f"Pipeline 异常退出: {e}")
            with self._lock:
                if chat_id in self._running_tasks:
                    self._running_tasks[chat_id]["status"] = "failed"
        finally:
            loop.close()

    async def _run_pipeline_async(self, chat_id: str, topic: str) -> None:
        """异步运行完整 pipeline。"""
        config = ResearchConfig.load()
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

            # 检查是否被取消
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

            # 导出结果
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

                # 列出生成的文件
                summary += "\n生成文件:\n"
                for f in sorted(export_path.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(export_path)
                        size = f.stat().st_size
                        summary += f"  {rel} ({size / 1024:.1f} KB)\n"

                self.send_message(chat_id, summary)

                # 上传 PDF（如果存在）
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

    def _upload_file(self, chat_id: str, file_path: Path) -> None:
        """上传文件到飞书会话。"""
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
                    # 发送文件消息
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


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app_id, app_secret = _load_feishu_credentials()
    bot = FeishuBot(app_id, app_secret)

    logger.info("NanoResearch 飞书机器人启动中...")

    # 注册消息事件处理
    def on_message(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # 只处理文本消息
            if message.message_type != "text":
                return

            chat_id = message.chat_id
            message_id = message.message_id
            sender_id = sender.sender_id.open_id if sender.sender_id else ""

            # 解析消息内容
            try:
                content = json.loads(message.content)
                text = content.get("text", "")
            except (json.JSONDecodeError, TypeError):
                text = ""

            if not text.strip():
                return

            bot.handle_message(chat_id, message_id, text, sender_id)

        except Exception as e:
            logger.exception("处理消息异常: %s", e)

    event_handler = lark.EventDispatcherHandler.builder("", "") \
        .register_p2_im_message_receive_v1(on_message) \
        .build()

    cli = lark.ws.Client(
        app_id,
        app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    logger.info("WebSocket 长连接启动，等待消息...")
    logger.info("在飞书中给机器人发消息即可触发 pipeline")
    cli.start()


if __name__ == "__main__":
    main()
