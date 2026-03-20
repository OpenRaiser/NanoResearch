"""FeishuBot core mixin -- messaging, AI chat, memory, and action parsing.

Separated from feishu_bot.py for size.  Mixed into FeishuBot via MRO.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import threading
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from nanoresearch.config import ResearchConfig, StageModelConfig
from nanoresearch.pipeline.multi_model import ModelDispatcher

logger = logging.getLogger(__name__)


class _FeishuBotCoreMixin:
    """First-half mixin: init, lifecycle, messaging, AI chat, memory."""

    # These attributes are set in FeishuBot.__init__ and accessed via self.
    # Declared here for type-checker clarity only.
    app_id: str
    app_secret: str
    client: Any
    _running_tasks: dict[str, dict[str, Any]]
    _lock: threading.Lock
    _config: ResearchConfig
    _dispatcher: ModelDispatcher
    _memories: dict
    _chat_locks: dict[str, threading.Lock]
    _chat_locks_lock: threading.Lock
    _action_nonce: str
    _pipeline_threads: dict[str, threading.Thread]
    _pipeline_loops: dict[str, asyncio.AbstractEventLoop]
    _shutting_down: bool
    _pending_env_select: dict[str, dict[str, Any]]
    _chat_model_config: StageModelConfig

    def _init_core(self, app_id: str, app_secret: str) -> None:
        """Shared initialisation called from FeishuBot.__init__."""
        self.app_id = app_id
        self.app_secret = app_secret
        self.client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        self._running_tasks = {}
        self._lock = threading.Lock()
        self._config = ResearchConfig.load()
        self._dispatcher = ModelDispatcher(self._config)
        self._memories = {}
        self._chat_locks = {}
        self._chat_locks_lock = threading.Lock()
        self._action_nonce = secrets.token_hex(4)
        self._pipeline_threads = {}
        self._pipeline_loops = {}
        self._shutting_down = False
        self._pending_env_select = {}
        self._chat_model_config = StageModelConfig(
            model=self._config.ideation.model,
            temperature=0.3,
            max_tokens=800,
            base_url=self._config.ideation.base_url,
            api_key=self._config.ideation.api_key,
        )

    # ─── lifecycle ───

    def shutdown(self) -> None:
        """Graceful shutdown: cancel all pipelines, save memories, wait for threads."""
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("正在关闭 bot...")

        with self._lock:
            self._pending_env_select.clear()
            for chat_id, task in self._running_tasks.items():
                if task.get("status") not in ("completed", "failed", "stopped"):
                    task["cancel"] = True
                    task["status"] = "stopped"
                    logger.info("取消任务: chat=%s topic=%s", chat_id, task.get("topic", "?")[:50])

        for chat_id, loop in list(self._pipeline_loops.items()):
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass

        for chat_id, thread in list(self._pipeline_threads.items()):
            if thread.is_alive():
                logger.info("等待 pipeline 线程结束: chat=%s", chat_id)
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning("Pipeline 线程未能在 5s 内结束: chat=%s", chat_id)

        with self._chat_locks_lock:
            for chat_id, memory in self._memories.items():
                try:
                    memory.save()
                except Exception as e:
                    logger.warning("保存记忆失败: chat=%s err=%s", chat_id, e)

        logger.info("Bot 已关闭")

    # ─── messaging ───

    def send_message(self, chat_id: str, text: str) -> None:
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
        if response.success():
            logger.info("回复消息成功: msg_id=%s", message_id)
        else:
            logger.error("回复消息失败: code=%s msg=%s", response.code, response.msg)

    def send_card(self, chat_id: str, card: dict) -> None:
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

    # ─── memory helpers ───

    def _get_memory(self, chat_id: str):
        with self._chat_locks_lock:
            if chat_id not in self._memories:
                from nanoresearch.feishu_bot import ChatMemory
                self._memories[chat_id] = ChatMemory(chat_id)
            return self._memories[chat_id]

    def _get_chat_lock(self, chat_id: str) -> threading.Lock:
        with self._chat_locks_lock:
            if chat_id not in self._chat_locks:
                self._chat_locks[chat_id] = threading.Lock()
            return self._chat_locks[chat_id]

    # ─── message routing ───

    @staticmethod
    def _clean_text(raw: str) -> str:
        text = raw.strip()
        text = re.sub(r'@_user_\d+\s*', '', text)
        text = re.sub(r'@\w+\s*', '', text) if text.startswith('@') else text
        text = text.replace('\uff0f', '/')
        text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00a0\u2060]', '', text)
        return text.strip()

    def handle_message(self, chat_id: str, message_id: str, text: str, sender_id: str) -> None:
        if self._shutting_down:
            return
        raw = text
        text = self._clean_text(raw)
        if not text:
            return

        logger.info("消息 raw=%r cleaned=%r bytes=%s chat=%s",
                    raw[:100], text[:100], text.encode('unicode_escape').decode()[:200], chat_id)
        lower = text.lower()

        with self._lock:
            pending = self._pending_env_select.get(chat_id)
        if pending:
            self._handle_env_selection(chat_id, message_id, text, pending)
            return

        _SLASH_CMDS = {
            "/help": "help", "/status": "status", "/list": "list",
            "/stop": "stop", "/export": "export", "/new": "new",
        }
        first_word = lower.split()[0] if lower.split() else ""
        cmd = _SLASH_CMDS.get(lower) or _SLASH_CMDS.get(first_word)
        logger.info("命令匹配: lower=%r first_word=%r cmd=%s", lower, first_word, cmd)
        if cmd:
            getattr(self, f"_cmd_{cmd}")(chat_id, message_id)
            return

        if lower.startswith("/run ") or first_word == "/run":
            topic = text.split(" ", 1)[1].strip() if " " in text else ""
            if len(topic) < 5:
                self.reply_message(message_id, "主题太短了，请描述更详细一些。\n用法: /run <研究主题>")
                return
            self._cmd_run(chat_id, message_id, topic)
            return

        thread = threading.Thread(
            target=self._handle_chat_thread,
            args=(chat_id, message_id, text),
            daemon=True,
        )
        thread.start()

    # ─── AI chat ───

    def _handle_chat_thread(self, chat_id: str, message_id: str, text: str) -> None:
        lock = self._get_chat_lock(chat_id)
        if not lock.acquire(timeout=30):
            self.reply_message(message_id, "正在处理上一条消息，请稍后再试...")
            return
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    self._handle_chat_async(chat_id, message_id, text)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.exception("Chat handler error: %s", e)
            self.reply_message(message_id, f"AI 回复出错: {e}")
        finally:
            lock.release()

    async def _handle_chat_async(self, chat_id: str, message_id: str, text: str) -> None:
        from nanoresearch.feishu_bot import _CHAT_SYSTEM
        memory = self._get_memory(chat_id)
        memory.add_message("user", text)

        context = self._build_chat_context(chat_id, memory)
        system = _CHAT_SYSTEM.replace("{context}", context).replace("{nonce}", self._action_nonce)
        user_prompt = memory.build_history_prompt(text)

        try:
            response = await self._dispatcher.generate(
                self._chat_model_config, system, user_prompt,
            )
        except Exception as e:
            logger.error("LLM chat error: %s", e)
            if memory.messages and memory.messages[-1].get("role") == "user":
                memory.messages.pop()
            memory.save()
            self.reply_message(message_id, f"AI 服务暂时不可用: {type(e).__name__}")
            return

        response = response or ""
        logger.info("LLM raw response (%d chars): %r", len(response), response[:200])
        reply, actions = self._parse_actions(response)
        logger.info("Parsed reply (%d chars): %r, actions=%s", len(reply), reply[:200], actions)

        memory.add_message("assistant", reply)

        for action, arg in actions:
            if action == "REMEMBER" and arg:
                memory.add_fact(arg)
                logger.info("记住: chat=%s fact=%r", chat_id, arg)

        if memory.needs_condensation():
            await self._condense_memory(memory)

        memory.save()

        if reply.strip():
            logger.info("Sending reply to message_id=%s", message_id)
            self.reply_message(message_id, reply)
        else:
            logger.warning("Empty reply, not sending. raw=%r", response[:300])

        reply_is_substantive = len(reply.strip()) > 20
        for action, arg in actions:
            if action == "RUN" and arg and len(arg) >= 5:
                self._cmd_run(chat_id, message_id, arg)
            elif action == "STATUS":
                if not reply_is_substantive:
                    self._cmd_status(chat_id, message_id)
                else:
                    logger.info("Skipping auto-STATUS (reply already sent)")
            elif action == "STOP":
                if not reply_is_substantive:
                    self._cmd_stop(chat_id, message_id)
                else:
                    logger.info("Skipping auto-STOP (reply already sent)")
            elif action == "EXPORT":
                if not reply_is_substantive:
                    self._cmd_export(chat_id, message_id)
                else:
                    logger.info("Skipping auto-EXPORT (reply already sent)")
            elif action == "LIST":
                if not reply_is_substantive:
                    self._cmd_list(chat_id, message_id)
                else:
                    logger.info("Skipping auto-LIST (reply already sent)")

    def _build_chat_context(self, chat_id: str, memory) -> str:
        parts = []
        with self._lock:
            task = self._running_tasks.get(chat_id)
        if task:
            parts.append(
                f"## 当前任务\n"
                f"主题: {task['topic']}\n"
                f"状态: {task['status']}"
            )
        else:
            parts.append("## 当前任务\n无正在运行的任务")

        if memory.summary:
            parts.append(f"## 对话摘要\n{memory.summary}")

        if memory.facts:
            facts_text = "\n".join(f"- {f}" for f in memory.facts[-15:])
            parts.append(f"## 已知信息\n{facts_text}")

        return "\n\n".join(parts)

    def _parse_actions(self, response: str) -> tuple[str, list[tuple[str, str]]]:
        actions: list[tuple[str, str]] = []
        reply_lines: list[str] = []
        prefix = f"##ACT_{self._action_nonce}_"

        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.startswith(prefix):
                tag = stripped[len(prefix):]
                if tag.startswith("RUN:"):
                    actions.append(("RUN", tag[4:].strip()))
                elif tag == "STATUS":
                    actions.append(("STATUS", ""))
                elif tag == "STOP":
                    actions.append(("STOP", ""))
                elif tag == "EXPORT":
                    actions.append(("EXPORT", ""))
                elif tag == "LIST":
                    actions.append(("LIST", ""))
                elif tag.startswith("REMEMBER:"):
                    actions.append(("REMEMBER", tag[9:].strip()))
                else:
                    reply_lines.append(line)
            else:
                reply_lines.append(line)

        return "\n".join(reply_lines).strip(), actions

    async def _condense_memory(self, memory) -> None:
        old_msgs = memory.messages[:-memory.KEEP_RECENT]
        if not old_msgs:
            return

        text_to_condense = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:300]}"
            for m in old_msgs
        )

        prompt_parts = ["请将以下对话历史压缩为简洁的中文摘要，保留关键信息、决定、用户偏好：\n"]
        if memory.summary:
            prompt_parts.append(f"之前的摘要：\n{memory.summary}\n")
        prompt_parts.append(f"新对话：\n{text_to_condense}\n\n生成更新后的摘要（200字以内）：")

        try:
            summary = await self._dispatcher.generate(
                self._chat_model_config,
                "你是对话摘要助手。只输出摘要内容，不加前缀。",
                "\n".join(prompt_parts),
            )
            memory.condense(summary.strip())
            logger.info("对话压缩完成: chat=%s, 摘要长度=%d", memory.chat_id, len(summary))
        except Exception as e:
            logger.warning("对话压缩失败: %s, 使用截断", e)
            memory.condense(memory.summary)
