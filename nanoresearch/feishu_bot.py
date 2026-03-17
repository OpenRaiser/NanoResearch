"""飞书机器人集成 -- AI 对话 + NanoResearch pipeline 触发。

使用飞书 WebSocket 长连接模式（无需公网服务器）。

功能：
    1. AI 对话：自然语言讨论研究、回答问题、提供建议
    2. Pipeline：启动自动论文生成、查看状态、导出结果
    3. 记忆：跨消息保持对话上下文和用户偏好

用法：
    nanoresearch feishu
    python -m nanoresearch.feishu_bot

环境变量（或在 ~/.nanobot/config.json 中配置）：
    FEISHU_APP_ID      飞书应用 App ID
    FEISHU_APP_SECRET  飞书应用 App Secret

Split into 3 modules:
    feishu_bot.py        -- ChatMemory, FeishuBot facade, main()
    feishu_bot_core.py   -- _FeishuBotCoreMixin (messaging, AI chat)
    feishu_bot_handlers.py -- _FeishuBotHandlersMixin (commands, pipeline)
"""

from __future__ import annotations

import atexit
import io
import json
import logging
import os
import re
import signal
import sys
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

from nanoresearch.feishu_bot_core import _FeishuBotCoreMixin
from nanoresearch.feishu_bot_handlers import _FeishuBotHandlersMixin

logger = logging.getLogger(__name__)

# ─── Config ───
_DEFAULT_ROOT = Path.home() / ".nanobot" / "workspace" / "research"
_CHAT_MEMORY_DIR = Path.home() / ".nanobot" / "chat_memory"


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
#  对话记忆
# ═══════════════════════════════════════════════════════════════

class ChatMemory:
    """Per-chat persistent memory: conversation history + condensed summary + facts."""

    MAX_MESSAGES = 40
    KEEP_RECENT = 10

    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id
        safe_id = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', chat_id)[:120]
        self._path = _CHAT_MEMORY_DIR / f"{safe_id}.json"
        self._summary: str = ""
        self._messages: list[dict[str, str]] = []
        self._facts: list[str] = []
        self._invalidated = False
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._summary = data.get("summary", "")
                self._messages = data.get("messages", [])
                self._facts = data.get("facts", [])
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        if self._invalidated:
            return
        _CHAT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "chat_id": self.chat_id,
            "summary": self._summary,
            "messages": self._messages[-self.MAX_MESSAGES:],
            "facts": self._facts[-50:],
        }
        try:
            self._path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("保存记忆失败: %s", e)

    @property
    def summary(self) -> str:
        return self._summary

    @summary.setter
    def summary(self, value: str) -> None:
        self._summary = value

    @property
    def messages(self) -> list[dict[str, str]]:
        return self._messages

    @property
    def facts(self) -> list[str]:
        return self._facts

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def add_fact(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self._facts:
            self._facts.append(fact)
            if len(self._facts) > 50:
                self._facts = self._facts[-50:]

    def needs_condensation(self) -> bool:
        return len(self._messages) > self.MAX_MESSAGES

    def condense(self, new_summary: str) -> None:
        self._summary = new_summary
        self._messages = self._messages[-self.KEEP_RECENT:]

    def build_history_prompt(self, current_text: str) -> str:
        recent = self._messages[-(self.KEEP_RECENT * 2):]
        parts: list[str] = []
        if self._summary:
            parts.append(f"[对话摘要] {self._summary}")
        for msg in recent:
            if msg["role"] == "user":
                parts.append(f"用户: {msg['content']}")
            else:
                parts.append(f"助手: {msg['content']}")
        parts.append(f"用户: {current_text}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  AI 对话系统提示
# ═══════════════════════════════════════════════════════════════

_CHAT_SYSTEM = """\
你是 NanoResearch 飞书助手，一个简洁的 AI 学术研究助理。

## 回复规则（最高优先级，必须遵守）
1. 每次回复不超过 150 字，像微信聊天一样简短
2. 需要用户选择时，用 A/B/C/D 格式
3. 用户回复单个字母（A/B/C/D）时，是在选择对应选项
4. 不确定就说"不太确定"，绝对不编造论文名、数据集名
5. 用中文回复（除非用户用英文）

## 启动研究的流程（极其重要）
当用户想生成论文时，严格按以下 5 步走，每步问 1 个 ABCD 问题：
1. 研究方向/核心主题
2. 论文类型（提出新方法 / 综述 / 改进现有方法 / 应用）
3. 数据场景和语言（中文/英文 + 数据类型）
4. 创新点偏好（如：跨模态融合 / 可解释性 / 鲁棒性 / 效率）
5. 落地偏好（追求SOTA / 可解释 / 小样本 / 跨域泛化）
问完第 5 个问题，用户回答后，立刻总结选择并加 RUN 标记启动 pipeline！
绝对不要问超过 5 个问题。技术细节（模型、损失函数、超参数等）由 pipeline 自动决定。
用户说"生成""帮我写""做一篇"等，就视为要启动 pipeline。

## 操作标记
在回复最末尾另起一行加标记（精确匹配前缀）：
- ##ACT_{nonce}_RUN:完整研究主题描述（至少15字，包含方向+方法关键词）
- ##ACT_{nonce}_STATUS　←仅当用户明确说"查状态/进度/status"时
- ##ACT_{nonce}_STOP　←仅当用户明确说"停止/stop/取消"时
- ##ACT_{nonce}_EXPORT
- ##ACT_{nonce}_LIST
- ##ACT_{nonce}_REMEMBER:要记住的内容
重要：每次回复最多只能有1个操作标记。
绝对禁止自动添加STATUS/STOP/LIST/EXPORT标记！只有当用户消息明确包含"状态""进度""停止""取消""导出""列表"等词时才能加。
在引导用户选择（A/B/C/D）或回答问题时，回复末尾不加任何标记。

{context}"""


# ═══════════════════════════════════════════════════════════════
#  飞书消息收发 + AI 对话 (facade)
# ═══════════════════════════════════════════════════════════════

class FeishuBot(_FeishuBotCoreMixin, _FeishuBotHandlersMixin):
    """飞书 NanoResearch 机器人 -- AI 对话 + Pipeline 管理。"""

    def __init__(self, app_id: str, app_secret: str) -> None:
        self._init_core(app_id, app_secret)


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

    logger.info("NanoResearch 飞书助手启动中...")

    def on_message(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        try:
            event = data.event
            message = event.message
            sender = event.sender

            if message.message_type != "text":
                return

            chat_id = message.chat_id
            message_id = message.message_id
            sender_id = sender.sender_id.open_id if sender.sender_id else ""

            try:
                content = json.loads(message.content)
                text = content.get("text", "")
            except (json.JSONDecodeError, TypeError):
                text = ""

            logger.info("on_message: raw_content=%r text=%r", message.content, text[:200] if text else "")

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

    def _shutdown_handler(*_args):
        bot.shutdown()

    signal.signal(signal.SIGINT, lambda *a: (_shutdown_handler(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *a: (_shutdown_handler(), sys.exit(0)))
    atexit.register(_shutdown_handler)

    logger.info("WebSocket 长连接启动，等待消息...")
    logger.info("在飞书中给机器人发消息即可开始对话")
    logger.info("按 Ctrl+C 可安全退出（自动保存记忆、取消任务）")
    cli.start()


if __name__ == "__main__":
    main()
