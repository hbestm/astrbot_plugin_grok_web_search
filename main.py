"""
AstrBot 插件：Grok 联网搜索

通过 Grok API 进行实时联网搜索，支持：
- /grok 指令
- LLM Tool (grok_web_search)
- Skill 脚本动态安装
"""

import shutil
import tempfile
import zipfile
from pathlib import Path

import aiohttp
import asyncio
import os
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.core.star.filter.command import GreedyStr
from astrbot.api.star import Context, Star
from astrbot.core.message.components import Image
from astrbot.core.utils.io import download_image_by_url, file_to_base64
from astrbot.core.utils.quoted_message.chain_parser import (
    _extract_image_refs_from_component_chain,
    _extract_text_from_component_chain,
)

from .api.grok_chat import grok_fetch, grok_search
from .api.grok_responses import grok_responses_search

try:
    from astrbot.core.provider.register import llm_tools as _llm_tools_registry
except ImportError:
    _llm_tools_registry = None
try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except ImportError:
    get_astrbot_data_path = None
from .tool.tool import (
    DEFAULT_JSON_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    build_headers,
    extract_urls,
    normalize_api_key,
    normalize_base_url,
    normalize_sources,
    parse_json_config,
    parse_json_object,
    resolve_system_prompt,
    safe_number,
)
from .tool.card_render import (
    render_search_card,
    init_fonts,
    set_logger as set_card_logger,
)

PLUGIN_NAME = "astrbot_plugin_grok_web_search"


def _fmt_tokens(n: int) -> str:
    """将 token 数量格式化为简短形式，如 1m2k、3.5k、800。"""
    if n >= 1_000_000:
        m, remain = divmod(n, 1_000_000)
        k = remain // 1_000
        return f"{m}m{k}k" if k else f"{m}m"
    if n >= 1_000:
        k, remain = divmod(n, 1_000)
        h = remain // 100
        return f"{k}.{h}k" if h else f"{k}k"
    return str(n)


class GrokSearchPlugin(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        self._session: aiohttp.ClientSession | None = None
        self._card_fonts_ready = False

    async def _extract_content_from_event(
        self, event: AstrMessageEvent
    ) -> tuple[str | None, list[str]]:
        """Extract text and images from the user's message.

        Reuses AstrBot core's chain_parser for text/image extraction from
        Reply, Node, Nodes, Forward, etc.

        Returns:
            A tuple of (text, images):
            - text: extracted text from the message chain (or None)
            - images: list of base64-encoded image strings (without prefix)
        """
        chain = event.get_messages()

        # 使用本体的 chain_parser 提取文本（处理 Reply/Node/Nodes/Forward）
        text = _extract_text_from_component_chain(chain)

        # 使用本体的 chain_parser 提取图片引用，再转为 base64
        image_refs = _extract_image_refs_from_component_chain(chain)
        images: list[str] = []
        seen: set[str] = set()

        # 提取消息链顶层的 Image 组件并转为 base64
        for comp in chain:
            if isinstance(comp, Image):
                try:
                    b64 = await comp.convert_to_base64()
                    if b64 and b64 not in seen:
                        seen.add(b64)
                        images.append(b64)
                except Exception as e:
                    logger.warning(
                        f"[{PLUGIN_NAME}] Failed to convert image to base64: {e}"
                    )

        # 将嵌套组件中的图片引用（URL/路径）转为 base64
        for ref in image_refs:
            try:
                img = Image.fromURL(ref)
                b64 = await img.convert_to_base64()
                if b64 and b64 not in seen:
                    seen.add(b64)
                    images.append(b64)
            except Exception as e:
                logger.warning(
                    f"[{PLUGIN_NAME}] Failed to convert image ref to base64: {e}"
                )

        return text, images

    def _unregister_disabled_tools(self):
        """根据配置在初始化时直接卸载不需要的 LLM Tool，避免 AI 看到无用工具"""
        if _llm_tools_registry is None:
            return

        if self.config.get("enable_skill", False):
            # Skill 接管，移除所有 LLM Tool
            _llm_tools_registry.remove_func("grok_web_search")
            _llm_tools_registry.remove_func("grok_web_fetch")
            logger.info(
                f"[{PLUGIN_NAME}] Skill 已启用，已卸载 grok_web_search 和 grok_web_fetch 工具"
            )
            return

        if not self.config.get("enable_fetch", False):
            _llm_tools_registry.remove_func("grok_web_fetch")
            logger.info(f"[{PLUGIN_NAME}] 网页抓取未启用，已卸载 grok_web_fetch 工具")

    def _init_fonts(self):
        """Initialize card rendering fonts (runs in background)."""
        logger.info(f"[{PLUGIN_NAME}] 正在后台初始化卡片渲染字体 ...")
        try:
            if get_astrbot_data_path:
                font_dir = str(
                    Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME / "font"
                )
            else:
                font_dir = os.path.join(os.path.dirname(__file__), "font")
            self._card_fonts_ready = init_fonts(font_dir)
            if self._card_fonts_ready:
                logger.info(f"[{PLUGIN_NAME}] 卡片渲染字体已就绪: {font_dir}")
            else:
                logger.warning(f"[{PLUGIN_NAME}] 卡片渲染字体初始化失败")
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 字体初始化异常: {e}")

    async def initialize(self):
        """插件初始化：验证配置并处理 Skill 安装"""
        # 在后台初始化字体，仅在开启图片渲染模式下
        if self.config.get("render_as_image", False):
            set_card_logger(logger)
            asyncio.get_event_loop().run_in_executor(None, self._init_fonts)

        # 根据配置卸载不需要的 LLM Tool
        self._unregister_disabled_tools()

        # 如果启用使用 AstrBot 自带供应商，则推迟创建会话和 Skill 安装
        if self.config.get("use_builtin_provider", False):
            logger.info(
                f"[{PLUGIN_NAME}] use_builtin_provider enabled, delaying full initialization until AstrBot is loaded"
            )
            return

        # 仅在使用外部 HTTP 客户端时校验 base_url/api_key
        await self._validate_config()

        # 根据配置决定是否创建复用的 HTTP 会话
        if self.config.get("reuse_session", False):
            self._session = aiohttp.ClientSession()

        # 首次安装：将插件目录的 skill 移动到持久化目录
        self._migrate_skill_to_persistent()

        if self.config.get("enable_skill", False):
            self._install_skill()
        else:
            self._uninstall_skill()

    async def _validate_config(self):
        """验证必要配置，并通过 v1/models 接口检查连通性"""
        base_url = normalize_base_url(self.config.get("base_url", ""))
        api_key = normalize_api_key(self.config.get("api_key", ""))
        if not base_url:
            logger.warning(
                f"[{PLUGIN_NAME}] 缺少 base_url 配置，请在插件设置中填写 Grok API 端点"
            )
            return
        if not api_key:
            logger.warning(
                f"[{PLUGIN_NAME}] 缺少 api_key 配置，请在插件设置中填写 API 密钥"
            )
            return

        # 通过 v1/models 接口验证连通性和密钥有效性
        models_url = f"{base_url}/v1/models"
        extra_headers = self._parse_json_config("extra_headers")
        headers = build_headers(api_key, extra_headers or None)

        # 获取代理配置
        proxy = self.config.get("proxy", "").strip() or None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    models_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                    proxy=proxy,
                ) as resp:
                    if resp.status == 401:
                        logger.warning(
                            f"[{PLUGIN_NAME}] API 密钥无效（401），请检查 api_key 配置"
                        )
                    elif resp.status == 403:
                        logger.warning(
                            f"[{PLUGIN_NAME}] API 密钥权限不足（403），请检查 api_key 权限"
                        )
                    elif resp.status == 404:
                        logger.warning(
                            f"[{PLUGIN_NAME}] v1/models 端点不存在（404），请检查 base_url 配置是否正确"
                        )
                    elif resp.status != 200:
                        logger.warning(
                            f"[{PLUGIN_NAME}] API 连通性检查返回 HTTP {resp.status}，请确认配置"
                        )
                    else:
                        logger.info(f"[{PLUGIN_NAME}] API 连通性检查通过")
        except aiohttp.ClientError as e:
            logger.warning(
                f"[{PLUGIN_NAME}] API 连通性检查失败（网络错误）: {e}，请检查 base_url 配置"
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[{PLUGIN_NAME}] API 连通性检查超时，请检查 base_url 是否可达"
            )

    def _get_skill_manager(self):
        """获取 SkillManager 实例（延迟导入）"""
        if hasattr(self, "_skill_mgr"):
            return self._skill_mgr
        try:
            from astrbot.core.skills import SkillManager

            self._skill_mgr = SkillManager()
        except ImportError:
            self._skill_mgr = None
        return self._skill_mgr

    def _get_plugin_data_path(self) -> Path:
        """获取插件持久化数据目录"""
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

            plugin_data_root = Path(get_astrbot_plugin_data_path())
        except ImportError:
            # 回退到相对路径
            plugin_data_root = Path(__file__).parent.parent.parent / "plugin_data"

        # 插件专属目录
        plugin_data_dir = plugin_data_root / PLUGIN_NAME
        plugin_data_dir.mkdir(parents=True, exist_ok=True)
        return plugin_data_dir

    def _get_skill_persistent_path(self) -> Path:
        """获取 Skill 持久化存储路径"""
        return self._get_plugin_data_path() / "skill"

    def _migrate_skill_to_persistent(self):
        """首次安装：将插件目录的 skill 复制到持久化目录"""
        source_dir = Path(__file__).parent / "skill"
        persistent_dir = self._get_skill_persistent_path()

        if source_dir.exists() and not persistent_dir.exists():
            try:
                shutil.copytree(source_dir, persistent_dir, symlinks=True)
                logger.info(
                    f"[{PLUGIN_NAME}] Skill 已复制到持久化目录: {persistent_dir}"
                )
            except Exception as e:
                logger.error(f"[{PLUGIN_NAME}] Skill 复制到持久化目录失败: {e}")

    def _install_skill(self):
        """通过 SkillManager 安装 Skill（打包为 zip 后调用官方接口）"""
        source_dir = self._get_skill_persistent_path()

        if not source_dir.exists():
            logger.error(f"[{PLUGIN_NAME}] Skill 持久化目录不存在: {source_dir}")
            return

        if source_dir.is_symlink():
            logger.error(
                f"[{PLUGIN_NAME}] Skill 源目录是 symlink，拒绝安装: {source_dir}"
            )
            return

        skill_mgr = self._get_skill_manager()
        if not skill_mgr:
            logger.error(f"[{PLUGIN_NAME}] SkillManager 不可用，无法安装 Skill")
            return

        tmp_zip = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_zip = Path(tmp.name)

            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in source_dir.rglob("*"):
                    if file.is_file():
                        arcname = f"grok-search/{file.relative_to(source_dir)}"
                        zf.write(file, arcname)

            skill_mgr.install_skill_from_zip(str(tmp_zip), overwrite=True)
            logger.info(f"[{PLUGIN_NAME}] Skill 已通过 SkillManager 安装并激活")
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] Skill 安装失败: {e}")
        finally:
            if tmp_zip:
                tmp_zip.unlink(missing_ok=True)

    def _uninstall_skill(self):
        """通过 SkillManager 卸载 Skill"""
        skill_mgr = self._get_skill_manager()
        if not skill_mgr:
            logger.error(f"[{PLUGIN_NAME}] SkillManager 不可用，无法卸载 Skill")
            return

        try:
            skill_mgr.delete_skill("grok-search")
            logger.info(f"[{PLUGIN_NAME}] Skill 已通过 SkillManager 卸载")
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] Skill 卸载失败: {e}")

    def _parse_json_config(self, key: str) -> dict:
        """解析 JSON 格式的配置项"""
        value = self.config.get(key, "")
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            result, error = parse_json_config(value)
            if error:
                logger.warning(f"[{PLUGIN_NAME}] {key} {error}")
            return result
        return {}

    async def _do_search(
        self,
        query: str,
        system_prompt: str | None = None,
        use_retry: bool = False,
        images: list[str] | None = None,
    ) -> dict:
        """Execute a search.

        Args:
            query: Search query content
            system_prompt: Custom system prompt, uses default when None
            use_retry: Whether to enable retry (command invocation only)
            images: Optional list of base64-encoded images for multimodal queries
        """
        # 安全解析 timeout 配置
        timeout = safe_number(
            self.config.get("timeout_seconds", 60),
            60.0,
            cast=float,
            min_val=0.001,
        )

        # 安全解析 thinking_budget 配置
        thinking_budget = safe_number(
            self.config.get("thinking_budget", 32000),
            32000,
            cast=int,
            min_val=0,
        )

        # 重试配置（仅指令调用时使用）
        max_retries = 0
        retry_delay = 1.0
        retryable_status_codes = None
        if use_retry:
            max_retries = self.config.get("max_retries", 3)
            retry_delay = self.config.get("retry_delay", 1.0)

            # 解析可重试状态码（直接从 list 类型配置获取）
            retryable_codes = self.config.get("retryable_status_codes", [])
            if retryable_codes and isinstance(retryable_codes, list):
                retryable_status_codes = set(retryable_codes)

        # 自定义系统提示词（传入优先，其次配置，最后默认 JSON 提示词）
        if system_prompt is None:
            system_prompt = resolve_system_prompt(
                self.config.get("custom_system_prompt", ""),
                DEFAULT_JSON_SYSTEM_PROMPT,
            )
        # 如果启用了使用 AstrBot 自带供应商，通过 AstrBot provider 接口调用
        if self.config.get("use_builtin_provider", False):
            attempts = 0
            started = time.time()
            while True:
                try:
                    # 严格按配置获取 provider
                    configured_provider_id = self.config.get("provider", "")
                    if not configured_provider_id:
                        return {
                            "ok": False,
                            "error": "启用了内置供应商但未选择供应商，请在插件设置中选择一个 LLM 供应商",
                        }
                    prov = self.context.get_provider_by_id(configured_provider_id)
                    if not prov:
                        return {
                            "ok": False,
                            "error": f"未找到配置的供应商: {configured_provider_id}",
                        }

                    provider_id = prov.meta().id

                    # 将 base64 图片转为内置供应商的 image_urls 格式
                    image_urls = (
                        [f"base64://{img}" for img in images] if images else None
                    )

                    llm_resp = await self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=query,
                        system_prompt=system_prompt,
                        image_urls=image_urls,
                    )

                    text = llm_resp.completion_text or ""
                    usage = {}
                    if llm_resp.usage:
                        usage = {
                            "prompt_tokens": llm_resp.usage.input,
                            "completion_tokens": llm_resp.usage.output,
                            "total_tokens": llm_resp.usage.total,
                        }

                    # 尝试解析 JSON 格式响应
                    parsed = parse_json_object(text)
                    if parsed is not None:
                        content = str(parsed.get("content", ""))
                        raw_sources = parsed.get("sources", [])
                        sources = normalize_sources(raw_sources)
                        return {
                            "ok": True,
                            "content": content,
                            "sources": sources,
                            "elapsed_ms": int((time.time() - started) * 1000),
                            "retries": attempts,
                            "usage": usage,
                            "raw": "",
                        }

                    # JSON 解析失败，降级处理：提取纯文本和 URL
                    logger.warning(
                        f"[{PLUGIN_NAME}] 内置供应商返回非 JSON 格式，使用降级处理"
                    )

                    # 检测典型错误模式，避免将错误文案误判为成功
                    text_lower = text.lower()
                    error_patterns = [
                        "rate limit",
                        "too many requests",
                        "quota exceeded",
                        "authentication failed",
                        "invalid api key",
                        "unauthorized",
                        "service unavailable",
                        "internal server error",
                        "timeout",
                        "connection refused",
                    ]
                    is_error_response = any(p in text_lower for p in error_patterns)

                    if not text.strip() or is_error_response:
                        error_msg = (
                            "提供商返回空响应"
                            if not text.strip()
                            else f"提供商返回错误: {text[:200]}"
                        )
                        return {
                            "ok": False,
                            "error": error_msg,
                            "content": "",
                            "sources": [],
                            "elapsed_ms": int((time.time() - started) * 1000),
                            "retries": attempts,
                            "usage": usage,
                            "raw": text[:500] if text else "",
                        }

                    sources = self._extract_sources_from_text(text)
                    return {
                        "ok": True,
                        "content": text,
                        "sources": sources,
                        "elapsed_ms": int((time.time() - started) * 1000),
                        "retries": attempts,
                        "usage": usage,
                        "raw": text,
                    }

                except Exception as e:
                    attempts += 1
                    if not use_retry or attempts > max_retries:
                        return {"ok": False, "error": str(e)}
                    await asyncio.sleep(retry_delay * attempts)

        # 否则使用 HTTP 客户端向外部 Grok API 发起请求
        try:
            # 获取代理配置
            proxy = self.config.get("proxy", "").strip() or None

            # 根据配置选择 API 模式
            if self.config.get("use_responses_api", False):
                # 使用 xAI Responses API（/v1/responses）
                result = await grok_responses_search(
                    query=query,
                    base_url=self.config.get("base_url", ""),
                    api_key=self.config.get("api_key", ""),
                    model=self.config.get("model", DEFAULT_MODEL),
                    timeout=timeout,
                    extra_body=self._parse_json_config("extra_body"),
                    extra_headers=self._parse_json_config("extra_headers"),
                    session=self._session,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retryable_status_codes=retryable_status_codes,
                    images=images,
                    proxy=proxy,
                )
            else:
                # 使用 Chat Completions API（/v1/chat/completions）
                result = await grok_search(
                    query=query,
                    base_url=self.config.get("base_url", ""),
                    api_key=self.config.get("api_key", ""),
                    model=self.config.get("model", DEFAULT_MODEL),
                    timeout=timeout,
                    enable_thinking=self.config.get("enable_thinking", True),
                    thinking_budget=thinking_budget,
                    extra_body=self._parse_json_config("extra_body"),
                    extra_headers=self._parse_json_config("extra_headers"),
                    session=self._session,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retryable_status_codes=retryable_status_codes,
                    images=images,
                    proxy=proxy,
                )
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] API 调用异常: {e}")
            return {"ok": False, "error": f"API 调用异常: {e}"}

        if not result.get("ok"):
            logger.warning(
                f"[{PLUGIN_NAME}] API 调用失败: {result.get('error', '未知错误')}"
            )
        return result

    def _render_sources(
        self,
        sources: list,
        *,
        header: str,
        with_snippet: bool,
    ) -> list[str]:
        """渲染来源列表，遵循 show_sources / max_sources 配置。"""
        if not self.config.get("show_sources", False) or not sources:
            return []
        max_sources = self.config.get("max_sources", 5)
        if max_sources > 0:
            sources = sources[:max_sources]
        lines = [f"\n{header}:"]
        for i, src in enumerate(sources, 1):
            url = src.get("url", "")
            title = src.get("title", "")
            if title:
                if with_snippet:
                    lines.append(f"  {i}. {title}")
                    lines.append(f"     {url}")
                else:
                    lines.append(f"  {i}. {title}\n     {url}")
            else:
                lines.append(f"  {i}. {url}")
            if with_snippet:
                snippet = src.get("snippet", "")
                if snippet:
                    lines.append(f"     {snippet}")
        return lines

    def _format_result(self, result: dict) -> str:
        """格式化搜索结果为用户友好的消息"""
        if not result.get("ok"):
            error = result.get("error", "未知错误")
            return f"搜索失败: {error}"

        content = result.get("content", "")
        sources = result.get("sources", [])
        elapsed = result.get("elapsed_ms", 0) / 1000

        lines = [content]
        lines.extend(self._render_sources(sources, header="来源", with_snippet=False))

        # 显示耗时、重试次数和 token 用量
        retry_info = ""
        retries = result.get("retries", 0)
        if retries > 0:
            retry_info = f"，重试 {retries} 次"

        token_info = ""
        usage = result.get("usage") or {}
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens:
            token_info = f"，tokens: {_fmt_tokens(total_tokens)}"

        lines.append(f"\n(耗时: {elapsed:.1f}s{retry_info}{token_info})")

        return "\n".join(lines)

    def _format_result_for_llm(self, result: dict) -> str:
        """格式化搜索结果供 LLM 使用（纯文本，无 Markdown）"""
        if not result.get("ok"):
            error = result.get("error", "未知错误")
            raw = result.get("raw", "")
            return f"搜索失败: {error}\n{raw}"

        content = result.get("content", "")
        sources = result.get("sources", [])

        lines = [f"搜索结果:\n{content}"]
        lines.extend(
            self._render_sources(sources, header="参考来源", with_snippet=True)
        )

        # 提示主 LLM 使用纯文本格式回复用户
        lines.append("\n[提示: 请使用纯文本格式回复用户，不要使用 Markdown 格式]")

        return "\n".join(lines)

    def _extract_sources_from_text(self, text: str) -> list[dict[str, str]]:
        """从文本中提取 URL 作为来源，仅允许 http/https 协议"""
        return [{"url": url, "title": "", "snippet": ""} for url in extract_urls(text)]

    def _help_text(self) -> str:
        """返回帮助文本"""
        use_builtin = self.config.get("use_builtin_provider", False)
        mode = "AstrBot 自带" if use_builtin else "自定义"
        provider_id = (
            (self.config.get("provider", "") or "未配置")
            if use_builtin
            else (self.config.get("base_url", "") or "未配置")
        )
        model = (
            "由供应商决定"
            if use_builtin
            else (self.config.get("model", DEFAULT_MODEL) or "默认")
        )
        has_custom_prompt = bool(
            (self.config.get("custom_system_prompt", "") or "").strip()
        )
        if has_custom_prompt:
            prompt_info = "自定义"
        else:
            prompt_info = "内置中文（/grok 指令）/ 内置英文 JSON（LLM Tool）"

        return (
            "Grok 联网搜索\n"
            "\n"
            "用法:\n"
            "  /grok help           显示此帮助\n"
            "  /grok <搜索内容>     执行联网搜索\n"
            "\n"
            "示例:\n"
            "  /grok Python 3.12 有什么新特性\n"
            "  /grok 最新的 AI 新闻\n"
            "  /grok React 19 发布了吗\n"
            "\n"
            "调用方式:\n"
            "  - /grok 指令：直接搜索并返回结果\n"
            "  - LLM Tool：模型自动调用 grok_web_search\n"
            "\n"
            f"当前配置:\n"
            f"  供应商来源: {mode}\n"
            f"  供应商: {provider_id}\n"
            f"  模型: {model}\n"
            f"  系统提示词: {prompt_info}"
        )

    @filter.command("grok")
    async def grok_cmd(self, event: AstrMessageEvent, query: GreedyStr = ""):
        """执行 Grok 搜索

        用法: /grok <搜索内容>
        """
        # 提取消息中的文本和图片（包括引用消息/转发消息）
        extra_text, images = await self._extract_content_from_event(event)
        if images:
            logger.info(
                f"[{PLUGIN_NAME}] /grok command: extracted {len(images)} image(s) from message"
            )

        # 仅在明确输入 help 时显示帮助
        if query.strip().lower() == "help":
            yield event.plain_result(self._help_text())
            return

        # 无查询文本但有图片或引用内容时，继续搜索
        has_content = bool(images) or bool(extra_text)
        if not query.strip() and not has_content:
            yield event.plain_result(self._help_text())
            return

        # 将引用/转发消息中提取的文本拼接到查询前面作为上下文
        if extra_text:
            if query.strip():
                query = f"[Referenced message content]\n{extra_text}\n\n[User query]\n{query}"
            else:
                query = extra_text

        # 仅有图片无文本时，使用默认提示词
        if not query.strip() and images:
            query = "请搜索这张图片的内容"

        # 优先使用自定义提示词，未设置则使用内置提示词（英文指令 + JSON 格式 + 中文回复）
        cmd_system_prompt = resolve_system_prompt(
            self.config.get("custom_system_prompt", ""),
            (
                "You are a web research assistant. Use live web search/browsing when answering. "
                "Return ONLY a single JSON object with keys: "
                "content (string), sources (array of objects with url/title/snippet when possible). "
                "Keep content concise and evidence-backed. "
                "IMPORTANT: Respond in Chinese. Do NOT use Markdown formatting in the content field - use plain text only. "
                "Keep proper nouns and names in their original language."
            ),
        )

        result = await self._do_search(
            query,
            system_prompt=cmd_system_prompt,
            use_retry=True,
            images=images or None,
        )
        event.should_call_llm(True)

        # 判断是否以图片卡片形式发送
        use_image = self.config.get("render_as_image", False) and self._card_fonts_ready
        image_sent = False

        if use_image and result.get("ok"):
            content = result.get("content", "")
            sources = result.get("sources", [])
            elapsed = result.get("elapsed_ms", 0)
            usage = result.get("usage") or {}
            total_tokens = usage.get("total_tokens", 0)
            model = self.config.get("model", "")
            theme = self.config.get("card_theme", "auto")

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                render_search_card(
                    content=content,
                    model=model,
                    elapsed_ms=elapsed,
                    total_tokens=total_tokens,
                    output_path=tmp_path,
                    theme=theme,
                )
                await event.send(MessageChain().file_image(tmp_path))
                image_sent = True
            except Exception as e:
                logger.warning(f"[{PLUGIN_NAME}] 图片卡片发送失败，降级为文本: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            # 来源链接单独以文本发送（可点击/复制）
            if image_sent:
                show_sources = self.config.get("show_sources", False)
                max_sources = self.config.get("max_sources", 5)
                if show_sources and sources:
                    if max_sources > 0:
                        sources = sources[:max_sources]
                    src_lines = ["来源:"]
                    for i, src in enumerate(sources, 1):
                        url = src.get("url", "")
                        title = src.get("title", "")
                        if title:
                            src_lines.append(f"  {i}. {title}\n     {url}")
                        else:
                            src_lines.append(f"  {i}. {url}")
                    try:
                        await event.send(MessageChain().message("\n".join(src_lines)))
                    except Exception as e:
                        logger.warning(f"[{PLUGIN_NAME}] 来源链接发送失败: {e}")

        # 文本模式或图片发送失败时降级
        if not image_sent:
            try:
                await event.send(MessageChain().message(self._format_result(result)))
            except Exception as e:
                logger.warning(f"[{PLUGIN_NAME}] 发送搜索结果失败: {e}")

    @filter.llm_tool(name="grok_web_search")
    async def grok_tool(
        self,
        event: AstrMessageEvent,
        query: str,
        image_urls: str = "",
    ) -> str:
        """实时联网搜索工具。搜索互联网和 X（Twitter）平台获取最新、准确的信息并返回搜索结果和来源链接。

        何时使用：
        - 用户询问实时信息、最新动态、新闻事件、天气、股价等时效性内容
        - 需要验证事实准确性或你对某个信息不确定时
        - 用户明确要求搜索或查询
        - 问题涉及你训练数据截止日期之后的内容
        - 需要获取特定网址、产品、人物的最新状态
        - 需要查找 X（Twitter）上的讨论、帖子、用户动态或社交媒体舆论

        返回内容：搜索结果的文本摘要，可能附带参考来源链接。如果搜索失败会返回错误信息。

        Args:
            query(string): 搜索查询内容，应是清晰、具体、自包含的自然语言问题或关键词
            image_urls(string): 可选，逗号分隔的图片URL，用于基于图片内容的搜索
        """
        # 收集图片：从 LLM 传入的 image_urls 参数 + 用户消息中提取
        images: list[str] = []

        # 1. 解析 LLM 传入的 image_urls
        if image_urls and isinstance(image_urls, str):
            for url in image_urls.split(","):
                url = url.strip()
                if not url:
                    continue
                if url.startswith("base64://"):
                    images.append(url.removeprefix("base64://"))
                elif url.startswith("http"):
                    # 下载并转为 base64
                    try:
                        file_path = await download_image_by_url(url)
                        b64 = file_to_base64(file_path)
                        b64 = b64.removeprefix("base64://")
                        if b64:
                            images.append(b64)
                    except Exception as e:
                        logger.warning(
                            f"[{PLUGIN_NAME}] Failed to download image from URL {url}: {e}"
                        )

        # 2. 从用户消息事件中自动提取内容
        extra_text, event_images = await self._extract_content_from_event(event)
        images.extend(event_images)

        # 将引用/转发消息中提取的文本拼接到查询前面作为上下文
        if extra_text:
            query = (
                f"[Referenced message content]\n{extra_text}\n\n[User query]\n{query}"
            )

        if images:
            logger.info(
                f"[{PLUGIN_NAME}] grok_web_search tool: processing with {len(images)} image(s)"
            )

        result = await self._do_search(query, use_retry=False, images=images or None)
        return self._format_result_for_llm(result)

    @filter.llm_tool(name="grok_web_fetch")
    async def grok_fetch_tool(self, event: AstrMessageEvent, url: str):
        """网页内容抓取工具。利用 Grok 联网能力获取指定 URL 的完整网页内容，转换为结构化 Markdown 格式返回。

        使用场景：
        - 需要读取某个网页的完整内容（文章、文档、帖子等）
        - 需要提取网页中的具体数据（表格、代码示例、列表等）
        - 用户提供了一个 URL 并要求查看或总结其内容

        注意：不需要额外配置外部 API，直接通过 Grok 的联网能力实现。

        Args:
            url(string): 要抓取的网页 URL，必须是完整的 HTTP/HTTPS 地址
        """
        if not url or not url.startswith("http"):
            return "错误：请提供完整的 HTTP/HTTPS URL"

        base_url = self.config.get("base_url", "")
        api_key = self.config.get("api_key", "")
        model = self.config.get("model", DEFAULT_MODEL)
        timeout = self.config.get("timeout_seconds", 60)
        proxy = self.config.get("proxy", "") or None

        extra_body_str = self.config.get("extra_body", "")
        extra_headers_str = self.config.get("extra_headers", "")
        extra_body, _ = parse_json_config(extra_body_str)
        extra_headers, _ = parse_json_config(extra_headers_str)

        result = await grok_fetch(
            url=url,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=float(timeout) if timeout else 60.0,
            extra_body=extra_body or None,
            extra_headers=extra_headers or None,
            proxy=proxy,
        )

        if result.get("ok"):
            content = result.get("content", "")
            elapsed = result.get("elapsed_ms", 0)
            if content:
                return f"{content}\n\n---\n耗时: {elapsed}ms"
            return "抓取成功但页面内容为空"
        else:
            error = result.get("error", "未知错误")
            return f"网页抓取失败: {error}"

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """当 AstrBot 初始化完成后执行的钩子：在启用了自带供应商时完成插件的剩余初始化工作"""
        try:
            if not self.config.get("use_builtin_provider", False):
                return

            logger.info(f"[{PLUGIN_NAME}] AstrBot 已初始化，继续完成插件初始化")

            # 创建复用的 HTTP 会话（如果配置要求）
            if self.config.get("reuse_session", False) and (
                self._session is None or self._session.closed
            ):
                self._session = aiohttp.ClientSession()

            # 迁移并根据 enable_skill 安装或卸载 Skill
            self._migrate_skill_to_persistent()
            if self.config.get("enable_skill", False):
                self._install_skill()
            else:
                self._uninstall_skill()

        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] on_astrbot_loaded 处理失败: {e}")

    async def terminate(self):
        """插件销毁：关闭 HTTP 会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
