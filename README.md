# XMAgent

XMAgent 是一个面向受信任局域网的自托管多渠道 Agent 服务。它将微信 iLink 联系人、Telegram 私聊和已白名单的 Telegram 群组接入独立的 Agent、对话会话和工作区，并提供密码保护的 WebUI 管理台。

> 这是一个可执行本机工具的 Agent 网关，不是公网多租户 SaaS。默认 `bypassPermissions` 会允许 Claude Code 在宿主机环境中执行工具；只应向受信任的管理员和渠道用户开放。

## 功能概览

- 多个微信 iLink 和 Telegram Bot 账号；Telegram 支持每 Bot 独立 HTTP proxy。
- 微信二维码登录、Telegram 待审批私聊、群组白名单和每群共享 Agent。
- 每个 Agent 独立工作区、持久化历史、串行邮箱、多轮 Claude SDK 会话和 OpenAI-compatible Chat Completions SSE。
- `/new`、`/effort`、`/status`、`/help`、`/sendfile`，以及 Claude 工具状态和 Telegram 流式编辑回复。
- WebUI 管理渠道、API profile/模型、Agent 高级 SDK 参数、MCP、插件、记忆、知识库、任务、历史、Outbox 与审计日志。
- SQLite WAL 持久化，持久 Outbox、限频、重试/死信/结果未知状态，以及受控本地 socket 文件发送和定时任务能力。
- 每 Agent 独立记忆和 `CLAUDE.md`；SQLite 知识库 FTS 检索与 Anthropic 只读知识库 MCP 工具。

## 快速部署

要求：Python 3.11+，Linux/macOS（运行期使用 Unix domain socket），以及能访问已配置的渠道和模型服务。

```bash
git clone https://github.com/XMWML/XMAgent.git XMAgent
cd XMAgent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'

# 在尚未暴露到网络前设置管理员密码
.venv/bin/xmagents init

# 首次建议先仅监听本机；确认后再放到 HTTPS 反向代理后
.venv/bin/xmagents serve --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765/`，依次创建 API profile、Agent 和渠道。执行以下命令检查本地数据目录和初始化状态：

```bash
.venv/bin/xmagents doctor
```

默认 `serve --host both` 同时监听 `0.0.0.0` 与 `::`。这是明文 HTTP，仅适用于可信 LAN。生产部署请把 XMAgent 绑定到 `127.0.0.1`，由 HTTPS 反向代理对外提供服务，并限制防火墙入站来源。

## 主要行为

- 微信联系人首条消息会自动创建并绑定独立 Agent；Telegram 私聊必须在 WebUI 审批后才会创建 Agent。
- Telegram 白名单群组每群一个共享 Agent，只处理命令、@Bot 或回复 Bot 的消息。
- Anthropic profile 通过 `ANTHROPIC_AUTH_TOKEN` 和 `ANTHROPIC_BASE_URL` 注入 Claude SDK 子进程；OpenAI-compatible profile 调用 Chat Completions SSE。
- Telegram 回复会以一条可编辑消息流式呈现，超长结果自动分段。微信单账号默认最小发送间隔为 1.5 秒，文本按段落/句子拆分以降低限频风险。
- 渠道凭据、API key 和 MCP headers 由 SQLite 明文保存，但不通过 WebUI/API 返回明文；务必保护 `data/`、备份和宿主机账户。

## 文档

- [使用手册](使用手册.md)：服务器部署、WebUI 操作、渠道、API/Agent、MCP、插件、记忆、知识库、文件、任务、安全、故障排查和维护。
- Claude Agent SDK 以项目依赖 `claude-agent-sdk==0.2.135` 为准；参数说明见[官方文档](https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python)。
- 微信 iLink 适配器位于 `xmagents/channels/wechat.py`，协议变化时应以该模块的 fixture 测试和官方接口为准。

## 验证边界

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q xmagents plugins
```

测试使用 mock/fixture 覆盖数据库、WebUI、消息解析、限频、流式、文件路径、任务等本地行为。真实微信扫码、Telegram Bot/代理、模型 API、MCP 和 WebSearch 仍需在目标服务器使用真实凭据完成 smoke test。

## 许可证

[MIT](LICENSE)
