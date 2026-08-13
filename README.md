# XMAgent

XMAgent 是自托管多渠道 Agent 服务。它将微信 iLink 联系人、Telegram 私聊和已白名单的 Telegram 群组接入独立的 Agent、对话会话和工作区，并提供带密码登录的 WebUI 管理台。

默认监听适合本机或局域网使用；需要公开访问时，请由常规 HTTPS 反向代理接入。

## 功能概览

- 多个微信 iLink 和 Telegram Bot 账号；Telegram 支持每 Bot 独立 HTTP proxy。
- 微信二维码登录（一次扫码确认只创建一个渠道）、微信和 Telegram 私聊的统一待绑定流程、群组白名单和每群共享 Agent。
- 可视化渠道路由：每个联系人或群组明确绑定一个 Agent，可在 WebUI 随时重绑。
- 每个 Agent 独立工作区、持久化历史、串行邮箱、多轮 Claude SDK 会话和 OpenAI-compatible Chat Completions SSE。
- `/new`、`/effort`、`/status`、`/help`、`/sendfile`，以及 Claude 工具状态和 Telegram 流式编辑回复。
- WebUI 管理渠道、API profile/模型、Agent 高级 SDK 参数、MCP、插件、记忆、知识库、任务、历史、Outbox 与审计日志；仪表盘显示累计入站消息和最近收到的消息。
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

# 设置管理员密码
.venv/bin/xmagents init

# 本机启动；局域网可改用 --host both
.venv/bin/xmagents serve --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765/`，依次创建 API profile、Agent 和渠道。执行以下命令检查本地数据目录和初始化状态：

```bash
.venv/bin/xmagents doctor
```

默认 `serve --host both` 同时监听 `0.0.0.0` 与 `::`。公开部署时请绑定到 `127.0.0.1` 并通过 HTTPS 反向代理提供服务。

## 主要行为

- 微信和 Telegram 私聊的首条消息都会进入“待绑定渠道用户”，并回复 `未绑定agent，请联系管理员在管理台绑定agent后继续。`。管理员可为其新建独立 Agent 或绑定既有 Agent；绑定后，用户下一条消息开始进入相应 Agent。所有联系人/群组到 Agent 的对应关系可在“渠道与扫码”页面查看和重绑。
- Telegram 白名单群组每群一个共享 Agent，只处理命令、@Bot 或回复 Bot 的消息。
- 内置“本机 Claude Code” profile 优先调用运行服务用户安装的 `claude`，复用其认证和设置；它既可使用该用户的 `claude login` 状态，也可继承服务进程的 `ANTHROPIC_AUTH_TOKEN` 与 `ANTHROPIC_BASE_URL` API/计费环境变量。普通 Anthropic profile 使用保存的 API 配置；OpenAI-compatible profile 调用 Chat Completions SSE。
- Telegram 回复会以一条可编辑消息流式呈现，超长结果自动分段。微信单账号默认最小发送间隔为 1.5 秒，文本按段落/句子拆分以降低限频风险。
- 渠道凭据、API key 和 MCP headers 保存在 SQLite 中，WebUI/API 只显示是否已配置或掩码。

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
