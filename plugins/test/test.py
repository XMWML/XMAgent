"""Example /test plugin."""

PLUGIN = {
    "name": "Test",
    "command": "test",
    "description": "回显输入文本",
    "usage": "/test text1",
}


def handle(ctx, args: str) -> str:
    return f"输入了{args}" if args else "请输入文本，例如：/test text1"
