from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(slots=True)
class ConversationHistory:
    """保存当前会话中不会因模型上下文处理而丢失的完整历史。"""

    _items: list[object] = field(default_factory=list, repr=False)

    def append(self, item: object) -> None:
        """把一条新产生的协议项追加到完整历史。"""
        self._items.append(item)

    def extend(self, items: Iterable[object]) -> None:
        """按原顺序追加一组协议项。"""
        self._items.extend(items)

    def clear(self) -> None:
        """清空内存历史，用于切换到新会话。"""
        self._items.clear()

    def snapshot(self) -> tuple[object, ...]:
        """返回不会受后续追加影响的当前历史快照。"""
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True, slots=True)
class ModelContextView:
    """表示某一次模型请求实际可见的不可变输入视图。"""

    items: tuple[object, ...]
