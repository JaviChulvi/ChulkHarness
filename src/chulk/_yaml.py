"""Shared strict YAML mechanics for declarative manifests."""

from __future__ import annotations

from typing import Any

import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node


class StrictManifestLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects aliases, anchors, and duplicate keys."""

    manifest_kind: str

    def compose_node(self, parent: Node | None, index: int) -> Node:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ComposerError(
                None,
                None,
                f"YAML aliases are not allowed in {self.manifest_kind} manifests",
                event.start_mark,
            )
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise ComposerError(
                None,
                None,
                f"YAML anchors are not allowed in {self.manifest_kind} manifests",
                event.start_mark,
            )
        node = super().compose_node(parent, index)
        if node is None:
            raise ComposerError(
                None, None, f"{self.manifest_kind} YAML node is missing", None
            )
        return node

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                "expected a mapping",
                node.start_mark,
            )
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _reject_custom_tag(
    _loader: StrictManifestLoader,
    tag_suffix: str,
    node: Node,
) -> object:
    raise ConstructorError(
        None,
        None,
        f"custom YAML tag is not allowed: {tag_suffix or node.tag}",
        node.start_mark,
    )


StrictManifestLoader.add_multi_constructor("!", _reject_custom_tag)
