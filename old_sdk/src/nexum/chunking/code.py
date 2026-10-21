import textwrap

import torch
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Node, Parser

from nexum.common.ai.base_model import StatefulModel
from nexum.common.helpers.device import resolve
from nexum.common.models import CodeDocument, CodeChunk, CodeChunkConfig


class CodeChunkerDL(StatefulModel[CodeDocument, list[CodeChunk]]):
    NODE_TARGETS: dict[str, dict[str, str]] = {
        "python": {
            "function_definition": "function",
            "class_definition": "class",
        },
        "javascript": {
            "function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
            "arrow_function": "function",
        },
        "typescript": {
            "function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "type_alias_declaration": "type_alias",
            "enum_declaration": "enum",
        },
        "java": {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "method_declaration": "method",
            "record_declaration": "record",
            "enum_declaration": "enum",
        },
        "csharp": {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "method_declaration": "method",
            "property_declaration": "property",
            "record_declaration": "record",
        },
        "go": {
            "function_declaration": "function",
            "method_declaration": "method",
            "type_declaration": "type_spec",
        },
        "rust": {
            "function_item": "function",
            "impl_item": "impl_block",
            "trait_item": "trait",
            "struct_item": "struct",
            "enum_item": "enum",
        },
    }

    def __init__(
            self,
            config: CodeChunkConfig = CodeChunkConfig(),
            compute_embeddings: bool = True,
            max_token_budget: int = 8192,
            cpu_threads: int | None = None,
            trust_remote_code: bool = False,
    ):
        self._model_name = config.model_name
        self._compute_embeddings = compute_embeddings
        self._max_token_budget = max_token_budget
        self._cpu_threads = cpu_threads
        self._trust_remote_code = trust_remote_code

        self._device: torch.device | None = None
        self._embedder: SentenceTransformer | None = None
        self._parsers: dict[str, Parser] = {}
        self._is_loaded: bool = False

    def load(self) -> None:
        self._device = resolve()

        if self._device.type == "cpu" and self._cpu_threads:
            torch.set_num_threads(self._cpu_threads)

        self._init_parsers()

        if self._compute_embeddings:
            self._embedder = SentenceTransformer(
                self._model_name,
                device=str(self._device),
                trust_remote_code=self._trust_remote_code,
            )
            self._embedder.max_seq_length = self._max_token_budget

        self._is_loaded = True

    def _init_parsers(self) -> None:
        import tree_sitter_c_sharp as tscsharp
        import tree_sitter_go as tsgo
        import tree_sitter_java as tsjava
        import tree_sitter_javascript as tsjs
        import tree_sitter_python as tspython
        import tree_sitter_rust as tsrust
        import tree_sitter_typescript as tsts

        lang_grammars = {
            "python": Language(tspython.language()),
            "go": Language(tsgo.language()),
            "java": Language(tsjava.language()),
            "rust": Language(tsrust.language()),
            "javascript": Language(tsjs.language()),
            "typescript": Language(tsts.language_typescript()),
            "csharp": Language(tscsharp.language()),
        }
        for lang, grammar in lang_grammars.items():
            self._parsers[lang] = Parser(grammar)

        self._parsers["kotlin"] = self._parsers["java"]

    def unload(self) -> None:
        self._embedder = None
        self._parsers.clear()
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _count_tokens(self, text: str) -> int:
        if self._embedder and hasattr(self._embedder, "tokenizer"):
            return len(self._embedder.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4

    def predict(self, input_data: CodeDocument) -> list[CodeChunk]:
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded via .load() before inference.")

        lang_key = input_data.language.lower()
        parser = self._parsers.get(lang_key)
        if not parser:
            raise ValueError(f"Unsupported language: {input_data.language}")

        source_bytes = input_data.code.encode("utf8")
        tree = parser.parse(source_bytes)

        target_nodes = self.NODE_TARGETS.get(lang_key, self.NODE_TARGETS["java"])
        raw_chunks: list[CodeChunk] = []

        self._traverse(
            node=tree.root_node,
            source_bytes=source_bytes,
            targets=target_nodes,
            chunks=raw_chunks,
            parent_name=None,
        )

        if not raw_chunks and input_data.code.strip():
            raw_chunks.append(
                CodeChunk(
                    name="root",
                    chunk_type="module",
                    content=textwrap.dedent(input_data.code).strip(),
                    start_line=1,
                    end_line=len(input_data.code.splitlines()),
                )
            )

        for chunk in raw_chunks:
            chunk.token_count = self._count_tokens(chunk.content)
            chunk.metadata["file_path"] = input_data.file_path
            chunk.metadata["language"] = input_data.language
            chunk.metadata["device"] = str(self._device)
            chunk.metadata["model_name"] = self._model_name

            if chunk.token_count > self._max_token_budget:
                chunk.metadata["warning"] = f"Exceeded budget of {self._max_token_budget} tokens."

        if self._compute_embeddings and self._embedder and raw_chunks:
            texts = [c.content for c in raw_chunks]
            batch_size = 8 if self._device.type == "cpu" else 32
            embeddings = self._embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for chunk, emb in zip(raw_chunks, embeddings):
                chunk.embedding = emb.tolist()

        return raw_chunks

    def _extract_class_skeleton(self, class_node: Node, source_bytes: bytes) -> str:
        body_node = class_node.child_by_field_name("body")
        if not body_node:
            for child in class_node.children:
                if child.type in ("block", "class_body"):
                    body_node = child
                    break

        if not body_node:
            return source_bytes[class_node.start_byte: class_node.end_byte].decode("utf8", errors="replace")

        header = source_bytes[class_node.start_byte: body_node.start_byte].decode("utf8", errors="replace").rstrip()
        skeleton_lines = [header]

        for child in body_node.children:
            if child.type == "expression_statement":
                text = child.text.decode("utf8", errors="replace").strip()
                if text.startswith(('"""', "'''")):
                    skeleton_lines.append(f"    {text}")
            elif child.type in ("assignment", "type_alias"):
                text = child.text.decode("utf8", errors="replace").strip()
                skeleton_lines.append(f"    {text}")
            elif child.type in ("function_definition", "decorated_definition"):
                sig = self._extract_method_signature(child, source_bytes)
                if sig:
                    skeleton_lines.append(f"    {sig}\n        ...")

        return "\n".join(skeleton_lines)

    def _extract_method_signature(self, node: Node, source_bytes: bytes) -> str:
        func_node = node
        decorators = []

        if node.type == "decorated_definition":
            for c in node.children:
                if c.type == "decorator":
                    decorators.append(c.text.decode("utf8", errors="replace").strip())
                elif c.type == "function_definition":
                    func_node = c

        body = func_node.child_by_field_name("body")
        if body:
            sig = source_bytes[func_node.start_byte: body.start_byte].decode("utf8", errors="replace").strip()
        else:
            sig = func_node.text.decode("utf8", errors="replace").splitlines()[0].strip()

        if decorators:
            return "\n    ".join(decorators) + f"\n    {sig}"
        return sig

    def _traverse(
            self,
            node: Node,
            source_bytes: bytes,
            targets: dict[str, str],
            chunks: list[CodeChunk],
            parent_name: str | None,
    ) -> None:
        current_node = node

        if current_node.type == "decorated_definition":
            inner_def = None
            for child in current_node.children:
                if child.type in targets:
                    inner_def = child
                    break

            if inner_def:
                entity_name = self._extract_identifier(inner_def) or "anonymous"
                chunk_type = targets[inner_def.type]
                raw_code = source_bytes[current_node.start_byte: current_node.end_byte].decode("utf8", errors="replace")

                clean_code = textwrap.dedent(raw_code).strip()
                context_header = f"# Scope: {parent_name}\n" if parent_name else ""
                content = f"{context_header}{clean_code}"

                chunks.append(
                    CodeChunk(
                        name=entity_name,
                        chunk_type=chunk_type,
                        content=content,
                        start_line=current_node.start_point.row + 1,
                        end_line=current_node.end_point.row + 1,
                        parent_scope=parent_name,
                    )
                )
                return

        if current_node.type in targets:
            chunk_type = targets[current_node.type]
            entity_name = self._extract_identifier(current_node) or "anonymous"

            if chunk_type == "class":
                content = self._extract_class_skeleton(current_node, source_bytes)
            else:
                raw_code = source_bytes[current_node.start_byte: current_node.end_byte].decode("utf8", errors="replace")
                clean_code = textwrap.dedent(raw_code).strip()
                context_header = f"# Scope: {parent_name}\n" if parent_name else ""
                content = f"{context_header}{clean_code}"

            chunks.append(
                CodeChunk(
                    name=entity_name,
                    chunk_type=chunk_type,
                    content=content,
                    start_line=current_node.start_point.row + 1,
                    end_line=current_node.end_point.row + 1,
                    parent_scope=parent_name,
                )
            )

            parent_name = f"{parent_name}.{entity_name}" if parent_name else entity_name

        for child in current_node.children:
            self._traverse(child, source_bytes, targets, chunks, parent_name)

    @staticmethod
    def _extract_identifier(node: Node) -> str | None:
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "name"):
                return child.text.decode("utf8", errors="replace")
            if child.type in ("function_definition", "class_definition"):
                for sub in child.children:
                    if sub.type in ("identifier", "name"):
                        return sub.text.decode("utf8", errors="replace")
        return None
