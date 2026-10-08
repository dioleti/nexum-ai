import asyncio
import textwrap

import tree_sitter_c_sharp as tscsharp
import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tsts
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Node, Parser

from nexum.common.ai.deep_learning.base_model import StatefulModel
from nexum.common.models import CodeDocument, CodeChunk


class CodeChunkerModel(StatefulModel[CodeDocument, list[CodeChunk]]):
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
            embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            compute_embeddings: bool = True,
    ):
        self._embedding_model_name = embedding_model_name
        self._compute_embeddings = compute_embeddings
        self._embedder: SentenceTransformer | None = None
        self._parsers: dict[str, Parser] = {}
        self._is_loaded: bool = False

    def load(self) -> None:
        """Carrega os parsers de gramática e o modelo pré-treinado em memória."""
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

        # Kotlin reusa a gramática Java em fallback ou parser nativo
        self._parsers["kotlin"] = self._parsers["java"]

        if self._compute_embeddings:
            self._embedder = SentenceTransformer(self._embedding_model_name)

        self._is_loaded = True

    def unload(self) -> None:
        """Libera o modelo de embedding e descarta parsers."""
        self._embedder = None
        self._parsers.clear()
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, input_data: CodeDocument) -> list[CodeChunk]:
        """Processa um arquivo de código e extrai os blocos semânticos com metadados."""
        if not self.is_loaded:
            raise RuntimeError("O modelo precisa ser carregado via .load() antes de inferir.")

        lang_key = input_data.language.lower()
        parser = self._parsers.get(lang_key)
        if not parser:
            raise ValueError(f"Linguagem não suportada: {input_data.language}")

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

        # Se o arquivo não tiver funções/classes (ex: script plano), mantém o corpo como chunk único
        if not raw_chunks and input_data.code.strip():
            raw_chunks.append(
                CodeChunk(
                    name="root",
                    chunk_type="module",
                    content=input_data.code.strip(),
                    start_line=1,
                    end_line=len(input_data.code.splitlines()),
                    metadata={"file_path": input_data.file_path},
                )
            )

        # Gera vetores de embedding para uso no índice vetorial do RAG
        if self._compute_embeddings and self._embedder and raw_chunks:
            texts = [c.content for c in raw_chunks]
            embeddings = self._embedder.encode(texts, convert_to_numpy=True)
            for chunk, emb in zip(raw_chunks, embeddings):
                chunk.embedding = emb.tolist()
                chunk.metadata["file_path"] = input_data.file_path

        return raw_chunks

    def _extract_class_skeleton(self, class_node: Node, source_bytes: bytes) -> str:
        """Gera um esqueleto sintático enxuto da classe para o índice de RAG."""
        body_node = class_node.child_by_field_name("body")
        if not body_node:
            # Fallback caso a gramática não exponha o campo 'body'
            for child in class_node.children:
                if child.type in ("block", "class_body"):
                    body_node = child
                    break

        if not body_node:
            return source_bytes[class_node.start_byte: class_node.end_byte].decode("utf8", errors="replace")

        # Cabeçalho completo da classe (inclui heranças multilinhas até o ':')
        header = source_bytes[class_node.start_byte: body_node.start_byte].decode("utf8", errors="replace").rstrip()

        skeleton_lines = [header]

        for child in body_node.children:
            # Mantém docstrings soltas no corpo da classe
            if child.type == "expression_statement":
                text = child.text.decode("utf8", errors="replace").strip()
                if text.startswith(('"""', "'''")):
                    skeleton_lines.append(f"    {text}")

            # Atribuições de atributos de classe (ex: MAX_RETRIES = 3)
            elif child.type in ("assignment", "type_alias"):
                text = child.text.decode("utf8", errors="replace").strip()
                skeleton_lines.append(f"    {text}")

            # Métodos normais ou decorados
            elif child.type in ("function_definition", "decorated_definition"):
                sig = self._extract_method_signature(child, source_bytes)
                if sig:
                    skeleton_lines.append(f"    {sig}\n        ...")

        return "\n".join(skeleton_lines)

    def _extract_method_signature(self, node: Node, source_bytes: bytes) -> str:
        """Extrai apenas decoradores e a linha de assinatura def ...(...) -> Retorno:"""
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
        """Percorre a AST isolando esqueleto da classe e métodos com contexto."""
        current_node = node

        # 1. Tratamento de Métodos Decorados
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

                # Injeta breadcrumb de contexto para RAG
                context_header = f"# Scope: {parent_name}\n" if parent_name else ""
                content = f"{context_header}{raw_code}"

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

        # 2. Tratamento de Classes e Funções Não-Decoradas
        if current_node.type in targets:
            chunk_type = targets[current_node.type]
            entity_name = self._extract_identifier(current_node) or "anonymous"

            if chunk_type == "class":
                content = self._extract_class_skeleton(current_node, source_bytes)
            else:
                raw_code = source_bytes[current_node.start_byte: current_node.end_byte].decode("utf8", errors="replace")
                clean_code = textwrap.dedent(raw_code)
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
        """Localiza o identificador direto ou em estruturas decoradas."""
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "name"):
                return child.text.decode("utf8", errors="replace")
            # Se for nó aninhado (ex: método decorado)
            if child.type in ("function_definition", "class_definition"):
                for sub in child.children:
                    if sub.type in ("identifier", "name"):
                        return sub.text.decode("utf8", errors="replace")
        return None


if __name__ == '__main__':
    async def main():
        model = CodeChunkerModel(compute_embeddings=True)
        model.load()

        doc = CodeDocument(
            code="""
    from typing import Sequence

import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from transformers import logging as hf_logging

from nexum.common.ai.deep_learning.base_model import StatefulModel
from nexum.document.models import TableDetectionInput, DetectedTable, TableDetectionConfig, BoundingBox


class TableDetector(
    StatefulModel[TableDetectionInput, list[DetectedTable]]
):
    def __init__(self, config: TableDetectionConfig | None = None) -> None:
        self.config = config or TableDetectionConfig()
        self.device = torch.device(
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor: AutoImageProcessor | None = None
        self.model: TableTransformerForObjectDetection | None = None

    @property
    def is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    def load(self) -> None:
        if not self.is_loaded:
            hf_logging.set_verbosity_error()
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.model_name
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                self.config.model_name
            ).to(self.device)
            self.model.eval()

    def unload(self) -> None:
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, input_data: TableDetectionInput) -> list[DetectedTable]:
        return self.predict_batch([input_data])[0]

    def predict_batch(
        self, inputs: Sequence[TableDetectionInput]
    ) -> list[list[DetectedTable]]:
        if not inputs:
            return []

        if not self.is_loaded:
            self.load()

        assert self.processor is not None
        assert self.model is not None

        images = [item.image for item in inputs]
        batch = self.processor(images=images, return_tensors="pt").to(
            self.device
        )

        with torch.inference_mode():
            outputs = self.model(**batch)

        target_sizes = torch.tensor(
            [img.size[::-1] for img in images], device=self.device
        )

        all_results: list[list[DetectedTable]] = []

        for idx, input_item in enumerate(inputs):
            threshold = (
                input_item.threshold
                if input_item.threshold is not None
                else self.config.default_threshold
            )

            post_processed = self.processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes[idx: idx + 1],
            )[idx]

            detected_tables = self._extract_tables_from_prediction(
                post_processed
            )
            all_results.append(detected_tables)

        return all_results

    def _extract_tables_from_prediction(
        self, prediction: dict[str, torch.Tensor]
    ) -> list[DetectedTable]:
        assert self.model is not None

        tables: list[DetectedTable] = []
        for score, label, box in zip(
            prediction["scores"],
            prediction["labels"],
            prediction["boxes"],
        ):
            label_name = self.model.config.id2label[label.item()]
            if "table" in label_name:
                coords = [round(float(c), 1) for c in box.tolist()]
                tables.append(
                    DetectedTable(
                        label=label_name,
                        score=round(float(score), 4),
                        box=BoundingBox(
                            xmin=coords[0],
                            ymin=coords[1],
                            xmax=coords[2],
                            ymax=coords[3],
                        ),
                    )
                )

        return tables

    """,
            language="python",
            file_path="src/payments.py",
        )

        # Execução assíncrona não bloqueia o event loop
        chunks = await model.apredict(doc)

        for chunk in chunks:
            print(f"[{chunk.chunk_type.upper()}] {chunk.name} (Linhas {chunk.start_line}-{chunk.end_line})")
            print(f"Parent: {chunk.parent_scope}")
            print(f"Embedding dimensions: {len(chunk.embedding) if chunk.embedding else None}")
            print(f"Chunk: \n\t {chunk.content}\n")
            print("-" * 40)

        model.unload()


    asyncio.run(main())
