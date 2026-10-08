import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from nexum.common.ai.deep_learning.flowchart_detection import FlowchartExtractor
from nexum.document.models import (
    FlowchartConfig,
    FlowchartGraph,
    FlowchartInput,
    FlowchartNode,
    FlowchartEdge,
)


class TestFlowchartExtractor(unittest.TestCase):

    @patch("nexum.common.ai.deep_learning.flowchart_detection.Reader")
    def test_init(self, mock_reader_cls):
        config = FlowchartConfig()
        extractor = FlowchartExtractor(config)
        self.assertIs(extractor.config, config)
        mock_reader_cls.assert_called_once_with(['pt', 'en'], gpu=False)

    @patch("nexum.common.ai.deep_learning.flowchart_detection.Reader")
    def test_predict_less_than_2_nodes(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader_cls.return_value = mock_reader
        mock_reader.readtext.return_value = [
            ([[10, 10], [50, 10], [50, 30], [10, 30]], "Start", 0.95)
        ]

        extractor = FlowchartExtractor()
        inp = FlowchartInput(image=np.zeros((100, 100, 3), dtype=np.uint8))
        result = extractor.predict(inp)

        self.assertIsNone(result)

    @patch("nexum.common.ai.deep_learning.flowchart_detection.Reader")
    def test_predict_success(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader_cls.return_value = mock_reader
        mock_reader.readtext.return_value = [
            ([[10, 100], [90, 100], [90, 140], [10, 140]], "Step 2", 0.9),
            ([[10, 10], [90, 10], [90, 50], [10, 50]], "Step 1", 0.95),
        ]

        extractor = FlowchartExtractor()
        inp = FlowchartInput(image=np.zeros((200, 200, 3), dtype=np.uint8))
        result = extractor.predict(inp)

        self.assertIsInstance(result, FlowchartGraph)
        self.assertEqual(len(result.nodes), 2)
        # Should be sorted by Y center -> Step 1 then Step 2
        self.assertEqual(result.nodes[0].label, "Step 1")
        self.assertEqual(result.nodes[1].label, "Step 2")

        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0].source, result.nodes[0].id)
        self.assertEqual(result.edges[0].target, result.nodes[1].id)

        self.assertIn("graph TD", result.mermaid)
        self.assertIn(f'{result.nodes[0].id}["Step 1"]', result.mermaid)
        self.assertIn(f'{result.nodes[1].id}["Step 2"]', result.mermaid)
        self.assertIn(f"{result.nodes[0].id} --> {result.nodes[1].id}", result.mermaid)

    def test_extract_nodes(self):
        with patch("nexum.common.ai.deep_learning.flowchart_detection.Reader"):
            extractor = FlowchartExtractor()

        boxes = [
            ([[10, 20], [60, 20], [60, 40], [10, 40]], "Process Order", 0.99)
        ]

        nodes = extractor._extract_nodes(boxes)
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.id, "node_1")
        self.assertEqual(node.label, "Process Order")
        self.assertEqual(node.shape, "rectangle")
        self.assertEqual(node.box, (10, 20, 50, 20))
        self.assertEqual(node.center, (35, 30))

    def test_build_sequential_edges(self):
        nodes = [
            FlowchartNode(id="n1", label="A", shape="rectangle", box=(0, 0, 10, 10), center=(5, 5)),
            FlowchartNode(id="n2", label="B", shape="rectangle", box=(0, 20, 10, 10), center=(5, 25)),
            FlowchartNode(id="n3", label="C", shape="rectangle", box=(0, 40, 10, 10), center=(5, 45)),
        ]
        edges = FlowchartExtractor._build_sequential_edges(nodes)
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges[0], FlowchartEdge(source="n1", target="n2", direction="down"))
        self.assertEqual(edges[1], FlowchartEdge(source="n2", target="n3", direction="down"))

    def test_generate_mermaid(self):
        nodes = [
            FlowchartNode(id="n1", label='Hello "World"', shape="rectangle", box=(0, 0, 10, 10), center=(5, 5)),
            FlowchartNode(id="n2", label="End", shape="rectangle", box=(0, 20, 10, 10), center=(5, 25)),
        ]
        edges = [FlowchartEdge(source="n1", target="n2", direction="down")]

        mermaid = FlowchartExtractor._generate_mermaid(nodes, edges)
        expected = "graph TD\n    n1[\"Hello 'World'\"]\n    n2[\"End\"]\n    n1 --> n2"
        self.assertEqual(mermaid, expected)


if __name__ == "__main__":
    unittest.main()
