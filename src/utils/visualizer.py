from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from src.config.config import config
from pathlib import Path
from loguru import logger
import graphviz
from pathlib import Path

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


NODE_SIZE = 1200
NODE_BORDER_NORMAL = 1.2
NODE_BORDER_SPECIAL = 1.5
FIGURE_SIZE = (12, 12)
FIGURE_DPI = 200
FONT_SIZE = 12
LABEL_WRAP_LENGTH = 20
ARROW_SIZE = 20
EDGE_WIDTH = 1.5
EDGE_CURVE = 0.15
EDGE_SOURCE_MARGIN = 10
EDGE_TARGET_MARGIN = 15
COLOR_ENTRY = "green"
COLOR_EXIT = "red"
COLOR_REGULAR = "black"
COLOR_EDGE = "black"
COLOR_BACKGROUND = "white"

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG


def visualize_cfg_with_graphviz(cfg: CFG, output_directory: Path, engine: str = 'dot') -> Optional[Path]:
    """Generate CFG visualization using Graphviz

    Renders the control flow graph to SVG and PNG files in the specified
    output directory Uses invisible dummy entry/exit nodes with explicit
    ranks to enforce a top-to-bottom hierarchy

    Args:
        cfg (CFG): The CFG object containing the graph and node information
        output_directory (Path): The directory path object where the images
            should be saved
        engine (str): The Graphviz layout engine to use (e.g, 'dot',
            'neato') Defaults to 'dot'

    Returns:
        Optional[Path]: The path to the generated SVG file if successful,
                        otherwise None

    Raises:
        graphviz.ExecutableNotFound: If the specified Graphviz engine
                                     executable is not found in the system PATH
    """

    output_directory.mkdir(parents=True, exist_ok=True)

    output_filename = output_directory / "cfg"
    output_svg_path = output_directory / "cfg.svg"
    output_png_path = output_directory / "cfg.png"
    dot = graphviz.Digraph(comment='Control Flow Graph')
    dot.attr(rankdir='TB', splines='spline')
    dot.graph_attr.update({ 'pad': '0', 'margin': '0' })
    dot.attr('node', shape='ellipse', style='solid', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='8')

    main_entry_node = cfg.entry_node
    main_exit_node = cfg.exit_node

    # Anchor real entry node at top
    if main_entry_node:
        with dot.subgraph() as s:
            s.attr(rank='source')
            s.node(str(main_entry_node.node_id))
    else:
        logger.warning("No entry node found in CFG to anchor at the top.")

    # Anchor real exit node at bottom
    if main_exit_node:
        with dot.subgraph() as s:
            s.attr(rank='sink')
            s.node(str(main_exit_node.node_id))
    else:
        logger.warning("No exit node found in CFG to anchor at the bottom.")

    for node in cfg.graph.nodes():
        node_id = node.node_id
        node_id_str = str(node_id)

        label_parts = []
        if 'id' in config.visualization.node_labels:
             label_parts.append(f"ID: {node_id_str}")

        jump_obj = node.jump
        instr_obj = node.instructions
        node_type = node.__class__.__name__
        content_str = None

        if jump_obj:
            node_type = jump_obj.__class__.__name__
            expression = jump_obj.expression
            if expression:
                 content_str = expression.to_c()
            else:
                content_str = str(jump_obj)
        elif instr_obj and instr_obj.operations:
            node_type = instr_obj.__class__.__name__
            ops_list = [op.to_c() for op in instr_obj.operations]
            content_str = "; ".join(ops_list)

        if 'type' in config.visualization.node_labels:
            label_parts.append(node_type)
        if content_str and 'content' in config.visualization.node_labels:
             max_content_len = 25
             if len(content_str) > max_content_len:
                 content_str = content_str[:max_content_len-3] + "..."
             label_parts.append(content_str)
        if 'depth' in config.visualization.node_labels:
             label_parts.append(f"depth: {node.depth}")

        final_label = "\n".join(label_parts) if label_parts else node_id_str

        color = COLOR_REGULAR
        penwidth = str(NODE_BORDER_NORMAL)

        if node == main_entry_node:
            color = COLOR_ENTRY
            penwidth = str(NODE_BORDER_SPECIAL)
        elif node == main_exit_node:
            color = COLOR_EXIT
            penwidth = str(NODE_BORDER_SPECIAL)

        dot.node(node_id_str, label=final_label, color=color, penwidth=penwidth)

    for u, v, data in cfg.graph.edges(data=True):
        source_id_str = str(u.node_id)
        target_id_str = str(v.node_id)

        condition = data.get('condition', None)
        condition_label = ''
        if condition:
             condition_label = condition.to_c()
             max_cond_len = 30
             if len(condition_label) > max_cond_len:
                  condition_label = condition_label[:max_cond_len-3] + "..."

        edge_label = data.get('label', '')

        edge_color = COLOR_EDGE
        style = 'solid'

        if edge_label:
            label_str = condition_label + "(" + edge_label + ")"
        else:
            label_str = condition_label
        dot.edge(source_id_str, target_id_str, label=label_str, color=edge_color, style=style, fontcolor='blue')

    try:
        dot.render(output_filename, view=False, format='svg', engine=engine, cleanup=True)
        dot.render(output_filename, view=False, format='png', engine=engine, cleanup=True)

        return output_svg_path
    except graphviz.ExecutableNotFound:
         logger.error(f"Graphviz executable ('{engine}') not found. Ensure Graphviz is installed and in your system's PATH.")
         return None

