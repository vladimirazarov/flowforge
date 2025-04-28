"""
Manage program output generation and saving
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import subprocess
import shutil
from typing import Optional, TYPE_CHECKING, List, Any, cast, Dict
import re 

from loguru import logger
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Flowable,
    PageBreak,
    Image
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from svglib.svglib import svg2rlg
import graphviz
from reportlab.graphics.shapes import Drawing, Group



from src.utils.serializers import (
    serialize_test_paths,
    serialize_config,
    serialize_cfg_structure,
    format_shtv_details,
    format_test_suites,
    format_with_loops
)


PDF_PATH_FONT_SIZE = 10 
PDF_CODE_FONT_SIZE = PDF_PATH_FONT_SIZE 

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG
    from src.config.config import AppConfig
    from src.core.fragments.fragment_forest import FragmentForest

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


class AutoLayoutDocTemplate(BaseDocTemplate):
    """Define a custom BaseDocTemplate using A4 pages and standard margins

    This template automatically sets up a single frame covering the page
    area within 1-inch margins
    """
    def __init__(self, filename, **kwargs):
        """Initialize the document template

        Args:
            filename (str): The path for the output PDF file
        """
        super().__init__(filename, pagesize=A4, **kwargs)
        self.addPageTemplates(self._create_page_templates())

    def _create_page_templates(self):
        """Helper method to create the default page template with a single frame

        Returns:
            list[PageTemplate]: A list containing the single page template
        """
        margin = inch  
        frame = Frame(
            margin,
            margin,
            A4[0] - 2 * margin,  
            A4[1] - 2 * margin   
        )
        return [PageTemplate(id='AutoLayout', frames=[frame])]


class ThreeZoneDocTemplate(BaseDocTemplate):
    """Define a document template with three frames for a specific layout

    - Frame A: Top-left (2/3 width) for CFG
    - Frame B: Top-right (1/3 width) for Code
    - Frame C: Bottom (full width) for Test Paths
    """
    def __init__(self, filename, top_height_ratio=0.6, **kwargs):
        """Initialize the document template.

        Args:
            filename (str): The path for the output PDF file
            top_height_ratio (float): Proportion of usable height for top frames (A & B)
            **kwargs: Additional keyword arguments for BaseDocTemplate
        """
        super().__init__(filename, pagesize=A4, **kwargs)
        self.addPageTemplates(self._create_page_templates(top_height_ratio))

    def _create_page_templates(self, top_height_ratio: float):
        """Helper method to create the three-zone page template"""
        m = inch  
        w, h = A4
        usable_w = w - 2 * m
        usable_h = h - 2 * m
        g = 0.2 * inch 

        top_h = usable_h * top_height_ratio
        bottom_h = usable_h - top_h

        left_w = usable_w * 0.65 - g / 2  
        right_w = usable_w * 0.35 - g / 2 

        inner_left_w  = left_w  - 12 
        inner_right_w = right_w - 12 
        inner_top_h   = top_h   - 12 

        # Frame A: Top-left
        fA = Frame(m, m + bottom_h,
                   inner_left_w, inner_top_h,
                   id='cfg', leftPadding=6, bottomPadding=6, rightPadding=6, topPadding=6)

        # Frame B: Top-right (shifted right by the gutter)
        fB = Frame(m + inner_left_w + g, m + bottom_h, 
                   inner_right_w, inner_top_h,
                   id='code', leftPadding=6, bottomPadding=6, rightPadding=6, topPadding=6)

        # Frame C: Bottom (full width)
        fC = Frame(m, m,
                   usable_w, bottom_h,
                   id='paths', leftPadding=6, bottomPadding=6, rightPadding=6, topPadding=6)

        return [PageTemplate(id='three_zone', frames=[fA, fB, fC])]


class OnePageContainer(Flowable):
    """Scale and center a list of flowables to fit vertically on one page

    Measures children, calculates a scale factor to fit available height,
    and optionally centers the scaled block horizontally
    """
    def __init__(self, flowables: List[Flowable], spacing: float = 6, center_content: bool = True):
        """Initialize the container

        Args:
            flowables (List[Flowable]): Flowables to arrange vertically
            spacing (float): Vertical spacing between flowables
            center_content (bool): Whether to horizontally center the content
        """
        super().__init__()
        self.flowables = flowables
        self.spacing = spacing
        self.center_content = center_content

        self.child_sizes = []    # List of (width, height) for each flowable
        self.total_width = 0     # The unscaled total width needed
        self.total_height = 0    # The unscaled total height needed
        self.scale_factor = 1.0  # Computed scale factor
        self.avail_width = 0     # Available width from wrap
        self.avail_height = 0    # Available height from wrap

    def wrap(self, availWidth, availHeight):
        """Calculate required dimensions and scale factor

        Measures child flowables at full size, computes total needed
        dimensions including spacing, and determines the scale factor
        required to fit vertically within availHeight

        Args:
            availWidth (float): The available width for the container
            availHeight (float): The available height for the container

        Returns:
            tuple[float, float]: The dimensions (availWidth, availHeight)
                                 claimed by this flowable
        """
        self.avail_width = availWidth
        self.avail_height = availHeight

        total_height = 0
        max_width = 0
        self.child_sizes = []

        # Wrap each child to get its desired (width, height)
        for f in self.flowables:
            w, h = f.wrap(availWidth, availHeight)
            self.child_sizes.append((w, h))
            total_height += h
            if w > max_width:
                max_width = w

        # Add spacing for gaps between flowables
        if self.flowables:
            total_height += self.spacing * (len(self.flowables) - 1)

        self.total_width = max_width
        self.total_height = total_height

        scale_y = availHeight / total_height if total_height > 0 else 1.0
        scale_x = availWidth / max_width if max_width > 0 else 1.0

        self.scale_factor = min(scale_x, scale_y, 1.0)

        return availWidth, availHeight

    def draw(self):
        """Draw the scaled and positioned child flowables onto the canvas

        Applies the calculated scale factor and horizontal offset (if centering)
        before drawing each child flowable in sequence from top to bottom
        """
        self.canv.saveState()

        # If horizontally centering, compute the leftover space after scaling total_width
        offset_x = 0
        if self.center_content:
            final_width = self.total_width * self.scale_factor
            leftover = self.avail_width - final_width
            if leftover > 0:
                offset_x = leftover / 2.0

        # Translate to horizontal center if leftover > 0
        self.canv.translate(offset_x, 0)

        # Apply vertical scaling (and possibly horizontal if code is wide).
        self.canv.scale(self.scale_factor, self.scale_factor)

        # Draw each child from top to bottom
        y = self.total_height
        for idx, f in enumerate(self.flowables):
            w, h = self.child_sizes[idx]
            y -= h
            f.drawOn(self.canv, 0, y)
            y -= self.spacing

        self.canv.restoreState()


class ProgramOutput:
    """Manage the collection and saving of program outputs

    Handles saving the CFG, generated code, test paths, configuration,
    and generating visualizations and summary reports
    """
    MAX_RUNTIME = 600

    def __init__(self, cfg: CFG, config: AppConfig):
        """Initialize the ProgramOutput instance

        Args:
            cfg (CFG): The Control Flow Graph object
            config (AppConfig): The application configuration object
        """
        self.cfg = cfg
        self.config = config
        self.output_dir: Optional[Path] = None
        self.debug_dir: Optional[Path] = None
        self.run_number: Optional[int] = None
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def _get_next_run_number(self, cfgs_dir: Path) -> int:
        """Helper method to determine the next sequential run number

        Scans a directory for existing 'run<number>' subdirectories and
        returns the next integer in the sequence

        Args:
            cfgs_dir (Path): The directory containing previous run outputs

        Returns:
            int: The next available run number (starting from 1)
        """
        run_dirs = [d for d in cfgs_dir.iterdir() if d.is_dir() and d.name.startswith("run")]
        run_numbers = []
        for d in run_dirs:
            try:
                if d.name.startswith("run") and d.name[3:].isdigit():
                    run_numbers.append(int(d.name[3:]))
            except (ValueError, IndexError):
                pass
        if not run_numbers:
            return 1
        return max(run_numbers) + 1

    def save_to_directory(self, output_dir: Optional[str] = None) -> Optional[Path]:
        """Save all generated program outputs to a specified directory

        Creates the output directory structure and calls individual saving methods
        for CFG, code, test paths, config, and summary PDF

        Args:
            output_dir (Optional[str]): The target directory path. If None,
                a new 'run<number>' directory is created under './output'

        Returns:
            Optional[Path]: The path to the created output directory, or None if
                           saving failed (e.g, CFG is None)
        """
        if self.cfg is None:
            logger.error("Cannot save outputs: CFG object is None.")
            return None

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to ./output/run<N> directory
            project_root = Path.cwd()
            output_base_dir = project_root / "output"
            output_base_dir.mkdir(exist_ok=True)
            self.run_number = self._get_next_run_number(output_base_dir)
            self.output_dir = output_base_dir / f"run{self.run_number}"

        self.output_dir.mkdir(exist_ok=True, parents=True)

        self._save_cfg()
        self._save_code()
        self._save_test_paths()
        self._save_config()
        self._save_global_shtv_details()
        self._visualize_fragment_forest()
        self._generate_pdf_outputs()
        self._generate_summary_pdf()
        self._save_cfg_textual()
        paths_json_data = self._save_test_paths()
        self._generate_and_save_test_suites_txt(paths_json_data)

        return self.output_dir

    def _save_cfg(self):
        """Helper method to save CFG visualization

        Generates and saves a Graphviz visualization of the CFG if possible
        """
        if not self.cfg:
            logger.warning("No CFG to save.")
            return
        if not self.output_dir:
            logger.error("Cannot save CFG: Output directory not set.")
            return

        from .visualizer import visualize_cfg_with_graphviz  
        viz_path = visualize_cfg_with_graphviz(self.cfg, self.output_dir)
        if not viz_path:
            logger.warning("CFG visualization generation failed or was skipped.")

    def _save_code(self):
        """Helper method to save the generated C code to a file

        Writes the code stored in `self.cfg.code` to 'generated.c'
        """
        code_to_save = getattr(self.cfg, 'code', None)
        if not code_to_save:
            logger.warning("No code found in CFG to save.")
            return
        if self.output_dir:
            code_path = self.output_dir / "generated.c"
            with open(code_path, "w", encoding='utf-8') as f:
                f.write(code_to_save)

    def _save_test_paths(self) -> Optional[Dict[str, Any]]:
        """Helper method to save test path information to JSON file"""
        if not self.cfg or not hasattr(self.cfg, 'test_paths') or not self.cfg.test_paths:
            logger.warning("No test paths found in CFG to save.")
            return None
        if not self.output_dir:
            logger.error("Cannot save test paths: Output directory not set.")
            return None

        test_paths_json_path = self.output_dir / "test_paths.json"
        json_string, json_data = serialize_test_paths(self.cfg)

        try:
            with open(test_paths_json_path, "w", encoding='utf-8') as f:
                f.write(json_string)
            return json_data
        except Exception as e:
            logger.error(f"Failed to write test paths JSON to {test_paths_json_path}: {e}")
            return None

    def _save_config(self):
        """Helper method to save a summary of the run configuration to YAML"""
        if not self.output_dir:
            logger.error("Cannot save config: Output directory not set.")
            return

        config_path = self.output_dir / "config.yaml"
        yaml_string = serialize_config(self.config, self.timestamp, self.run_number)

        try:
            with open(config_path, "w", encoding='utf-8') as f:
                f.write(yaml_string)
        except Exception as e:
            logger.error(f"Failed to write config YAML to {config_path}: {e}")

    def _save_cfg_textual(self):
        """Helper method to save CFG structure to a JSON file"""
        if not self.cfg or not self.cfg.graph:
            logger.warning("No CFG graph data to save textually.")
            return
        if not self.output_dir:
            logger.error("Cannot save textual CFG: Output directory not set.")
            return

        cfg_structure_path = self.output_dir / "cfg_structure.json"
        json_string = serialize_cfg_structure(self.cfg)

        try:
            with open(cfg_structure_path, "w", encoding='utf-8') as f:
                f.write(json_string)
        except Exception as e:
            logger.error(f"Failed to write CFG structure JSON to {cfg_structure_path}: {e}")

    def _save_global_shtv_details(self):
        """Helper method to save detailed SHTV calculation components to text"""
        if not self.output_dir:
            logger.error("Cannot save SHTV details: Output directory not set.")
            return
        if not self.cfg or not self.cfg.context:
            logger.error("Cannot save SHTV details: CFG or CFG context is None.")
            return

        shtv_file_path = self.output_dir / "global_shtv.txt"
        shtv_text = format_shtv_details(self.cfg, self.config)

        try:
            with open(shtv_file_path, "w", encoding='utf-8') as f:
                f.write(shtv_text)
        except Exception as e:
            logger.error(f"Failed to write SHTV details to {shtv_file_path}: {e}")

    def _visualize_fragment_forest(self) -> Optional[Path]:
        """Helper method to generate and save a Fragment Forest visualization

        Uses Graphviz to create an SVG representation of the fragment forest
        structure stored in the CFG

        Returns:
            Optional[Path]: The path to the generated SVG file, or None if
                           visualization failed or was skipped
        """
        if not self.output_dir:
            logger.error("Cannot visualize Fragment Forest: Output directory not set.")
            return None
        fragment_forest: Optional[FragmentForest] = getattr(self.cfg, 'fragment_forest', None)
        if not fragment_forest or not hasattr(fragment_forest, '_fragments') or not hasattr(fragment_forest, '_nodes'):
            logger.warning("Fragment Forest data not found or incomplete in CFG. Skipping visualization.")
            return None
        fragments = getattr(fragment_forest, '_fragments', {})
        forest_nodes = getattr(fragment_forest, '_nodes', None)
        if not fragments or not forest_nodes:
            logger.warning("Fragment Forest data is empty. Skipping visualization.")
            return None
        dot = graphviz.Digraph(name='FragmentForest', comment='Fragment Forest Structure')

        # Add nodes for actual fragments
        for frag_id, fragment in fragments.items():
            frag_role = getattr(fragment, 'fragment_role', 'Unknown')
            related_nodes = getattr(fragment, '_related_nodes', None) 
            label = f"ID: {frag_id}\nRole: {str(frag_role).split('.')[-1]}"
            if related_nodes is not None: 
                node_ids_list = [str(getattr(n, 'node_id', '?')) for n in related_nodes]
                label += f"\nNodes: [{', '.join(node_ids_list)} ]"
            dot.node(str(frag_id), label=label, shape='box')

        # Add edges based on the tree structure (parent-child)
        for frag_id, tree_node in forest_nodes.items():
            parent_str = str(frag_id)
            # Add edges for each child in the ordered list
            for child_id in tree_node.child_ids:
                child_str = str(child_id)
                dot.edge(parent_str, child_str)

        ordered_roots = getattr(fragment_forest, '_roots', [])
        if len(ordered_roots) > 1:
            for i in range(len(ordered_roots) - 1):
                u = str(ordered_roots[i])
                v = str(ordered_roots[i+1])
                dot.edge(u, v, style='invis', constraint='false')

        output_png_path = self.output_dir / "fragment_forest.png"
        dot.render(outfile=str(output_png_path), format='png', view=False, cleanup=True)
        return output_png_path

    def _generate_pdf_outputs(self):
        """Helper method to generate separate PDF reports for code, CFG, and test paths"""
        if not self.output_dir:
            logger.error("Cannot generate PDFs: Output directory not set.")
            return

        styles = getSampleStyleSheet()
        
        # Prepare content elements using helper methods
        code_elements = self._prepare_code_elements(styles)
        cfg_drawing = self._prepare_cfg_drawing(styles)
        test_paths_table = self._prepare_test_paths_table(styles)
        fragment_forest_image = self._prepare_fragment_forest_image(styles)

        # Generate individual PDFs
        if cfg_drawing:
            self._generate_cfg_pdf(cfg_drawing)
        if test_paths_table:
            self._generate_test_paths_pdf(test_paths_table)
        if fragment_forest_image:
            self._generate_fragment_forest_pdf(fragment_forest_image)

    def _prepare_code_elements(self, styles) -> List[Paragraph]:
        """Formats the generated C code into a list of Paragraphs for PDF generation"""
        code_elements = []
        code_to_save = self.cfg.code
        if not code_to_save:
             logger.warning("No code found in CFG to generate code PDF.")
             return []

        formatted_code = code_to_save
        clang_format_path = shutil.which('clang-format')
        
        code_style = ParagraphStyle(
            "Code",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=PDF_CODE_FONT_SIZE,
            leading=PDF_CODE_FONT_SIZE + 2,
            firstLineIndent=0
        )
        
        if clang_format_path:
            try:
                process = subprocess.run(
                    [clang_format_path],
                    input=code_to_save,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10 
                )
                formatted_code = process.stdout
            except subprocess.CalledProcessError as e:
                logger.warning(f"clang-format failed (return code {e.returncode}). Using unformatted code for PDF. Error: {e.stderr}")
            except subprocess.TimeoutExpired:
                logger.warning("clang-format timed out. Using unformatted code for PDF.")
            except Exception as e:
                logger.warning(f"An unexpected error during code formatting for PDF: {e}. Using unformatted code.")
        else:
            logger.warning("'clang-format' executable not found in PATH. Skipping formatting for PDF.")

        for line in formatted_code.splitlines():
            line = line.replace('\t', '    ')
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            leading_spaces = len(escaped_line) - len(escaped_line.lstrip(' '))
            non_breaking_spaces = '&nbsp;' * leading_spaces
            final_line = non_breaking_spaces + escaped_line.lstrip(' ')
            code_elements.append(Paragraph(final_line, code_style))
            
        return code_elements

    def _crop_drawing(self, drawing: Drawing) -> Drawing:
        """Trim whitespace by shifting and resizing to the visible bounds"""
        try:
            x_min, y_min, x_max, y_max = drawing.getBounds()  # [x0,y0,x1,y1]
            # shift every child back to (0,0)
            for item in drawing.contents:
                obj = cast(Group, item) # Cast to Group which has translate
                obj.translate(-x_min, -y_min)
            # update the drawing's reported size
            drawing.width  = x_max - x_min
            drawing.height = y_max - y_min
        except Exception as e:
            logger.warning(f"Could not crop SVG drawing: {e}")
        return drawing

    def _prepare_cfg_drawing(self, styles) -> Optional[Any]:
        """Loads the CFG SVG visualization and crops it"""
        if not self.output_dir: return None
        
        cfg_drawing = Paragraph("CFG Visualization (SVG) not generated or not found.", styles["Normal"])
        cfg_svg_path = self.output_dir / "cfg.svg"
        if cfg_svg_path.exists():
            try:
                d = svg2rlg(str(cfg_svg_path))
                if d:
                    cfg_drawing = self._crop_drawing(d) 
                else:
                    logger.warning("svg2rlg returned None for CFG SVG.")
            except Exception as e:
                logger.error(f"Failed to load or parse CFG SVG for PDF: {e}")
                return None 
        else:
            logger.warning(f"CFG SVG file not found at {cfg_svg_path}. Cannot generate CFG PDF.")
            return None 
            
        return cfg_drawing

    def _prepare_fragment_forest_image(self, styles) -> Optional[Image]:
        """Loads the Fragment Forest PNG visualization into a ReportLab Image"""
        if not self.output_dir: return None

        ff_png_path = self.output_dir / "fragment_forest.png"
        if ff_png_path.exists():
            try:
                img = Image(str(ff_png_path))
                return img
            except Exception as e:
                logger.error(f"Failed to load Fragment Forest PNG for PDF: {e}")
                return None
        else:
            logger.warning(f"Fragment Forest PNG file not found at {ff_png_path}. Cannot generate Fragment Forest PDF.")
            return None

    def _prepare_test_paths_table(self, styles) -> Optional[Table]:
        """Creates a ReportLab Table object for test paths with cleaned inputs"""
        if not self.output_dir: return None
        
        test_paths_table = None
        test_paths_to_save = getattr(self.cfg, 'test_paths', [])
        if not test_paths_to_save:
            logger.info("No test paths found in CFG. Skipping test paths PDF.")
            return None

        path_style = ParagraphStyle(
            "PathStyle",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=PDF_PATH_FONT_SIZE,
            leading=PDF_PATH_FONT_SIZE + 2,
            firstLineIndent=0
        )
        
        table_data: List[List[Any]] = [["Test case", "Test path", "Test inputs"]]
        path_counter = 1
        for path in test_paths_to_save:
            suffixed_inputs = getattr(path, 'test_inputs', None)

            if suffixed_inputs is not None: 
                node_ids_int = [getattr(n, 'node_id', -1) for n in getattr(path, 'nodes', [])] 
                valid_node_ids_int = [nid for nid in node_ids_int if nid != -1]
                path_nodes_str_unicode = format_with_loops(valid_node_ids_int) if valid_node_ids_int else "[No Nodes]"
                path_nodes_str_pdf = path_nodes_str_unicode.replace("⁺", "<super>+</super>")
                path_nodes_para = Paragraph(path_nodes_str_pdf, path_style)

                effective_formula = path.get_effective_formula_str() if hasattr(path, 'get_effective_formula_str') else "[Formula Error]"
                formula_para = Paragraph(f"Effective Formula: {effective_formula}", path_style)
                
                inputs_str = json.dumps(suffixed_inputs, separators=(',', ':')) if suffixed_inputs else "[dim]None[/dim]" 
                inputs_para = Paragraph(inputs_str, path_style)
                path_cell_content = [path_nodes_para, formula_para]
                
                table_data.append([str(path_counter), path_cell_content, inputs_para])
                path_counter += 1

        if len(table_data) > 1:
            col_widths = [0.8 * inch, 4.2 * inch, 1.5 * inch]
            test_paths_table = Table(table_data, colWidths=col_widths)
            test_paths_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), (0.9, 0.9, 0.9)),
                ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, (0.7, 0.7, 0.7)),
                ('TOPPADDING', (0, 1), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
            ]))
        else:
             logger.info("No test paths with generated inputs found. Skipping test paths PDF.")
             return None 

        return test_paths_table

    def _generate_cfg_pdf(self, cfg_drawing: Any):
        """Generates a PDF containing the CFG visualization"""
        if not self.output_dir: return
        pdf_path = str(self.output_dir / "cfg.pdf")
        doc = AutoLayoutDocTemplate(pdf_path)
        
        # Scale CFG drawing to fit the page using OnePageContainer
        story = []
        container = OnePageContainer([cfg_drawing], spacing=0, center_content=True)
        story.append(container)
        
        try:
            doc.build(story)
        except Exception as e:
            logger.error(f"Failed to build CFG summary PDF: {e}", exc_info=True)

    def _generate_test_paths_pdf(self, test_paths_table: Table):
        """Generates a PDF containing the test paths table"""
        if not self.output_dir: return
        pdf_path = str(self.output_dir / "test_paths.pdf")
        doc = AutoLayoutDocTemplate(pdf_path)
        # Cast list with single Table to list of Flowables
        story: List[Flowable] = cast(List[Flowable], [test_paths_table]) 
        
        try:
            doc.build(story)
        except Exception as e:
            logger.error(f"Failed to build test paths summary PDF: {e}", exc_info=True)

    def _generate_fragment_forest_pdf(self, fragment_forest_image: Image):
        """Generates a PDF containing the Fragment Forest visualization"""
        if not self.output_dir: return
        pdf_path = str(self.output_dir / "fragment_forest.pdf")
        doc = AutoLayoutDocTemplate(pdf_path)

        story = []
        container = OnePageContainer([fragment_forest_image], spacing=0, center_content=True)
        story.append(container)

        try:
            doc.build(story)
        except Exception as e:
            logger.error(f"Failed to build Fragment Forest summary PDF: {e}", exc_info=True)

    def _generate_summary_pdf(self):
        """Helper method to generate a multi-page combined summary PDF report

        Layout adapts based on `config.test_input_complexity`:
        - Levels 1-2: 1 page (Code+CFG/Paths)
        - Levels 3-4: 2 pages (CFG+Paths | Code)
        """
        if not self.output_dir:
            logger.error("Cannot generate combined PDF: Output directory not set.")
            return

        pdf_path = str(self.output_dir / "summary.pdf")
        styles = getSampleStyleSheet()
        normal_style = styles["Normal"]

        code_elements = self._prepare_code_elements(styles)
        cfg_drawing = self._prepare_cfg_drawing(styles)
        test_paths_table = self._prepare_test_paths_table(styles)
        fragment_forest_image = self._prepare_fragment_forest_image(styles)
        
        story: List[Flowable] = []
        available_width = A4[0] - 2 * inch
        available_height = A4[1] - 2 * inch

        if not code_elements:
            code_elements = [Paragraph("Code not available.", normal_style)]
        if not isinstance(cfg_drawing, Flowable): 
            cfg_drawing = Paragraph("CFG Visualization not available.", normal_style)

        doc = AutoLayoutDocTemplate(pdf_path)
        # Page 1: CFG -> Test Paths
        page1_content: List[Flowable] = []
        scaled_cfg_drawing = cfg_drawing 

        # Scaling Logic for CFG on Page 1
        # Only attempt scaling if cfg_drawing is a Drawing with valid dimensions
        if isinstance(cfg_drawing, Drawing) and hasattr(cfg_drawing, 'width') and hasattr(cfg_drawing, 'height') and cfg_drawing.width > 0 and cfg_drawing.height > 0:
            page_content_height = available_height * 0.8 
            scale_x = available_width / cfg_drawing.width
            scale_y = page_content_height / cfg_drawing.height
            scale_factor = min(scale_x, scale_y, 1.0) 
            
            if scale_factor < 1.0: 
                try:
                    cfg_drawing.scale(scale_factor, scale_factor)
                    cfg_drawing.width *= scale_factor
                    cfg_drawing.height *= scale_factor
                    scaled_cfg_drawing = cfg_drawing 
                except Exception as e:
                    scaled_cfg_drawing = cfg_drawing 
                    logger.error(f"Error scaling CFG SVG (combined PDF, page 1): {e}")
        
        elif not isinstance(cfg_drawing, Drawing):
            logger.warning("CFG drawing is not a Drawing object, cannot apply scaling (combined PDF page 1).")
            
        page1_content.append(scaled_cfg_drawing)
        page1_content.append(Spacer(1, 0.15 * inch))
        if test_paths_table:
            page1_content.append(test_paths_table)
        else:
            page1_content.append(Paragraph("No test paths with inputs generated.", normal_style))
        
        story.append(OnePageContainer(page1_content, spacing=6, center_content=True)) 
        story.append(PageBreak())
        story.extend(cast(List[Flowable], code_elements)) 

        try:
            if story: 
                doc.build(story)
            else:
                logger.warning("Skipping combined summary PDF build because the story is empty.")
        except Exception as e:
            logger.error("Failed to build combined PDF document: {}", e, exc_info=True)
            if self.output_dir:
                error_code_path = self.output_dir / "generated_code_on_combined_pdf_error.c"
                if code_elements and hasattr(code_elements[0], 'text'): 
                    try:
                        raw_code = "\\n".join(p.text for p in code_elements) 
                        with open(error_code_path, "w", encoding='utf-8') as f:
                            f.write(raw_code) 
                        logger.info(f"Saved approximate raw code to {error_code_path} due to combined PDF generation error.")
                    except Exception as code_save_e:
                        logger.error(f"Failed to save code during combined PDF error handling: {code_save_e}")

    # Updated method to generate test_suites.txt using the formatter
    def _generate_and_save_test_suites_txt(self, paths_json_data: Optional[Dict[str, Any]]):
        """Generate test suites text using formatter and save to test_suites.txt"""
        if not self.output_dir:
            logger.error("Cannot save test suites txt: Output directory not set.")
            return
        if not paths_json_data:
             logger.warning("No test paths data provided (from _save_test_paths). Skipping test suites text.")
             return

        test_suites_text = format_test_suites(paths_json_data)

        test_suites_txt_path = self.output_dir / "test_suites.txt"
        try:
            with open(test_suites_txt_path, "w", encoding='utf-8') as f:
                f.write(test_suites_text)
            logger.info(f"Successfully wrote test suites to {test_suites_txt_path}")
        except Exception as e:
             logger.error(f"Failed to write {test_suites_txt_path}: {e}")

