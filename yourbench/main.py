"""
Universal main.py script for ui and cli

Usage:
    python main.py --gui # launches the gradio ui
    python main.py       # runs in cli mode
"""
import argparse
import os
import gradio as gr

def load_logo(logo_path: str, max_width: int = 300) -> str:
    """
    Load a logo file (SVG or other image format) and return an HTML snippet 
    that displays it resized as a banner.
    """
    ext = os.path.splitext(logo_path)[1].lower()
    if ext == ".svg":
        try:
            with open(logo_path, "r", encoding="utf-8") as f:
                logo_content = f.read()
            # For SVGs, insert a style block to override intrinsic sizing.
            logo_html = f'''
            <div style="max-width:{max_width}px; margin:auto;">
                <style>
                    svg {{ width: 100% !important; height: auto !important; }}
                </style>
                {logo_content}
            </div>
            '''
        except FileNotFoundError:
            logo_html = "<p>SVG file not found.</p>"
    else:
        # For raster images, simply use an <img> tag.
        if os.path.exists(logo_path):
            logo_html = f'''
            <div style="max-width:{max_width}px; margin:auto;">
                <img src="{logo_path}" style="width:100%; height:auto;" alt="Logo">
            </div>
            '''
        else:
            logo_html = "<p>Logo file not found.</p>"
    return logo_html

def launch_ui() -> None:
    """
    Launch a minimal Gradio UI that displays a logo.
    
    The logo is loaded from docs/assets/banner_logo.svg (or another supported format)
    and wrapped in a container that limits its maximum width for consistent display.
    """
    logo_path = "docs/assets/banner_logo.svg"
    logo_html = load_logo(logo_path)
    
    # Create Gradio Blocks UI with a single HTML component showing the logo
    with gr.Blocks() as demo:
        gr.HTML(value=logo_html)
    
    # Launch the UI on all network interfaces
    demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic Evaluation Set Generation with Large Language Models"
    )
    parser.add_argument("--gui", action="store_true", help="Launch the gradio UI")
    args = parser.parse_args()

    if args.gui:
        launch_ui()
    else:
        # CLI mode logic can be added here
        print("Running in CLI mode. No GUI is launched.")