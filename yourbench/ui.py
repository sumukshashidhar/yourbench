"""
UI module for Yourbench containing all Gradio and banner-related functionality.
"""
import os
import html
import base64
import gradio as gr


def inline_svg_to_data_url(svg_path: str) -> str:
    """
    Reads an SVG from disk and returns a data URL suitable for embedding directly 
    in a <source> or <img> tag. 
    """
    try:
        with open(svg_path, "rb") as f:
            svg_bytes = f.read()
        b64 = base64.b64encode(svg_bytes).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"
    except FileNotFoundError:
        return ""


def load_logo_banner(
    light_logo_path: str,
    dark_logo_path: str,
    alt_text: str = "Yourbench Logo",
    image_width: int = 200,
    max_width: int = 300
) -> str:
    """
    Generate an HTML snippet that displays a responsive banner with dark and light mode support.
    This version inlines SVG content as data URLs, which avoids broken references to local files.

    Args:
        light_logo_path: Path to the light mode logo SVG file
        dark_logo_path: Path to the dark mode logo SVG file
        alt_text: Alternative text for the image
        image_width: Width of the image in pixels
        max_width: Maximum width of the container in pixels

    Returns:
        str: HTML snippet containing the responsive banner
    """
    # If the light logo doesn't exist, we can't do anything
    if not os.path.exists(light_logo_path):
        return f"<p>Light mode logo file not found: {light_logo_path}</p>"

    # If the dark logo doesn't exist, fall back to the light logo
    if not os.path.exists(dark_logo_path):
        dark_logo_path = light_logo_path

    # Convert both images to data URLs
    light_logo_url = inline_svg_to_data_url(light_logo_path)
    dark_logo_url = inline_svg_to_data_url(dark_logo_path)

    # If for some reason the read fails, inline_svg_to_data_url() returns an empty string.
    # In that case, just revert to a small text fallback to avoid blank output.
    if not light_logo_url:
        return f"<p>Unable to load {light_logo_path}</p>"
    if not dark_logo_url:
        dark_logo_url = light_logo_url

    # Return an HTML snippet using <picture> so that dark mode is detected automatically
    html_snippet = f"""
    <div style="max-width:{max_width}px; margin:auto;" align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="{dark_logo_url}">
        <source media="(prefers-color-scheme: light)" srcset="{light_logo_url}">
        <!-- Fallback to the light version if color-scheme media query isn't supported -->
        <img 
           alt="{html.escape(alt_text)}" 
           src="{light_logo_url}" 
           width="{image_width}" 
           style="max-width:100%; height:auto;"
        />
      </picture>
    </div>
    """

    return html_snippet


def launch_ui() -> None:
    """
    Launch a minimal Gradio UI that displays a responsive banner with dark/light mode support.
    
    The banner is generated using the load_logo_banner function, which creates a <picture>
    element with appropriate <source> tags. The light and dark mode images are expected to be located at:
      - Light mode: docs/assets/yourbench_banner_light_mode.svg
      - Dark mode: docs/assets/yourbench_banner_dark_mode.svg
    """
    light_logo_path = "docs/assets/yourbench_banner_light_mode.svg"
    dark_logo_path = "docs/assets/yourbench_banner_dark_mode.svg"
    logo_html = load_logo_banner(light_logo_path, dark_logo_path)
    
    # Create Gradio Blocks UI with a single HTML component showing the banner
    with gr.Blocks() as demo:
        gr.HTML(value=logo_html)
    
    # Launch the UI on all network interfaces
    demo.launch(server_name="0.0.0.0") 