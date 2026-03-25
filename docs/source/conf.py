import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


from pathsim import __version__ 

# -- Project information -----------------------------------------------------

project = 'pathsim'
copyright = '2025, Milan Rother' 
author = 'Milan Rother'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

sys.path.insert(0, os.path.abspath('./_ext'))

extensions = [
    'sphinx.ext.autodoc',  # Core Sphinx library for auto doc generation
    'sphinx.ext.napoleon', # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.mathjax', # Render math
    'myst_parser',          # Support for MyST Markdown (optional, but recommended)
	'sphinx.ext.autosummary', # Create neat summary tables,
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_design',  # Modern design components (cards, tabs, grids)
    'nbsphinx',  # Jupyter notebook support
    'github_issues', # Automatic roadmap generation from active github issues
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_logo = 'logos/pathsim_logo.png'
html_favicon = "logos/pathsim_icon.png"
html_title = "PathSim Documentation"
html_css_files = ['custom.css']  # Add custom CSS for link previews and styling
html_js_files = ['redirect.js']  # Auto-redirect RTD visitors to docs.pathsim.org
html_baseurl = 'https://docs.pathsim.org/'  # Canonical URL for SEO

html_theme_options = {
    "light_css_variables": {
        # PathSim brand colors - using blue from the palette
        "color-brand-primary": "#377eb8",  # PathSim blue
        "color-brand-content": "#377eb8",  # PathSim blue for links

        # Accent colors for various elements using PathSim palette
        "color-api-keyword": "#377eb8",  # PathSim blue for keywords
        "color-highlight-on-target": "#fff3cd",  # Soft yellow highlight

        # Font stacks
        "font-stack": "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif",
        "font-stack--monospace": "SFMono-Regular, Menlo, Consolas, Monaco, Liberation Mono, Lucida Console, monospace",
    },
    "dark_css_variables": {
        # PathSim brand colors for dark mode - slightly lighter for better contrast
        "color-brand-primary": "#377eb8",  # PathSim blue
        "color-brand-content": "#377eb8",  # PathSim blue

        # Accent colors for dark mode
        "color-api-keyword": "#377eb8",  # PathSim blue
    },
    "sidebar_hide_name": True,  # Hide project name, show logo only
    "navigation_with_keys": True,  # Allow keyboard navigation
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/milanofthe/pathsim",
    "source_branch": "master",
    "source_directory": "docs/source/",
}

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,       # Document all members (functions, classes, methods)
    'member-order': 'bysource', # Order members as they appear in the source code
    'undoc-members': True,  # Include members that don't have docstrings
    'show-inheritance': True, # Show base classes
}
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# -- Options for MyST Parser -----------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_update_mathjax = False
myst_heading_anchors = 3  
myst_url_schemes = ("http", "https") 
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]

# Add support to link variables in other projects, used in the docstrings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for nbsphinx -----------------------------------------------

# Execute notebooks before conversion
nbsphinx_execute = 'always'  # 'always', 'never', or 'auto' (only if no output)

# Timeout for notebook execution (in seconds)
nbsphinx_timeout = 180

# Allow errors in notebooks (useful during development)
nbsphinx_allow_errors = True

# Kernel to use for notebook execution
nbsphinx_kernel_name = 'python3'

# Custom CSS for notebooks in Furo theme
nbsphinx_prolog = """
.. raw:: html

    <style>
        /* Make notebook outputs fit better with Furo theme */
        .nboutput .output_area pre {
            background-color: var(--color-background-secondary);
            border: 1px solid var(--color-background-border);
            border-radius: 0.25rem;
            padding: 0.5rem;
        }

        /* Transparent figure backgrounds */
        .nboutput img {
            background-color: transparent !important;
        }

        /* Better styling for stderr/logging output - adapts to light/dark mode */
        .nboutput .stderr {
            background-color: transparent !important;
            color: var(--color-foreground-secondary) !important;
            border-left: 3px solid var(--color-brand-primary);
            opacity: 0.85;
        }
    </style>
"""

# Exclude certain notebook patterns from execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'png'}",
    "--InlineBackend.rc={'figure.dpi': 200}",
]
