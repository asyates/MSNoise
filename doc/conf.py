# -*- coding: utf-8 -*-
#
# MSNoise documentation build configuration file, created by
# sphinx-quickstart on Wed Oct 02 13:37:02 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
import os

import matplotlib

matplotlib.use('Agg')

import msnoise.move2obspy
import msnoise.preprocessing
import sphinx_bootstrap_theme
import datetime
import click

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
from msnoise.default import default

grid = [ [key, default[key][0], default[key][1]] for key in default.keys() ]

numcolumns = len(grid[0])
colsizes = [max(len(r[i]) for r in grid) for i in range(numcolumns)]
formatter = ' '.join('{:<%d}' % c for c in colsizes)
rowsformatted = [formatter.format(*row) for row in grid]
headformatted = formatter.format(*['Parameter Name', 'Description', 'Default Value'])
header = formatter.format(*['=' * c for c in colsizes])

output = header +'\n' + headformatted +'\n' + header + '\n' + '\n'.join(rowsformatted) + '\n' + header

f = open('defaults.rst','w')
f.write(output)
f.close()

output = ""
for key in default.keys():
    descr, defvalue = default[key]
    output += ".. |%s| replace:: ``%s``: %s (default=%s)\n" % (key, key, descr,
                                                            defvalue)

#output = output.replace('*','&#42;')
f = open('configs.hrst','w')
f.write(output)
f.close()

space = " " *4
# Generate the help files

from msnoise.scripts import msnoise as M
if not os.path.isdir("clickhelp"):
    os.makedirs("clickhelp")

def write_click_help(group='', command='', data=''):
    out = ".. code-block:: sh\n"
    out += "\n"
    if group:
        f = open('clickhelp/msnoise-%s-%s.rst'% (group, command), 'w')
        out += space+"msnoise %s %s --help" % (group, command)
    else:
        f = open('clickhelp/msnoise-%s.rst'% (command), 'w')
        out += space+"msnoise %s --help" % command
    out += "\n"
    out += "\n"
    for line in data.split('\n'):
        line = line.replace('\r','').replace('\n','')
        out += space+line+"\n"
    f.write(out)
    f.close()
    return out

out = open('clickhelp/msnoise.rst', 'w')
out.write('Help on the msnoise commands\n')
out.write('============================\n\n')
out.write('This page shows all the command line interface commands\n\n')
C = M.cli.commands
for command in sorted(C):
    group = ""
    if hasattr(C[command], "group"):
        group = command
        out.write("\n")
        out.write("\n")
        out.write("------------")
        out.write("\n")
        out.write("\n")
        fullgroup = "msnoise %s" % group
        out.write('%s\n'%fullgroup)
        out.write('-'*len(fullgroup)+'\n')
        out.write("\n")
        if command in ["plugin", "p"]:
            out.write(
                "Will be automatically populated with the commands declared "
                "by the plugins (`p` is an alias for `plugin`)\n\n")
            continue
        CC = C[command].commands
        for command in sorted(CC):
            fullcommand = "msnoise %s %s" % (group, command)
            out.write('%s\n'%fullcommand)
            out.write('~'*len(fullcommand)+'\n')

            c = click.Context(command=eval('M.%s'%(command)))
            data = c.get_help()
            out.write(write_click_help(group, command, data))
            # out.write('.. include:: msnoise-%s-%s.rst\n\n'%(group,command))
            out.write("\n\n")
        out.write("\n")
        out.write("\n")
        out.write("------------")
        out.write("\n")
        out.write("\n")

    else:
        fullcommand = "msnoise %s" % command
        out.write('%s\n'%fullcommand)
        out.write('-'*len(fullcommand)+'\n')
        c = click.Context(command=eval('M.%s'%command))
        data = c.get_help()
        out.write(write_click_help(group, command, data))
        # out.write('.. include:: msnoise-%s.rst\n\n'%(command))
        out.write("\n\n")
out.close()

out = open('contributors.rst', 'w')
out.write('Contributors\n')
out.write('============\n\n')
out.write('The following poeple have contributed to MSNoise (sorted '
          'alphabetically):\n\n')

cont = open("../CONTRIBUTORS.txt",'r')
for line in cont.readlines():
    out.write("* "+ line)
out.close()
# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.intersphinx',
              'sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'numpydoc',]
math_number_all = False
todo_include_todos = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['.templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'MSNoise'
copyright = u'%i, Lecocq, Caudron & the MSNoise devs' % datetime.datetime.now().year
# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = 'master'# The full version, including alpha/beta/rc tags.
release = 'master'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['.build', 'clickhelp/msnoise-*.rst']

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []
autodoc_member_order = 'bysource'

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'sphinxdoc'
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}
html_theme_options = {'bootswatch_theme': "lumen",
                    "navbar_site_name":"Menu",
                    "navbar_sidebarrel":True,
                    "source_link_position":"footer",
                    }

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "MSNoise | A Python Package for Monitoring Seismic Velocity Changes using Ambient Seismic Noise"

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = ".static/msnoise.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = ".static/favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['.static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%Y-%m-%d %H:%M'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'MSNoisedoc'


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    'preamble': '''
\setcounter{tocdepth}{2}
''',
    # disable font inclusion
    'fontpkg': '',
    'fontenc': '',
    # Fix Unicode handling by disabling the defaults for a few items
    # set by sphinx
    'inputenc': '',
    'utf8extra': '',
    'papersize': 'a4paper',
    'pointsize': '11pt',
}


# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'MSNoise.tex', u'MSNoise Documentation',
   u'Thomas Lecocq, Corentin Caudron and MSNoise Devs', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = r".static/msnoise_logo_large.png"

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = True

# If true, show page references after internal links.
latex_show_pagerefs = True

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
latex_domain_indices = False

# latex_show_urls = 'footnote'

pdf_documents = [
    ('index', u'msnoise', u'Msnoise Documentation', u'Lecocq, Caudron'),
]
# A comma-separated list of custom stylesheets. Example:
pdf_stylesheets = ['sphinx', 'a4']
# A list of folders to search for stylesheets. Example:
pdf_style_path = ['_styles']
pdf_toc_depth = 4
pdf_fit_mode = "shrink"
pdf_break_level = 1
pdf_verbosity = 0
pdf_use_modindex = False


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'msnoise', u'MSNoise Documentation',
     [u'Lecocq, Caudron, Brenguier'], 1)
]

intersphinx_mapping = {
'python': ('https://docs.python.org/2.7/', None),
'numpy': ('http://docs.scipy.org/doc/numpy/', None),
'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
'matplotlib': ('http://matplotlib.org/', None),
'sqlalchemy': ('http://docs.sqlalchemy.org/en/latest/', None),
'click': ('http://click.pocoo.org/5/', None,),
'obspy': ('http://docs.obspy.org', None),
'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None)
}

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'MSNoise', u'MSNoise Documentation',
   u'Lecocq, Caudron, Brenguier', 'MSNoise',
   'A Python Package for Monitoring Seismic Velocity Changes using Ambient Seismic Noise.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'
