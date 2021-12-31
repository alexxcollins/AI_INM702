#!/usr/bin/env python
# coding: utf-8


from traitlets.config import Config
import nbformat as nbf
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor


def save_html(read_name, write_name, write_path='html_files'):
    '''
    convert a notebook into html file.
    read_name is name of current notebook. If running in same folder, just need the name - no extension. If running in another file need relative or full path.
    write_name is name of html file to write to - no extension.
    write_path, optional. If you want to write to a path other than relative path 'html_files' specify here.
    
    '''
    
    read_name = r'{}.ipynb'.format(read_name)
    write_name = r'{}/{}.html'.format(write_path, write_name)
    
    c = Config()

    # Configure our tag removal
    c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
    c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)
    c.TagRemovePreprocessor.enabled = True

    # Configure and run out exporter
    c.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

    exporter = HTMLExporter(config=c)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c),True)

    # Configure and run out exporter - returns a tuple - first element with html, second with notebook metadata
    output = HTMLExporter(config=c).from_filename(read_name)

    # Write to output html file
    with open(write_name,
              "w", errors='ignore') as f:
        f.write(output[0])
    
    return None
