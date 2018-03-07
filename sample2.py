from flask import Flask, render_template
from bokeh.embed import components, server_document
from bokeh.models.widgets import Paragraph, Button, Div
from bokeh.layouts import column
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello, World!'


@app.route('/bokeh')
def bokeh():

    but = Button(label="click me", width = 80, height = 10, button_type = 'primary')
    
    def onclick():
        if par.text == "":
            par.text = "Hello! World"
        else:
            par.text = ""
        
    but.on_click(onclick)
    par = Paragraph(text="", width = 80, height = 61, style = {'background-color':'#F2F3F4'}) 
    grid = column(but, Div(text = "", width = 10, height = 100), par)

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(grid)
#    script = server_document(url="http://localhost:5006/sample")
    html = render_template('index.html', plot_script=script, 
                           div=div, js_resources=js_resources, css_resources=css_resources)
    return encode_utf8(html)


if __name__ == '__main__':
    app.run(debug = True)

